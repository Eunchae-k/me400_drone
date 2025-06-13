#!/usr/bin/env python3
import time
import csv
from datetime import datetime
from typing import List, Tuple, Optional, Any

import cv2
import numpy as np
import smbus
from picamera2 import Picamera2
from libcamera import controls
from gpiozero import AngularServo, Servo
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite

# === Configuration ===
DEVICE_NAME: str = "/dev/ttyUSB0"
BAUDRATE: int = 1_000_000
PROTOCOL_VERSION: float = 2.0
LEFT_ID: int = 2
RIGHT_ID: int = 1
WHEEL_RADIUS: float = 0.09  # m
WHEEL_SEPARATION: float = 0.24  # m

FRAME_SIZE: Tuple[int, int] = (640, 480)
CROP_HEIGHT: int = 250
VIDEO_FPS: float = 20.0

ESC_PIN: int = 17  # GPIO PWM pin for ESC
SERVO_PIN: int = 14  # GPIO pin for latch servo
SERVO_MIN_PULSE: float = 0.0005  # seconds
SERVO_MAX_PULSE: float = 0.0025  # seconds
LATCH_ANGLE: int = 0  # degrees
RELEASE_ANGLE: int = 180  # degrees

CSV_HEADER: List[str] = [
    "Timestamp",
    "P0",
    "P1"
]

MPU6050_ADDR: int = 0x6B
ACCEL_REG: int = 0x3B
GYRO_REG: int = 0x43


# === Hardware Helper Classes ===

class IMUSensor:
    def __init__(self, bus_id: int = 1) -> None:
        self.bus = smbus.SMBus(bus_id)
        # wake up MPU6050
        self.bus.write_byte_data(MPU6050_ADDR, 0x6B, 0)

    def _read_word(self, reg: int) -> int:
        hi: int = self.bus.read_byte_data(MPU6050_ADDR, reg)
        lo: int = self.bus.read_byte_data(MPU6050_ADDR, reg + 1)
        val: int = (hi << 8) | lo
        return val - 65536 if val & 0x8000 else val

    def read_accel(self) -> Tuple[float, float, float]:
        return tuple(self._read_word(ACCEL_REG + 2 * i) / 16384.0 for i in range(3))

    def read_gyro(self) -> Tuple[float, float, float]:
        return tuple(self._read_word(GYRO_REG + 2 * i) / 131.0 for i in range(3))


class MotorController:
    ADDR_TORQUE_ENABLE: int = 64
    ADDR_OPERATING_MODE: int = 11
    ADDR_GOAL_VELOCITY: int = 104
    OPER_MODE_VELOCITY: int = 1
    TORQUE_ENABLE: int = 1
    TORQUE_DISABLE: int = 0
    MAX_VELOCITY: int = 265 

    def __init__(
        self,
        device: str = DEVICE_NAME,
        baud: int = BAUDRATE,
        ids: Tuple[int, int] = (LEFT_ID, RIGHT_ID),
    ) -> None:
        self.port = PortHandler(device)
        self.packet = PacketHandler(PROTOCOL_VERSION)
        if not (self.port.openPort() and self.port.setBaudRate(baud)):
            raise RuntimeError(f"Cannot open {device}@{baud}")
        # set mode & enable torque
        for id_ in ids:
            self.packet.write1ByteTxRx(
                self.port, id_, self.ADDR_OPERATING_MODE, self.OPER_MODE_VELOCITY
            )
            self.packet.write1ByteTxRx(
                self.port, id_, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE
            )
        self.group = GroupSyncWrite(self.port, self.packet, self.ADDR_GOAL_VELOCITY, 4)
        self.ids = ids

    def set_speeds(self, left_rpm, right_rpm):
        def to_bytes(value):
            v = int(value) & 0xFFFFFFFF
            return [v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF]

        self.group.clearParam()
        self.group.addParam(self.ids[0], bytes(to_bytes(left_rpm)))
        self.group.addParam(self.ids[1], bytes(to_bytes(right_rpm)))
        self.group.txPacket()

    def stop(self) -> None:
        for id_ in self.ids:
            self.packet.write1ByteTxRx(
                self.port, id_, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE
            )
        self.port.closePort()

    def compute_rpms(self, angular_z):
        rot = ((angular_z * WHEEL_SEPARATION) / (2 * WHEEL_RADIUS)) / (2 * np.pi) * 60
        lin = self.MAX_VELOCITY - abs(rot)
        left = max(min(lin + rot, self.MAX_VELOCITY), -self.MAX_VELOCITY)
        right = max(min(lin - rot, self.MAX_VELOCITY), -self.MAX_VELOCITY)
        return left, right

class Latch:
    def __init__(self, pin: int = SERVO_PIN) -> None:
        self.servo = AngularServo(
            pin,
            min_angle=0,
            max_angle=180,
            min_pulse_width=SERVO_MIN_PULSE,
            max_pulse_width=SERVO_MAX_PULSE,
        )
        self.release()

    def latch(self) -> None:
        self.servo.angle = LATCH_ANGLE

    def release(self) -> None:
        self.servo.angle = RELEASE_ANGLE  

# === Pole‐Detection ===

class PoleDetector:
    def __init__(
        self,
        rough_thresh: float = 0.05,
        blur_ksize: int = 5, 
        max_height: int = CROP_HEIGHT,
    ) -> None:
        self.rough_thresh = rough_thresh
        self.blur_ksize = blur_ksize | 1  # ensure odd
        self.max_height = max_height

    @staticmethod
    def compute_smoothness(gray: np.ndarray, blur_ksize: int) -> np.ndarray:
        gray_lp: np.ndarray = cv2.blur(gray, (1, blur_ksize))
        diff_vert: np.ndarray = np.square(np.diff(gray_lp.astype(np.int16), axis=0))
        raw: np.ndarray = diff_vert.sum(axis=0).astype(np.float32)
        mn, mx = raw.min(), raw.max()
        return (raw - mn) / (mx - mn + 1e-6)

    @staticmethod
    def find_blobs(mask: np.ndarray) -> List[Tuple[int, int]]:
        blobs: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                blobs.append((start, i - 1))
                start = None
        if start is not None:
            blobs.append((start, len(mask) - 1))
        return blobs

    @staticmethod
    def select_best_blob(
        blobs: List[Tuple[int, int]], gray: np.ndarray
    ) -> Optional[Tuple[int, int]]: 
        best_score: float = 0.0
        best_blob: Optional[Tuple[int, int]] = None
        for start, end in blobs:
            width: int = end - start + 1
            mean_int: float = gray[:, start : end + 1].mean() / 255.0
            darkness: float = 1.0 - mean_int
            score: float = width * darkness
            if score > best_score:
                best_score, best_blob = score, (start, end)
        return best_blob

    def process_frame(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        crop: np.ndarray = frame[: self.max_height]
        gray: np.ndarray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        smooth: np.ndarray = self.compute_smoothness(gray, self.blur_ksize)
        mask: np.ndarray = smooth < self.rough_thresh
        blobs: List[Tuple[int, int]] = self.find_blobs(mask)
        return self.select_best_blob(blobs, gray)

    @staticmethod
    def overlay_edges(
        frame: np.ndarray, edges: Tuple[Optional[int], Optional[int]]
    ) -> np.ndarray:
        vis: np.ndarray = frame.copy()
        h, _ = vis.shape[:2]
        start, end = edges
        if start is not None and end is not None:
            mid: int = (start + end) // 2
            for x, col in [
                (start, (0, 255, 0)),
                (end, (0, 255, 0)),
                (mid, (255, 0, 0)),
            ]:
                cv2.line(vis, (x, 0), (x, h - 1), col, 2)
        return vis

# === PID Controller ===
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0.0
        self.integral = 0.0

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output
        
# === Ascending ===
def ascending(esc, latch):
    esc.value = -1  # Initialize ESC value
    esc_value  = -1.0
    
    print("Start ascending")
    
    # Gradually increase ESC value
    while esc_value < 0.8:
        time.sleep(0.1) #.1 
        esc_value += 0.1 #.2
        if esc_value > 0.8: esc_value = 0.8
        esc.value = esc_value
        print(f"ESC value: {esc_value:.2f}")

    print("Maintaining ESC value at 0.8. Press 'q' to stop.")
    esc.value = 0.8
    
    # Decelerate ESC value to -1.0 (double stage)
    user_input = input("Press enter to slow down: ")
    while esc_value > 0.2:
        esc_value -= 0.2
        if esc_value < 0.2 : esc_value = 0.2
        esc.value = esc_value
        print(f"ESC value: {esc_value:.2f}")
        time.sleep(0.1)
    
    esc.value = 0.2
    print("ESC slowed to 0.2(60%). Press Enter to stop completely...")

    input_str = input()
    while esc_value > -1.0:
        esc_value -= 0.1
        if esc_value < -1.0:
            esc_value = -1.0
        esc.value = esc_value
        print(f"ESC value: {esc_value:.2f}")
        time.sleep(0.2)
    esc.value = -1.0
    time.sleep(1)
    latch.release()
    print("ESC stopped. Value set to -1.")

    
# === CSV Logging ===
def create_csv(path: str) -> None:
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)


def append_csv(path: str, row: List[Any]) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


# === Main Routine ===

def main() -> None:
    now: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path: str = f"pole_data_{now}.csv"
    vid_path: str = f"pole_vid_{now}.mp4"
    create_csv(csv_path)

    fourcc: int = cv2.VideoWriter_fourcc(*"MP4V")
    vw: cv2.VideoWriter = cv2.VideoWriter(vid_path, fourcc, VIDEO_FPS, (640, 250))

    cam: Picamera2 = Picamera2()
    cam.set_controls({"AeExposureMode": controls.AeExposureModeEnum.Short})
    cfg = cam.create_preview_configuration(
        main={"format": "RGB888", "size": FRAME_SIZE}
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(0.1)

    motors: MotorController = MotorController()
    latch: Latch = Latch()
    detector: PoleDetector = PoleDetector()
    pid = PIDController(Kp=10.0, Ki=0.3, Kd=0.1)
    esc = Servo(ESC_PIN)
    esc.value = -1.0
    detect = 0


    print("Press Enter to start, 'q' to quit.")
    if input().lower() == "q":
        return
    last_time = time.time()

    try:
        while True:
            frame: np.ndarray = cam.capture_array()[:CROP_HEIGHT]

            pole = detector.process_frame(frame)
            print(pole)
            if pole is None:
                continue
            p0, p1 = pole
            mid = (p0 + p1) // 2.0
            thickness = p1 - p0

            frame = detector.overlay_edges(frame, pole)
            vw.write(frame)

            ts = datetime.now().isoformat()
            row: List[Any] = [ts, p0, p1]
            append_csv(csv_path, row)

            mid: float = (p0 + p1) / 2.0
            fw: int = FRAME_SIZE[0]
            err: float = -((mid - fw / 2) / (fw / 2)) 
            now = time.time()
            dt = now - last_time
            last_time = now

            angular_z = pid.compute(err, dt)
            left_rpm, right_rpm = motors.compute_rpms(angular_z)
            motors.set_speeds(left_rpm, right_rpm)

            if (p1 - p0) > 390 and (p1 - p0)<450 and p0>0: 
                print("Pole reached → latching")
                motors.set_speeds(motors.MAX_VELOCITY, motors.MAX_VELOCITY)
                latch.latch()
                time.sleep(0.3)
                print("Stop driving wheel")
                motors.stop()
                ascending(esc, latch)
                break

    finally:
        latch.release()
        cam.stop()
        vw.release()

if __name__ == "__main__":
    main()
