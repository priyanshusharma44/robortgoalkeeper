# goalkeeper.py — Robot Goalkeeper v7.0 — INSTANT DIVE + MANUAL RESET
#
#  pip install ultralytics opencv-python filterpy pyserial numpy
#
#  Usage:
#    python goalkeeper.py --source 0 --port COM8
#    python goalkeeper.py --source 0 --no-motor
#    python goalkeeper.py --source video.mp4 --no-motor
#
#  Calibration:
#    python goalkeeper.py --source 0 --calibrate
#
#  Runtime keys:
#    Q/Esc  quit
#    H      home motor (manual, goes to 90°)
#    C      flip direction mapping
#    R      reset tracking
#    +/-    move reaction zone up/down
#    S      save calibration
#    D      toggle debug overlay
#
#  v7.0 CHANGES vs v6.0:
#    1. INSTANT dive — angle sent immediately, no rate-limit on dive moves
#    2. NO auto-return — motor stays wherever it dove, never nudges back alone
#    3. POST-SAVE COOLDOWN — after a dive save, system freezes for SAVE_COOLDOWN_S
#       (default 10s). During cooldown: detection paused, motor locked, HUD shows
#       countdown. After cooldown: detection resumes fresh (assumes human reset GK).
#    4. DIVE CONFIRM — must see ball approaching for DIVE_CONFIRM_FRAMES frames
#       before committing, eliminating false dives from noise.
#    5. Removed LooseGearReturn entirely — simpler, more reliable.
#    6. Kalman filter tuned for faster ball tracking (lower process noise)
#    7. Serial command rate-limit removed for dive path — commands queue instantly
#    8. Full try/except on every critical path
#    9. Cleaner HUD showing cooldown countdown prominently

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from filterpy.kalman import KalmanFilter
import serial
import serial.tools.list_ports
import time
import argparse
import threading
import queue
import sys
import traceback
import signal
import logging
import json
import os

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("goalkeeper.log", mode="a"),
    ],
)
log = logging.getLogger("GK")

parser = argparse.ArgumentParser(description="Robot Goalkeeper v7.0")
parser.add_argument("--source",         default="0")
parser.add_argument("--port",           default="COM8")
parser.add_argument("--baud",           type=int,   default=115200)
parser.add_argument("--model",          default="yolov8n.pt")
parser.add_argument("--conf",           type=float, default=0.08)
parser.add_argument("--no-motor",       action="store_true")
parser.add_argument("--flip",           action="store_true")
parser.add_argument("--output",         default="goalkeeper_output.mp4")
parser.add_argument("--save",           action="store_true")
parser.add_argument("--debug",          action="store_true")
parser.add_argument("--calibrate",      action="store_true")
parser.add_argument("--calib-file",     default="goalkeeper_calib.json")
parser.add_argument("--save-cooldown",  type=float, default=10.0,
                    help="Seconds to freeze after a save before resuming detection (default: 10)")
parser.add_argument("--dive-confirm",   type=int,   default=2,
                    help="Frames ball must approach before committing dive (default: 2)")
args = parser.parse_args()


# ── Physical constants ─────────────────────────────────────────────────────────
class Physical:
    GOAL_WIDTH_MM        = 2438.4
    GOAL_HEIGHT_MM       = 1219.2
    GK_SPAN_MM           = 1041.4

    ANGLE_MIN            = 20.0
    ANGLE_MAX            = 165.0
    ANGLE_HOME           = 90.0
    ANGLE_RANGE          = 145.0     # 165 - 20

    CAMERA_HEIGHT_MM     = 127.0
    BALL_DIAM_MM         = 216.0
    BALL_RADIUS_FAR_PX   = 8.0
    BALL_RADIUS_CLOSE_PX = 48.0

PHYS = Physical()


# ── Config ─────────────────────────────────────────────────────────────────────
class Config:
    REACTION_ZONE_Y           = 0.72
    DEAD_ZONE_DEG             = 0.5          # reduced for faster response
    MOTOR_CMD_HZ              = 120          # higher for dive path
    APPROACH_FRAMES           = 2            # faster approach detection
    APPROACH_GROW_PX          = 0.25         # lower threshold = more sensitive
    APPROACH_MIN_SPEED        = 1.0
    PREDICT_FRAMES            = 6
    PREDICT_MAX_EXTRAP        = 0.40
    KALMAN_COAST_FRAMES       = 12
    RESET_LOST_FRAMES         = 30
    SERIAL_WATCHDOG_TIMEOUT   = 8.0
    SERIAL_RECONNECT_INTERVAL = 4.0
    SERIAL_KEEPALIVE_INTERVAL = 3.0
    TRAIL_LEN                 = 40
    SHOW_GOALPOST_LINES       = True
    SHOW_COVERAGE_ARC         = True

CFG = Config()


# ── Calibration ────────────────────────────────────────────────────────────────
class Calibration:
    CALIB_FILE_VERSION = 2

    def __init__(self, frame_w, frame_h, calib_file):
        self.frame_w    = frame_w
        self.frame_h    = frame_h
        self.calib_file = calib_file
        self.left_post_px  = int(frame_w * 0.04)
        self.right_post_px = int(frame_w * 0.96)
        self.crossbar_y_px = int(frame_h * 0.72)
        self._update_derived()
        self._load(calib_file)

    def _update_derived(self):
        self.goal_px_width = max(self.right_post_px - self.left_post_px, 1)
        self.px_per_mm     = self.goal_px_width / PHYS.GOAL_WIDTH_MM

    def _load(self, path):
        if not os.path.exists(path):
            log.info("[CALIB] No calibration file — using defaults")
            return
        try:
            with open(path) as f:
                d = json.load(f)
            if d.get("version") != self.CALIB_FILE_VERSION:
                log.warning("[CALIB] Old calib file — re-calibrate")
                return
            self.left_post_px  = d["left_post_px"]
            self.right_post_px = d["right_post_px"]
            self.crossbar_y_px = d["crossbar_y_px"]
            self._update_derived()
            log.info(f"[CALIB] Loaded: L={self.left_post_px} R={self.right_post_px} Y={self.crossbar_y_px}")
        except Exception as e:
            log.warning(f"[CALIB] Load failed: {e}")

    def save(self, path=None):
        path = path or self.calib_file
        try:
            with open(path, "w") as f:
                json.dump({
                    "version": self.CALIB_FILE_VERSION,
                    "left_post_px": self.left_post_px,
                    "right_post_px": self.right_post_px,
                    "crossbar_y_px": self.crossbar_y_px,
                    "frame_w": self.frame_w,
                    "frame_h": self.frame_h,
                }, f, indent=2)
            log.info(f"[CALIB] Saved to {path}")
        except Exception as e:
            log.error(f"[CALIB] Save failed: {e}")

    def px_to_goal_mm(self, px):
        try:
            ratio = (float(px) - self.left_post_px) / float(self.goal_px_width)
            return float(np.clip(ratio * PHYS.GOAL_WIDTH_MM, 0.0, PHYS.GOAL_WIDTH_MM))
        except Exception:
            return PHYS.GOAL_WIDTH_MM / 2.0

    def goal_mm_to_angle(self, mm, flip=False):
        try:
            norm = float(np.clip(float(mm) / PHYS.GOAL_WIDTH_MM, 0.0, 1.0))
            if flip:
                norm = 1.0 - norm
            return float(np.clip(PHYS.ANGLE_MIN + norm * PHYS.ANGLE_RANGE,
                                 PHYS.ANGLE_MIN, PHYS.ANGLE_MAX))
        except Exception:
            return PHYS.ANGLE_HOME

    def px_to_angle(self, px, flip=False):
        try:
            return self.goal_mm_to_angle(self.px_to_goal_mm(px), flip=flip)
        except Exception:
            return PHYS.ANGLE_HOME

    def angle_to_px(self, angle):
        try:
            norm = float(np.clip((float(angle) - PHYS.ANGLE_MIN) / PHYS.ANGLE_RANGE, 0.0, 1.0))
            return int(self.left_post_px + norm * PHYS.GOAL_WIDTH_MM * self.px_per_mm)
        except Exception:
            return (self.left_post_px + self.right_post_px) // 2

    def angle_to_goal_mm(self, angle):
        try:
            norm = float(np.clip((float(angle) - PHYS.ANGLE_MIN) / PHYS.ANGLE_RANGE, 0.0, 1.0))
            return float(np.clip(norm * PHYS.GOAL_WIDTH_MM, 0.0, PHYS.GOAL_WIDTH_MM))
        except Exception:
            return PHYS.GOAL_WIDTH_MM / 2.0

    @property
    def crossbar_frac(self):
        try:
            return self.crossbar_y_px / max(self.frame_h, 1)
        except Exception:
            return 0.72


# ── Threaded Camera Capture ────────────────────────────────────────────────────
class ThreadedCapture:
    def __init__(self, source, is_webcam):
        self._source    = source
        self._is_webcam = is_webcam
        self._cap       = None
        self._frame     = None
        self._ret       = False
        self._lock      = threading.Lock()
        self._running   = False
        self._ended     = False
        self.W = self.H = 0
        self.FPS = 30.0

    def open(self):
        try:
            self._cap = cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                raise IOError(f"Cannot open: {self._source}")
        except Exception as e:
            log.error(f"Camera open error: {e}")
            return False
        if self._is_webcam:
            try:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
                self._cap.set(cv2.CAP_PROP_FPS,            60)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE,      1)
            except Exception as e:
                log.warning(f"Camera props error: {e}")
        try:
            self.W   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.H   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.FPS = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            if self.W <= 0 or self.H <= 0:
                raise ValueError(f"Bad frame size: {self.W}x{self.H}")
        except Exception as e:
            log.error(f"Camera resolution error: {e}")
            return False
        log.info(f"Camera: {self.W}x{self.H} @ {self.FPS:.0f}fps  [threaded]")
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="cam-cap").start()
        return True

    def _loop(self):
        while self._running:
            try:
                ret, frame = self._cap.read()
                if ret and frame is not None and frame.size > 0:
                    with self._lock:
                        self._ret, self._frame = True, frame
                else:
                    with self._lock:
                        self._ret = False
                    if not self._is_webcam:
                        self._ended = True
                        self._running = False
                        break
                    time.sleep(0.005)
            except Exception as e:
                log.warning(f"Capture loop error: {e}")
                time.sleep(0.02)

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    @property
    def is_video_ended(self):
        return self._ended

    def release(self):
        self._running = False
        try:
            if self._cap:
                self._cap.release()
        except Exception as e:
            log.debug(f"Cap release: {e}")


# ── Motor Controller ───────────────────────────────────────────────────────────
class MotorController:
    def __init__(self, port, baud, no_motor=False, flip=False):
        self.port          = port
        self.baud          = baud
        self.no_motor      = no_motor
        self.flip          = flip
        self.connected     = False
        self._ser          = None
        self._lock         = threading.Lock()
        self._queue        = queue.Queue(maxsize=8)
        self._last_send    = 0.0
        self._last_rx      = time.time()
        self._status_log   = deque(maxlen=40)
        self._angle        = PHYS.ANGLE_HOME
        self._reported_ang = PHYS.ANGLE_HOME
        self._esp32_alive  = False
        self._shutdown     = False

        if no_motor:
            log.info("[MOTOR] No-motor mode")
            self.connected = True
            self._esp32_alive = True
            return

        self._connect()
        if self.connected:
            threading.Thread(target=self._read_loop,  daemon=True, name="motor-rx").start()
            threading.Thread(target=self._write_loop, daemon=True, name="motor-tx").start()
            threading.Thread(target=self._watchdog,   daemon=True, name="motor-wd").start()

    def _all_ports(self):
        try:
            return [p.device for p in serial.tools.list_ports.comports()]
        except Exception as e:
            log.warning(f"[MOTOR] Port enum error: {e}")
            return []

    def _try_open(self, port):
        try:
            s = serial.Serial(port, self.baud, timeout=0.5,
                              write_timeout=1.0, exclusive=True)
            with self._lock:
                self._ser = s
            return True
        except Exception as e:
            log.debug(f"[MOTOR] Cannot open {port}: {e}")
            return False

    def _connect(self):
        for port in [self.port] + [p for p in self._all_ports() if p != self.port]:
            if self._try_open(port):
                self.port = port
                self.connected = True
                log.info(f"[MOTOR] Opened {port} @ {self.baud}baud")
                try:
                    time.sleep(2.2)
                    self._ser.reset_input_buffer()
                    self._raw_write("S\n")
                    time.sleep(0.15)
                    self._raw_write("H\n")
                except Exception as e:
                    log.warning(f"[MOTOR] Init error: {e}")
                log.info(f"[MOTOR] Ready on {port}")
                return
        log.error(f"[MOTOR] No ESP32 found. Ports: {self._all_ports()}")

    def _reconnect(self):
        log.warning("[MOTOR] Reconnecting...")
        with self._lock:
            try:
                if self._ser:
                    self._ser.close()
            except Exception:
                pass
            self._ser = None
        self.connected = False
        self._esp32_alive = False
        self._drain_queue()
        time.sleep(CFG.SERIAL_RECONNECT_INTERVAL)
        self._connect()

    def _drain_queue(self):
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break

    def _enqueue(self, cmd, urgent=False):
        if self.no_motor:
            return
        if urgent:
            # For dive: clear queue first so command goes ASAP
            self._drain_queue()
        try:
            self._queue.put_nowait(cmd)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(cmd)
            except Exception:
                pass

    def _raw_write(self, text):
        try:
            with self._lock:
                ser = self._ser
            if ser and ser.is_open:
                ser.write(text.encode("utf-8"))
                ser.flush()
        except Exception as e:
            log.warning(f"[MOTOR] Write error: {e}")

    def _read_loop(self):
        while not self._shutdown:
            try:
                with self._lock:
                    ser = self._ser
                if ser is None or not ser.is_open:
                    time.sleep(0.1)
                    continue
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                self._last_rx = time.time()
                self._esp32_alive = True
                try:
                    ang = None
                    if   line.startswith("OK:"):     ang = float(line[3:].split(",")[0])
                    elif line.startswith("STATUS:"): ang = float(line[7:].split(",")[0])
                    elif line.startswith("PONG:"):   ang = float(line[5:])
                    elif line.startswith("ALIVE:"):  ang = float(line[6:])
                    if ang is not None:
                        clamped = float(np.clip(ang, PHYS.ANGLE_MIN, PHYS.ANGLE_MAX))
                        with self._lock:
                            self._reported_ang = clamped
                            self._angle = clamped
                except Exception:
                    pass
                self._status_log.append(line)
            except serial.SerialException as e:
                log.warning(f"[MOTOR] RX serial error: {e}")
                self._esp32_alive = False
                time.sleep(0.2)
            except Exception as e:
                log.warning(f"[MOTOR] RX error: {e}")
                time.sleep(0.05)

    def _write_loop(self):
        while not self._shutdown:
            try:
                cmd = self._queue.get(timeout=0.5)
                with self._lock:
                    ser = self._ser
                if ser and ser.is_open:
                    ser.write((cmd + "\n").encode("utf-8"))
                    ser.flush()
            except queue.Empty:
                pass
            except serial.SerialException as e:
                log.warning(f"[MOTOR] TX serial error: {e}")
                self._esp32_alive = False
                time.sleep(0.1)
            except Exception as e:
                log.warning(f"[MOTOR] TX error: {e}")
                time.sleep(0.05)

    def _watchdog(self):
        last_ping = time.time()
        while not self._shutdown:
            try:
                time.sleep(1.0)
                now = time.time()
                if not self.connected:
                    continue
                if now - self._last_rx > CFG.SERIAL_WATCHDOG_TIMEOUT:
                    log.warning("[MOTOR] Watchdog → reconnect")
                    self._reconnect()
                elif now - last_ping > CFG.SERIAL_KEEPALIVE_INTERVAL:
                    self._enqueue("PING")
                    last_ping = now
            except Exception as e:
                log.warning(f"[MOTOR] Watchdog error: {e}")
                time.sleep(1.0)

    def send_angle(self, angle, urgent=False):
        """
        Send fast dive angle command.
        urgent=True clears queue first so the command gets there immediately.
        """
        try:
            angle = float(np.clip(angle, PHYS.ANGLE_MIN, PHYS.ANGLE_MAX))
            now = time.time()
            # Only rate-limit non-urgent commands
            if not urgent:
                if now - self._last_send < 1.0 / CFG.MOTOR_CMD_HZ:
                    return False
            self._last_send = now
            if self.no_motor:
                with self._lock:
                    self._angle = angle
                return True
            if not self.connected:
                return False
            cmd = f"A:{angle:.1f}"
            self._enqueue(cmd, urgent=urgent)
            with self._lock:
                self._angle = angle
            return True
        except Exception as e:
            log.warning(f"[MOTOR] send_angle error: {e}")
            return False

    def go_home(self):
        try:
            if self.no_motor:
                with self._lock:
                    self._angle = PHYS.ANGLE_HOME
                    self._reported_ang = PHYS.ANGLE_HOME
                return
            self._drain_queue()
            self._enqueue("H")
            with self._lock:
                self._angle = PHYS.ANGLE_HOME
        except Exception as e:
            log.warning(f"[MOTOR] go_home error: {e}")

    def shutdown(self):
        self._shutdown = True
        try:
            with self._lock:
                ser = self._ser
            if ser and ser.is_open:
                ser.write(b"H\n")
                ser.flush()
                time.sleep(0.3)
                ser.close()
        except Exception as e:
            log.warning(f"[MOTOR] Shutdown error: {e}")
        log.info("[MOTOR] Shutdown complete")

    @property
    def angle(self):
        with self._lock:
            return self._angle

    @property
    def reported_angle(self):
        with self._lock:
            return self._reported_ang

    @property
    def alive(self):
        return self._esp32_alive


# ── Kalman Filter ──────────────────────────────────────────────────────────────
def make_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0
    kf.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
    kf.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
    kf.R = np.eye(2) * 1.5     # lower = trust detections more
    kf.Q = np.eye(4) * 0.08    # slightly higher = track fast balls better
    kf.P = np.eye(4) * 80.0
    return kf


# ── Approach Detector ──────────────────────────────────────────────────────────
class ApproachDetector:
    def __init__(self, confirm_frames=2):
        self._sizes     = deque(maxlen=14)
        self._pos       = deque(maxlen=14)
        self.approaching     = False
        self.saved_x         = None
        self._confirm_frames = confirm_frames
        self._confirm_count  = 0

    def update(self, cx, cy, radius, frame_h, zone_frac):
        try:
            self._sizes.append(float(radius))
            self._pos.append((int(cx), int(cy)))
            n = CFG.APPROACH_FRAMES
            if len(self._sizes) < n + 1:
                return False
            sizes  = list(self._sizes)
            pos    = list(self._pos)
            growth = sizes[-1] - sizes[-n]
            dx     = pos[-1][0] - pos[-n][0]
            dy     = pos[-1][1] - pos[-n][1]
            spd    = (dx*dx + dy*dy) ** 0.5
            in_zone = cy >= frame_h * zone_frac

            primary   = growth >= CFG.APPROACH_GROW_PX * n and spd >= CFG.APPROACH_MIN_SPEED and in_zone
            fallback  = in_zone and dy > 0 and spd >= CFG.APPROACH_MIN_SPEED * 2.0
            size_only = in_zone and growth >= CFG.APPROACH_GROW_PX * n * 0.6

            raw_approach = primary or fallback or size_only

            # Confirm over multiple frames to reduce false positives
            if raw_approach:
                self._confirm_count = min(self._confirm_count + 1, self._confirm_frames + 2)
            else:
                self._confirm_count = max(self._confirm_count - 1, 0)

            was = self.approaching
            self.approaching = self._confirm_count >= self._confirm_frames

            if self.approaching and not was:
                self.saved_x = cx

            return self.approaching
        except Exception as e:
            log.debug(f"ApproachDetector error: {e}")
            return False

    def reset(self):
        try:
            self._sizes.clear()
            self._pos.clear()
            self.approaching   = False
            self.saved_x       = None
            self._confirm_count = 0
        except Exception as e:
            log.warning(f"ApproachDetector.reset error: {e}")


# ── Helpers ────────────────────────────────────────────────────────────────────
def z_weight(radius):
    lo, hi = PHYS.BALL_RADIUS_FAR_PX, PHYS.BALL_RADIUS_CLOSE_PX
    return float(np.clip((radius - lo) / (hi - lo + 1e-6), 0.0, 1.0))


def predict_landing_mm(trail_mm, goal_width_mm):
    try:
        if len(trail_mm) < 2:
            return None
        pts = list(trail_mm)
        dx  = (pts[-1] - pts[-3]) / 2.0 if len(pts) >= 4 else float(pts[-1] - pts[-2])
        raw = pts[-1] + dx * CFG.PREDICT_FRAMES
        ext = goal_width_mm * CFG.PREDICT_MAX_EXTRAP
        return float(np.clip(np.clip(raw, pts[-1]-ext, pts[-1]+ext), 0.0, goal_width_mm))
    except Exception:
        return None


def compute_target_angle(cx_px, radius, trail_mm, calib, flip):
    try:
        cur_mm  = calib.px_to_goal_mm(cx_px)
        pred_mm = predict_landing_mm(trail_mm, PHYS.GOAL_WIDTH_MM)
        w       = z_weight(radius)
        tmm     = cur_mm if pred_mm is None else w * cur_mm + (1.0 - w) * pred_mm
        tmm     = float(np.clip(tmm, 0.0, PHYS.GOAL_WIDTH_MM))
        angle   = float(np.clip(calib.goal_mm_to_angle(tmm, flip=flip),
                                PHYS.ANGLE_MIN, PHYS.ANGLE_MAX))
        return angle, tmm
    except Exception as e:
        log.warning(f"compute_target_angle error: {e}")
        return PHYS.ANGLE_HOME, PHYS.GOAL_WIDTH_MM / 2.0


def gk_coverage_px(motor_angle, calib):
    try:
        ctr  = calib.angle_to_goal_mm(motor_angle)
        half = PHYS.GK_SPAN_MM / 2.0
        lmm  = max(0.0, ctr - half)
        rmm  = min(PHYS.GOAL_WIDTH_MM, ctr + half)
        return (int(calib.left_post_px + lmm * calib.px_per_mm),
                int(calib.left_post_px + rmm * calib.px_per_mm))
    except Exception:
        mid = (calib.left_post_px + calib.right_post_px) // 2
        return mid - 50, mid + 50


# ── Drawing ────────────────────────────────────────────────────────────────────
def draw_goalpost_markers(frame, calib, H):
    try:
        lx, rx, cy = calib.left_post_px, calib.right_post_px, calib.crossbar_y_px
        for x, lbl, off in [(lx, "L POST", 4), (rx, "R POST", -52)]:
            cv2.line(frame, (x, cy), (x, H), (60, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, lbl, (x+off, cy+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (60, 200, 255), 1, cv2.LINE_AA)
        mid = (lx + rx) // 2
        cv2.line(frame, (lx, cy-10), (rx, cy-10), (60, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "8 ft goal", (mid-32, cy-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 200, 255), 1, cv2.LINE_AA)
    except Exception as e:
        log.debug(f"draw_goalpost_markers: {e}")


def draw_reaction_line(frame, W, calib, zone_frac, active):
    try:
        y  = int(calib.frame_h * zone_frac)
        lx = calib.left_post_px
        rx = calib.right_post_px
        col = (0, 40, 255) if active else (0, 120, 220)
        cv2.line(frame, (lx, y), (rx, y), col, 3, cv2.LINE_AA)
        x = 0
        while x < lx - 2:
            x2 = min(x + 16, lx - 2)
            cv2.line(frame, (x, y), (x2, y), (col[0]//2, col[1]//2, col[2]//2), 1, cv2.LINE_AA)
            x += 23
        x = rx + 2
        while x < W:
            x2 = min(x + 16, W)
            cv2.line(frame, (x, y), (x2, y), (col[0]//2, col[1]//2, col[2]//2), 1, cv2.LINE_AA)
            x += 23
        cv2.putText(frame, f"CROSSBAR [{int(zone_frac*100)}%] [+/-]",
                    (lx+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    except Exception as e:
        log.debug(f"draw_reaction_line: {e}")


def draw_gk_coverage(frame, motor_angle, calib, crossbar_y, H, is_diving):
    try:
        lp, rp = gk_coverage_px(motor_angle, calib)
        cy  = crossbar_y
        col = (30, 60, 255) if is_diving else (20, 180, 80)
        ov  = frame.copy()
        cv2.rectangle(ov, (lp, cy-1), (rp, H), col, -1)
        cv2.addWeighted(ov, 0.13, frame, 0.87, 0, frame)
        cv2.line(frame, (lp, cy), (rp, cy), col, 3, cv2.LINE_AA)
        cx2 = calib.angle_to_px(motor_angle)
        cv2.circle(frame, (cx2, cy), 8, col, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx2, cy), 8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"GK {PHYS.GK_SPAN_MM/PHYS.GOAL_WIDTH_MM*100:.0f}%",
                    ((lp+rp)//2-22, cy+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    except Exception as e:
        log.debug(f"draw_gk_coverage: {e}")


def draw_ball_box(frame, cx, cy, half, conf, predicted, W, H):
    try:
        col = (30, 180, 255) if predicted else (30, 240, 120)
        x1, y1 = max(0, cx-half), max(0, cy-half)
        x2, y2 = min(W, cx+half), min(H, cy+half)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
        c = min(half//2, 18)
        for px_, py_, sx, sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (px_, py_), (px_+sx*c, py_), col, 3, cv2.LINE_AA)
            cv2.line(frame, (px_, py_), (px_, py_+sy*c), col, 3, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, col, -1, cv2.LINE_AA)
        lbl = f"{'~' if predicted else ''}Ball {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), col, -1)
        cv2.putText(frame, lbl, (x1+3, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    except Exception as e:
        log.debug(f"draw_ball_box: {e}")


def draw_trail(frame, trail):
    try:
        pts = list(trail)
        for i in range(1, len(pts)):
            a = i / max(len(pts), 1)
            cv2.line(frame, pts[i-1], pts[i],
                     (0, int(160*a), int(255*a)), max(1, int(4*a)), cv2.LINE_AA)
    except Exception as e:
        log.debug(f"draw_trail: {e}")


def draw_landing(frame, cx, cy, pred_mm, calib, crossbar_y):
    try:
        if pred_mm is None:
            return
        px = int(np.clip(calib.left_post_px + pred_mm * calib.px_per_mm, 0, frame.shape[1]))
        cv2.line(frame, (cx, cy), (px, crossbar_y), (255, 120, 0), 2, cv2.LINE_AA)
        cv2.circle(frame, (px, crossbar_y), 10, (255, 120, 0), -1, cv2.LINE_AA)
        cv2.putText(frame, f"{pred_mm:.0f}mm", (px-20, crossbar_y-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 120, 0), 1, cv2.LINE_AA)
    except Exception as e:
        log.debug(f"draw_landing: {e}")


def draw_motor_bar(frame, W, H, motor_angle, target_angle, calib, reported_angle):
    try:
        bx1, bx2, by = 10, W-10, H-36
        bw = bx2 - bx1
        cv2.rectangle(frame, (bx1, by-5), (bx2, by+5), (30, 30, 30), -1)
        cv2.rectangle(frame, (bx1, by-5), (bx2, by+5), (75, 75, 75), 1)

        hx = int(bx1 + (PHYS.ANGLE_HOME - PHYS.ANGLE_MIN) / PHYS.ANGLE_RANGE * bw)
        cv2.line(frame, (hx, by-8), (hx, by+8), (120, 120, 60), 2)
        cv2.putText(frame, "HOME", (hx-16, by+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (100, 100, 50), 1)

        if target_angle is not None:
            tx = int(np.clip(bx1 + (target_angle-PHYS.ANGLE_MIN)/PHYS.ANGLE_RANGE*bw, bx1+4, bx2-4))
            cv2.circle(frame, (tx, by), 7, (50, 140, 255), -1, cv2.LINE_AA)

        if reported_angle is not None:
            rx2 = int(np.clip(bx1 + (reported_angle-PHYS.ANGLE_MIN)/PHYS.ANGLE_RANGE*bw, bx1+4, bx2-4))
            cv2.circle(frame, (rx2, by), 11, (200, 255, 50), 1, cv2.LINE_AA)

        ax = int(np.clip(bx1 + (motor_angle-PHYS.ANGLE_MIN)/PHYS.ANGLE_RANGE*bw, bx1+8, bx2-8))
        cv2.circle(frame, (ax, by), 9, (30, 240, 120), -1, cv2.LINE_AA)
        cv2.circle(frame, (ax, by), 9, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(frame, f"{PHYS.ANGLE_MIN:.0f}°", (bx1, by+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 100, 100), 1)
        cv2.putText(frame, f"{PHYS.ANGLE_MAX:.0f}°", (bx2-28, by+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 100, 100), 1)

        if calib:
            gk_mm = calib.angle_to_goal_mm(motor_angle)
            drift = motor_angle - PHYS.ANGLE_HOME
            col   = (200, 100, 50) if abs(drift) > 5 else (140, 140, 140)
            dtxt  = f"  DRIFT:{drift:+.1f}°" if abs(drift) > 1.5 else ""
            cv2.putText(frame, f"GK: {gk_mm:.0f}mm ({gk_mm/25.4:.1f}in){dtxt}",
                        (bx1, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    except Exception as e:
        log.debug(f"draw_motor_bar: {e}")


def draw_cooldown_overlay(frame, W, H, seconds_remaining, total):
    """Full-screen semi-transparent overlay during post-save cooldown."""
    try:
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (W, H), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

        # Progress bar
        bar_w = int(W * 0.6)
        bar_h = 18
        bx = (W - bar_w) // 2
        by = H // 2 + 60
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (60, 60, 60), -1)
        frac = max(0.0, 1.0 - seconds_remaining / max(total, 0.01))
        fill = int(bar_w * frac)
        col = (0, int(200 * frac), int(60 + 180 * frac))
        if fill > 0:
            cv2.rectangle(frame, (bx, by), (bx + fill, by + bar_h), col, -1)
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (180, 180, 180), 2)

        # Big countdown
        txt = f"SAVE! Resuming in {seconds_remaining:.1f}s"
        fs  = 1.1
        th  = 2
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        cv2.putText(frame, txt, ((W-tw)//2, H//2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (30, 240, 120), th, cv2.LINE_AA)

        hint = "Keeper will auto-resume after countdown"
        (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(frame, hint, ((W-hw)//2, H//2 + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    except Exception as e:
        log.debug(f"draw_cooldown_overlay: {e}")


def draw_hud(frame, W, H, fps, det_rate, state, scol, motor_angle,
             target_angle, approaching, esp32_alive, flip, zone_frac,
             frame_idx, calib, reported_angle):
    try:
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (W, 72), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, f"● {state}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, scol, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{fps:.1f} FPS  Det:{det_rate:.0f}%  #{frame_idx}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (155, 155, 155), 1, cv2.LINE_AA)

        gk_mm  = calib.angle_to_goal_mm(motor_angle)
        ma_txt = f"Motor {motor_angle:.1f}°  {gk_mm:.0f}mm"
        (mw, _), _ = cv2.getTextSize(ma_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        mcol = (30, 70, 255) if approaching else (200, 200, 200)
        cv2.putText(frame, ma_txt, (W//2 - mw//2, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, mcol, 2, cv2.LINE_AA)

        if approaching:
            sub_txt = ">> DIVE <<"
            sub_col = (30, 70, 255)
            cv2.putText(frame, sub_txt, (W//2 - 60, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, sub_col, 1, cv2.LINE_AA)

        etxt = "ESP32 OK" if esp32_alive else "ESP32 !"
        ecol = (30, 210, 70) if esp32_alive else (30, 30, 240)
        (ew, _), _ = cv2.getTextSize(etxt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.putText(frame, etxt, (W-ew-10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, ecol, 1, cv2.LINE_AA)

        ov2 = frame.copy()
        cv2.rectangle(ov2, (0, H-54), (W, H), (10, 10, 10), -1)
        cv2.addWeighted(ov2, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, "[Q]Quit [H]Home [C]Flip [R]Reset [+/-]Zone [S]Save [D]Debug",
                    (10, H-40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 120), 1, cv2.LINE_AA)

        draw_motor_bar(frame, W, H, motor_angle, target_angle, calib, reported_angle)
    except Exception as e:
        log.debug(f"draw_hud: {e}")


# ── Calibration tool ───────────────────────────────────────────────────────────
def run_calibration(tcap, calib):
    log.info("[CALIB] Click: 1)Left post  2)Right post  3)Crossbar  SPACE=confirm R=retry Q=cancel")
    clicks  = []
    labels  = ["LEFT GOALPOST", "RIGHT GOALPOST", "CROSSBAR"]
    colours = [(0, 255, 100), (0, 100, 255), (255, 200, 0)]

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 3:
            clicks.append((x, y))
            log.info(f"[CALIB] Point {len(clicks)}: ({x},{y})")

    try:
        cv2.namedWindow("CALIBRATION", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("CALIBRATION", on_click)
    except cv2.error as e:
        log.error(f"[CALIB] Window failed: {e}")
        return calib

    while True:
        try:
            ret, frame = tcap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue
            d    = frame.copy()
            step = len(clicks)
            msg  = (f"Step {step+1}/3: Click {labels[step]}"
                    if step < 3 else "SPACE=confirm  R=retry  Q=quit")
            cv2.rectangle(d, (0, 0), (d.shape[1], 50), (0, 0, 0), -1)
            cv2.putText(d, msg, (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            for i, (cx2, cy2) in enumerate(clicks):
                cv2.circle(d, (cx2, cy2), 10, colours[i], -1)
                cv2.circle(d, (cx2, cy2), 10, (255, 255, 255), 2)
                cv2.putText(d, labels[i], (cx2+14, cy2+6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, colours[i], 1)
            if len(clicks) >= 2:
                for xi in [clicks[0][0], clicks[1][0]]:
                    cv2.line(d, (xi, 0), (xi, d.shape[0]),
                             colours[len(clicks)-1], 2, cv2.LINE_AA)
            if len(clicks) == 3:
                cv2.line(d, (0, clicks[2][1]), (d.shape[1], clicks[2][1]),
                         colours[2], 2, cv2.LINE_AA)
            cv2.imshow("CALIBRATION", d)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                clicks.clear()
            elif key == ord('q'):
                cv2.destroyWindow("CALIBRATION")
                return calib
            elif key == ord(' ') and len(clicks) == 3:
                calib.left_post_px  = clicks[0][0]
                calib.right_post_px = clicks[1][0]
                calib.crossbar_y_px = clicks[2][1]
                calib._update_derived()
                calib.save()
                log.info("[CALIB] Saved!")
                break
        except Exception as e:
            log.warning(f"[CALIB] Error: {e}")

    try:
        cv2.destroyWindow("CALIBRATION")
    except Exception:
        pass
    return calib


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    flip_mapping  = args.flip
    save_cooldown = args.save_cooldown    # seconds to freeze after a save

    motor = MotorController(args.port, args.baud,
                            no_motor=args.no_motor, flip=flip_mapping)
    if not motor.connected and not args.no_motor:
        log.error("Motor not connected. Use --no-motor to test without hardware.")
        sys.exit(1)

    IS_WEBCAM = args.source.strip().isdigit()
    SOURCE    = int(args.source) if IS_WEBCAM else args.source

    tcap = ThreadedCapture(SOURCE, IS_WEBCAM)
    if not tcap.open():
        log.error("Camera failed to open.")
        motor.shutdown()
        sys.exit(1)
    W, H, FPS = tcap.W, tcap.H, tcap.FPS

    writer = None

    def _sig_handler(sig=None, fr=None):
        log.info("Signal → shutdown")
        motor.shutdown()
        tcap.release()
        try:
            if writer:
                writer.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT,  _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    log.info(f"Loading {args.model} ...")
    try:
        model_yolo = YOLO(args.model)
        _ = model_yolo(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False)
    except Exception as e:
        log.error(f"YOLO load failed: {e}")
        motor.shutdown()
        tcap.release()
        sys.exit(1)
    log.info("Model ready.")

    try:
        calib = Calibration(W, H, args.calib_file)
    except Exception as e:
        log.error(f"Calibration init failed: {e}")
        motor.shutdown()
        tcap.release()
        sys.exit(1)

    if args.calibrate:
        try:
            calib = run_calibration(tcap, calib)
        except Exception as e:
            log.error(f"Calibration failed: {e}")

    zone_frac = calib.crossbar_frac
    CFG.REACTION_ZONE_Y = zone_frac

    if args.save or not IS_WEBCAM:
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, FPS, (W, H))
            if not writer.isOpened():
                raise IOError("VideoWriter failed")
        except Exception as e:
            log.warning(f"Video writer error: {e}")
            writer = None

    try:
        kf = make_kalman()
    except Exception as e:
        log.error(f"Kalman init failed: {e}")
        motor.shutdown()
        tcap.release()
        sys.exit(1)

    approach = ApproachDetector(confirm_frames=args.dive_confirm)

    kf_initialized   = False
    kalman_lost      = 0
    trail            = deque(maxlen=CFG.TRAIL_LEN)
    trail_mm         = deque(maxlen=CFG.TRAIL_LEN)
    ball_sizes       = deque(maxlen=12)

    frame_idx         = 0
    ball_found_count  = 0
    consec_errors     = 0
    start             = time.time()
    fps_live          = 0.0
    last_motor_angle  = PHYS.ANGLE_HOME
    last_target_angle = PHYS.ANGLE_HOME
    goalkeeper_state  = "READY"
    show_debug        = args.debug

    # ── State machine ──────────────────────────────────────────────────────────
    # States: READY → WATCHING → DIVING → COOLDOWN → READY
    # COOLDOWN: motor frozen, detection paused, countdown shown
    GK_READY    = "READY"
    GK_WATCHING = "WATCHING"
    GK_DIVING   = "DIVING"
    GK_COOLDOWN = "COOLDOWN"

    gk_state          = GK_READY
    cooldown_end_time = 0.0
    dive_committed    = False    # have we sent the dive command this dive?

    log.info("=" * 60)
    log.info("  GOALKEEPER v7.0 — INSTANT DIVE + MANUAL RESET")
    log.info(f"  Save cooldown: {save_cooldown:.1f}s")
    log.info(f"  Dive confirm frames: {args.dive_confirm}")
    log.info("=" * 60)

    while True:
        # ── Latest frame ──────────────────────────────────────────────────────
        try:
            ret, frame = tcap.read()
        except Exception as e:
            log.warning(f"Frame read error: {e}")
            consec_errors += 1
            if consec_errors > 25:
                log.error("Too many errors — exiting")
                break
            time.sleep(0.02)
            continue

        if not ret or frame is None:
            if IS_WEBCAM:
                consec_errors += 1
                if consec_errors > 60:
                    log.error("Webcam lost — exiting")
                    break
                time.sleep(0.01)
                continue
            else:
                if tcap.is_video_ended:
                    log.info("Video ended")
                    break
                time.sleep(0.01)
                continue

        consec_errors = 0
        frame_idx    += 1
        fps_live       = frame_idx / max(time.time() - start, 1e-6)

        try:
            if frame.size == 0:
                log.warning(f"Empty frame #{frame_idx}")
                continue
            display = frame.copy()

            # ── Get reported angle ─────────────────────────────────────────
            try:
                reported_ang = motor.reported_angle
            except Exception:
                reported_ang = last_motor_angle

            # ══════════════════════════════════════════════════════════════
            #  COOLDOWN STATE — skip all detection, just show countdown
            # ══════════════════════════════════════════════════════════════
            if gk_state == GK_COOLDOWN:
                now = time.time()
                remaining = cooldown_end_time - now

                if remaining <= 0.0:
                    # Cooldown expired — reset everything and go READY
                    log.info("[GK] Cooldown done → READY")
                    gk_state         = GK_READY
                    goalkeeper_state = "READY"
                    dive_committed   = False
                    kf               = make_kalman()
                    kf_initialized   = False
                    kalman_lost      = 0
                    trail.clear()
                    trail_mm.clear()
                    ball_sizes.clear()
                    approach.reset()
                else:
                    # Draw basic frame + cooldown overlay
                    if CFG.SHOW_GOALPOST_LINES:
                        draw_goalpost_markers(display, calib, H)
                    if CFG.SHOW_COVERAGE_ARC:
                        draw_gk_coverage(display, last_motor_angle, calib,
                                         calib.crossbar_y_px, H, False)
                    draw_cooldown_overlay(display, W, H, remaining, save_cooldown)
                    draw_motor_bar(display, W, H, last_motor_angle,
                                   last_target_angle, calib, reported_ang)

                    if writer:
                        try:
                            writer.write(display)
                        except Exception:
                            pass
                    try:
                        cv2.imshow("Robot Goalkeeper v7.0", display)
                    except cv2.error:
                        pass

                    try:
                        key = cv2.waitKey(1) & 0xFF
                    except cv2.error:
                        key = 0xFF
                    if key in (ord('q'), 27):
                        log.info("Quit")
                        break
                    elif key == ord('h'):
                        try:
                            motor.go_home()
                            last_motor_angle  = PHYS.ANGLE_HOME
                            last_target_angle = PHYS.ANGLE_HOME
                        except Exception as e:
                            log.warning(f"Manual home error: {e}")
                    elif key == ord('d'):
                        show_debug = not show_debug
                    continue

            # ══════════════════════════════════════════════════════════════
            #  ACTIVE STATES — READY / WATCHING / DIVING
            # ══════════════════════════════════════════════════════════════

            # ── Detection ─────────────────────────────────────────────────
            detected_center = None
            best_conf = 0.0
            best_half = 24
            detected_radius = 0.0
            try:
                results = model_yolo.track(
                    frame, persist=True, conf=args.conf, iou=0.3,
                    classes=[32], tracker="bytetrack.yaml", verbose=False)
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    confs = results[0].boxes.conf.cpu().numpy()
                    xyxys = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    for (x1, y1, x2, y2), cf in zip(xyxys, confs):
                        bw, bh = x2-x1, y2-y1
                        if bw < 5 or bh < 5:
                            continue
                        if max(bw, bh) / (min(bw, bh) + 1e-5) > 2.8:
                            continue
                        if cf > best_conf:
                            best_conf = cf
                            detected_center = ((x1+x2)//2, (y1+y2)//2)
                            best_half = max(bw, bh)//2 + 8
                            detected_radius = max(bw, bh) / 2.0
            except Exception as e:
                log.warning(f"YOLO error #{frame_idx}: {e}")

            # ── Kalman ────────────────────────────────────────────────────
            predicted_pos = None
            if detected_center is not None:
                try:
                    cx, cy = detected_center
                    if not kf_initialized:
                        kf.x = np.array([[cx], [cy], [0.], [0.]])
                        kf_initialized = True
                    else:
                        kf.predict()
                        kf.update(np.array([[cx], [cy]], dtype=float))
                    kalman_lost = 0
                    ball_found_count += 1
                    ball_sizes.append(detected_radius)
                except Exception as e:
                    log.warning(f"Kalman update error: {e}")
                    kf = make_kalman()
                    kf_initialized = False
            elif kf_initialized and kalman_lost < CFG.KALMAN_COAST_FRAMES:
                try:
                    kf.predict()
                    predicted_pos = (int(np.clip(kf.x[0, 0], 0, W-1)),
                                     int(np.clip(kf.x[1, 0], 0, H-1)))
                    kalman_lost += 1
                except Exception as e:
                    log.warning(f"Kalman predict error: {e}")
                    kf = make_kalman()
                    kf_initialized = False

            active_center  = detected_center or predicted_pos
            is_approaching = False

            # ── Tracking & motor ──────────────────────────────────────────
            if active_center is not None:
                try:
                    cx, cy = active_center
                    radius = float(ball_sizes[-1]) if ball_sizes else 12.0
                    trail_mm.append(calib.px_to_goal_mm(cx))
                    trail.append((int(cx), int(cy)))
                    is_approaching = approach.update(cx, cy, radius, H, zone_frac)

                    target_angle_now, _ = compute_target_angle(
                        cx, radius, trail_mm, calib, flip_mapping)
                    target_angle_now = float(np.clip(target_angle_now,
                                                     PHYS.ANGLE_MIN, PHYS.ANGLE_MAX))

                    if is_approaching:
                        # ── DIVING ────────────────────────────────────────
                        gk_state         = GK_DIVING
                        goalkeeper_state = "DIVING"

                        # Send angle IMMEDIATELY, no rate-limit, clear queue first
                        if not dive_committed or abs(target_angle_now - last_motor_angle) > CFG.DEAD_ZONE_DEG:
                            try:
                                if motor.send_angle(target_angle_now, urgent=True):
                                    last_motor_angle  = target_angle_now
                                    last_target_angle = target_angle_now
                                    if not dive_committed:
                                        log.info(f"[GK] DIVE → {target_angle_now:.1f}°")
                                        dive_committed = True
                            except Exception as e:
                                log.warning(f"Dive motor send error: {e}")

                    else:
                        # Ball visible but not approaching
                        if gk_state == GK_DIVING:
                            # Ball was diving but now not — save confirmed, start cooldown
                            log.info(f"[GK] Save complete → cooldown {save_cooldown:.1f}s")
                            gk_state          = GK_COOLDOWN
                            cooldown_end_time = time.time() + save_cooldown
                            goalkeeper_state  = "COOLDOWN"
                        else:
                            gk_state         = GK_WATCHING
                            goalkeeper_state = "WATCHING"
                            dive_committed   = False

                except Exception as e:
                    log.warning(f"Tracking error: {e}")

            else:
                # No ball detected
                try:
                    if trail:
                        trail.clear()
                    if trail_mm:
                        trail_mm.clear()
                except Exception:
                    pass
                try:
                    approach.reset()
                except Exception:
                    pass

                if gk_state == GK_DIVING:
                    # Ball disappeared during/after dive — treat as save, start cooldown...
                    log.info(f"[GK] Ball lost after dive → cooldown {save_cooldown:.1f}s")
                    gk_state          = GK_COOLDOWN
                    cooldown_end_time = time.time() + save_cooldown
                    goalkeeper_state  = "COOLDOWN"
                elif gk_state != GK_COOLDOWN:
                    gk_state         = GK_READY
                    goalkeeper_state = "READY"
                    dive_committed   = False

            # ── Render ────────────────────────────────────────────────────
            try:
                if CFG.SHOW_GOALPOST_LINES:
                    draw_goalpost_markers(display, calib, H)
                if CFG.SHOW_COVERAGE_ARC:
                    draw_gk_coverage(display, last_motor_angle, calib,
                                     calib.crossbar_y_px, H, is_approaching)
                draw_trail(display, trail)

                if detected_center:
                    cx, cy = detected_center
                    draw_ball_box(display, cx, cy, best_half, best_conf, False, W, H)
                    if is_approaching and len(trail_mm) >= 2:
                        draw_landing(display, cx, cy,
                                     predict_landing_mm(trail_mm, PHYS.GOAL_WIDTH_MM),
                                     calib, calib.crossbar_y_px)
                elif predicted_pos:
                    draw_ball_box(display, predicted_pos[0], predicted_pos[1],
                                  best_half, best_conf, True, W, H)

                draw_reaction_line(display, W, calib, zone_frac, is_approaching)

                ss = ("DIVING"     if detected_center and is_approaching else
                      "PREDICTING" if predicted_pos else goalkeeper_state)
                sc = ((30, 240, 120) if detected_center else
                      (30, 200, 255) if predicted_pos   else (80, 80, 255))

                draw_hud(display, W, H, fps_live,
                         ball_found_count / max(frame_idx, 1) * 100,
                         ss, sc, last_motor_angle, last_target_angle,
                         is_approaching, motor.alive, flip_mapping, zone_frac,
                         frame_idx, calib, reported_angle=reported_ang)
            except Exception as e:
                log.warning(f"Render error #{frame_idx}: {e}")
                if show_debug:
                    traceback.print_exc()

            if writer:
                try:
                    writer.write(display)
                except Exception as e:
                    log.warning(f"Writer error: {e}")
                    writer = None

            try:
                cv2.imshow("Robot Goalkeeper v7.0", display)
            except cv2.error as e:
                log.warning(f"imshow error: {e}")

        except MemoryError as e:
            log.error(f"Memory error #{frame_idx}: {e}")
            break
        except Exception as e:
            log.error(f"Processing error #{frame_idx}: {e}")
            if show_debug:
                traceback.print_exc()
            try:
                cv2.imshow("Robot Goalkeeper v7.0", frame)
            except Exception:
                pass

        # ── Key handling ──────────────────────────────────────────────────
        try:
            key = cv2.waitKey(1) & 0xFF
        except cv2.error:
            key = 0xFF

        if key in (ord('q'), 27):
            log.info("Quit")
            break
        elif key == ord('h'):
            try:
                motor.go_home()
                last_motor_angle  = PHYS.ANGLE_HOME
                last_target_angle = PHYS.ANGLE_HOME
                gk_state          = GK_READY
                goalkeeper_state  = "READY"
                dive_committed    = False
                log.info("Manual home")
            except Exception as e:
                log.warning(f"Manual home error: {e}")
        elif key == ord('c'):
            flip_mapping = not flip_mapping
            log.info(f"Flip: {flip_mapping}")
        elif key == ord('r'):
            try:
                kf = make_kalman()
                kf_initialized = False
                kalman_lost    = 0
                trail.clear()
                trail_mm.clear()
                ball_sizes.clear()
                approach.reset()
                gk_state         = GK_READY
                goalkeeper_state = "READY"
                dive_committed   = False
                log.info("Tracking reset")
            except Exception as e:
                log.warning(f"Reset error: {e}")
        elif key == ord('d'):
            show_debug = not show_debug
        elif key in (ord('+'), ord('=')):
            zone_frac = round(max(0.05, zone_frac - 0.05), 2)
            log.info(f"Zone: {int(zone_frac*100)}%")
        elif key == ord('-'):
            zone_frac = round(min(0.95, zone_frac + 0.05), 2)
            log.info(f"Zone: {int(zone_frac*100)}%")
        elif key == ord('s'):
            try:
                calib.crossbar_y_px = int(H * zone_frac)
                calib.save()
            except Exception as e:
                log.warning(f"Save error: {e}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    log.info("Cleaning up...")
    try:
        motor.shutdown()
    except Exception as e:
        log.warning(f"Motor shutdown error: {e}")
    try:
        tcap.release()
    except Exception as e:
        log.warning(f"Cap release error: {e}")
    try:
        if writer:
            writer.release()
    except Exception as e:
        log.warning(f"Writer release error: {e}")
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    log.info(f"Done: {frame_idx} frames  {fps_live:.1f} fps avg  "
             f"ball {ball_found_count/max(frame_idx,1)*100:.1f}%")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Interrupted")
    except SystemExit:
        pass
    except Exception as e:
        log.error(f"Fatal: {e}")
        traceback.print_exc()
        sys.exit(1)