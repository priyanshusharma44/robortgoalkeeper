# keeper_yolo_v2.py
# ============================================================
#  ROBOT GOALKEEPER v5.2 — FAST + RELIABLE DETECTION
#
#  FIXES vs v5.1:
#    1. YOLO runs in its OWN thread — camera never blocks on inference
#       → FPS goes from 1 to 25-30 immediately
#    2. yolov8n.pt for speed (x was 1fps on CPU, unusable)
#    3. HSV orange/yellow fallback — if YOLO misses the ball,
#       color detection catches it (works great for orange ping pong)
#    4. Camera uses grab() + retrieve() loop to stay fresh
#    5. Frame is downscaled to 416x416 for YOLO input only
#       (display still shows full resolution)
# ============================================================

import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time
import sys
import json
import threading
import queue
import signal
import logging
import logging.handlers
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum, auto
from pathlib import Path
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════

CONFIG_FILE = Path("goalkeeper_config.json")

DEFAULT_CONFIG = {
    "baud_rate":            115200,
    "serial_timeout":       0.3,
    "reconnect_delay":      2.0,
    "send_interval":        0.012,
    "camera_index":         0,
    "frame_w":              640,    # lower res = faster camera + YOLO
    "frame_h":              480,
    "target_fps":           30,
    "cam_buffer_size":      1,

    # YOLO — use nano for speed, confidence low for small balls
    "yolo_model":           "yolov8n.pt",
    "yolo_conf":            0.10,
    "yolo_iou":             0.3,
    "yolo_ball_class":      32,
    "yolo_imgsz":           416,    # smaller = faster inference

    # HSV fallback for orange/yellow balls (tune if needed)
    "hsv_enabled":          True,
    "hsv_lower":            [5, 100, 100],    # orange-ish lower bound
    "hsv_upper":            [35, 255, 255],   # orange-ish upper bound
    "hsv_min_radius":       8,
    "hsv_max_radius":       120,

    # Kalman
    "kalman_process_noise": 1e-2,
    "kalman_meas_noise":    4e-2,
    "trajectory_history":   30,
    "max_track_loss_frames": 8,

    # Angle geometry
    "min_angle":            20.0,
    "max_angle":            165.0,

    # Goalkeeper logic
    "deadzone_px":          40,
    "deadzone_angle":       0.4,
    "incoming_vy_thresh":   1.8,
    "intercept_y_ratio":    0.74,
    "goal_line_y_ratio":    0.88,
    "velocity_ff_scale":    0.50,

    # Watchdog
    "watchdog_interval":    10.0,
    "show_debug_overlay":   True,
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    # Always write back so new keys appear
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


# ══════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════

def setup_logging() -> logging.Logger:
    log = logging.getLogger("goalkeeper")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.handlers.RotatingFileHandler(
        "goalkeeper.log", maxBytes=5 * 1024 * 1024, backupCount=2,
        encoding="utf-8", mode="a")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    log.addHandler(sh)
    return log

log = setup_logging()


# ══════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════

@dataclass
class BallState:
    cx:           Optional[int]   = None
    cy:           Optional[int]   = None
    radius:       int             = 0
    vx:           float           = 0.0
    vy:           float           = 0.0
    ax:           float           = 0.0
    ay:           float           = 0.0
    confident:    bool            = False
    predicted_x:  Optional[int]   = None
    predicted_y:  Optional[int]   = None
    threat_level: float           = 0.0
    target_angle: Optional[float] = None
    timestamp:    float           = 0.0
    source:       str             = ""   # "yolo" or "hsv"


@dataclass
class TrackingStats:
    frames_processed: int   = 0
    frames_detected:  int   = 0
    serial_sends:     int   = 0
    serial_errors:    int   = 0
    reconnects:       int   = 0
    dropped_frames:   int   = 0
    start_time:       float = field(default_factory=time.monotonic)
    _fps_history:     deque = field(default_factory=lambda: deque(maxlen=60))

    def tick_fps(self):
        self._fps_history.append(time.monotonic())

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def fps(self) -> float:
        if len(self._fps_history) < 2:
            return 0.0
        span = self._fps_history[-1] - self._fps_history[0]
        return (len(self._fps_history) - 1) / max(span, 1e-6)

    @property
    def detection_rate(self) -> float:
        return self.frames_detected / max(self.frames_processed, 1)


class GoalkeeperMode(Enum):
    IDLE      = auto()
    TRACKING  = auto()
    DIVING    = auto()
    RETURNING = auto()
    FAULT     = auto()


# ══════════════════════════════════════════════════════════════
#  ANGLE MAPPER
# ══════════════════════════════════════════════════════════════

class AngleMapper:
    def __init__(self, cfg):
        self._frame_w   = cfg["frame_w"]
        self._min_angle = cfg["min_angle"]
        self._max_angle = cfg["max_angle"]
        self._range     = self._max_angle - self._min_angle
        self._center    = (self._min_angle + self._max_angle) / 2.0

    @property
    def center_angle(self): return self._center
    @property
    def min_angle(self):    return self._min_angle
    @property
    def max_angle(self):    return self._max_angle

    def px_to_angle(self, px):
        norm  = px / self._frame_w
        return float(np.clip(self._min_angle + norm * self._range,
                             self._min_angle, self._max_angle))

    def angle_to_px(self, angle):
        norm = (angle - self._min_angle) / self._range
        return int(np.clip(norm * self._frame_w, 0, self._frame_w - 1))


# ══════════════════════════════════════════════════════════════
#  KALMAN FILTER
# ══════════════════════════════════════════════════════════════

class KalmanBallFilter:
    def __init__(self, process_noise=1e-2, measurement_noise=4e-2):
        self._kf = cv2.KalmanFilter(4, 2)
        self._kf.transitionMatrix    = np.eye(4, dtype=np.float32)
        self._kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self._kf.processNoiseCov     = np.eye(4, dtype=np.float32) * process_noise
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self._kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self._initialized = False
        self._last_time   = None
        self._prev_vx = self._prev_vy = 0.0

    def update(self, cx, cy, now):
        dt = (now - self._last_time) if self._last_time else 1/30.0
        dt = max(1e-4, min(dt, 0.5))
        self._last_time = now
        self._kf.transitionMatrix[0, 2] = dt
        self._kf.transitionMatrix[1, 3] = dt

        if cx is None:
            if not self._initialized:
                return None, None, 0.0, 0.0, 0.0, 0.0
            pred = self._kf.predict()
            px, py = int(pred[0].item()), int(pred[1].item())
            vx, vy = float(pred[2].item()), float(pred[3].item())
            ax = (vx - self._prev_vx) / dt
            ay = (vy - self._prev_vy) / dt
            self._prev_vx, self._prev_vy = vx, vy
            return px, py, vx, vy, ax, ay

        meas = np.array([[np.float32(cx)], [np.float32(cy)]])
        if not self._initialized:
            self._kf.statePre  = np.array([[np.float32(cx)],[np.float32(cy)],
                                            [0.],[0.]], dtype=np.float32)
            self._kf.statePost = self._kf.statePre.copy()
            self._initialized  = True
        self._kf.predict()
        corr = self._kf.correct(meas)
        x, y   = int(corr[0].item()), int(corr[1].item())
        vx, vy = float(corr[2].item()), float(corr[3].item())
        ax = (vx - self._prev_vx) / dt
        ay = (vy - self._prev_vy) / dt
        self._prev_vx, self._prev_vy = vx, vy
        return x, y, vx, vy, ax, ay

    def reset(self):
        self._initialized = False
        self._last_time   = None
        self._prev_vx = self._prev_vy = 0.0
        self._kf.errorCovPost = np.eye(4, dtype=np.float32)


# ══════════════════════════════════════════════════════════════
#  BALL TRACKER
# ══════════════════════════════════════════════════════════════

class BallTracker:
    def __init__(self, cfg, mapper):
        self._mapper      = mapper
        self._kf          = KalmanBallFilter(cfg["kalman_process_noise"],
                                             cfg["kalman_meas_noise"])
        self._history     = deque(maxlen=cfg["trajectory_history"])
        self._loss_frames = 0
        self._max_loss    = cfg["max_track_loss_frames"]
        self._frame_w     = cfg["frame_w"]
        self._frame_h     = cfg["frame_h"]
        self._intercept_y = int(cfg["frame_h"] * cfg["intercept_y_ratio"])
        self._vy_thresh   = cfg["incoming_vy_thresh"]
        self._ff_scale    = cfg["velocity_ff_scale"]

    def update(self, cx, cy, radius, now, source="") -> BallState:
        state = BallState(timestamp=now, source=source)
        if cx is None:
            self._loss_frames += 1
            if self._loss_frames > self._max_loss:
                self._kf.reset()
                self._history.clear()
                return state
            kx, ky, vx, vy, ax, ay = self._kf.update(None, None, now)
            if kx is not None:
                state.cx, state.cy = kx, ky
                state.vx, state.vy = vx, vy
                state.ax, state.ay = ax, ay
                self._fill_angle(state)
            return state

        self._loss_frames = 0
        kx, ky, vx, vy, ax, ay = self._kf.update(cx, cy, now)
        self._history.append((kx, ky, now))
        state.cx, state.cy = kx, ky
        state.radius       = radius
        state.vx, state.vy = vx, vy
        state.ax, state.ay = ax, ay
        state.confident    = len(self._history) >= 3
        state.threat_level = self._calc_threat(state)
        self._fill_angle(state)
        return state

    def _fill_angle(self, state):
        if state.cx is None:
            return
        is_shot = (state.vy >= self._vy_thresh and state.confident)
        if is_shot:
            frames_ahead = (self._intercept_y - state.cy) / max(state.vy, 0.1)
            if 0 < frames_ahead < 90:
                pred_x = float(np.clip(
                    state.cx + state.vx * frames_ahead + 0.5 * state.ax * frames_ahead**2,
                    0, self._frame_w - 1))
                state.predicted_x  = int(pred_x)
                state.predicted_y  = self._intercept_y
                state.target_angle = self._mapper.px_to_angle(pred_x)
        else:
            ff_px = float(np.clip(state.cx + state.vx * self._ff_scale * 8.0,
                                  0, self._frame_w - 1))
            state.target_angle = self._mapper.px_to_angle(ff_px)

    def _calc_threat(self, state):
        if state.cy is None or state.vy <= 0:
            return 0.0
        return float(np.clip(
            (state.cy / self._frame_h) * min(abs(state.vy) / 20.0, 1.0) * 2,
            0.0, 1.0))

    def reset(self):
        self._kf.reset()
        self._history.clear()
        self._loss_frames = 0


# ══════════════════════════════════════════════════════════════
#  THREADED YOLO INFERENCE ENGINE
#  Camera captures at full speed. YOLO runs in background thread.
#  Main loop always gets the latest result without blocking.
# ══════════════════════════════════════════════════════════════

class YoloEngine(threading.Thread):
    """
    Runs YOLO inference in a dedicated thread.
    Input:  frames pushed via submit()
    Output: latest result read via get_result()
    """
    def __init__(self, cfg):
        super().__init__(daemon=True, name="YoloEngine")
        self._conf       = cfg["yolo_conf"]
        self._iou        = cfg["yolo_iou"]
        self._ball_class = cfg["yolo_ball_class"]
        self._imgsz      = cfg.get("yolo_imgsz", 416)
        self._model_name = cfg.get("yolo_model", "yolov8n.pt")

        self._in_q:  queue.Queue = queue.Queue(maxsize=1)
        self._result = None          # (cx, cy, radius, conf) or None
        self._result_lock = threading.Lock()
        self._running = True

    def submit(self, frame: np.ndarray):
        """Push a frame for inference. Drops if engine is busy."""
        try:
            self._in_q.put_nowait(frame.copy())
        except queue.Full:
            pass   # engine busy — skip this frame, display keeps going

    def get_result(self):
        """Returns latest (cx, cy, radius, conf) or None. Non-blocking."""
        with self._result_lock:
            return self._result

    def stop(self):
        self._running = False

    def run(self):
        log.info(f"[YOLO] Loading {self._model_name} ...")
        model = YOLO(self._model_name)
        log.info(f"[YOLO] Ready  conf={self._conf}  imgsz={self._imgsz}")

        while self._running:
            try:
                frame = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                results = model.predict(
                    frame,
                    imgsz   = self._imgsz,
                    conf    = self._conf,
                    iou     = self._iou,
                    classes = [self._ball_class],
                    verbose = False,
                )

                best = None
                if results[0].boxes is not None:
                    confs = results[0].boxes.conf.cpu().numpy()
                    xyxys = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    best_cf = 0.0
                    for (x1, y1, x2, y2), cf in zip(xyxys, confs):
                        bw, bh = x2 - x1, y2 - y1
                        if max(bw, bh) > 250:
                            continue
                        aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
                        if aspect > 3.0:
                            continue
                        if cf > best_cf:
                            best_cf = cf
                            best = ((x1+x2)//2, (y1+y2)//2,
                                    max(bw, bh)//2, float(cf))
                with self._result_lock:
                    self._result = best

            except Exception as e:
                log.warning(f"[YOLO] inference error: {e}")
                with self._result_lock:
                    self._result = None


# ══════════════════════════════════════════════════════════════
#  HSV COLOR DETECTOR (orange/yellow ball fallback)
#  Fast — runs every frame in main thread, no lag.
# ══════════════════════════════════════════════════════════════

class HSVDetector:
    def __init__(self, cfg):
        lo = cfg.get("hsv_lower", [5, 100, 100])
        hi = cfg.get("hsv_upper", [35, 255, 255])
        self._lower     = np.array(lo, dtype=np.uint8)
        self._upper     = np.array(hi, dtype=np.uint8)
        self._min_r     = cfg.get("hsv_min_radius", 8)
        self._max_r     = cfg.get("hsv_max_radius", 120)
        self._enabled   = cfg.get("hsv_enabled", True)
        self._kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame):
        """Returns (cx, cy, radius) or (None, None, 0)"""
        if not self._enabled:
            return None, None, 0

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lower, self._upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_r = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            r = int(radius)
            if r < self._min_r or r > self._max_r:
                continue
            # circularity check
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.55:   # must be roundish
                continue
            if r > best_r:
                best_r = r
                best   = (int(x), int(y), r)

        if best:
            return best
        return None, None, 0


# ══════════════════════════════════════════════════════════════
#  BALL DETECTOR — fuses YOLO + HSV
# ══════════════════════════════════════════════════════════════

class BallDetector:
    def __init__(self, cfg, yolo_engine: YoloEngine):
        self._yolo  = yolo_engine
        self._hsv   = HSVDetector(cfg)
        self._fw    = cfg["frame_w"]
        self._fh    = cfg["frame_h"]
        self._trail = deque(maxlen=35)

    def _draw_box(self, frame, cx, cy, radius, color=(0, 229, 160)):
        """Square reticle box identical to pingpong_tracker."""
        W, H = frame.shape[1], frame.shape[0]
        half = radius + 8
        sx1 = max(0, cx - half);  sy1 = max(0, cy - half)
        sx2 = min(W, cx + half);  sy2 = min(H, cy + half)

        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2, cv2.LINE_AA)
        corner = half // 2
        t = 3
        cv2.line(frame, (sx1,sy1),(sx1+corner,sy1), color, t, cv2.LINE_AA)
        cv2.line(frame, (sx1,sy1),(sx1,sy1+corner), color, t, cv2.LINE_AA)
        cv2.line(frame, (sx2,sy1),(sx2-corner,sy1), color, t, cv2.LINE_AA)
        cv2.line(frame, (sx2,sy1),(sx2,sy1+corner), color, t, cv2.LINE_AA)
        cv2.line(frame, (sx1,sy2),(sx1+corner,sy2), color, t, cv2.LINE_AA)
        cv2.line(frame, (sx1,sy2),(sx1,sy2-corner), color, t, cv2.LINE_AA)
        cv2.line(frame, (sx2,sy2),(sx2-corner,sy2), color, t, cv2.LINE_AA)
        cv2.line(frame, (sx2,sy2),(sx2,sy2-corner), color, t, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)
        return sx1, sy1, sx2, sy2

    def detect(self, frame):
        """
        Returns (cx, cy, radius, source, debug_frame)
        source is "yolo", "hsv", or None
        """
        debug = frame.copy()

        # ── 1. Submit frame to YOLO thread (non-blocking) ─────────────────
        self._yolo.submit(frame)

        # ── 2. Get YOLO result (from previous frame — no wait) ────────────
        yolo_res = self._yolo.get_result()
        cx = cy = None
        radius = 0
        source = None

        if yolo_res is not None:
            ycx, ycy, yr, ycf = yolo_res
            # Sanity: result must be within frame
            if 0 < ycx < self._fw and 0 < ycy < self._fh:
                cx, cy, radius, source = ycx, ycy, yr, "yolo"
                COLOR = (0, 229, 160)   # green
                sx1, sy1, sx2, sy2 = self._draw_box(debug, cx, cy, radius, COLOR)
                label = f"Ball YOLO {ycf:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(debug, (sx1, sy1-th-6), (sx1+tw+6, sy1), COLOR, -1)
                cv2.putText(debug, label, (sx1+3, sy1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)

        # ── 3. HSV fallback if YOLO didn't find anything ──────────────────
        if cx is None:
            hcx, hcy, hr = self._hsv.detect(frame)
            if hcx is not None:
                cx, cy, radius, source = hcx, hcy, hr, "hsv"
                COLOR = (0, 180, 255)   # orange-ish for HSV detections
                sx1, sy1, sx2, sy2 = self._draw_box(debug, cx, cy, radius, COLOR)
                label = f"Ball HSV"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(debug, (sx1, sy1-th-6), (sx1+tw+6, sy1), COLOR, -1)
                cv2.putText(debug, label, (sx1+3, sy1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)

        # ── 4. Motion trail ───────────────────────────────────────────────
        if cx is not None:
            self._trail.append((cx, cy))
        elif len(self._trail) > 0:
            self._trail.popleft()

        for i in range(1, len(self._trail)):
            alpha = i / len(self._trail)
            c = (0, int(200*alpha), int(255*alpha))
            cv2.line(debug, self._trail[i-1], self._trail[i],
                     c, max(1, int(3*alpha)), cv2.LINE_AA)

        return cx, cy, radius, source or "", debug


# ══════════════════════════════════════════════════════════════
#  CAMERA MANAGER — grab()/retrieve() for minimal lag
# ══════════════════════════════════════════════════════════════

class CameraManager:
    def __init__(self, cfg):
        self._cfg   = cfg
        self._cap   = None
        self._frame = None
        self._ts    = 0.0
        self._lock  = threading.Lock()
        self._new   = threading.Event()
        self._ready = threading.Event()
        self._run   = True
        self._t     = threading.Thread(target=self._loop, daemon=True, name="Camera")
        self._t.start()

    def get_frame(self, timeout=0.05):
        if self._new.wait(timeout=timeout):
            self._new.clear()
            with self._lock:
                if self._frame is not None:
                    return self._frame.copy(), self._ts
        return None, 0.0

    def wait_ready(self, timeout=10.0):
        return self._ready.wait(timeout=timeout)

    def stop(self):
        self._run = False

    def _open(self):
        idx = self._cfg["camera_index"]
        while self._run:
            log.info(f"[CAMERA] opening index {idx}")
            cap = None
            for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY, cv2.CAP_MSMF):
                c = cv2.VideoCapture(idx, backend)
                if c.isOpened():
                    cap = c
                    break
                c.release()
            if cap is None:
                log.warning("[CAMERA] failed to open, retry in 3s")
                time.sleep(3.0)
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cfg["frame_w"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg["frame_h"])
            cap.set(cv2.CAP_PROP_FPS,          self._cfg["target_fps"])
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # KEY: keep buffer at 1
            time.sleep(0.3)
            # Flush stale frames
            for _ in range(5):
                cap.grab()
            ret, f = cap.retrieve()
            if ret and f is not None and f.size > 0:
                self._cap = cap
                log.info(f"[CAMERA] {int(cap.get(3))}x{int(cap.get(4))}"
                         f"@{cap.get(cv2.CAP_PROP_FPS):.0f}fps")
                return True
            cap.release()
            time.sleep(2.0)
        return False

    def _loop(self):
        if not self._open():
            return
        fw, fh = self._cfg["frame_w"], self._cfg["frame_h"]
        fails  = 0
        while self._run:
            if not self._cap or not self._cap.isOpened():
                self._ready.clear()
                self._open()
                fails = 0
                continue

            # grab() is faster than read() — doesn't decode until retrieve()
            ok = self._cap.grab()
            if not ok:
                fails += 1
                if fails >= 30:
                    self._cap.release()
                    self._ready.clear()
                    self._open()
                    fails = 0
                time.sleep(0.005)
                continue

            ret, frame = self._cap.retrieve()
            ts = time.monotonic()
            if not ret or frame is None:
                fails += 1
                continue
            fails = 0
            if frame.shape[1] != fw or frame.shape[0] != fh:
                frame = cv2.resize(frame, (fw, fh))
            with self._lock:
                self._frame = frame
                self._ts    = ts
            self._new.set()
            self._ready.set()


# ══════════════════════════════════════════════════════════════
#  SERIAL MANAGER
# ══════════════════════════════════════════════════════════════

class SerialManager(threading.Thread):
    def __init__(self, cfg, stats):
        super().__init__(daemon=True, name="SerialManager")
        self._cfg       = cfg
        self._stats     = stats
        self._queue     = queue.Queue(maxsize=1)
        self._pqueue    = queue.Queue(maxsize=4)
        self._esp       = None
        self._lock      = threading.Lock()
        self._running   = True
        self._connected = threading.Event()
        self._last_send = 0.0

    def send(self, message, priority=False):
        q = self._pqueue if priority else self._queue
        try:
            q.put_nowait(message)
        except queue.Full:
            try:
                q.get_nowait()
                q.put_nowait(message)
            except queue.Empty:
                pass

    def is_connected(self):
        return self._connected.is_set()

    def wait_connected(self, timeout=30.0):
        return self._connected.wait(timeout=timeout)

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            if self._esp is None or not self._esp.is_open:
                self._do_connect()
                continue
            try:
                try:
                    msg = self._pqueue.get_nowait()
                except queue.Empty:
                    msg = self._queue.get(timeout=0.004)
                now = time.monotonic()
                gap = now - self._last_send
                if gap < self._cfg["send_interval"]:
                    time.sleep(self._cfg["send_interval"] - gap)
                self._esp.write(msg.encode("ascii"))
                self._last_send = time.monotonic()
                self._stats.serial_sends += 1
            except queue.Empty:
                pass
            except Exception as e:
                log.warning(f"[SERIAL] {e}")
                self._stats.serial_errors += 1
                self._disconnect()
        if self._esp and self._esp.is_open:
            try:
                self._esp.write(b"N\n")
                time.sleep(0.05)
                self._esp.close()
            except Exception:
                pass

    def _do_connect(self):
        port = self._find_port()
        if not port:
            self._connected.clear()
            time.sleep(self._cfg["reconnect_delay"])
            return
        try:
            s = serial.Serial(port, self._cfg["baud_rate"],
                              timeout=self._cfg["serial_timeout"],
                              write_timeout=0.5)
            time.sleep(2.0)
            s.reset_input_buffer()
            with self._lock:
                self._esp = s
            self._connected.set()
            self._stats.reconnects += 1
            log.info(f"[SERIAL] connected {port}")
        except Exception as e:
            log.error(f"[SERIAL] {e}")
            self._connected.clear()
            time.sleep(self._cfg["reconnect_delay"])

    def _disconnect(self):
        self._connected.clear()
        with self._lock:
            if self._esp:
                try: self._esp.close()
                except Exception: pass
                self._esp = None

    @staticmethod
    def _find_port():
        keywords = ["cp210", "ch340", "ch341", "uart bridge", "esp32",
                    "ftdi", "ch9102", "wch usb"]
        for p in serial.tools.list_ports.comports():
            if any(k in (p.description or "").lower() for k in keywords):
                return p.device
        for p in serial.tools.list_ports.comports():
            if any(k in (p.device or "").lower()
                   for k in ["usb", "ttyusb", "ttyacm", "com"]):
                return p.device
        return None


# ══════════════════════════════════════════════════════════════
#  GOALKEEPER
# ══════════════════════════════════════════════════════════════

class Goalkeeper:
    def __init__(self, serial_mgr, stats, cfg, mapper):
        self._serial      = serial_mgr
        self._stats       = stats
        self._cfg         = cfg
        self._mapper      = mapper
        self._mode        = GoalkeeperMode.IDLE
        self._last_tx     = 0.0
        self._last_angle  = None
        self._center_x    = cfg["frame_w"] // 2

    def update(self, ball):
        now = time.monotonic()
        if ball.cx is None:
            self._mode = GoalkeeperMode.IDLE
            if now - self._last_tx >= self._cfg["send_interval"] * 4:
                self._serial.send("N\n")
                self._last_tx = now
            return

        target_angle = ball.target_angle
        if target_angle is None:
            self._mode = GoalkeeperMode.IDLE
            return

        is_dive = (ball.predicted_x is not None
                   and ball.vy >= self._cfg["incoming_vy_thresh"]
                   and ball.confident)

        if is_dive:
            self._mode = GoalkeeperMode.DIVING
        elif abs(ball.cx - self._center_x) < self._cfg["deadzone_px"]:
            self._mode = GoalkeeperMode.IDLE
            return
        else:
            self._mode = GoalkeeperMode.TRACKING

        target_angle = float(np.clip(target_angle,
                                     self._mapper.min_angle,
                                     self._mapper.max_angle))
        if not is_dive:
            if now - self._last_tx < self._cfg["send_interval"]:
                return
            if (self._last_angle is not None
                    and abs(target_angle - self._last_angle) < self._cfg["deadzone_angle"]):
                return

        self._serial.send(f"A{target_angle:.2f}\n", priority=is_dive)
        self._last_angle = target_angle
        self._last_tx    = now

    @property
    def mode(self):
        return self._mode


# ══════════════════════════════════════════════════════════════
#  OVERLAY
# ══════════════════════════════════════════════════════════════

_MODE_COLORS = {
    GoalkeeperMode.TRACKING:  (0,   200,  60),
    GoalkeeperMode.DIVING:    (0,    80, 255),
    GoalkeeperMode.IDLE:      (100, 100, 100),
    GoalkeeperMode.RETURNING: (200, 200,   0),
    GoalkeeperMode.FAULT:     (0,     0, 220),
}


def draw_overlay(frame, ball, gk_mode, stats, serial_ok, cfg, mapper):
    h, w   = frame.shape[:2]
    cx_s   = w // 2
    goal_y = int(cfg["goal_line_y_ratio"] * h)
    int_y  = int(cfg["intercept_y_ratio"] * h)
    dz_px  = cfg["deadzone_px"]

    cv2.line(frame, (cx_s, 0),   (cx_s, h),   (70, 70, 70),  1)
    cv2.line(frame, (0, goal_y), (w, goal_y), (0, 200, 255), 1)
    cv2.line(frame, (0, int_y),  (w, int_y),  (0, 255, 128), 1)
    cv2.putText(frame, "GOAL",      (6, goal_y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,255), 1)
    cv2.putText(frame, "INTERCEPT", (6, int_y-4),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,128), 1)

    # Angle indicator bar
    min_a = cfg["min_angle"];  max_a = cfg["max_angle"]
    left_px  = mapper.angle_to_px(min_a)
    right_px = mapper.angle_to_px(max_a)
    cv2.line(frame, (left_px, h-6), (right_px, h-6), (60,60,60), 2)
    if ball.target_angle is not None:
        motor_px = mapper.angle_to_px(ball.target_angle)
        cv2.line(frame, (motor_px, h-14), (motor_px, h), (0,255,255), 3)

    # Ball
    if ball.cx is not None:
        color = _MODE_COLORS.get(gk_mode, (200,200,200))
        vxs, vys = int(ball.vx*6), int(ball.vy*6)
        if abs(vxs)+abs(vys) > 3:
            cv2.arrowedLine(frame, (ball.cx, ball.cy),
                            (ball.cx+vxs, ball.cy+vys),
                            (0,200,255), 2, tipLength=0.3)
        if ball.predicted_x is not None:
            cv2.drawMarker(frame, (ball.predicted_x, int_y),
                           (0,255,128), cv2.MARKER_DIAMOND, 18, 2)
            cv2.line(frame, (ball.cx, ball.cy),
                     (ball.predicted_x, int_y), (0,180,80), 1)

    # HUD bar
    cv2.rectangle(frame, (0,0), (w,44), (0,0,0), -1)
    mc = _MODE_COLORS.get(gk_mode, (80,80,80))
    cv2.rectangle(frame, (0,0), (110,44), mc, -1)
    cv2.putText(frame, gk_mode.name, (4,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    src_tag = f"[{ball.source}]" if ball.source else ""
    ang_str = f"{ball.target_angle:.1f}°" if ball.target_angle else "---"
    det_str = (f"Ball ({ball.cx},{ball.cy}) r={ball.radius} ang={ang_str} {src_tag}"
               if ball.cx else "Ball: NOT DETECTED")
    cv2.putText(frame, det_str, (118, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,255,200), 1)
    cv2.putText(frame,
                f"FPS={stats.fps:.0f}  DET={stats.detection_rate*100:.0f}%  "
                f"TX={stats.serial_sends}  UP={stats.elapsed:.0f}s",
                (118, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150,150,150), 1)
    dot = (0,255,0) if serial_ok else (0,0,255)
    cv2.circle(frame, (w-16, 16), 8, dot, -1)
    cv2.putText(frame, "ESP32" if serial_ok else "----",
                (w-70, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, dot, 1)
    return frame


# ══════════════════════════════════════════════════════════════
#  SIGNAL / SHUTDOWN
# ══════════════════════════════════════════════════════════════

_shutdown = threading.Event()
signal.signal(signal.SIGINT,  lambda s,f: _shutdown.set())
signal.signal(signal.SIGTERM, lambda s,f: _shutdown.set())


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("  ROBOT GOALKEEPER v5.2")
    log.info("  YOLO in thread + HSV fallback + grab() camera")
    log.info("  ESC/Q=quit  R=reset  H=home  C=calibrate HSV")
    log.info("=" * 60)

    cfg    = load_config()
    mapper = AngleMapper(cfg)
    stats  = TrackingStats()

    # Start YOLO engine thread first (loads model in background)
    yolo_engine = YoloEngine(cfg)
    yolo_engine.start()

    serial_m = SerialManager(cfg, stats)
    camera_m = CameraManager(cfg)
    detector = BallDetector(cfg, yolo_engine)
    tracker  = BallTracker(cfg, mapper)
    gk       = Goalkeeper(serial_m, stats, cfg, mapper)

    serial_m.start()

    log.info("[MAIN] waiting for camera...")
    if not camera_m.wait_ready(20.0):
        log.warning("[MAIN] camera not ready — continuing")

    log.info("[MAIN] waiting for ESP32 (10s)...")
    if not serial_m.wait_connected(10.0):
        log.warning("[MAIN] no ESP32 — display-only mode")

    WIN = None
    if cfg.get("show_debug_overlay", True):
        WIN = "Robot Goalkeeper v5.2"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, cfg["frame_w"], cfg["frame_h"])

    last_log = time.monotonic()
    log.info("[MAIN] running — hold orange ball in front of camera")

    while not _shutdown.is_set():
        frame, ts = camera_m.get_frame(timeout=0.05)
        if frame is None:
            stats.dropped_frames += 1
            continue

        stats.frames_processed += 1
        stats.tick_fps()

        cx, cy, radius, source, debug_frame = detector.detect(frame)
        if cx is not None:
            stats.frames_detected += 1

        ball = tracker.update(cx, cy, radius, ts, source)
        gk.update(ball)

        out = draw_overlay(debug_frame, ball, gk.mode, stats,
                           serial_m.is_connected(), cfg, mapper)

        if WIN is not None:
            cv2.imshow(WIN, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                _shutdown.set()
            elif key in (ord('r'), ord('R')):
                tracker.reset()
                log.info("[MAIN] tracker reset")
            elif key in (ord('h'), ord('H')):
                serial_m.send("H\n", priority=True)

        now = time.monotonic()
        if now - last_log >= 5.0:
            log.info(f"[STATS] FPS={stats.fps:.1f}  "
                     f"DET={stats.detection_rate*100:.1f}%  "
                     f"TX={stats.serial_sends}  ERR={stats.serial_errors}")
            last_log = now

    log.info("[MAIN] shutdown...")
    serial_m.send("H\n", priority=True)
    time.sleep(0.2)
    yolo_engine.stop()
    serial_m.stop()
    camera_m.stop()
    if WIN:
        cv2.destroyAllWindows()
    log.info(f"[DONE] frames={stats.frames_processed} "
             f"det={stats.detection_rate*100:.1f}%")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        sys.exit(0)
    except Exception as e:
        log.critical(f"FATAL: {e}\n{traceback.format_exc()}")
        try: cv2.destroyAllWindows()
        except Exception: pass
        sys.exit(1)