# ═══════════════════════════════════════════════════════════════════════════════
#  ball_tracker_v2.py  —  Universal Ball Tracker (webcam + video)
#  NO HSV color detection. Pure YOLO + Kalman filter + ByteTrack.
#  Works on ping pong, tennis, soccer, basketball, any round fast object.
#
#  pip install ultralytics opencv-python scipy filterpy
# ═══════════════════════════════════════════════════════════════════════════════

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from filterpy.kalman import KalmanFilter
import time
import argparse

# ── ARG PARSE (run: python ball_tracker_v2.py --source 0  for webcam) ─────────
parser = argparse.ArgumentParser()
parser.add_argument("--source",  default="0",
    help="0/1/2 for webcam index, or path to video file")
parser.add_argument("--output",  default="tracked_output.mp4",
    help="Output video path (ignored in webcam live-view mode)")
parser.add_argument("--model",   default="yolov8n.pt",
    help="yolov8n=fastest, yolov8s=balanced, yolov8x=most accurate")
parser.add_argument("--conf",    type=float, default=0.12)
parser.add_argument("--save",    action="store_true",
    help="Save output video even in webcam mode")
args = parser.parse_args()

# ── CONFIG ────────────────────────────────────────────────────────────────────
IS_WEBCAM   = args.source.isdigit()
SOURCE      = int(args.source) if IS_WEBCAM else args.source
MODEL_NAME  = args.model
CONF        = args.conf
IOU         = 0.3
BALL_CLASS  = 32          # COCO: sports ball
TRAIL_LEN   = 45
MAX_LOST    = 8           # frames to keep predicting after ball disappears
# ─────────────────────────────────────────────────────────────────────────────

# ── KALMAN FILTER SETUP ───────────────────────────────────────────────────────
# State: [x, y, vx, vy]  (position + velocity)
# Measurement: [x, y]
def make_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0
    # State transition: constant velocity model
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1,  0],
                     [0, 0, 0,  1]], dtype=float)
    # Measurement function: observe x, y only
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]], dtype=float)
    kf.R  *= 10    # measurement noise
    kf.Q  *= 0.1   # process noise (trust the model)
    kf.P  *= 100   # initial uncertainty
    return kf

kf             = make_kalman()
kf_initialized = False
lost_frames    = 0
predicted_pos  = None

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print(f"[INFO] Loading {MODEL_NAME} ...")
model = YOLO(MODEL_NAME)
# Warm-up pass so first frame isn't slow
_ = model(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False)
print(f"[INFO] Model ready.\n")

# ── OPEN SOURCE ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise IOError(f"Cannot open source: {SOURCE}")

if IS_WEBCAM:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          60)   # request 60fps if camera supports it
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)     # minimal buffer = less latency

W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS   = cap.get(cv2.CAP_PROP_FPS) or 30
TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ── WRITER (optional) ─────────────────────────────────────────────────────────
writer = None
if args.save or not IS_WEBCAM:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, FPS, (W, H))

# ── BACKGROUND SUBTRACTOR ─────────────────────────────────────────────────────
bg_sub = cv2.createBackgroundSubtractorMOG2(
    history=200, varThreshold=40, detectShadows=False
)

# ── TRAIL ─────────────────────────────────────────────────────────────────────
trail = deque(maxlen=TRAIL_LEN)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def motion_ratio(fg_mask, x1, y1, x2, y2):
    roi = fg_mask[max(0,y1):y2, max(0,x1):x2]
    if roi.size == 0:
        return 0.0
    return np.count_nonzero(roi) / roi.size

def is_ball_shaped(x1, y1, x2, y2):
    """Aspect ratio + size guard — rejects sticks, humans, etc."""
    bw, bh   = x2 - x1, y2 - y1
    aspect   = max(bw, bh) / (min(bw, bh) + 1e-5)
    max_side = max(bw, bh)
    # Ball can be close (large) or far (tiny) — only reject extreme aspect ratios
    return aspect < 2.2

def draw_box(frame, cx, cy, half, conf, is_predicted=False):
    """Draw reticle box. Yellow = Kalman prediction, green = detected."""
    color  = (0, 200, 255) if is_predicted else (0, 229, 160)
    sx1    = max(0, cx - half)
    sy1    = max(0, cy - half)
    sx2    = min(W, cx + half)
    sy2    = min(H, cy + half)

    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2, cv2.LINE_AA)

    corner = half // 2
    thick  = 3
    for (px, py), (dx, dy) in [
        ((sx1, sy1), ( 1,  1)),
        ((sx2, sy1), (-1,  1)),
        ((sx1, sy2), ( 1, -1)),
        ((sx2, sy2), (-1, -1)),
    ]:
        cv2.line(frame, (px, py), (px + dx*corner, py),        color, thick, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + dy*corner),        color, thick, cv2.LINE_AA)

    cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)

    tag = f"{'~' if is_predicted else ''}Ball {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (sx1, sy1 - th - 8), (sx1 + tw + 6, sy1), color, -1)
    cv2.putText(frame, tag, (sx1 + 3, sy1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return sx1, sy1, sx2, sy2

def draw_trail(frame, trail):
    for i in range(1, len(trail)):
        alpha = i / len(trail)
        color = (0, int(180 * alpha), int(255 * alpha))
        thick = max(1, int(5 * alpha))
        cv2.line(frame, trail[i-1], trail[i], color, thick, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════
frame_idx        = 0
ball_found_count = 0
start            = time.time()

print(f"[INFO] Source: {'Webcam #' + str(SOURCE) if IS_WEBCAM else SOURCE}")
print(f"[INFO] Resolution: {W}x{H} @ {FPS:.0f}fps")
print(f"[INFO] Press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    display  = frame.copy()
    fg_mask  = bg_sub.apply(frame)

    # ── YOLO inference ───────────────────────────────────────────────────────
    results = model.track(
        frame,
        persist  = True,
        conf     = CONF,
        iou      = IOU,
        classes  = [BALL_CLASS],
        tracker  = "bytetrack.yaml",
        verbose  = False,
    )

    detected_center = None
    best_conf       = 0.0
    best_half       = 24   # default box half-size

    if results[0].boxes is not None:
        boxes = results[0].boxes
        ids   = (boxes.id.cpu().numpy().astype(int)
                 if boxes.id is not None else [1]*len(boxes))
        confs  = boxes.conf.cpu().numpy()
        xyxys  = boxes.xyxy.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), tid, cf in zip(xyxys, ids, confs):

            # ── Filters (NO color — pure shape + motion) ─────────────────────
            if motion_ratio(fg_mask, x1, y1, x2, y2) < 0.10:
                continue
            if not is_ball_shaped(x1, y1, x2, y2):
                continue

            if cf > best_conf:
                best_conf = cf
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detected_center = (cx, cy)
                best_half = max((x2-x1), (y2-y1)) // 2 + 8

    # ── Kalman update / predict ───────────────────────────────────────────────
    if detected_center is not None:
        cx, cy = detected_center
        if not kf_initialized:
            kf.x = np.array([[cx], [cy], [0.], [0.]])
            kf_initialized = True
        else:
            kf.predict()
            kf.update(np.array([[cx], [cy]]))

        lost_frames    = 0
        ball_found_count += 1
        predicted_pos  = None

    elif kf_initialized and lost_frames < MAX_LOST:
        # Ball not detected — use Kalman to PREDICT where it should be
        kf.predict()
        px = int(kf.x[0, 0])
        py = int(kf.x[1, 0])
        # Keep prediction inside frame
        px = np.clip(px, 0, W)
        py = np.clip(py, 0, H)
        predicted_pos = (px, py)
        lost_frames  += 1
    else:
        predicted_pos = None

    # ── Draw detected box ─────────────────────────────────────────────────────
    if detected_center is not None:
        cx, cy = detected_center
        trail.append((cx, cy))
        draw_box(display, cx, cy, best_half, best_conf, is_predicted=False)

        # Speed annotation
        if len(trail) >= 2:
            dx  = trail[-1][0] - trail[-2][0]
            dy  = trail[-1][1] - trail[-2][1]
            spd = (dx**2 + dy**2) ** 0.5
            cv2.putText(display, f"{spd:.0f} px/f",
                        (cx + best_half + 4, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

    elif predicted_pos is not None:
        # Show Kalman ghost box
        px, py = predicted_pos
        trail.append((px, py))
        draw_box(display, px, py, best_half, best_conf, is_predicted=True)

    else:
        # Fade trail gradually
        if trail:
            trail.popleft()

    # ── Draw trail ────────────────────────────────────────────────────────────
    draw_trail(display, trail)

    # ── HUD ───────────────────────────────────────────────────────────────────
    elapsed   = time.time() - start
    fps_live  = frame_idx / elapsed if elapsed > 0 else 0
    det_rate  = ball_found_count / frame_idx * 100

    if detected_center:
        status, hud_col = "● BALL DETECTED",  (0, 229, 160)
    elif predicted_pos:
        status, hud_col = "~ PREDICTING",     (0, 200, 255)
    else:
        status, hud_col = "○ searching...",   (80, 80, 255)

    frame_info = (f"Frame {frame_idx}/{TOTAL}  |  {fps_live:.1f} FPS  |  {status}"
                  if not IS_WEBCAM else
                  f"{fps_live:.1f} FPS  |  {status}")

    cv2.putText(display, frame_info,
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_col, 2, cv2.LINE_AA)
    cv2.putText(display,
                f"Det rate: {det_rate:.1f}%  |  model: {MODEL_NAME}  |  [Q] quit",
                (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    # ── Show / write ──────────────────────────────────────────────────────────
    if writer:
        writer.write(display)

    cv2.imshow("Ball Tracker v2", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quit by user.")
        break

    if not IS_WEBCAM and frame_idx % 60 == 0:
        pct = frame_idx / TOTAL * 100 if TOTAL > 0 else 0
        kstate = "det" if detected_center else ("pred" if predicted_pos else "lost")
        print(f"  [{pct:5.1f}%]  f={frame_idx}  {kstate}  det={det_rate:.1f}%  fps={fps_live:.1f}")

# ── CLEANUP ───────────────────────────────────────────────────────────────────
cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()

elapsed = time.time() - start
print(f"\n[DONE] {frame_idx} frames in {elapsed:.1f}s  ({frame_idx/elapsed:.1f} FPS avg)")
print(f"[DONE] Ball detected in {ball_found_count/max(1,frame_idx)*100:.1f}% of frames")
if writer:
    print(f"[DONE] Output → {args.output}")