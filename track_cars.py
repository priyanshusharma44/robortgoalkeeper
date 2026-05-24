# pingpong_tracker.py
# YOLOv8 Ping Pong Ball Tracker — clean square box like car tracker
# pip install ultralytics opencv-python

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time

# ── CONFIG ────────────────────────────────────────────────────────────────────
VIDEO_PATH  = r"C:\Users\serjo\Downloads\WhatsApp Video 2026-05-06 at 3.04.30 PM.mp4"
OUTPUT_PATH = r"C:\Users\serjo\Downloads\pingpong_tracked_output.mp4"
MODEL_NAME  = "yolov8x.pt"   # best accuracy for tiny objects
CONF        = 0.15            # low conf — ball is tiny
IOU         = 0.3
BALL_CLASS  = 32              # COCO class 32 = sports ball
TRAIL_LEN   = 35
# ─────────────────────────────────────────────────────────────────────────────

print(f"[INFO] Loading {MODEL_NAME} ...")
model = YOLO(MODEL_NAME)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS   = cap.get(cv2.CAP_PROP_FPS) or 30
TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (W, H))

trail            = deque(maxlen=TRAIL_LEN)
prev_gray        = None
frame_idx        = 0
ball_found_count = 0
start            = time.time()

# ── Background subtractor — used to FILTER yolo boxes by motion ──────────────
bg_sub = cv2.createBackgroundSubtractorMOG2(
    history=120, varThreshold=50, detectShadows=False
)

def is_moving(fg_mask, x1, y1, x2, y2, threshold=0.15):
    """Return True if enough pixels inside the box are in motion."""
    roi = fg_mask[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    return np.count_nonzero(roi) / roi.size > threshold

def draw_square_box(frame, x1, y1, x2, y2, color=(0, 229, 160)):
    """Draw a clean square box — same style as car tracker."""
    # Force square from center
    cx, cy  = (x1 + x2) // 2, (y1 + y2) // 2
    half    = max((x2 - x1), (y2 - y1)) // 2 + 8   # slight padding
    sx1, sy1 = max(0, cx - half), max(0, cy - half)
    sx2, sy2 = min(W, cx + half), min(H, cy + half)

    # Main box
    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2, cv2.LINE_AA)

    # Corner accents (like a camera/target reticle)
    corner = half // 2
    thick  = 3
    # Top-left
    cv2.line(frame, (sx1, sy1), (sx1 + corner, sy1), color, thick, cv2.LINE_AA)
    cv2.line(frame, (sx1, sy1), (sx1, sy1 + corner), color, thick, cv2.LINE_AA)
    # Top-right
    cv2.line(frame, (sx2, sy1), (sx2 - corner, sy1), color, thick, cv2.LINE_AA)
    cv2.line(frame, (sx2, sy1), (sx2, sy1 + corner), color, thick, cv2.LINE_AA)
    # Bottom-left
    cv2.line(frame, (sx1, sy2), (sx1 + corner, sy2), color, thick, cv2.LINE_AA)
    cv2.line(frame, (sx1, sy2), (sx1, sy2 - corner), color, thick, cv2.LINE_AA)
    # Bottom-right
    cv2.line(frame, (sx2, sy2), (sx2 - corner, sy2), color, thick, cv2.LINE_AA)
    cv2.line(frame, (sx2, sy2), (sx2, sy2 - corner), color, thick, cv2.LINE_AA)

    # Center crosshair dot
    cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)

    return cx, cy, sx1, sy1, sx2, sy2

print(f"[INFO] Video : {W}x{H} @ {FPS:.1f}fps  |  {TOTAL} frames")
print(f"[INFO] Output: {OUTPUT_PATH}\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    display  = frame.copy()
    fg_mask  = bg_sub.apply(frame)

    # ── Run YOLO — balls only ─────────────────────────────────────────────────
    results = model.track(
        frame,
        persist   = True,
        conf      = CONF,
        iou       = IOU,
        classes   = [BALL_CLASS],   # ← ONLY sports ball, nothing else
        tracker   = "bytetrack.yaml",
        verbose   = False,
    )

    ball_center = None
    best_conf   = 0

    if results[0].boxes is not None:
        boxes = results[0].boxes

        # IDs may be None on first frame
        ids   = boxes.id.cpu().numpy().astype(int) if boxes.id is not None \
                else [1] * len(boxes)
        confs = boxes.conf.cpu().numpy()
        xyxys = boxes.xyxy.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), tid, cf in zip(xyxys, ids, confs):

            # ── Extra filter: must be MOVING (removes static false positives) ──
            if not is_moving(fg_mask, x1, y1, x2, y2):
                continue

            # ── Extra filter: must be roughly small & square (ball-shaped) ──
            bw, bh = x2 - x1, y2 - y1
            aspect = max(bw, bh) / (min(bw, bh) + 1e-5)
            if aspect > 2.5:          # too elongated — not a ball
                continue
            if max(bw, bh) > 120:     # too large — not a ping pong ball
                continue

            # Pick highest-confidence ball this frame
            if cf > best_conf:
                best_conf = cf
                # Draw the box and get center
                cx, cy, sx1, sy1, sx2, sy2 = draw_square_box(display, x1, y1, x2, y2)

                # Label above box
                label = f"Ball  {cf:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(display, (sx1, sy1 - th - 6), (sx1 + tw + 6, sy1), (0, 229, 160), -1)
                cv2.putText(display, label, (sx1 + 3, sy1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

                ball_center = (cx, cy)

    # ── Motion trail ─────────────────────────────────────────────────────────
    if ball_center:
        trail.append(ball_center)
        ball_found_count += 1

        for i in range(1, len(trail)):
            alpha = i / len(trail)
            color = (0, int(200 * alpha), int(255 * alpha))
            thick = max(1, int(4 * alpha))
            cv2.line(display, trail[i-1], trail[i], color, thick, cv2.LINE_AA)

        # Speed
        if len(trail) >= 2:
            dx  = trail[-1][0] - trail[-2][0]
            dy  = trail[-1][1] - trail[-2][1]
            spd = (dx**2 + dy**2) ** 0.5
            cv2.putText(display, f"{spd:.0f} px/f",
                        (ball_center[0] + 20, ball_center[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
    else:
        if len(trail) > 0:
            trail.popleft()   # fade trail gradually when ball lost

    # ── HUD ──────────────────────────────────────────────────────────────────
    elapsed   = time.time() - start
    fps_live  = frame_idx / elapsed if elapsed > 0 else 0
    det_rate  = ball_found_count / frame_idx * 100
    status    = "BALL DETECTED" if ball_center else "searching..."
    hud_color = (0, 229, 160) if ball_center else (0, 100, 255)

    cv2.putText(display,
        f"Frame {frame_idx}/{TOTAL}  |  {fps_live:.1f} FPS  |  {status}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2, cv2.LINE_AA)

    cv2.putText(display,
        f"Detection rate: {det_rate:.1f}%  |  model: {MODEL_NAME}",
        (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    out.write(display)

    if frame_idx % 60 == 0:
        pct = frame_idx / TOTAL * 100
        print(f"  [{pct:5.1f}%]  frame {frame_idx}/{TOTAL}  "
              f"ball={'YES' if ball_center else 'no ':3}  "
              f"det_rate={det_rate:.1f}%  fps={fps_live:.1f}")

cap.release()
out.release()

print(f"\n[DONE] {frame_idx} frames in {time.time()-start:.1f}s")
print(f"[DONE] Ball detected in {det_rate:.1f}% of frames")
print(f"[DONE] Output → {OUTPUT_PATH}")