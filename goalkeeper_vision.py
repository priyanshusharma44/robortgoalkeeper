"""
============================================================
GOALKEEPER ROBOT — Python Vision Script
File: goalkeeper_vision.py

Run on your Laptop / Raspberry Pi
Requires: pip install opencv-python pyserial numpy

WHAT THIS FILE DOES:
  1. Opens USB webcam
  2. Every frame: converts to HSV colour space
  3. Masks out everything except your football (orange/yellow by default)
  4. Finds the ball's X position in the frame
  5. Sends 'L', 'R', or 'G' to Arduino over USB serial
  6. Shows a live debug window so you can see what the camera sees

HOW TO RUN:
  python goalkeeper_vision.py

Press ESC in the camera window to quit.
============================================================
"""

import cv2
import numpy as np
import serial
import time
import sys

# ============================================================
# CONFIGURATION — change these to match your setup
# ============================================================

# --- Serial port ---
# Windows: "COM3"  (check Device Manager → Ports)
# Linux/Mac: "/dev/ttyUSB0" or "/dev/ttyACM0"
# If you DON'T have Arduino yet, set ARDUINO_CONNECTED = False
ARDUINO_CONNECTED = True
SERIAL_PORT = "COM3"       # <-- CHANGE THIS if needed
BAUD_RATE   = 9600         # Must match Arduino Serial.begin(9600)

# --- Camera ---
CAMERA_INDEX = 0   # 0 = first USB camera. Try 1 or 2 if wrong camera opens.

# --- Ball colour (HSV range) ---
# These values work for an ORANGE ball under normal indoor lighting.
# Tune them if your ball is a different colour — see CALIBRATION section below.
LOWER_HSV = np.array([5,  120, 120])   # [Hue_min, Sat_min, Val_min]
UPPER_HSV = np.array([20, 255, 255])   # [Hue_max, Sat_max, Val_max]

# --- Detection threshold ---
# Ignore detected blobs smaller than this (filters out reflections, noise)
MIN_BALL_AREA = 500   # pixels²  — increase if you get false detections

# --- Frame zones ---
# Camera frame is split into 3 thirds:
#   LEFT  third: ball X < frame_width * LEFT_ZONE_FRACTION  → send 'L'
#   RIGHT third: ball X > frame_width * RIGHT_ZONE_FRACTION → send 'R'
#   CENTRE:      anything in between                         → send 'G'
LEFT_ZONE_FRACTION  = 0.33
RIGHT_ZONE_FRACTION = 0.66

# ============================================================
# SETUP
# ============================================================

# Connect to Arduino (skip if not connected yet)
arduino = None
if ARDUINO_CONNECTED:
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)   # Arduino resets when serial opens — wait for it
        print(f"[OK] Arduino connected on {SERIAL_PORT}")
    except serial.SerialException as e:
        print(f"[WARN] Could not connect to Arduino: {e}")
        print("       Running in CAMERA-ONLY mode (no motor control)")
        arduino = None

# Open webcam
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"[ERROR] Could not open camera index {CAMERA_INDEX}")
    sys.exit(1)

print("[OK] Camera opened. Press ESC to quit.")
print("     Position your ball in the camera view...")

# Track last command sent (avoids spamming same command repeatedly)
last_command = None

def send_command(cmd):
    """Send a single-byte command to Arduino, only if it changed."""
    global last_command
    if cmd != last_command:
        if arduino:
            arduino.write(cmd.encode())
        last_command = cmd
        print(f"  → Sent: {cmd}")

# ============================================================
# MAIN LOOP
# ============================================================

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read from camera")
        break

    # --- FLIP FRAME ---
    # If camera is mounted facing the field (not a mirror), flip horizontally
    # so LEFT on screen = LEFT of the goal.
    # Comment this line out if movement direction is reversed.
    frame = cv2.flip(frame, 1)

    frame_height, frame_width = frame.shape[:2]

    # --- COLOUR DETECTION (HSV masking) ---
    # HSV separates colour (Hue) from brightness (Value),
    # making detection more robust under changing lighting than RGB.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a binary mask: white where ball colour, black everywhere else
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)

    # Clean up the mask (remove small holes and noise)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask,  kernel, iterations=1)   # remove tiny white dots
    mask = cv2.dilate(mask, kernel, iterations=2)   # fill gaps in ball

    # --- FIND BALL ---
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ball_found = False
    command = 'G'   # Default: no ball = stop motor

    if contours:
        # Pick the largest blob (most likely the ball)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > MIN_BALL_AREA:
            ball_found = True

            # Get bounding box of the ball
            x, y, w, h = cv2.boundingRect(largest)
            cx = x + w // 2    # Ball centre X coordinate
            cy = y + h // 2    # Ball centre Y coordinate

            # --- DECIDE DIRECTION ---
            left_boundary  = int(frame_width * LEFT_ZONE_FRACTION)
            right_boundary = int(frame_width * RIGHT_ZONE_FRACTION)

            if cx < left_boundary:
                command = 'L'       # Ball on left → move keeper left
                zone_text = "LEFT"
                box_colour = (0, 0, 255)     # Red box
            elif cx > right_boundary:
                command = 'R'       # Ball on right → move keeper right
                zone_text = "RIGHT"
                box_colour = (255, 0, 0)     # Blue box
            else:
                command = 'G'       # Ball centre → GOAL, stop keeper
                zone_text = "CENTRE - GOAL!"
                box_colour = (0, 255, 0)     # Green box

            # --- DRAW DEBUG VISUALS ---
            # Ball bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_colour, 2)

            # Ball centre dot
            cv2.circle(frame, (cx, cy), 6, box_colour, -1)

            # Ball info text
            cv2.putText(frame, f"Ball: {zone_text}  Area:{int(area)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_colour, 2)

    # --- SEND COMMAND TO ARDUINO ---
    send_command(command)

    # --- DRAW ZONE LINES ---
    left_boundary  = int(frame_width * LEFT_ZONE_FRACTION)
    right_boundary = int(frame_width * RIGHT_ZONE_FRACTION)
    cv2.line(frame, (left_boundary,  0), (left_boundary,  frame_height), (200, 200, 200), 1)
    cv2.line(frame, (right_boundary, 0), (right_boundary, frame_height), (200, 200, 200), 1)

    # Zone labels
    cv2.putText(frame, "L", (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(frame, "GOAL", (left_boundary + 10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(frame, "R", (right_boundary + 10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # Command indicator
    cmd_colour = {'L': (0,0,255), 'R': (255,0,0), 'G': (0,255,0)}.get(command, (255,255,255))
    cv2.putText(frame, f"CMD: {command}", (frame_width - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, cmd_colour, 2)

    # No-ball indicator
    if not ball_found:
        cv2.putText(frame, "NO BALL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    # --- SHOW WINDOWS ---
    cv2.imshow("Goalkeeper Vision (Main)", frame)
    cv2.imshow("Ball Mask (Debug)", mask)   # Shows what the HSV mask sees

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        print("ESC pressed — exiting")
        break

# ============================================================
# CLEANUP
# ============================================================
cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.write(b'S')   # Send Stop before closing
    arduino.close()
    print("Arduino disconnected")
print("Done.")


# ============================================================
# HSV CALIBRATION GUIDE
# If your ball colour isn't being detected:
#
# Step 1: Run this colour picker to find your ball's HSV values
#         (paste this into a separate file and run it)
#
# import cv2, numpy as np
# cap = cv2.VideoCapture(0)
# def nothing(x): pass
# cv2.namedWindow("Calibrate")
# cv2.createTrackbar("H_min","Calibrate", 0,179,nothing)
# cv2.createTrackbar("H_max","Calibrate",179,179,nothing)
# cv2.createTrackbar("S_min","Calibrate",120,255,nothing)
# cv2.createTrackbar("S_max","Calibrate",255,255,nothing)
# cv2.createTrackbar("V_min","Calibrate",120,255,nothing)
# cv2.createTrackbar("V_max","Calibrate",255,255,nothing)
# while True:
#     _, frame = cap.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lo = np.array([cv2.getTrackbarPos("H_min","Calibrate"),
#                    cv2.getTrackbarPos("S_min","Calibrate"),
#                    cv2.getTrackbarPos("V_min","Calibrate")])
#     hi = np.array([cv2.getTrackbarPos("H_max","Calibrate"),
#                    cv2.getTrackbarPos("S_max","Calibrate"),
#                    cv2.getTrackbarPos("V_max","Calibrate")])
#     mask = cv2.inRange(hsv, lo, hi)
#     print(f"Lower:{lo}  Upper:{hi}", end="\r")
#     cv2.imshow("Frame", frame)
#     cv2.imshow("Mask",  mask)
#     if cv2.waitKey(1) & 0xFF == 27: break
# cap.release(); cv2.destroyAllWindows()
#
# Tune until only your ball shows as white in the Mask window.
# Then paste those values into LOWER_HSV and UPPER_HSV above.
#
# Common ball HSV ranges:
#   Orange ball:  [5,120,120] → [20,255,255]
#   Yellow ball:  [22,100,100] → [40,255,255]
#   Red ball:     [0,120,70] → [10,255,255]  (also check [170,120,70]→[180,255,255])
#   White ball:   [0,0,200] → [180,30,255]
# ============================================================