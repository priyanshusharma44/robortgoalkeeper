"""
============================================================
HSV COLOUR CALIBRATION TOOL
File: calibrate_hsv.py

Run this BEFORE goalkeeper_vision.py to find the right
HSV values for YOUR specific ball under YOUR lighting.

HOW TO USE:
  1. python calibrate_hsv.py
  2. Hold your ball in front of the camera
  3. Drag the trackbars until ONLY the ball is white in the Mask window
  4. Note down the 6 numbers printed in the terminal
  5. Paste them into goalkeeper_vision.py:
       LOWER_HSV = np.array([H_min, S_min, V_min])
       UPPER_HSV = np.array([H_max, S_max, V_max])
  6. Press ESC to quit
============================================================
"""

import cv2
import numpy as np

CAMERA_INDEX = 0  # Change if wrong camera

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

def nothing(x):
    pass

cv2.namedWindow("Calibration Controls")

# Default starting values for orange ball
cv2.createTrackbar("H min", "Calibration Controls",  5, 179, nothing)
cv2.createTrackbar("H max", "Calibration Controls", 20, 179, nothing)
cv2.createTrackbar("S min", "Calibration Controls", 120, 255, nothing)
cv2.createTrackbar("S max", "Calibration Controls", 255, 255, nothing)
cv2.createTrackbar("V min", "Calibration Controls", 120, 255, nothing)
cv2.createTrackbar("V max", "Calibration Controls", 255, 255, nothing)

print("Hold your ball in the camera view.")
print("Adjust trackbars until ONLY the ball is white in the Mask window.")
print("Press ESC to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("H min", "Calibration Controls")
    h_max = cv2.getTrackbarPos("H max", "Calibration Controls")
    s_min = cv2.getTrackbarPos("S min", "Calibration Controls")
    s_max = cv2.getTrackbarPos("S max", "Calibration Controls")
    v_min = cv2.getTrackbarPos("V min", "Calibration Controls")
    v_max = cv2.getTrackbarPos("V max", "Calibration Controls")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower, upper)

    # Clean mask
    kernel = np.ones((5,5), np.uint8)
    mask_clean = cv2.erode(mask,  kernel, iterations=1)
    mask_clean = cv2.dilate(mask_clean, kernel, iterations=2)

    # Draw detected blobs on frame
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 200:
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Print values to terminal
    print(f"LOWER_HSV = [{h_min}, {s_min}, {v_min}]   UPPER_HSV = [{h_max}, {s_max}, {v_max}]", end="\r")

    cv2.imshow("Camera (with detection)", frame)
    cv2.imshow("Mask (ball should be white here)", mask_clean)

    if cv2.waitKey(1) & 0xFF == 27:
        break

print(f"\n\nFinal values:")
print(f"  LOWER_HSV = np.array([{h_min}, {s_min}, {v_min}])")
print(f"  UPPER_HSV = np.array([{h_max}, {s_max}, {v_max}])")
print("Paste these into goalkeeper_vision.py")

cap.release()
cv2.destroyAllWindows()