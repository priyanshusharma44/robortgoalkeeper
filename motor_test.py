# motor_test.py — Simple Motor Movement Test
# Usage: python motor_test.py --port COM8

import serial
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", default="COM8")
parser.add_argument("--baud", type=int, default=115200)
args = parser.parse_args()

print(f"Connecting to {args.port}...")

try:
    ser = serial.Serial(args.port, args.baud, timeout=3)
    time.sleep(2.5)  # wait for ESP32 boot
    ser.reset_input_buffer()
    print("Connected!\n")
except Exception as e:
    print(f"FAILED to open port: {e}")
    exit(1)

def send(cmd):
    ser.write((cmd + "\n").encode())
    ser.flush()
    time.sleep(0.1)
    response = ""
    while ser.in_waiting:
        response += ser.readline().decode("utf-8", errors="ignore").strip() + " "
    print(f"  Sent: {cmd:15s}  Got: {response.strip()}")

print("=" * 40)
print("  MOTOR TEST SEQUENCE")
print("=" * 40)

print("\nStep 1 — Check status")
send("S")
time.sleep(0.5)

print("\nStep 2 — Go HOME (90 degrees)")
send("H")
time.sleep(3)

print("\nStep 3 — Move to 120 degrees (RIGHT)")
send("A:120.0")
time.sleep(3)

print("\nStep 4 — Move to 60 degrees (LEFT)")
send("A:60.0")
time.sleep(3)

print("\nStep 5 — Move to 140 degrees (FAR RIGHT)")
send("A:140.0")
time.sleep(3)

print("\nStep 6 — Move to 40 degrees (FAR LEFT)")
send("A:40.0")
time.sleep(3)

print("\nStep 7 — Return HOME (90 degrees)")
send("H")
time.sleep(3)

print("\nStep 8 — Final status check")
send("S")

print("\n" + "=" * 40)
print("TEST COMPLETE")
print("If motor moved through all positions -> hardware OK")
print("If motor did not move at any step    -> wiring/power issue")
print("=" * 40)

ser.close()