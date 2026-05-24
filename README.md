# 🥅 Robot Goalkeeper (robortgoalkeeper)

An AI-powered **robot goalkeeper system** that tracks fast-moving balls using computer vision and commands a motorized keeper to dive left/right in real time.

---

## ✨ Highlights

- ⚡ Real-time ball tracking with **YOLOv8**
- 🎯 Optional **HSV fallback detection** for orange/yellow balls
- 🧠 Motion smoothing + prediction using **Kalman filtering**
- 🔌 Serial communication with **ESP32/Arduino motor controller**
- 🛡️ Goalkeeper logic with dive confirmation and cooldown protection
- 🧪 Includes calibration and motor test utilities

---

## 📁 Project Structure

- `goalkeeper.py` — Main advanced robot goalkeeper pipeline (tracking + prediction + motor control)
- `keeper_yolo.py` — Alternative fast/reliable threaded YOLO+HSV goalkeeper implementation
- `goalkeeper_vision.py` — Simpler HSV-based goalkeeper vision controller
- `calibrate_hsv.py` — HSV calibration tool for your ball and lighting
- `motor_test.py` — Motor/serial communication test script
- `goalkeeper_config.json` — Runtime config for YOLO/HSV/motor behavior
- `track_cars.py`, `track_cars2.py` — Ball/object tracking experiments

---

## 🧰 Tech Stack

- Python 3.9+
- OpenCV
- Ultralytics YOLOv8
- NumPy
- FilterPy
- PySerial

---

## 📦 Installation

```bash
pip install ultralytics opencv-python filterpy pyserial numpy scipy
```

---

## 🤖 Model Files (Important)

These trained model files are required and were not uploaded to the repository:

- `yolov8n.pt`
- `yolov8s.pt`
- `yolov8x.pt`

Place them in the **project root directory** (same folder as `goalkeeper.py`).

---

## 🚀 Quick Start

### 1) Run the main goalkeeper system

```bash
python goalkeeper.py --source 0 --port COM8
```

Useful options:

- `--no-motor` → run vision only (no hardware control)
- `--model yolov8n.pt` / `yolov8s.pt` / `yolov8x.pt`
- `--calibrate` → run calibration flow
- `--save` → save output video
- `--debug` → extra debugging info

### 2) Calibrate HSV (recommended)

```bash
python calibrate_hsv.py
```

### 3) Test motor connection

```bash
python motor_test.py --port COM8
```

---

## 🎮 Runtime Controls (`goalkeeper.py`)

- `Q` / `Esc` — Quit
- `H` — Home motor (90°)
- `C` — Flip direction mapping
- `R` — Reset tracking
- `+` / `-` — Move reaction zone
- `S` — Save calibration
- `D` — Toggle debug overlay

---

## ⚙️ Hardware Notes

- Connect camera and motor controller before running.
- Update serial port (`COMx` on Windows or `/dev/ttyUSBx` on Linux) based on your system.
- If testing without hardware, use `--no-motor`.

---

## 🧪 Typical Development Flow

1. Place YOLO model files in root folder.
2. Run `calibrate_hsv.py` in your actual lighting setup.
3. Run `motor_test.py` to verify hardware response.
4. Start `goalkeeper.py` and tune config if needed.

---

## 📌 Configuration

Tune behavior in:

- `goalkeeper_config.json` (YOLO confidence, HSV ranges, angle limits, deadzones, FPS, watchdog, etc.)

---

## 🙌 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV community

---

If you want, I can also provide a **version with badges, architecture diagram, and GIF preview placeholders** for an even more portfolio-ready README.
