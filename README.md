# 🧠 YOLOv10 Smart Object Detection App

This is a **Streamlit-based web application** for real-time object detection using the **YOLOv10 (Ultralytics)** model. It allows users to perform detection on:

- 📸 Uploaded images
- 🎞️ Uploaded videos
- 📷 Live webcam feed

The UI is clean, interactive, and designed to provide a smart dashboard experience with metrics and real-time feedback.

---

## 🚀 Features

- ✅ Supports **image**, **video**, and **live camera** detection modes
- ✅ Displays total object counts and per-class statistics
- ✅ Option to **save frames** during live detection
- ✅ Clean and responsive UI with custom layout
- ✅ Real-time detection using `YOLOv10m.pt` model (compatible with YOLOv8 interface)
- ✅ Lightweight and easy to use with just a few dependencies


## 📦 Requirements

- Python 3.8+
- [YOLO (Ultralytics)](https://docs.ultralytics.com)
- Streamlit
- OpenCV
- NumPy

Install them via pip:

```bash
pip install streamlit opencv-python numpy ultralytics
