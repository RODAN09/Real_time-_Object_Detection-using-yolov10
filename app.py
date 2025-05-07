import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time
from collections import Counter
import os

# Load model
model = YOLO("yolov10m.pt")

# Page setup
st.set_page_config(page_title="🧠 YOLOv10 Smart Detection", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 YOLOv10 Object Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Options")
    mode = st.radio("Detection Mode", ["📸 Image", "🎞️ Video", "📷 Live Camera"])
    uploaded_file = st.file_uploader("Upload Image/Video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])
    start_detection = st.checkbox("🔴 Start Live Detection", key="live_checkbox")
    save_frame_button = st.button("📸 Save Current Frame (Live Only)", key="save_button")
    st.markdown("---")
    st.info("💡 Select a mode and upload a file or enable live detection.")

frame_placeholder = st.empty()
stats_placeholder = st.empty()
saved_once = False

# --- Detection Functions ---

def detect_objects(image):
    results = model(image)
    boxes = results[0].boxes
    class_ids = boxes.cls.tolist()
    class_names = [model.names[int(cls)] for cls in class_ids]
    count_per_class = Counter(class_names)
    total_objects = len(class_names)

    annotated = results[0].plot()

    return annotated, count_per_class, total_objects

def display_stats(counts, total):
    cols = st.columns(len(counts) + 1)
    for i, (cls, count) in enumerate(counts.items()):
        cols[i].metric(label=f"🔍 {cls}", value=count)
    cols[-1].metric(label="📦 Total Objects", value=total)

def detect_on_image(image):
    annotated, count_per_class, total_objects = detect_objects(image)
    frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
    display_stats(count_per_class, total_objects)

def detect_on_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated, count_per_class, total_objects = detect_objects(frame)
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
        display_stats(count_per_class, total_objects)
        time.sleep(0.03)

    cap.release()

def live_camera_detection():
    global saved_once
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Cannot open webcam.")
        return

    st.success("✅ Live camera started.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, count_per_class, total_objects = detect_objects(frame)
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
        display_stats(count_per_class, total_objects)

        if save_frame_button and not saved_once:
            cv2.imwrite("saved_live_frame.jpg", annotated)
            st.sidebar.success("📸 Frame saved as saved_live_frame.jpg")
            saved_once = True

        time.sleep(0.01)

        if not st.session_state.live_checkbox:
            break

    cap.release()
    st.success("✅ Camera stopped.")


# --- Main Logic ---

if uploaded_file:
    file_bytes = uploaded_file.read()
    temp_path = tempfile.NamedTemporaryFile(delete=False)
    temp_path.write(file_bytes)
    temp_path.flush()

    if mode == "📸 Image":
        st.subheader("🖼️ Processed Image Output")
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 1)
        detect_on_image(img)

    elif mode == "🎞️ Video":
        st.subheader("🎬 Processing Video")
        detect_on_video(temp_path.name)

    os.unlink(temp_path.name)

elif mode == "📷 Live Camera" and start_detection:
    st.subheader("🟢 Real-time Camera Detection")
    live_camera_detection()

else:
    st.warning("👈 Please upload a file or start live detection.")
