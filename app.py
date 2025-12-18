import os
import cv2
import streamlit as st
import warnings

from drowsiness.mediapipe_analyzer import MediapipeAnalyzer

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")

# ---------------- UI ---------------- #
st.sidebar.title("Settings")

method = st.sidebar.selectbox(
    "Select Detection Method",
    ["MediaPipe"],  # Only supported method
)

run_camera = st.sidebar.checkbox("Run Camera")
camera_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=5, value=0)

st.title("🚗 Driver Drowsiness Detection System")

st.markdown(
    """
**Recommended on Windows:** MediaPipe  
CNN and dlib are disabled due to compatibility constraints.
"""
)

if not run_camera:
    st.info("📷 Camera is off. Enable **Run Camera** to start detection.")
    st.stop()

# ---------------- Camera ---------------- #
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    st.error("❌ Unable to open camera.")
    st.stop()

analyzer = MediapipeAnalyzer()

frame_placeholder = st.empty()

while run_camera:
    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ Camera frame not received.")
        break

    frame, is_drowsy, is_yawning = analyzer.analyze_frame(frame)

    if is_drowsy:
        cv2.putText(frame, "DROWSINESS ALERT!", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    if is_yawning:
        cv2.putText(frame, "YAWNING DETECTED", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    frame_placeholder.image(frame, channels="BGR")

cap.release()
