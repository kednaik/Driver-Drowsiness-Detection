import streamlit as st
import cv2
import numpy as np
from drowsiness.dlib_analyzer import DlibAnalyzer
from drowsiness.mediapipe_analyzer import MediapipeAnalyzer
from drowsiness.cnn_analyzer import CnnAnalyzer
import tempfile

st.set_page_config(page_title="Drowsiness Detection App", layout="wide")

st.title("Driver Drowsiness Detection System")
st.sidebar.title("Settings")

method = st.sidebar.selectbox(
    "Select Detection Method", ("Dlib (Landmarks)", "MediaPipe", "CNN")
)

run_app = st.sidebar.checkbox("Run Camera", value=True)


# Initialize Analyzers
@st.cache_resource
def load_analyzers():
    dlib_analyzer = DlibAnalyzer()
    mediapipe_analyzer = MediapipeAnalyzer()
    cnn_analyzer = CnnAnalyzer()
    return dlib_analyzer, mediapipe_analyzer, cnn_analyzer


dlib_analyzer, mediapipe_analyzer, cnn_analyzer = load_analyzers()

FRAME_WINDOW = st.image([])
status_text = st.empty()

camera = None

if run_app:
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        st.error("Could not open webcam.")
    else:
        while run_app:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture frame.")
                break

            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)

            is_drowsy = False
            is_yawning = False
            analyzer = None
            if method == "Dlib (Landmarks)":
                analyzer = dlib_analyzer
            elif method == "MediaPipe":
                analyzer = mediapipe_analyzer
            elif method == "CNN":
                analyzer = cnn_analyzer
                
            frame, is_drowsy, is_yawning = analyzer.analyze_frame(frame)

            # Convert to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            FRAME_WINDOW.image(frame_rgb)

            status_msg = []
            if is_drowsy:
                status_msg.append("DROWSY")
            if is_yawning:
                status_msg.append("YAWNING")

            if status_msg:
                status_text.markdown(
                    f"<h2 style='color: red; text-align: center;'>ALERT: {' & '.join(status_msg)}</h2>",
                    unsafe_allow_html=True,
                )
            else:
                status_text.markdown(
                    "<h2 style='color: green; text-align: center;'>Status: Active</h2>",
                    unsafe_allow_html=True,
                )

else:
    if camera is not None:
        camera.release()
    st.write("Camera is off.")
