import streamlit as st
import cv2
import numpy as np
from drowsiness.dlib_analyzer import DlibAnalyzer
from drowsiness.mediapipe_analyzer import MediapipeAnalyzer
from drowsiness.cnn_analyzer import CnnAnalyzer
import warnings
import os
 
# Suppress TensorFlow/OpenCV warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
warnings.filterwarnings('ignore')
 
st.set_page_config(page_title="Drowsiness Detection App", layout="wide")
 
st.title("Driver Drowsiness Detection System")
st.sidebar.title("Settings")
 
method = st.sidebar.selectbox(
    "Select Detection Method", ("Dlib (Landmarks)", "MediaPipe", "CNN")
)
 
run_app = st.sidebar.checkbox("Run Camera", value=False)  # Changed default to False
 
 
# Initialize Analyzers
@st.cache_resource
def load_analyzers():
    try:
        dlib_analyzer = DlibAnalyzer()
        mediapipe_analyzer = MediapipeAnalyzer()
        cnn_analyzer = CnnAnalyzer()
        return dlib_analyzer, mediapipe_analyzer, cnn_analyzer
    except Exception as e:
        st.error(f"Error loading analyzers: {str(e)}")
        return None, None, None
 
 
# Add camera index selector in sidebar
camera_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
 
# Load analyzers
with st.spinner("Loading detection models..."):
    dlib_analyzer, mediapipe_analyzer, cnn_analyzer = load_analyzers()
 
if any(a is None for a in [dlib_analyzer, mediapipe_analyzer, cnn_analyzer]):
    st.error("Failed to load one or more analyzers. Please check your installation.")
    st.stop()
 
FRAME_WINDOW = st.image([])
status_text = st.empty()
 
# Initialize session state for camera
if 'camera' not in st.session_state:
    st.session_state.camera = None
 
if run_app:
    # Try to open camera if not already open
    if st.session_state.camera is None or not st.session_state.camera.isOpened():
        try:
            st.session_state.camera = cv2.VideoCapture(camera_index)
            
            # Verify camera opened successfully
            if not st.session_state.camera.isOpened():
                st.error(f"Could not open webcam at index {camera_index}. Please try a different camera index or check if your camera is connected and not in use by another application.")
                st.info("💡 Tip: Try camera index 1, 2, or check if another application is using your camera.")
                st.session_state.camera = None
                st.stop()
            else:
                st.success(f"Camera {camera_index} opened successfully!")
        except Exception as e:
            st.error(f"Error accessing camera: {str(e)}")
            st.session_state.camera = None
            st.stop()
 
    if st.session_state.camera is not None and st.session_state.camera.isOpened():
        # Create a placeholder for stop button
        stop_button_placeholder = st.empty()
        
        while run_app:
            ret, frame = st.session_state.camera.read()
            if not ret:
                st.error("Failed to capture frame from camera.")
                break
 
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
 
            is_drowsy = False
            is_yawning = False
            
            # Select analyzer based on method
            try:
                if method == "Dlib (Landmarks)":
                    analyzer = dlib_analyzer
                elif method == "MediaPipe":
                    analyzer = mediapipe_analyzer
                elif method == "CNN":
                    analyzer = cnn_analyzer
                    
                frame, is_drowsy, is_yawning = analyzer.analyze_frame(frame)
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                frame = cv2.putText(frame, f"Error: {str(e)}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
            # Convert to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
            FRAME_WINDOW.image(frame_rgb, channels="RGB", use_container_width=True)
 
            # Update status
            status_msg = []
            if is_drowsy:
                status_msg.append("DROWSY")
            if is_yawning:
                status_msg.append("YAWNING")
 
            if status_msg:
                status_text.markdown(
                    f"<h2 style='color: red; text-align: center;'>⚠️ ALERT: {' & '.join(status_msg)}</h2>",
                    unsafe_allow_html=True,
                )
            else:
                status_text.markdown(
                    "<h2 style='color: green; text-align: center;'>✓ Status: Active</h2>",
                    unsafe_allow_html=True,
                )
            
            # Small delay to prevent overwhelming the system
            cv2.waitKey(1)
            
            # Check if user unchecked the run box
            if not st.session_state.get('run_app', True):
                break
 
else:
    # Release camera when not running
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
    status_text.empty()
    FRAME_WINDOW.empty()
    st.info("📷 Camera is off. Check 'Run Camera' to start detection.")
    
    # Display instructions
    st.markdown("""
    ### Instructions:
    1. Select your preferred detection method from the sidebar
    2. Adjust the camera index if needed (usually 0 for built-in camera)
    3. Check 'Run Camera' to start the detection system
    4. The system will alert you if drowsiness or yawning is detected
    """)
 
# Cleanup on app termination
if st.session_state.camera is not None:
    st.session_state.camera.release()
 