import cv2
import numpy as np
import mediapipe as mp

from drowsiness.base_analyzer import BaseAnalyzer
from drowsiness.utils import compute_eye_aspect_ratio


class MediapipeAnalyzer(BaseAnalyzer):
    """
    Drowsiness analyzer based ONLY on MediaPipe FaceMesh.
    Safe for Windows. No dlib, no TensorFlow.
    """

    def __init__(self):
        super().__init__()

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Eye landmark indices (MediaPipe standard)
        self.left_eye_idx = [33, 160, 158, 133, 153, 144]
        self.right_eye_idx = [362, 385, 387, 263, 373, 380]

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        ear = None
        mar = None

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]

            def lm_to_xy(idx):
                lm = face.landmark[idx]
                return (int(lm.x * w), int(lm.y * h))

            left_eye = [lm_to_xy(i) for i in self.left_eye_idx]
            right_eye = [lm_to_xy(i) for i in self.right_eye_idx]

            if len(left_eye) == 6 and len(right_eye) == 6:
                ear = (compute_eye_aspect_ratio(left_eye) +
                       compute_eye_aspect_ratio(right_eye)) / 2.0

            # Mouth (yawning)
            top = lm_to_xy(13)
            bottom = lm_to_xy(14)
            left = lm_to_xy(61)
            right = lm_to_xy(291)

            mouth_open = np.linalg.norm(np.array(top) - np.array(bottom))
            mouth_width = np.linalg.norm(np.array(left) - np.array(right))

            if mouth_width > 0:
                mar = mouth_open / mouth_width

            self.mp_draw.draw_landmarks(
                frame,
                face,
                self.mp_face_mesh.FACEMESH_TESSELATION,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                self.mp_draw.DrawingSpec(color=(0, 128, 255), thickness=1),
            )

        return ear, mar

    def analyze_frame(self, frame):
        ear, mar = self.process_frame(frame)
        self._apply_detection_logic(ear, mar, frame)
        return frame, self._is_drowsy, self._is_yawning
