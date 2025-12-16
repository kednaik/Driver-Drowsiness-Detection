import cv2
import mediapipe as mp
import numpy as np
from drowsiness.utils import compute_eye_aspect_ratio


class MediapipeAnalyzer:
    def __init__(
        self,
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self):
        """Release MediaPipe resources if available."""
        try:
            if hasattr(self, "face_mesh") and self.face_mesh is not None:
                # FaceMesh has a close() method to free native resources
                try:
                    self.face_mesh.close()
                except Exception:
                    # Older versions may not have close(); ignore safely
                    pass
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _calculate_distance(self, p1, p2):
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def calculate_ear(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR) from 6 MediaPipe eye landmarks.

        Expected ordering for eye_landmarks (6 points):
          [p1, p2, p3, p4, p5, p6] where
          p1 = left-most (outer) corner
          p4 = right-most (inner) corner
          p2,p3,p5,p6 are vertical pairs

        Uses the standard formula: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        try:
            if len(eye_landmarks) < 6:
                return None

            p1 = eye_landmarks[0]
            p2 = eye_landmarks[1]
            p3 = eye_landmarks[2]
            p4 = eye_landmarks[3]
            p5 = eye_landmarks[4]
            p6 = eye_landmarks[5]

            vertical_dist1 = self._calculate_distance(p2, p6)
            vertical_dist2 = self._calculate_distance(p3, p5)
            horizontal_dist = self._calculate_distance(p1, p4)

            if horizontal_dist == 0:
                return None

            ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
            return float(ear)
        except Exception:
            return None

    def calculate_mar(self, mouth_landmarks):
        """
        Calculate Mouth Aspect Ratio (MAR) using MediaPipe landmarks.

        Uses mouth corners (61, 291) for horizontal width and
        inner top/bottom lip pair (13, 14) for vertical opening when available.
        """
        try:
            # Need at least corners and one vertical pair
            if len(mouth_landmarks) < 2:
                return None

            # Map expected indices if a full face_landmarks list was provided; otherwise
            # assume mouth_landmarks already contains the selected points.
            # If the caller passed a list constructed around indices [61,291,13,14], we use them.
            # Here we assume callers pass the full list extracted in process_frame, so we
            # directly pick by values if present.
            # For safety, try to find standard positions by index attribute if available.
            # We'll attempt to locate landmarks by their approximate positions: corners and center.

            # Best-case: caller passed [left_corner, right_corner, top_inner, bottom_inner]
            if len(mouth_landmarks) >= 4:
                p_left = mouth_landmarks[0]
                p_right = mouth_landmarks[1]
                p_top = mouth_landmarks[2]
                p_bottom = mouth_landmarks[3]
            else:
                # Fallback: use first and last for horizontal and middle pair for vertical
                p_left = mouth_landmarks[0]
                p_right = mouth_landmarks[-1]
                mid = len(mouth_landmarks) // 2
                p_top = mouth_landmarks[mid - 1]
                p_bottom = mouth_landmarks[mid]

            horizontal_dist = self._calculate_distance(p_left, p_right)
            vertical_dist = self._calculate_distance(p_top, p_bottom)

            if horizontal_dist == 0:
                return None

            mar = float(vertical_dist) / float(horizontal_dist)
            return mar
        except Exception:
            return None

    def process_frame(self, frame, draw_landmarks=False):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        ear = None
        mar = None

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Define eye landmark index sets (used for both drawing and EAR computation)
                left_indices = [33, 160, 158, 133, 153, 144]
                right_indices = [362, 385, 387, 263, 373, 380]

                # Optionally draw the face mesh onto the original frame
                if draw_landmarks:
                    try:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame,
                            face_landmarks,
                            mp.solutions.face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                color=(0, 255, 0), thickness=1, circle_radius=1
                            ),
                            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                color=(0, 128, 255), thickness=1, circle_radius=1
                            ),
                        )
                    except Exception:
                        # Drawing failures should not stop processing
                        pass

                # Extract 6-point eye landmark sets commonly used for EAR
                left_eye_landmarks = [
                    face_landmarks.landmark[i]
                    for i in left_indices
                    if i < len(face_landmarks.landmark)
                ]
                right_eye_landmarks = [
                    face_landmarks.landmark[i]
                    for i in right_indices
                    if i < len(face_landmarks.landmark)
                ]

                # Convert normalized MediaPipe landmarks to pixel (x,y) tuples
                # and compute EAR using the same utility used for dlib (pixel-based)
                h, w = frame.shape[:2]
                left_eye_pts = []
                for lm in left_eye_landmarks:
                    # clamp coordinates to [0,1] then to pixels
                    x = int(min(max(lm.x, 0.0), 1.0) * w)
                    y = int(min(max(lm.y, 0.0), 1.0) * h)
                    left_eye_pts.append((x, y))

                right_eye_pts = []
                for lm in right_eye_landmarks:
                    x = int(min(max(lm.x, 0.0), 1.0) * w)
                    y = int(min(max(lm.y, 0.0), 1.0) * h)
                    right_eye_pts.append((x, y))

                left_ear = None
                right_ear = None
                try:
                    if len(left_eye_pts) == 6:
                        left_ear = compute_eye_aspect_ratio(left_eye_pts)
                except Exception:
                    left_ear = None
                try:
                    if len(right_eye_pts) == 6:
                        right_ear = compute_eye_aspect_ratio(right_eye_pts)
                except Exception:
                    right_ear = None

                if left_ear is not None and right_ear is not None:
                    ear = (left_ear + right_ear) / 2.0
                elif left_ear is not None:
                    ear = left_ear
                elif right_ear is not None:
                    ear = right_ear
                else:
                    ear = None

                # For MAR: use outer mouth corners for horizontal width and inner lip center for vertical
                # typical indices: left corner=61, right corner=291, top inner=13, bottom inner=14
                try:
                    if all(i < len(face_landmarks.landmark) for i in (61, 291, 13, 14)):
                        p_left = face_landmarks.landmark[61]
                        p_right = face_landmarks.landmark[291]
                        p_top = face_landmarks.landmark[13]
                        p_bottom = face_landmarks.landmark[14]
                        mar = self.calculate_mar([p_left, p_right, p_top, p_bottom])
                    else:
                        # Fallback: attempt to build a small mouth list with available indices
                        mouth_indices = [61, 291, 13, 14]
                        mouth_landmarks = [
                            face_landmarks.landmark[i]
                            for i in mouth_indices
                            if i < len(face_landmarks.landmark)
                        ]
                        mar = self.calculate_mar(mouth_landmarks)
                except Exception:
                    mar = None

        return ear, mar
 
    def __getattr__(self, name):
        """
        Provide a few compatibility aliases and forward unknown attribute lookups
        to the underlying MediaPipe `face_mesh` object when appropriate.
 
        Common aliases provided:
        - `analyze`, `analyze_frame`, `process` -> `process_frame`
 
        If the attribute exists on the internal `face_mesh`, it will be
        returned. Otherwise an AttributeError is raised.
        """
        # Provide simple aliases to match the DrowsinessAnalyzer API
        if name in ("analyze", "analyze_frame", "process"):
            return self.process_frame
 
        # Forward unknown attributes to the underlying MediaPipe FaceMesh
        if hasattr(self, "face_mesh") and hasattr(self.face_mesh, name):
            return getattr(self.face_mesh, name)
 
        raise AttributeError(f"'MediapipeAnalyzer' object has no attribute '{name}'")
 