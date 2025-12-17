"""
This module contains the DlibAnalyzer class, which is responsible for
detecting faces and facial landmarks in an image.
"""

import cv2
import dlib
import os
import numpy as np
from drowsiness.base_analyzer import BaseAnalyzer
from drowsiness.utils import compute_eye_aspect_ratio, compute_mouth_aspect_ratio


class DlibAnalyzer(BaseAnalyzer):
    """
    A class to detect faces and facial landmarks in an image.

    Attributes:
        detector: dlib's frontal face detector.
        predictor: dlib's shape predictor for facial landmarks.
    """

    def __init__(self, predictor_path="models/shape_predictor_68_face_landmarks.dat"):
        """
        Initializes the DlibAnalyzer with the path to the shape predictor model.

        Args:
            predictor_path (str): The path to the dlib shape predictor model file.

        Raises:
            FileNotFoundError: If the predictor model file is not found.
        """
        super().__init__()
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(
                f"The predictor model file was not found at {predictor_path}. "
                f"Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
                f" and place it in the 'models' directory."
            )
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def analyze_frame(self, frame):
        """
        Analyzes a single video frame for drowsiness and yawns.

        Args:
            frame: The video frame to analyze.

        Returns:
            A tuple containing the frame with annotations, and the drowsiness/yawn status.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks_list = self.get_cv2_landmarks(gray)

        ear, mar = 0.0, 0.0  # Default values

        if landmarks_list:
            landmarks = landmarks_list[0]  # Assuming one face
            shape = np.array(
                [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
            )

            # Extract eye and mouth coordinates using list comprehension
            left_eye = [tuple(p) for p in shape[36:42]]
            right_eye = [tuple(p) for p in shape[42:48]]
            mouth = [tuple(p) for p in shape[48:68]]

            # Compute EAR and MAR
            left_ear = compute_eye_aspect_ratio(left_eye)
            right_ear = compute_eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = compute_mouth_aspect_ratio(mouth)

            # Draw contours for visualization
            left_eye_hull = cv2.convexHull(np.array(left_eye))
            right_eye_hull = cv2.convexHull(np.array(right_eye))
            mouth_hull = cv2.convexHull(np.array(mouth))
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)

            # Apply detection/overlay logic (extracted helper)
            self._apply_detection_logic(ear, mar, frame)

        return frame, self._is_drowsy, self._is_yawning


    def detect_faces(self, gray_image):
        """
        Detects faces in a grayscale image and returns their facial landmarks.

        Args:
            gray_image: A grayscale image.

        Returns:
            A list of dlib shape objects, where each shape object contains
            the 68 facial landmarks for a detected face.
        """
        faces = self.detector(gray_image)
        landmarks = [self.predictor(gray_image, face) for face in faces]
        return landmarks

    def __getattr__(self, name):
        if name == "dlib_module":
            return dlib
        elif name in ("get_cv2_landmarks", "cv2_landmarks", "detect"):
            return self.detect_faces
        elif name in ('analyze'):
            return self.analyze_frame
        elif name in ("get_cv2_landmarks", "cv2_landmarks"):
            return self.face_detector.get_cv2_landmarks
        else:
            super().__getattr__(name)
