"""
This module contains the FaceDetector class, which is responsible for
detecting faces and facial landmarks in an image.
"""

import cv2
import dlib
import os


class FaceDetector:
    """
    A class to detect faces and facial landmarks in an image.

    Attributes:
        detector: dlib's frontal face detector.
        predictor: dlib's shape predictor for facial landmarks.
    """

    def __init__(self, predictor_path="models/shape_predictor_68_face_landmarks.dat"):
        """
        Initializes the FaceDetector with the path to the shape predictor model.

        Args:
            predictor_path (str): The path to the dlib shape predictor model file.

        Raises:
            FileNotFoundError: If the predictor model file is not found.
        """
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(
                f"The predictor model file was not found at {predictor_path}. "
                f"Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
                f" and place it in the 'models' directory."
            )
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

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
        raise AttributeError(f"'FaceDetector' object has no attribute '{name}'")
