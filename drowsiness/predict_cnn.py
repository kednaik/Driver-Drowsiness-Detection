"""
CNN-based drowsiness predictor utilities.

This module exposes `CNNDrowsinessDetector`, a lightweight wrapper that
loads a Keras model and applies it to face / mouth crops to detect
yawning and eye closure. The convenience function `predict_drowsiness`
runs a webcam loop using the detector.
"""

import cv2
import numpy as np
import os
import csv
import time
import datetime
from tensorflow.keras.models import load_model
from drowsiness.cnn_utils import get_haarcascade_path, get_mouth_from_face_image
from drowsiness.utils import play_alarm, log_event


class CNNDrowsinessDetector:
    def __init__(self, model_path="models/drowsiness_cnn_trained.h5"):
        """
        Initialize the CNN drowsiness detector.

        Parameters
        ----------
        model_path : str
            Path to a Keras `.h5` model used for predicting yawn/eye states.
        """
        self.model_path = model_path
        self.model = None
        self.face_cascade = None
        self.eye_cascade = None
        self.labels_new = ["yawn", "no_yawn", "Closed", "Open"]
        self.IMG_SIZE = 145
        self.eye_consecutive_frames = 20
        self.eye_counter = 0
        self.is_drowsy = False
        self.last_drowsy_time = 0.0

        self.load_resources()

    def load_resources(self):
        """
        Load model and Haar cascades required for face/eye detection.

        - Loads the Keras model from `self.model_path` when present.
        - Initializes `self.face_cascade` and `self.eye_cascade` using
          OpenCV Haar cascade files.
        """
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}...")
            self.model = load_model(self.model_path)
        else:
            print(
                f"Model not found at {self.model_path}. Please train the model first."
            )

        face_cas_path = get_haarcascade_path()
        eye_cas_path = cv2.data.haarcascades + "haarcascade_eye.xml"

        self.face_cascade = cv2.CascadeClassifier(face_cas_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_cas_path)

    def predict_frame(self, frame):
        """
        Predict drowsiness/yawn on a single BGR frame.

        The method performs the following steps for each detected face:
        - crop the face and attempt mouth extraction for yawn prediction using
          the loaded CNN model;
        - detect eyes using a Haar cascade and (optionally) run the CNN on
          eye crops to determine closed eyes;
        - update an internal consecutive-eye-closure counter to set
          transient drowsiness state and optionally play an alarm and log
          the event using `drowsiness.utils`.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR image as produced by OpenCV camera capture.

        Returns
        -------
        (frame, is_drowsy, is_yawning)
            `frame` is the annotated BGR image. `is_drowsy` is the
            detector's persistent drowsiness state, and `is_yawning` is a
            boolean indicating whether this frame contained a yawn.
        """
        if self.model is None:
            cv2.putText(
                frame,
                "Model not loaded",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return frame, False, False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        status = "Active"
        color = (0, 255, 0)

        frame_is_yawning = False
        frame_is_drowsy = False

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Face ROI
            roi_color = frame[y : y + h, x : x + w]

            # 1. Yawn Detection (using cropped mouth)
            cropped_mouth = get_mouth_from_face_image(roi_color)

            if cropped_mouth is not None:
                resized_array = cv2.resize(
                    cropped_mouth, (self.IMG_SIZE, self.IMG_SIZE)
                )
                normalized_array = resized_array / 255.0
                reshaped_array = normalized_array.reshape(
                    -1, self.IMG_SIZE, self.IMG_SIZE, 3
                )

                prediction = self.model.predict(reshaped_array, verbose=0)
                class_idx = np.argmax(prediction)
                prediction_label = self.labels_new[class_idx]

                if prediction_label == "yawn":
                    status = "Yawn Detected"
                    color = (0, 0, 255)
                    frame_is_yawning = True

            # 2. Eye Detection (using Haar cascade on Face ROI)
            roi_gray = gray[y : y + h, x : x + w]
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=20
            )

            # Only apply eye-based model prediction when 1 or 2 eyes are detected
            n_eyes = len(eyes) if hasattr(eyes, "__len__") else 0
            frame_eye_closed = False
            if n_eyes in (1, 2):
                for ex, ey, ew, eh in eyes:
                    cv2.rectangle(
                        roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
                    )
                    eye_roi_color = roi_color[ey : ey + eh, ex : ex + ew]

                    # Resize and predict for eyes
                    try:
                        resized_eye = cv2.resize(
                            eye_roi_color, (self.IMG_SIZE, self.IMG_SIZE)
                        )
                        normalized_eye = resized_eye / 255.0
                        reshaped_eye = normalized_eye.reshape(
                            -1, self.IMG_SIZE, self.IMG_SIZE, 3
                        )

                        eye_prediction = self.model.predict(reshaped_eye, verbose=0)
                        eye_class_idx = np.argmax(eye_prediction)
                        eye_label = self.labels_new[eye_class_idx]

                        if eye_label == "Closed":
                            frame_eye_closed = True
                            status = "Drowsy (Eyes Closed)"
                            color = (0, 0, 255)

                    except Exception:
                        # Eye might be too small or other issue — skip this eye
                        pass
            else:
                # If zero or more than two eyes detected, skip eye-based prediction for this face.
                # This avoids unreliable predictions when detection is ambiguous.
                frame_eye_closed = False

            # Update drowsiness counter/state based on eye closure
            if frame_eye_closed:
                self.eye_counter += 1
            else:
                self.eye_counter = 0
                self.is_drowsy = False

            if self.eye_counter >= self.eye_consecutive_frames and not self.is_drowsy:
                self.is_drowsy = True
                self.last_drowsy_time = time.time()
                frame_is_drowsy = True
                # Play alarm (non-blocking where possible)
                try:
                    play_alarm()
                except Exception:
                    pass

                # Log the event and let utils save the screenshot
                try:
                    log_event("Drowsiness Detected", ear=0.0, mar=0.0, frame=frame)
                except Exception:
                    pass

            cv2.putText(
                frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
            )

        return frame, self.is_drowsy, frame_is_yawning

    def __getattr__(self, name):
        """
        Provide a small compatibility alias map for external callers.

        Currently maps `analyze_frame` to `predict_frame` so the detector can
        be used interchangeably with other analyzer implementations.
        """
        if name == "analyze_frame":
            return self.predict_frame
        raise AttributeError(
            f"'CNNDrowsinessDetector' object has no attribute '{name}'"
        )


def predict_drowsiness(model_path="models/drowsiness_cnn_trained.h5"):
    """
    Start a webcam loop and run the CNN drowsiness detector live.

    Parameters
    ----------
    model_path : str
        Path to the Keras model to load for predictions.
    """
    detector = CNNDrowsinessDetector(model_path)

    if detector.model is None:
        return

    try:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Cannot open camera")
            return

        print("Starting video stream. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame, is_drowsy, is_yawning = detector.predict_frame(frame)

            cv2.imshow("Drowsiness Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 27 = ESC
                break

    except (IOError, cv2.error) as e:
        print(f"An error occurred: {e}")
    except FileNotFoundError as e:
        print(f"A required file was not found: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        # Release the webcam and destroy all windows
        if "cap" in locals() and cap.isOpened():
            cap.release()

        # Destroy the named window explicitly
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass

        # Destroy all windows and give the OS a moment to process the close event
        cv2.destroyAllWindows()
        # ensure the GUI event queue is processed
        try:
            cv2.waitKey(1)
        except Exception:
            pass
        # small pause helps the OS close the window reliably
        time.sleep(0.1)

        print("Webcam released and windows closed.")


if __name__ == "__main__":
    predict_drowsiness()
