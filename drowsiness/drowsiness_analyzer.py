"""
This module contains the DrowsinessAnalyzer class, which is responsible for
analyzing facial landmarks to detect drowsiness.
"""

import csv
import datetime
import time
from collections import deque
import os

import cv2
import numpy as np
from typing import Optional

from drowsiness.face_detector import FaceDetector
from drowsiness.utils import (
    compute_eye_aspect_ratio,
    compute_mouth_aspect_ratio,
    play_alarm,
)


class DrowsinessAnalyzer:
    """
    A class to analyze facial landmarks and detect driver drowsiness.

    This class uses a FaceDetector object to get facial landmarks from video frames
    and then computes the eye aspect ratio (EAR) and mouth aspect ratio (MAR) to
    determine if the driver is drowsy or yawning.

    Attributes:
        face_detector (FaceDetector): An object to detect faces and landmarks.
        ear_threshold (float): The EAR threshold to determine if eyes are closed.
        ear_consecutive_frames (int): The number of consecutive frames the EAR
                                     must be below the threshold to trigger an alert.
        mar_threshold (float): The MAR threshold to detect a yawn.
        log_file (str): The path to the CSV file for logging drowsiness events.
    """

    def __init__(
        self,
        ear_threshold=0.20,
        ear_consecutive_frames=20,
        mar_threshold=0.5,
        log_file="logs/drowsiness_log.csv",
    ):
        """
        Initializes the DrowsinessAnalyzer.

        Args:
            ear_threshold (float): EAR threshold for drowsiness.
            ear_consecutive_frames (int): Consecutive frames for drowsiness.
            mar_threshold (float): MAR threshold for yawning.
            log_file (str): Path to the log file.
        """
        self.face_detector = FaceDetector()
        self.ear_threshold = ear_threshold
        self.ear_consecutive_frames = ear_consecutive_frames
        self.mar_threshold = mar_threshold
        self.log_file = log_file
        self._ear_counter = 0
        self._yawn_counter = 0
        self._is_drowsy = False
        self._is_yawning = False
        # Timestamps used to implement flashing overlay after an event
        self._last_drowsy_time = 0.0
        self._last_yawn_time = 0.0
        # Flashing configuration: total duration and individual flash interval
        self._flash_duration = 1.0  # seconds to keep flashing after detection
        self._flash_interval = 0.2  # seconds per flash toggle
        self._init_log_file()

    def _init_log_file(self):
        """Initializes the log file with headers if it doesn't exist."""
        with open(self.log_file, "a", newline="") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(["Timestamp", "Event", "EAR", "MAR", "Screenshot"])

    def _log_event(self, event, ear, mar, frame=None):
        """
        Logs a drowsiness or yawn event to the CSV file.

        Args:
            event (str): The type of event ('Drowsiness' or 'Yawn').
            ear (float): The current eye aspect ratio.
            mar (float): The current mouth aspect ratio.
        """
        # Timestamp for human-readable CSV and filesystem-safe filename
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_fname = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save screenshot if a frame was provided
        if frame is not None:
            screenshots_dir = os.path.join(
                os.path.dirname(self.log_file), "screenshots"
            )
            try:
                os.makedirs(screenshots_dir, exist_ok=True)
                # Filename: <event>_YYYYmmdd_HHMMSS.png
                safe_event = event.replace(" ", "_")
                img_name = f"{safe_event}_{ts_fname}.png"
                img_path = os.path.join(screenshots_dir, img_name)
                # Write the BGR frame as PNG
                cv2.imwrite(img_path, frame)
            except Exception:
                # Don't let screenshot saving break detection; ignore filesystem errors
                pass

        with open(self.log_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, event, f"{ear:.2f}", f"{mar:.2f}"])

    def analyze_frame(self, frame):
        """
        Analyzes a single video frame for drowsiness and yawns.

        Args:
            frame: The video frame to analyze.

        Returns:
            A tuple containing the frame with annotations, and the drowsiness/yawn status.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks_list = self.face_detector.detect_faces(gray)

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

    def _apply_detection_logic(self, ear: float, mar: float, frame) -> None:
        """
        Shared logic that applies EAR/MAR thresholds, updates counters and
        state, logs events, and draws overlays on the provided frame.

        This is extracted so the same behavior can be used with different
        landmark providers (dlib or MediaPipe).
        """
        # Check for drowsiness
        if ear is not None and ear < self.ear_threshold:
            self._ear_counter += 1
            if self._ear_counter >= self.ear_consecutive_frames and not self._is_drowsy:
                self._is_drowsy = True
                # record the time of detection (start flashing)
                self._last_drowsy_time = time.time()
                play_alarm()
                # Save a screenshot of the frame when logging the event
                try:
                    self._log_event("Drowsiness Detected", ear, mar, frame.copy())
                except Exception:
                    # Ensure logging errors don't break processing
                    self._log_event("Drowsiness Detected", ear, mar, None)
                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
        else:
            # Reset counter/state when EAR is okay or unavailable
            self._ear_counter = 0
            self._is_drowsy = False

        # Check for yawning
        if mar is not None and mar > self.mar_threshold:
            self._yawn_counter += 1
            if (
                self._yawn_counter > 1 and not self._is_yawning
            ):  # Avoid single-frame flickers
                self._is_yawning = True
                # record the time of yawn detection (start flashing)
                self._last_yawn_time = time.time()
                try:
                    self._log_event("Yawn Detected", ear, mar, frame.copy())
                except Exception:
                    self._log_event("Yawn Detected", ear, mar, None)
        else:
            self._yawn_counter = 0
            self._is_yawning = False

        # Display EAR and MAR on the frame if values are available
        if ear is not None:
            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        if mar is not None:
            cv2.putText(
                frame,
                f"MAR: {mar:.2f}",
                (300, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Determine whether to show/flash overlays for a short time after
        # detection. This allows a short flashing notification after the event
        # was observed.
        now = time.time()

        def _should_show(last_time, current_flag):
            # If currently active, show immediately
            if current_flag:
                return True
            # If recently active, flash for the configured duration
            if last_time and (now - last_time) < self._flash_duration:
                elapsed = now - last_time
                return int(elapsed / self._flash_interval) % 2 == 0
            return False

        show_drowsy = _should_show(self._last_drowsy_time, self._is_drowsy)
        show_yawning = _should_show(self._last_yawn_time, self._is_yawning)

        # Draw overlay with explicit show flags
        self.draw_status_overlay(
            frame,
            ear if ear is not None else 0.0,
            mar if mar is not None else 0.0,
            show_drowsy,
            show_yawning,
        )

    def analyze_frame_mediapipe(
        self,
        frame,
        mediapipe_analyzer: Optional[object] = None,
        draw_landmarks: bool = True,
    ):
        """
        Analyze a frame using a MediaPipe-based analyzer instance.

        Args:
            frame: BGR image (numpy array) to analyze.
            mediapipe_analyzer: An instance providing `process_frame(frame, draw_landmarks)` -> (ear, mar).
                                If None, this function will import and create a temporary `MediapipeAnalyzer`.
            draw_landmarks: Whether to ask the mediapipe analyzer to draw the face mesh on the frame.

        Returns:
            (frame, is_drowsy, is_yawning) with overlays applied and internal state updated.
        """
        created_local = False
        if mediapipe_analyzer is None:
            # Lazy import to avoid adding a hard dependency at module import time
            try:
                from drowsiness.mediapipe_analyzer import MediapipeAnalyzer

                mediapipe_analyzer = MediapipeAnalyzer()
                created_local = True
            except Exception:
                # If MediaPipe isn't available, skip detection and return frame unchanged
                return frame, self._is_drowsy, self._is_yawning

        try:
            ear, mar = mediapipe_analyzer.process_frame(
                frame, draw_landmarks=draw_landmarks
            )
            # Apply the same downstream logic (counters, overlays, logging)
            self._apply_detection_logic(ear, mar, frame)
        finally:
            if created_local:
                try:
                    mediapipe_analyzer.close()
                except Exception:
                    pass

        return frame, self._is_drowsy, self._is_yawning

    def draw_status_overlay(
        self,
        frame,
        ear: float = 0.0,
        mar: float = 0.0,
        show_drowsy=None,
        show_yawning=None,
    ):
        """
        Draws the status overlay (DROWSY / YAWNING) on the provided frame.

        This is extracted so tests can call the overlay drawing without running
        face detection. The overlay is centered at the top of the frame for
        greater visibility.

        Args:
            frame: BGR image (numpy array) to draw on (modified in-place).
            ear: current EAR value (unused right now, but provided for future use).
            mar: current MAR value (unused right now, but provided for future use).
        """
        # Resolve explicit show flags if provided, otherwise fall back to
        # the analyzer's current booleans.
        if show_drowsy is None:
            show_drowsy = self._is_drowsy
        if show_yawning is None:
            show_yawning = self._is_yawning

        status_msgs = []
        if show_drowsy:
            status_msgs.append("DROWSY")
        if show_yawning:
            status_msgs.append("YAWNING")

        if not status_msgs:
            return

        status_text = " & ".join(status_msgs)

        # Choose color: red for drowsy (high-priority), orange for yawning
        color = (0, 165, 255)  # default orange
        if show_drowsy:
            color = (0, 0, 255)

        # Compute text size and draw a filled rectangle as background for readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            status_text, font, font_scale, thickness
        )

        # Position at top-center
        h, w = frame.shape[:2]
        x = max(10, w // 2 - text_width // 2)
        y = 50

        # Rectangle coordinates (bg) and draw
        rect_top_left = (x - 8, y - text_height - 8)
        rect_bottom_right = (x + text_width + 8, y + baseline + 8)
        cv2.rectangle(frame, rect_top_left, rect_bottom_right, (0, 0, 0), -1)

        # Put the status text on top
        cv2.putText(frame, status_text, (x, y), font, font_scale, color, thickness)

    def __str__(self):
        """Returns a string representation of the analyzer's settings."""
        return (
            f"DrowsinessAnalyzer(EAR_Threshold={self.ear_threshold}, "
            f"EAR_Frames={self.ear_consecutive_frames}, "
            f"MAR_Threshold={self.mar_threshold})"
        )


if __name__ == "__main__":
    # This block is for testing the DrowsinessAnalyzer class independently.
    # It requires a webcam.
    analyzer = DrowsinessAnalyzer()
    print(analyzer)

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, is_drowsy, is_yawning = analyzer.analyze_frame(frame)

            cv2.imshow("Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except (IOError, cv2.error) as e:
        print(f"An error occurred: {e}")
    finally:
        if "cap" in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
