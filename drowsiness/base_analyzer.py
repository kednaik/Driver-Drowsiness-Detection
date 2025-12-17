"""
This module contains the BaseAnalyzer class, which is responsible for
analyzing facial landmarks to detect drowsiness.
"""

import time

import cv2

from drowsiness.utils import (
    play_alarm,
    ensure_log_file,
    log_event,
)


class BaseAnalyzer:
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
        Initializes the BaseAnalyzer.

        Args:
            ear_threshold (float): EAR threshold for drowsiness.
            ear_consecutive_frames (int): Consecutive frames for drowsiness.
            mar_threshold (float): MAR threshold for yawning.
            log_file (str): Path to the log file.
        """
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
        # Ensure log file exists with header
        try:
            ensure_log_file(self.log_file)
        except Exception:
            pass

    def analyze_frame(self, frame):
        """
        Analyzes a single video frame for drowsiness and yawns.
        Args:
            frame: The video frame to analyze.
        Returns:
            A tuple containing the frame with annotations, and the drowsiness/yawn status.
        """
        pass  # To be implemented in subclasses

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
                # Save a screenshot of the frame when logging the event using shared util
                try:
                    log_event(
                        "Drowsiness Detected",
                        ear,
                        mar,
                        frame.copy(),
                        log_file=self.log_file,
                    )
                except Exception:
                    try:
                        log_event(
                            "Drowsiness Detected",
                            ear,
                            mar,
                            None,
                            log_file=self.log_file,
                        )
                    except Exception:
                        # Suppress all exceptions here to avoid crashing if logging fails.
                        pass
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
                    log_event(
                        "Yawn Detected", ear, mar, frame.copy(), log_file=self.log_file
                    )
                except Exception:
                    try:
                        log_event(
                            "Yawn Detected", ear, mar, None, log_file=self.log_file
                        )
                    except Exception:
                        # Suppress all exceptions to ensure detection logic continues even if logging fails.
                        pass
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
            show_drowsy,
            show_yawning,
        )

    def draw_status_overlay(
        self,
        frame,
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
            f"BaseAnalyzer(EAR_Threshold={self.ear_threshold}, "
            f"EAR_Frames={self.ear_consecutive_frames}, "
            f"MAR_Threshold={self.mar_threshold})"
        )

    def __getattr__(self, name):
        """
        Provides access to FaceDetector attributes if not found in BaseAnalyzer.

        Args:
            name (str): The attribute name.
        Returns:
            The attribute from FaceDetector if it exists.
        Raises:
            AttributeError: If the attribute is not found in both classes.
        """
        if name == "analyze":
            return self.analyze_frame
        elif name == "ear_thres":
            return self.ear_threshold
        elif name == "mar_thres":
            return self.mar_threshold

        raise AttributeError(f"'BaseAnalyzer' object has no attribute '{name}'")

