"""
This module contains utility functions for the drowsiness detection system.
"""

import numpy as np
from scipy.spatial import distance as dist
import os
import subprocess
import shutil
import sys
import csv
import datetime
import cv2


def compute_eye_aspect_ratio(eye_points):
    """
    Computes the eye aspect ratio (EAR) given the coordinates of the eye landmarks.

    The EAR is the ratio of the distance between the vertical eye landmarks to the
    distance between the horizontal eye landmarks.

    Args:
        eye_points (list): A list of 6 (x, y) tuples representing the eye landmarks.

    Returns:
        The eye aspect ratio (float).
    """
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    a = dist.euclidean(eye_points[1], eye_points[5])
    b = dist.euclidean(eye_points[2], eye_points[4])

    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    c = dist.euclidean(eye_points[0], eye_points[3])

    # Compute the eye aspect ratio
    ear = (a + b) / (2.0 * c)

    return ear


def compute_mouth_aspect_ratio(mouth_points):
    """
    Computes the mouth aspect ratio (MAR) to detect yawns.

    Args:
        mouth_points (list): A list of (x, y) tuples for mouth landmarks.

    Returns:
        The mouth aspect ratio (float).
    """
    # Compute the euclidean distances between the vertical mouth landmarks
    a = dist.euclidean(mouth_points[2], mouth_points[10])  # 51, 59
    b = dist.euclidean(mouth_points[4], mouth_points[8])  # 53, 57

    # Compute the euclidean distance between the horizontal mouth landmarks
    c = dist.euclidean(mouth_points[0], mouth_points[6])  # 49, 55

    # Compute the mouth aspect ratio
    mar = (a + b) / (2.0 * c)

    return mar


def play_alarm(sound_file="./alarm/alert.wav"):
    """
    Play an alarm sound using a platform-safe method.

    Strategy (in order):
      - On macOS: call `afplay` (non-blocking)
      - On Windows: try `winsound` for WAV (non-blocking)
      - On Linux: try `paplay` / `aplay` / `ffplay` (non-blocking where supported)
      - Fallback: try the `playsound` package (may be blocking)

    Returns:
      subprocess.Popen if a non-blocking subprocess player was launched, else None.
    """

    if not os.path.exists(sound_file):
        print(
            f"Alarm sound file not found at {sound_file}. Please add an alarm sound file."
        )
        return None

    # Helper to launch a subprocess player non-blocking
    def _try_subprocess(cmd_list):
        try:
            proc = subprocess.Popen(
                cmd_list, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return proc
        except Exception:
            return None

    # macOS: afplay is standard and reliable
    if sys.platform == "darwin":
        afplay = shutil.which("afplay")
        if afplay:
            return _try_subprocess([afplay, sound_file])

    # Windows: try winsound for WAV (no external process)
    if sys.platform.startswith("win"):
        if sound_file.lower().endswith(".wav"):
            try:
                import winsound

                winsound.PlaySound(
                    sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC
                )
                return None
            except Exception:
                pass

        # fallback to starting the file with the default handler (may be blocking depending on environment)
        try:
            return _try_subprocess(["cmd", "/c", "start", "", sound_file])
        except Exception:
            pass

    # Linux / misc unix: try common players
    if sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        for player in ("paplay", "aplay", "ffplay", "ffmpeg", "mpv", "mpg123"):
            path = shutil.which(player)
            if not path:
                continue
            # ffplay/ffmpeg/mpv need different args to play and exit
            if player == "ffplay":
                return _try_subprocess(
                    [path, "-nodisp", "-autoexit", "-loglevel", "quiet", sound_file]
                )
            if player == "ffmpeg":
                # use ffmpeg to play (requires -nostdin and -hide_banner to be quiet)
                return _try_subprocess(
                    [
                        path,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        sound_file,
                        "-f",
                        "null",
                        "-",
                    ]
                )
            if player in ("mpv", "mpg123", "paplay", "aplay"):
                return _try_subprocess([path, sound_file])

    # Generic fallback: try playsound (blocking)
    try:
        from playsound import playsound

        # playsound may block; run it in a background thread to avoid blocking caller
        import threading

        t = threading.Thread(target=playsound, args=(sound_file,), daemon=True)
        t.start()
        return None
    except Exception:
        pass

    # If nothing worked, print a message and return None
    print(
        "No available audio player found. Please install 'afplay' (macOS), 'paplay'/'aplay' (Linux), or the 'playsound' package."
    )
    return None


def ensure_log_file(log_file="logs/drowsiness_log.csv"):
    """Ensure the CSV log file and directory exist with a header.

    Creates `logs/` directory and writes header row if file is missing or empty.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    header = ["Timestamp", "Event", "EAR", "MAR", "Screenshot"]
    try:
        if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
            with open(log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
    except Exception:
        # Ignore filesystem errors
        pass


def log_event(event, ear=0.0, mar=0.0, frame=None, log_file="logs/drowsiness_log.csv"):
    """Log a detection event to CSV and optionally save a screenshot.

    Args:
        event (str): Human readable event name (e.g., "Drowsiness Detected").
        ear (float): Eye Aspect Ratio value.
        mar (float): Mouth Aspect Ratio value.
        frame (np.ndarray or None): BGR image to save as screenshot (optional).
        log_file (str): Path to CSV log file.
    """
    ensure_log_file(log_file=log_file)

    screenshot_name = ""
    if frame is not None:
        try:
            screenshots_dir = os.path.join(os.path.dirname(log_file), "screenshots")
            os.makedirs(screenshots_dir, exist_ok=True)
            ts_fname = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_event = event.replace(" ", "_")
            screenshot_name = f"{safe_event}_{ts_fname}.png"
            screenshot_path = os.path.join(screenshots_dir, screenshot_name)
            # Write BGR image
            cv2.imwrite(screenshot_path, frame)
        except Exception:
            screenshot_name = ""

    try:
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow(
                [timestamp, event, f"{ear:.2f}", f"{mar:.2f}", screenshot_name]
            )
    except Exception:
        # Do not let logging errors break detection
        pass
