"""
This module contains utility functions for the drowsiness detection system.
"""

import numpy as np
from scipy.spatial import distance as dist
import pygame
import os


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


def play_alarm(sound_file="./alarm/alert.mp3"):
    """
    Plays an alarm sound.

    Args:
        sound_file (str): The path to the alarm sound file.
    """
    if not os.path.exists(sound_file):
        print(
            f"Alarm sound file not found at {sound_file}. Please add an alarm sound file."
        )
        return
    pygame.mixer.init()
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
