"""
This module contains tests for the drowsiness detection system.
"""

import os
import sys
import types
import csv
import pytest
import numpy as np

# Ensure tests can import modules that reference `dlib` even if dlib isn't
# installed in the test environment. We provide a lightweight dummy module
# so imports succeed; the tests below do not instantiate the FaceDetector,
# so no dlib functionality is required.
sys.modules.setdefault("dlib", types.SimpleNamespace())

from drowsiness.utils import compute_eye_aspect_ratio
from drowsiness.base_analyzer import BaseAnalyzer


def test_compute_eye_aspect_ratio():
    """
    Tests the compute_eye_aspect_ratio function with ideal eye landmarks.
    """
    # Test with open eye landmarks
    open_eye = [(0, 10), (20, 0), (40, 0), (60, 10), (40, 20), (20, 20)]
    ear = compute_eye_aspect_ratio(open_eye)
    assert ear > 0.2, "EAR for open eye should be greater than 0.2"

    # Test with closed eye landmarks
    closed_eye = [(0, 10), (20, 10), (40, 10), (60, 10), (40, 10), (20, 10)]
    ear = compute_eye_aspect_ratio(closed_eye)
    assert ear == 0.0, "EAR for a completely closed eye should be 0.0"


def test_draw_status_overlay_modifies_frame():
    # Create a blank BGR frame (height=480, width=640)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Avoid importing dlib at import-time by inserting a dummy 'dlib' module
    sys.modules.setdefault("dlib", types.SimpleNamespace())

    # Now import the analyzer module (FaceDetector won't be initialized because
    # we will not call BaseAnalyzer.__init__)
    from drowsiness import base_analyzer as dda

    analyzer = object.__new__(dda.BaseAnalyzer)

    # Set only the attributes used by draw_status_overlay
    analyzer._is_drowsy = True
    analyzer._is_yawning = True

    # Ensure frame is initially all zeros
    assert frame.sum() == 0

    # Call the overlay drawing method
    analyzer.draw_status_overlay(frame)

    # The overlay draws a filled rectangle and text; therefore frame sum should be > 0
    assert frame.sum() > 0
