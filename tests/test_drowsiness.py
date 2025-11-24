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
from drowsiness.drowsiness_analyzer import DrowsinessAnalyzer


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


def test_event_logging():
    """
    Tests the event logging functionality of the DrowsinessAnalyzer.
    """
    log_file = "logs/test_log.csv"
    # Ensure the log file is clean before the test
    if os.path.exists(log_file):
        os.remove(log_file)

    # Use a mock analyzer that doesn't need a real face detector
    class MockDrowsinessAnalyzer(DrowsinessAnalyzer):
        def __init__(self, log_file):
            self.log_file = log_file
            self._init_log_file()

        def _log_event(self, event, ear, mar):
            super()._log_event(event, ear, mar)

    analyzer = MockDrowsinessAnalyzer(log_file)
    analyzer._log_event("Test Event", 0.25, 0.5)

    assert os.path.exists(log_file), "Log file should be created."

    with open(log_file, "r") as file:
        reader = csv.reader(file)
        header = next(reader)
        # Allow an optional Screenshot column; ensure the first four columns are present
        assert header[:4] == [
            "Timestamp",
            "Event",
            "EAR",
            "MAR",
        ], "Log file header is incorrect."
        data = next(reader)
        assert data[1] == "Test Event", "Event type was not logged correctly."
        assert float(data[2]) == 0.25, "EAR value was not logged correctly."
        assert float(data[3]) == 0.5, "MAR value was not logged correctly."

    # Clean up the test log file
    os.remove(log_file)
