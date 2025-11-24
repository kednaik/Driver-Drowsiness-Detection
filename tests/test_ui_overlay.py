import numpy as np
import types
import sys


def test_draw_status_overlay_modifies_frame():
    # Create a blank BGR frame (height=480, width=640)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Avoid importing dlib at import-time by inserting a dummy 'dlib' module
    import sys
    import types

    sys.modules.setdefault("dlib", types.SimpleNamespace())

    # Now import the analyzer module (FaceDetector won't be initialized because
    # we will not call DrowsinessAnalyzer.__init__)
    from drowsiness import drowsiness_analyzer as dda

    analyzer = object.__new__(dda.DrowsinessAnalyzer)

    # Set only the attributes used by draw_status_overlay
    analyzer._is_drowsy = True
    analyzer._is_yawning = True

    # Ensure frame is initially all zeros
    assert frame.sum() == 0

    # Call the overlay drawing method
    analyzer.draw_status_overlay(frame, ear=0.1, mar=0.7)

    # After drawing, some pixels should be non-zero (rectangle/text drawn)
    assert frame.sum() > 0
