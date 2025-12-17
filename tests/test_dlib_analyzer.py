import sys
import types
import numpy as np

# Ensure 'dlib' is present as a lightweight stub for imports
sys.modules.setdefault("dlib", types.SimpleNamespace())

from drowsiness.dlib_analyzer import DlibAnalyzer


class _MockShape:
    def __init__(self, coords):
        self._coords = coords

    def part(self, idx):
        x, y = self._coords[idx]
        return types.SimpleNamespace(x=int(x), y=int(y))


def test_init_raises_when_predictor_missing(tmp_path):
    missing = tmp_path / "no_such_predictor.dat"
    try:
        DlibAnalyzer(str(missing))
        assert False, "Expected FileNotFoundError when predictor missing"
    except FileNotFoundError:
        pass


def test_analyze_frame_with_mock_landmarks_marks_drowsy():
    # Build dummy 68 landmarks: eyes closed (vertical distances zero)
    coords = [(i * 2, 100) for i in range(68)]

    # left eye indices 36-41: craft points so vertical pairs are identical
    coords[36] = (10, 100)
    coords[37] = (20, 100)
    coords[38] = (22, 100)
    coords[39] = (30, 100)
    coords[40] = coords[38]
    coords[41] = coords[37]

    # right eye indices 42-47: similar pattern shifted to the right
    coords[42] = (40, 100)
    coords[43] = (50, 100)
    coords[44] = (52, 100)
    coords[45] = (60, 100)
    coords[46] = coords[44]
    coords[47] = coords[43]

    # mouth indices 48-67: small vertical opening
    for i in range(48, 68):
        coords[i] = (coords[i][0], 110)

    mock_shape = _MockShape(coords)

    # Create analyzer instance without running __init__ (to avoid file checks)
    analyzer = object.__new__(DlibAnalyzer)

    # Provide a get_cv2_landmarks implementation that returns our mock shape
    analyzer.get_cv2_landmarks = lambda gray: [mock_shape]

    # Ensure thresholds are such that a single closed-eye frame triggers drowsiness
    analyzer.ear_consecutive_frames = 1
    analyzer.ear_threshold = 0.2
    analyzer.mar_threshold = 0.5

    # Initialize internal state expected by BaseAnalyzer
    analyzer._ear_counter = 0
    analyzer._yawn_counter = 0
    analyzer._is_drowsy = False
    analyzer._is_yawning = False
    analyzer._last_drowsy_time = 0.0
    analyzer._last_yawn_time = 0.0

    # Create a blank BGR image
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    # Replace _apply_detection_logic with a capture function so we can assert
    # the EAR and MAR values computed by analyze_frame without depending on
    # BaseAnalyzer internal counters.
    captured = {}

    def _capture(ear, mar, frm):
        captured["ear"] = ear
        captured["mar"] = mar

    analyzer._apply_detection_logic = _capture

    annotated, is_drowsy, is_yawning = analyzer.analyze_frame(frame.copy())
    assert isinstance(annotated, np.ndarray)
    # EAR should be computed as 0.0 for our closed-eye mock
    assert "ear" in captured
    assert abs(captured["ear"] - 0.0) < 1e-6
    assert "mar" in captured
