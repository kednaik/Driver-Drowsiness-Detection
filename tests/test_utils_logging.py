import os
import sys
import types
import tempfile
import numpy as np


def test_play_alarm_missing_file_returns_none(capsys, monkeypatch):
    # Ensure missing file returns None and prints helpful message
    from drowsiness.utils import play_alarm

    res = play_alarm(sound_file="nonexistent_file_hopefully_missing.wav")
    assert res is None
    captured = capsys.readouterr()
    assert "Alarm sound file not found" in captured.out


def test_log_event_writes_csv_and_screenshot(tmp_path, monkeypatch):
    # Provide dummy 'dlib' so DrowsinessAnalyzer can be imported if needed
    sys.modules.setdefault("dlib", types.SimpleNamespace())

    from drowsiness.drowsiness_analyzer import DrowsinessAnalyzer

    # Create fake BGR frame
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    log_file = str(tmp_path / "test_log.csv")

    # Create an analyzer object without running FaceDetector-heavy init by
    # allocating object and calling _init_log_file manually.
    analyzer = object.__new__(DrowsinessAnalyzer)
    analyzer.log_file = log_file
    # Ensure directories exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    analyzer._init_log_file()

    # Call _log_event with a frame to force screenshot saving
    analyzer._log_event("UnitTestEvent", 0.12, 0.34, frame=frame)

    # Check CSV created and contains expected event
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        content = f.read()
    assert "UnitTestEvent" in content

    # Check screenshots directory exists and has a png
    screenshots_dir = os.path.join(os.path.dirname(log_file), "screenshots")
    assert os.path.isdir(screenshots_dir)
    files = [p for p in os.listdir(screenshots_dir) if p.endswith(".png")]
    assert len(files) >= 1
