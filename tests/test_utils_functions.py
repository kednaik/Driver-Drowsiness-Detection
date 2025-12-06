import os
import csv
import sys
import types
import numpy as np

# Ensure imports that reference `dlib` succeed in minimal test envs
sys.modules.setdefault("dlib", types.SimpleNamespace())

from drowsiness.utils import (
    compute_mouth_aspect_ratio,
    ensure_log_file,
    log_event,
)


def test_compute_mouth_aspect_ratio_open_and_closed():
    # Closed mouth: vertical distances zero -> MAR == 0.0
    # Construct a mouth where the vertical landmark pairs are identical
    # so the vertical distances are zero (MAR == 0.0).
    closed_mouth = [
        (0, 0),  # 49
        (10, 0),
        (20, 0),  # 51 (will be same as index 10)
        (30, 0),
        (40, 0),  # 53 (will be same as index 8)
        (50, 0),
        (60, 0),  # 55
        (70, 0),
        (40, 0),  # 57 (same as index 4)
        (90, 0),
        (20, 0),  # 59 (same as index 2)
        (110, 0),
    ]

    mar_closed = compute_mouth_aspect_ratio(closed_mouth)
    assert mar_closed == 0.0

    # Open mouth: vertical distances non-zero -> MAR > 0
    open_mouth = [
        (0, 0),
        (10, 5),
        (20, 20),  # 51
        (30, 5),
        (40, 20),  # 53
        (50, 5),
        (60, 0),
        (70, 5),
        (80, 20),  # 57
        (90, 5),
        (100, 20),  # 59
        (110, 5),
    ]

    mar_open = compute_mouth_aspect_ratio(open_mouth)
    assert mar_open > 0.0


def test_ensure_log_file_creates_header(tmp_path):
    log_file = str(tmp_path / "mylog.csv")
    # Ensure it's not present
    if os.path.exists(log_file):
        os.remove(log_file)

    ensure_log_file(log_file=log_file)
    assert os.path.exists(log_file)

    with open(log_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["Timestamp", "Event", "EAR", "MAR", "Screenshot"]


def test_log_event_writes_row_and_empty_screenshot(tmp_path):
    log_file = str(tmp_path / "events.csv")
    # Ensure directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Call log_event without a frame
    log_event("TestOnly", ear=0.12, mar=0.34, frame=None, log_file=log_file)

    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        rows = list(csv.reader(f))

    # header + one row
    assert len(rows) == 2
    header, data = rows
    assert data[1] == "TestOnly"
    # Screenshot field should be present (empty string)
    assert data[4] == ""
