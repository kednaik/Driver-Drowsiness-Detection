import numpy as np
from drowsiness.utils import compute_mouth_aspect_ratio


def test_compute_mouth_aspect_ratio_open_and_closed():
    # Mouth landmarks are expected as a list where indices used are 0..10 mapping
    # For a wide open mouth, vertical distances are large compared to horizontal
    open_mouth = [
        (0, 0),  # 49 -> 0
        (1, 0),
        (2, 10),  # 51
        (3, 10),
        (4, 10),  # 53
        (5, 0),
        (8, 0),  # 55 -> 6
        (7, 0),
        (6, 10),  # 57 -> 8
        (9, 0),
        (2, 20),  # 59 -> 10
    ]

    mar_open = compute_mouth_aspect_ratio(open_mouth)
    assert mar_open > 0.3, "MAR for an open mouth should be reasonably large"

    # Closed mouth: vertical coordinates equal -> division by zero may occur -> expect NaN
    closed_mouth = [(0, 0)] * 11
    mar_closed = compute_mouth_aspect_ratio(closed_mouth)
    import math

    assert math.isnan(mar_closed), "MAR for degenerate (all-equal) mouth points should be NaN"
