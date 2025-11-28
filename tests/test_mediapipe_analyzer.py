import sys
import types
import numpy as np


def test_mediapipe_analyzer_process_frame_monkeypatched():
    """
    Test MediapipeAnalyzer.process_frame by injecting a fake `mediapipe` module
    with a minimal `solutions.face_mesh.FaceMesh` implementation that returns
    predictable landmarks. This avoids the heavy native MediaPipe dependency
    while verifying EAR/MAR computation and landmark handling.
    """

    # Build a fake mediapipe module structure
    fake_mp = types.SimpleNamespace()
    solutions_ns = types.SimpleNamespace()

    # Create a fake FaceMesh class
    class FakeFaceMesh:
        def __init__(self, *args, **kwargs):
            pass

        def process(self, image_rgb):
            # Create a fake results object with one face_landmarks entry
            class FakeResults:
                pass

            class FakeFaceLandmarks:
                def __init__(self):
                    # Create a list of 300 dummy landmarks with x,y normalized
                    self.landmark = [types.SimpleNamespace(x=0.5, y=0.5) for _ in range(400)]

                    # Set left eye landmarks (indices used in analyzer)
                    # left_indices = [33, 160, 158, 133, 153, 144]
                    self.landmark[33].x, self.landmark[33].y = 0.2, 0.4  # p1
                    self.landmark[160].x, self.landmark[160].y = 0.2, 0.38  # p2
                    self.landmark[158].x, self.landmark[158].y = 0.25, 0.38  # p3
                    self.landmark[133].x, self.landmark[133].y = 0.3, 0.4  # p4
                    self.landmark[153].x, self.landmark[153].y = 0.25, 0.42  # p5
                    self.landmark[144].x, self.landmark[144].y = 0.2, 0.42  # p6

                    # Right eye indices (we'll mirror left across x)
                    self.landmark[362].x, self.landmark[362].y = 0.7, 0.4
                    self.landmark[385].x, self.landmark[385].y = 0.7, 0.38
                    self.landmark[387].x, self.landmark[387].y = 0.65, 0.38
                    self.landmark[263].x, self.landmark[263].y = 0.6, 0.4
                    self.landmark[373].x, self.landmark[373].y = 0.65, 0.42
                    self.landmark[380].x, self.landmark[380].y = 0.7, 0.42

                    # Mouth indices: 61, 291, 13, 14
                    self.landmark[61].x, self.landmark[61].y = 0.4, 0.6
                    self.landmark[291].x, self.landmark[291].y = 0.6, 0.6
                    self.landmark[13].x, self.landmark[13].y = 0.5, 0.55
                    self.landmark[14].x, self.landmark[14].y = 0.5, 0.65

            results = FakeResults()
            results.multi_face_landmarks = [FakeFaceLandmarks()]
            return results

    # Attach fake FaceMesh and drawing utils placeholders to solutions namespace
    face_mesh_ns = types.SimpleNamespace(FaceMesh=FakeFaceMesh, FACEMESH_TESSELATION=None)
    drawing_utils = types.SimpleNamespace()
    # Provide a dummy draw_landmarks to avoid attribute errors if called
    def dummy_draw_landmarks(*args, **kwargs):
        return None

    drawing_utils.draw_landmarks = dummy_draw_landmarks
    drawing_utils.DrawingSpec = lambda **kwargs: None

    solutions_ns.face_mesh = face_mesh_ns
    solutions_ns.drawing_utils = drawing_utils

    fake_mp.solutions = solutions_ns

    # Insert fake mediapipe into sys.modules so the module under test will pick it up
    sys.modules["mediapipe"] = fake_mp

    # Ensure the module is (re)imported fresh
    if "drowsiness.mediapipe_analyzer" in sys.modules:
        del sys.modules["drowsiness.mediapipe_analyzer"]

    # Now import the MediapipeAnalyzer under test
    from drowsiness.mediapipe_analyzer import MediapipeAnalyzer

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    analyzer = MediapipeAnalyzer()
    ear, mar = analyzer.process_frame(frame, draw_landmarks=False)

    # Both EAR and MAR should be computed and be floats
    assert ear is not None and isinstance(ear, float)
    assert mar is not None and isinstance(mar, float)
