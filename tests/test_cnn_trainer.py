import sys
import importlib
import types
import numpy as np


def test_import_cnn_trainer_without_sklearn(monkeypatch):
    """Ensure importing the module does not require scikit-learn at import-time.

    We simulate an environment where `sklearn` is not installed by removing it
    from `sys.modules` (if present) and then importing `drowsiness.cnn_trainer`.
    The import should succeed because `train_test_split` is imported lazily.
    """
    # Remove sklearn from sys.modules temporarily
    saved = sys.modules.pop("sklearn", None)
    saved_mms = {}
    # Also remove submodules that may exist
    for key in list(sys.modules.keys()):
        if key.startswith("sklearn."):
            saved_mms[key] = sys.modules.pop(key)

    try:
        if "drowsiness.cnn_trainer" in sys.modules:
            del sys.modules["drowsiness.cnn_trainer"]
        mod = importlib.import_module("drowsiness.cnn_trainer")
        assert hasattr(mod, "CNNTrainer")
    finally:
        # restore sklearn modules
        if saved is not None:
            sys.modules["sklearn"] = saved
        for k, v in saved_mms.items():
            sys.modules[k] = v


def test_predict_image_monkeypatched(monkeypatch, tmp_path):
    """Monkeypatch `load_model` and preprocessing functions to test predict_image."""
    import drowsiness.cnn_trainer as ct

    # Dummy image array to be returned by crop_mouth_from_face
    dummy_img = np.ones((145, 145, 3), dtype=np.uint8) * 128

    # Patch crop_mouth_from_face and prepare_image used in cnn_trainer
    monkeypatch.setattr(ct, "crop_mouth_from_face", lambda img: dummy_img)
    monkeypatch.setattr(ct, "prepare_image", lambda path: dummy_img.reshape(-1, 145, 145, 3) / 255.0)

    # Create a dummy model with a predict method
    class DummyModel:
        def predict(self, arr):
            # return a softmax-like vector with highest at index 2 (Closed)
            return np.array([[0.05, 0.05, 0.85, 0.05]])

    monkeypatch.setattr(ct, "load_model", lambda path: DummyModel())

    trainer = ct.CNNTrainer()

    # Create a temporary image file path (we won't actually read it because crop_mouth is patched)
    img_path = str(tmp_path / "img.png")
    # create an empty file to satisfy existence checks
    open(img_path, "wb").close()

    label, score = trainer.predict_image(img_path, model_path="models/drowsiness_cnn_trained.h5")
    assert isinstance(label, str)
    assert label == "Closed"
    assert isinstance(score, float)
