import os
from drowsiness.predict_cnn import predict_drowsiness
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
)
from tensorflow.keras.models import Sequential
from drowsiness.cnn_utils import load_and_preprocess_data
from drowsiness.cnn_utils import crop_mouth_from_face, prepare_image
from tensorflow.keras.models import load_model
import cv2
import argparse


class CNNTrainer:
    """Class that encapsulates model creation and training for the drowsiness CNN.

    Usage:
        trainer = CNNTrainer()
        trainer.train(data_dir='data/drowsiness', epochs=10, batch_size=32)
    """

    def __init__(self, input_shape=(145, 145, 3), num_classes=4):
        """
        Initialize a trainer instance.

        Parameters
        ----------
        input_shape : tuple
            The expected shape of input images (height, width, channels).
        num_classes : int
            Number of output classes for the classifier (default 4: yawn/no_yawn/Closed/Open).
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        """
        Build a simple CNN classifier model.

        The architecture is a small convolutional network with successive
        Conv2D + MaxPooling blocks, followed by Flatten, Dropout and Dense
        layers. The final layer uses softmax to produce class probabilities.

        Returns
        -------
        model : keras.Model
            A compiled Keras `Sequential` model ready for training.
        """
        model = Sequential()

        model.add(Conv2D(256, (3, 3), activation="relu", input_shape=self.input_shape))
        model.add(MaxPooling2D(2, 2))

        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(MaxPooling2D(2, 2))

        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(2, 2))

        model.add(Conv2D(32, (3, 3), activation="relu"))
        model.add(MaxPooling2D(2, 2))

        model.add(Flatten())
        model.add(Dropout(0.5))

        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.num_classes, activation="softmax"))

        model.compile(
            loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam"
        )

        self.model = model
        return model

    def train(
        self,
        data_dir="data/drowsiness",
        epochs=50,
        batch_size=32,
        model_save_path="models/drowsiness_cnn.h5",
    ):
        """
        Train the CNN model using preprocessed data from `data_dir`.

        The method loads data via `load_and_preprocess_data`, splits into
        train/test sets (uses scikit-learn locally), builds the model if
        necessary, fits the model using Keras data generators, saves the
        trained weights, and writes a training history plot.

        Parameters
        ----------
        data_dir : str
            Directory containing the preprocessed dataset (default 'data/drowsiness').
        epochs : int
            Number of training epochs.
        batch_size : int
            Batch size used by the training data generator.
        model_save_path : str
            Filesystem path where the trained model `.h5` will be saved.

        Returns
        -------
        tuple
            `(model, history)` on success, or `None` when data loading fails.
        """
        # Load data
        X, y = load_and_preprocess_data(data_dir)
        if X is None or y is None:
            print("Failed to load data. Exiting training.")
            return None

        # Update input shape / classes based on data
        self.input_shape = X.shape[1:]
        self.num_classes = y.shape[1]

        # Split
        # Import locally to avoid requiring scikit-learn at module import time
        try:
            from sklearn.model_selection import train_test_split
        except Exception:
            raise ImportError(
                "scikit-learn is required for training. Install it with: .venv/bin/python -m pip install scikit-learn"
            )

        seed = 42
        test_size = 0.30
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=seed, test_size=test_size
        )

        # Generators
        train_generator = ImageDataGenerator(
            rescale=1 / 255, zoom_range=0.2, horizontal_flip=True, rotation_range=30
        )
        test_generator = ImageDataGenerator(rescale=1 / 255)

        train_generator = train_generator.flow(
            np.array(X_train), y_train, shuffle=False, batch_size=batch_size
        )
        test_generator = test_generator.flow(
            np.array(X_test), y_test, shuffle=False, batch_size=batch_size
        )

        # Build model
        if self.model is None:
            self.build_model()

        # Train
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=test_generator,
            shuffle=True,
            validation_steps=len(test_generator),
        )

        # Save
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        self.model.save(model_save_path)

        # Plot history
        accuracy = history.history.get("accuracy", [])
        val_accuracy = history.history.get("val_accuracy", [])
        loss = history.history.get("loss", [])
        val_loss = history.history.get("val_loss", [])
        epochs_range = range(len(accuracy))

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, accuracy, "b", label="Training Accuracy")
        plt.plot(epochs_range, val_accuracy, "r", label="Validation Accuracy")
        plt.legend()
        plt.title("Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, "b", label="Training Loss")
        plt.plot(epochs_range, val_loss, "r", label="Validation Loss")
        plt.legend()
        plt.title("Loss")

        plot_path = os.path.join(
            os.path.dirname(model_save_path), "training_history.png"
        )
        plt.savefig(plot_path)
        print(f"Training history plot saved to {plot_path}")

        return self.model, history

    def predict_image(self, image_path, model_path="models/drowsiness_cnn_trained.h5"):
        """
        Predict a single image file using a trained CNN model.

        Behavior
        --------
        - Validates that `image_path` and `model_path` exist.
        - Loads the model from `model_path` using `load_model`.
        - Attempts to detect and crop the mouth from the image; if cropping
          fails the entire image is preprocessed and used for prediction.

        Parameters
        ----------
        image_path : str
            Path to an input image file to classify.
        model_path : str
            Path to the trained Keras `.h5` model file.

        Returns
        -------
        tuple
            `(label, score)` where `label` is one of `['yawn','no_yawn','Closed','Open']`
            and `score` is the predicted probability for the returned class.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = load_model(model_path)

        # Try cropping mouth from detected face first
        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            cropped = crop_mouth_from_face(img)
        except Exception:
            cropped = None

        if cropped is not None:
            IMG_SIZE = 145
            resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
            arr = resized / 255.0
            arr = arr.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        else:
            arr = prepare_image(image_path)

        preds = model.predict(arr)
        labels_new = ["yawn", "no_yawn", "Closed", "Open"]
        class_idx = int(np.argmax(preds))
        return labels_new[class_idx], float(preds[0][class_idx])


def _cli_args():
    parser = argparse.ArgumentParser(description="CNN trainer / predictor CLI")
    sub = parser.add_subparsers(dest="cmd")

    t = sub.add_parser("train", help="Train the CNN model")
    t.add_argument("--data-dir", default="data/drowsiness")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--model-save-path", default="models/drowsiness_cnn.h5")

    p = sub.add_parser("predict", help="Run prediction")
    p.add_argument(
        "--image",
        help="Path to image file to predict (optional). If omitted, opens webcam via legacy predictor.",
    )
    p.add_argument("--model", default="models/drowsiness_cnn_trained.h5")

    return parser.parse_args()


if __name__ == "__main__":
    args = _cli_args()
    if args.cmd == "train":
        trainer = CNNTrainer()
        trainer.train(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_save_path=args.model_save_path,
        )
    elif args.cmd == "predict":
        if args.image:
            trainer = CNNTrainer()
            try:
                label, score = trainer.predict_image(args.image, model_path=args.model)
                print(f"Prediction: {label} ({score:.4f})")
            except Exception as e:
                print(f"Prediction failed: {e}")
        else:
            # Fallback to legacy interactive webcam predictor
            try:
                predict_drowsiness(args.model)
            except Exception as e:
                print(f"Interactive prediction failed: {e}")
    else:
        print("No command given. Use 'train' or 'predict'.")
