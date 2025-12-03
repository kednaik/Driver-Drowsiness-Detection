import cv2
import numpy as np
import os
import csv
import time
import datetime
from tensorflow.keras.models import load_model
from drowsiness.cnn_utils import get_haarcascade_path, get_mouth_from_face_image
from drowsiness.utils import play_alarm, log_event


# logging is handled by `drowsiness.utils.log_event`


def predict_drowsiness(model_path="models/drowsiness_cnn_trained.h5"):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    labels_new = ["yawn", "no_yawn", "Closed", "Open"]
    IMG_SIZE = 145

    # Load cascades and detectors
    face_cas_path = get_haarcascade_path()
    eye_cas_path = cv2.data.haarcascades + "haarcascade_eye.xml"

    face_cascade = cv2.CascadeClassifier(face_cas_path)
    eye_cascade = cv2.CascadeClassifier(eye_cas_path)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Starting video stream. Press 'q' to quit.")

    # Drowsiness detection state (eye-closure based)
    eye_consecutive_frames = 20
    eye_counter = 0
    is_drowsy = False
    last_drowsy_time = 0.0
    # Screenshots directory will be handled by utils.log_event

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        status = "Active"
        color = (0, 255, 0)

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Face ROI
            roi_color = frame[y : y + h, x : x + w]

            # 1. Yawn Detection (using cropped mouth)
            cropped_mouth = get_mouth_from_face_image(roi_color)

            if cropped_mouth is not None:
                resized_array = cv2.resize(cropped_mouth, (IMG_SIZE, IMG_SIZE))
                normalized_array = resized_array / 255.0
                reshaped_array = normalized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

                prediction = model.predict(reshaped_array)
                class_idx = np.argmax(prediction)
                prediction_label = labels_new[class_idx]

                if prediction_label == "yawn":
                    status = "Yawn Detected"
                    color = (0, 0, 255)

            # 2. Eye Detection (using Haar cascade on Face ROI)
            roi_gray = gray[y : y + h, x : x + w]
            eyes = eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=20
            )

            # Only apply eye-based model prediction when 1 or 2 eyes are detected
            n_eyes = len(eyes) if hasattr(eyes, "__len__") else 0
            frame_eye_closed = False
            if n_eyes in (1, 2):
                for ex, ey, ew, eh in eyes:
                    cv2.rectangle(
                        roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
                    )
                    eye_roi_color = roi_color[ey : ey + eh, ex : ex + ew]

                    # Resize and predict for eyes
                    try:
                        resized_eye = cv2.resize(eye_roi_color, (IMG_SIZE, IMG_SIZE))
                        normalized_eye = resized_eye / 255.0
                        reshaped_eye = normalized_eye.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

                        eye_prediction = model.predict(reshaped_eye)
                        eye_class_idx = np.argmax(eye_prediction)
                        eye_label = labels_new[eye_class_idx]

                        if eye_label == "Closed":
                            frame_eye_closed = True
                            status = "Drowsy (Eyes Closed)"
                            color = (0, 0, 255)

                    except Exception:
                        # Eye might be too small or other issue — skip this eye
                        pass
            else:
                # If zero or more than two eyes detected, skip eye-based prediction for this face.
                # This avoids unreliable predictions when detection is ambiguous.
                frame_eye_closed = False

            # Update drowsiness counter/state based on eye closure
            if frame_eye_closed:
                eye_counter += 1
            else:
                eye_counter = 0
                is_drowsy = False

            if eye_counter >= eye_consecutive_frames and not is_drowsy:
                is_drowsy = True
                last_drowsy_time = time.time()
                # Play alarm (non-blocking where possible)
                try:
                    play_alarm()
                except Exception:
                    pass

                # Log the event and let utils save the screenshot
                try:
                    log_event("Drowsiness Detected", ear=0.0, mar=0.0, frame=frame)
                except Exception:
                    pass

            cv2.putText(
                frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
            )

        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict_drowsiness()
