import os
import cv2
import numpy as np
import kagglehub
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from drowsiness.face_detector import FaceDetector


def download_dataset():
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("dheerajperumandla/drowsiness-dataset")
    print("Path to dataset files:", path)

    # Define target directory
    target_dir = "data/drowsiness"

    # If data/drowsiness does not exist or is empty, copy files
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Check if the downloaded path contains the folders directly or a subfolder
    # The dataset structure usually has 'train' folder or the class folders directly
    # Based on user description: Closed, Open, yawn, no_yawn

    # We will try to find these folders in the downloaded path
    required_folders = ["Closed", "Open", "yawn", "no_yawn"]

    # Search recursively for these folders
    found_root = None
    for root, dirs, files in os.walk(path):
        if all(folder in dirs for folder in required_folders):
            found_root = root
            break

    if found_root:
        print(f"Found dataset root at {found_root}")
        for folder in required_folders:
            src = os.path.join(found_root, folder)
            dst = os.path.join(target_dir, folder)
            if os.path.exists(dst):
                print(f"Folder {dst} already exists. Skipping copy.")
            else:
                print(f"Copying {folder} to {target_dir}...")
                shutil.copytree(src, dst)
    else:
        print(
            "Could not find the expected dataset structure (Closed, Open, yawn, no_yawn) in the downloaded files."
        )
        print(f"Please check {path} manually.")

    return target_dir


def get_haarcascade_path():
    # Use opencv's built-in haarcascade if available, otherwise expect it in models folder
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(path):
        # Fallback to local models folder
        path = os.path.join("models", "haarcascade_frontalface_default.xml")
    return path


def get_smile_cascade_path():
    path = cv2.data.haarcascades + "haarcascade_smile.xml"
    if not os.path.exists(path):
        path = os.path.join("models", "haarcascade_smile.xml")
    return path


def get_mouth_from_face_image(face_image):
    """
    Takes a cropped face image and returns the mouth region.
    Uses smile detector or falls back to lower half of the face.
    """
    h, w = face_image.shape[:2]
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    # Mouth is usually in the lower half of the face
    mouth_search_y = int(h / 2)
    mouth_search_roi_gray = gray[mouth_search_y:, :]
    mouth_search_roi_color = face_image[mouth_search_y:, :]

    # Try to detect smile/mouth
    smile_cas_path = get_smile_cascade_path()
    if os.path.exists(smile_cas_path):
        smile_cascade = cv2.CascadeClassifier(smile_cas_path)
        # Lower neighbors and higher scale factor to detect open mouths (yawns)
        smiles = smile_cascade.detectMultiScale(mouth_search_roi_gray, 1.7, 11)

        if len(smiles) > 0:
            # Get the largest smile
            smiles = sorted(smiles, key=lambda x: x[2] * x[3], reverse=True)
            (sx, sy, sw, sh) = smiles[0]

            # Add some padding
            pad_w = int(sw * 0.1)
            pad_h = int(sh * 0.1)

            sx = max(0, sx - pad_w)
            sy = max(0, sy - pad_h)
            sw = min(w - sx, sw + 2 * pad_w)
            sh = min(h - mouth_search_y - sy, sh + 2 * pad_h)

            return mouth_search_roi_color[sy : sy + sh, sx : sx + sw]

    # Fallback: Return the lower half of the face
    return mouth_search_roi_color


def crop_mouth_from_face(image_array, face_cas_path=None):
    """
    Detects face using Haar Cascade, then attempts to detect and crop the mouth.
    If mouth detection fails, falls back to the lower half of the face.
    """
    if face_cas_path is None:
        face_cas_path = get_haarcascade_path()

    face_cascade = cv2.CascadeClassifier(face_cas_path)
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    # Assume the largest face is the target
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    (x, y, w, h) = faces[0]

    # Face ROI
    face_roi_color = image_array[y : y + h, x : x + w]

    return get_mouth_from_face_image(face_roi_color)


def face_for_yawn(direc="data/drowsiness", face_cas_path=None):
    if face_cas_path is None:
        face_cas_path = get_haarcascade_path()

    yaw_no = []
    IMG_SIZE = 145
    categories = ["yawn", "no_yawn"]
    for category in categories:
        path_link = os.path.join(direc, category)
        if not os.path.exists(path_link):
            print(f"Warning: Path {path_link} does not exist. Skipping.")
            continue

        class_num1 = categories.index(category)
        print(f"Processing {category}...")
        for image in os.listdir(path_link):
            try:
                image_path = os.path.join(path_link, image)
                image_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image_array is None:
                    continue

                # Crop mouth using Haar Cascade
                cropped_mouth = crop_mouth_from_face(image_array, face_cas_path)

                if cropped_mouth is not None:
                    resized_array = cv2.resize(cropped_mouth, (IMG_SIZE, IMG_SIZE))
                    yaw_no.append([resized_array, class_num1])
            except Exception as e:
                print(f"Error processing image {image}: {e}")
    return yaw_no


def get_data(dir_path="data/drowsiness"):
    labels = ["Closed", "Open"]
    IMG_SIZE = 145
    data = []
    for label in labels:
        path = os.path.join(dir_path, label)
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist. Skipping.")
            continue

        class_num = labels.index(label)
        class_num += 2  # 0: yawn, 1: no_yawn, 2: Closed, 3: Open
        print(f"Processing {label}...")
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_array, class_num])
            except Exception as e:
                print(f"Error processing image {img}: {e}")
    return data


def load_and_preprocess_data(data_dir="data/drowsiness"):
    # Check if data directory exists, if not try to download
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(
            f"Data directory {data_dir} not found or empty. Attempting to download..."
        )
        data_dir = download_dataset()

    if not os.path.exists(data_dir):
        print("Failed to download dataset.")
        return None, None

    yaw_no = face_for_yawn(direc=data_dir)
    data = get_data(dir_path=data_dir)

    if not yaw_no and not data:
        print("No data found.")
        return None, None

    yaw_no.extend(data)
    new_data = np.array(yaw_no, dtype=object)

    X = []
    y = []
    for feature, label in new_data:
        X.append(feature)
        y.append(label)

    X = np.array(X)
    X = X.reshape(-1, 145, 145, 3)

    from sklearn.preprocessing import LabelBinarizer

    label_bin = LabelBinarizer()
    y = label_bin.fit_transform(y)
    y = np.array(y)

    return X, y


def prepare_image(filepath, face_cas_path=None):
    IMG_SIZE = 145
    if face_cas_path is None:
        face_cas_path = get_haarcascade_path()

    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_array = img_array / 255.0
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
