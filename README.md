# Driver Drowsiness Alert System

This project is a real-time driver drowsiness detection system that uses computer vision to prevent accidents caused by driver fatigue. It analyzes a driver's facial landmarks from a webcam feed to detect signs of drowsiness, such as closed eyes and yawning, and issues an alert.

## Running the Streamlit app with the project's `.venv`

If you created a virtual environment at `.venv`, run Streamlit using the venv Python so Streamlit uses the same packages:

1. Create and install (if you haven't already):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

2. Start the Streamlit app (two options):

- Using the helper script (recommended):

```bash
./scripts/run_streamlit_venv.sh
```

- Or directly with the venv Python (no activation required):

```bash
.venv/bin/python -m streamlit run app.py
```

3. In VS Code, you can point the workspace interpreter to the venv. The workspace settings file `.vscode/settings.json` is already configured to use `.venv/bin/python` if present.

If Streamlit still starts with the wrong Python, ensure you invoked the command with the `.venv` Python above, or configure your editor/IDE to use the `.venv` interpreter.


## Features

- **Real-time Drowsiness Detection:** Monitors the driver's face using a webcam.
- **Eye Aspect Ratio (EAR) Calculation:** Detects eye closure by calculating the EAR.
- **Yawn Detection:** Monitors the Mouth Aspect Ratio (MAR) to detect yawns.
- **Audible Alerts:** Plays an alarm sound when drowsiness is detected.
- **Event Logging:** Logs all drowsiness and yawn events to a CSV file for later analysis.

## Bonus: Streamlit Demo

As an additional feature (bonus), this repository includes a Streamlit web app that demonstrates all three drowsiness detection approaches (dlib landmarks, MediaPipe Face Mesh, and the CNN classifier). The app provides a simple UI to switch between detection modes, view live webcam output, and inspect logged events.

## How It Works

The system uses the `dlib` library to detect facial landmarks and `OpenCV` to process video frames. It calculates the Eye Aspect Ratio (EAR) to determine if the eyes are closed. If the EAR falls below a certain threshold for a specified number of consecutive frames, the system concludes that the driver is drowsy and plays an alarm. It also monitors the Mouth Aspect Ratio (MAR) to detect yawns, which are also logged as potential signs of fatigue.

## Project Structure

```
aai-551-project-v1/
│
├── drowsiness/
│   ├── __init__.py
│   ├── face_detector.py       # Class for face and landmark detection
│   ├── drowsiness_analyzer.py # Class for analyzing drowsiness
│   └── utils.py               # Utility functions (EAR, MAR, alarm)
│
├── tests/
│   └── test_drowsiness.py     # Pytest tests for the system
│
├── models/
│   └── shape_predictor_68_face_landmarks.dat # dlib's landmark model (needs to be downloaded)
│
├── logs/
│   └── drowsiness_log.csv     # Log file for drowsiness events
│
├── alarm/
│   └── alarm.wav              # Alarm sound file (needs to be added)
│
├── main.ipynb                 # Main Jupyter Notebook to run the system
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd aai-551-project-v1
    ```

2.  **Create and activate a virtual environment (recommended):**

    It's strongly recommended to run the project inside a Python virtual environment to avoid dependency conflicts.

    Option A — quick setup script (macOS/Linux `zsh`):
    ```bash
    # Create venv and install requirements
    bash scripts/setup_venv.sh
    # Activate the venv afterwards
    source .venv/bin/activate
    ```

    Option B — manual steps:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # on macOS / Linux (zsh)
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    Once the venv is activated, run Python scripts normally (they'll use the venv's interpreter).

    If you don't want to use a virtual environment, you can still install dependencies globally with:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the dlib shape predictor model:**
    - Download `shape_predictor_68_face_landmarks.dat.bz2` from [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
    - Extract the file and place `shape_predictor_68_face_landmarks.dat` in the `models/` directory.

4.  **Add an alarm sound:**
    - Place a `.wav` file named `alarm.wav` in the `alarm/` directory.

## Usage

### Method 1: Facial Landmarks (dlib)

Run the `main.ipynb` notebook to start the drowsiness detection system using facial landmarks.

### Method 2: CNN Model

This method uses a Convolutional Neural Network (CNN) trained on the Drowsiness Dataset.

1.  **Download the Dataset:**
    - Download the dataset from Kaggle: [Drowsiness Dataset](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset/data).
    - Extract the dataset into `data/drowsiness`. The structure should be:
        ```
        data/drowsiness/
        ├── Closed/
        ├── Open/
        ├── yawn/
        └── no_yawn/
        ```

2.  **Train the Model:**
    Run the training script to train the CNN model.
    ```bash
    # Use the trainer class to run training from Python or run the module if supported.
    # Programmatic usage (recommended):
    #
    # ```python
    # from drowsiness.cnn_trainer import CNNTrainer
    # trainer = CNNTrainer()
    # trainer.train(epochs=50, batch_size=32)
    # ```
    #
    # If you prefer a module entrypoint and it's available, you can run:
    python -m drowsiness.cnn_trainer
    ```
    This will save the trained model to `models/drowsiness_cnn.h5` and a training history plot.

3.  **Run Real-time Prediction:**
    Run the prediction script to use the trained model with your webcam.
    ```bash
    python -m drowsiness.cnn_analyzer
    ```

    should be updated to:

    ```bash
    python -m drowsiness.cnn_trainer
    ```

### Preprocessing for Yawn Detection

To improve the accuracy of yawn detection, the system preprocesses the input images (both for training and real-time prediction) by focusing on the mouth region:

1.  **Face Detection:** The system first detects the face using OpenCV's `haarcascade_frontalface_default.xml`.
2.  **Mouth Cropping:**
    - It attempts to detect the mouth within the lower half of the face using `haarcascade_smile.xml`.
    - If a smile/mouth is detected, it crops that specific region.
    - If no specific mouth region is detected (which can happen with wide yawns), it falls back to cropping the entire lower half of the face.
3.  **Resizing:** The cropped mouth image is then resized to 145x145 pixels before being fed into the CNN model.

This ensures that the model learns features specific to the mouth shape (open vs. closed) rather than irrelevant facial features.

## Testing
Run the test suite with `pytest`. Recommended workflow when using the project's virtual environment:

```bash
# activate the venv (macOS / Linux zsh)
source .venv/bin/activate

# (optional) ensure dev/test dependencies are installed
pip install -r requirements.txt

# run the full test suite
pytest

# run tests with more output (verbose)
pytest -q

# run a single test file
pytest tests/test_utils_logging.py

# run a single test function
pytest tests/test_utils_logging.py::test_log_event
```

Notes:
- If you created the virtual environment using `scripts/setup_venv.sh`, activate it with `source .venv/bin/activate` before running tests.
- On macOS with Apple Silicon you may need to install platform-specific packages (for example `tensorflow-macos` / `tensorflow-metal`) in the venv before running model-related tests.
- If a test requires hardware (webcam) or audio output, run it manually while the venv is active.

Running tests via helper script

We provide a small helper script to make running the test suite convenient. The script will activate the project's `.venv` (if present), add the project root to `PYTHONPATH`, and run `pytest`.

Usage:
```bash
# run default (quiet) test run
bash scripts/run_tests.sh

# pass pytest args through (verbose)
bash scripts/run_tests.sh -q

# make it executable then run
chmod +x scripts/run_tests.sh
./scripts/run_tests.sh
```

Notes:
- The script is a convenience wrapper; you can still activate the venv yourself and run `pytest` directly if you prefer.
- The script sets `PYTHONPATH` so tests can import `drowsiness` from the project root.
