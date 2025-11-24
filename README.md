# Driver Drowsiness Alert System

This project is a real-time driver drowsiness detection system that uses computer vision to prevent accidents caused by driver fatigue. It analyzes a driver's facial landmarks from a webcam feed to detect signs of drowsiness, such as closed eyes and yawning, and issues an alert.

## Features

- **Real-time Drowsiness Detection:** Monitors the driver's face using a webcam.
- **Eye Aspect Ratio (EAR) Calculation:** Detects eye closure by calculating the EAR.
- **Yawn Detection:** Monitors the Mouth Aspect Ratio (MAR) to detect yawns.
- **Audible Alerts:** Plays an alarm sound when drowsiness is detected.
- **Event Logging:** Logs all drowsiness and yawn events to a CSV file for later analysis.

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

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the shape predictor model:**
    Download the `shape_predictor_68_face_landmarks.dat` file from [dlib's website](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
    Extract the file and place it in the `models/` directory.

5.  **Add an alarm sound:**
    Place an alarm sound file (e.g., `alarm.wav`) in the `alarm/` directory.

## How to Run

1.  **Run the Jupyter Notebook:**
    Start Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter lab
    ```
    Open `main.ipynb` and run the cells.

2.  **Using the system:**
    -   A window will open showing your webcam feed.
    -   The system will draw contours around your eyes and mouth.
    -   The current EAR and MAR values will be displayed on the screen.
    -   If the system detects drowsiness, it will display a "DROWSINESS ALERT!" message and play an alarm sound.
    -   A top-centered status overlay shows `DROWSY` and/or `YAWNING`. The overlay will flash for 1 second after an event is detected to draw attention; the flash duration and interval are configurable in the analyzer implementation.
    -   Press the 'q' key to stop the program.

## Running Tests

To run the tests, use `pytest`:
```bash
pytest
```

## Dependencies

-   OpenCV
-   dlib
-   NumPy
-   SciPy
-   Pygame
-   pytest

See `requirements.txt` for specific versions.
