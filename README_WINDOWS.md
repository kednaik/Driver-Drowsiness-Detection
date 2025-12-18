# Driver Drowsiness Detection System (Windows)

## 📌 Overview

This project is a **real-time Driver Drowsiness Detection System** implemented in **Python** and designed to run on **Windows**.  
It uses a webcam feed to detect signs of driver fatigue such as **eye closure (EAR)** and **yawning (MAR)**.

### Supported Detection Methods (Windows)
- ✅ **MediaPipe (Face Mesh)** — **Recommended and supported**
- ❌ CNN — Disabled due to TensorFlow/Keras model compatibility issues
- ❌ Dlib — Disabled due to native dependency and build constraints on Windows

> **Important:**  
> On Windows, **MediaPipe is the only fully supported and stable detection method**.

---

## 🧠 How It Works

- Captures live video from the webcam using OpenCV
- Uses **MediaPipe Face Mesh** to extract facial landmarks
- Computes:
  - **EAR (Eye Aspect Ratio)** → eye closure detection
  - **MAR (Mouth Aspect Ratio)** → yawning detection
- Displays real-time visual overlays and alerts in a **Streamlit web interface**

---

## 💻 System Requirements

- **Operating System:** Windows 10 / 11 (64-bit)
- **Python:** 3.10 or 3.11 (tested with 3.11)
- **Webcam:** Built-in or external USB camera
- **Browser:** Chrome / Edge / Firefox

---

## 🧪 Environment Setup (Windows)

1️⃣ Create a Virtual Environment

From the project root directory:

```powershell
python -m venv .venv


2️⃣ Activate the Virtual Environment
.venv\Scripts\activate


You should see:

(.venv)


in your terminal prompt.

3️⃣ Install Dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

🚀 How to Run the Application

From the project root:

streamlit run app.py


Streamlit will start the application and display:

Local URL: http://localhost:8501


Open that URL in your browser.

🧪 How to Test the Application

Open the Streamlit UI in your browser

In the left sidebar:

Select MediaPipe as the detection method

Choose the correct Camera Index (usually 0)

Enable Run Camera

Position your face in front of the webcam

Observe:

EAR value (eye openness)

MAR value (mouth openness)

Alerts will appear when:

Eyes remain closed for a sustained period

Yawning is detected

⚙️ Camera Index Troubleshooting

If the camera does not start:

Try camera indices 0, 1, or 2

Close other applications using the camera (Zoom, Teams, etc.)

Restart the Streamlit app after changing the index

🛠️ Troubleshooting
MediaPipe Loads but No Face Is Detected

Ensure your face is well-lit

Face the camera directly

Avoid strong backlighting

TensorFlow / oneDNN Warnings

You may see messages like:

oneDNN custom operations are on


These are informational warnings and do not affect functionality.

CNN or Dlib Errors

These methods are intentionally disabled on Windows due to:

Native build issues (dlib)

Model deserialization incompatibilities (CNN)

Use MediaPipe instead.

Streamlit App Does Not Start

Ensure the virtual environment is activated

Confirm streamlit is installed:

pip show streamlit

📁 Project Structure (Simplified)
Driver-Drowsiness-Detection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── drowsiness/
│   ├── base_analyzer.py
│   ├── mediapipe_analyzer.py
│   └── utils.py
│
└── .venv/

✅ Recommended Configuration (Windows)
Component	Setting
Detection Method	MediaPipe
Camera Index	0
Python Version	3.10 – 3.11
Platform	Windows
📌 Final Notes

This project is optimized for local execution

MediaPipe provides real-time performance without GPU

Designed for academic and experimental use