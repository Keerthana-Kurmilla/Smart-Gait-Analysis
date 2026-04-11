# Smart-Gait-Analysis

##  Overview

Smart Gait Analysis is an AI-based computer vision project that analyzes human knee posture in real-time using a webcam. It detects body landmarks, calculates knee angles, and classifies posture as **Normal**, **Abnormal**, or **Not Detected**.

---
##  Features

*  Real-time pose detection using MediaPipe
*  Knee angle calculation using geometry
*  Posture classification (Normal / Abnormal / Not Detected)
*  Angle smoothing using buffer (deque)
*  Graph visualization of posture results
*  Handles low visibility and no-person scenarios

---

## Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy
* Matplotlib

---

##  Project Structure

```
Smart-Gait-Analysis/
│── app.py
│── README.md
```

---

##  How to Run the Project

### 1️ Clone the repository

```
git clone https://github.com/Keerthana-Kurmilla/Smart-Gait-Analysis.git
cd Smart-Gait-Analysis
```

### 2️ Create virtual environment (optional but recommended)

```
python -m venv venv
venv\Scripts\activate
```

### 3️ Install dependencies

```
pip install opencv-python mediapipe numpy matplotlib
```

### 4️ Run the project

```
python app.py
```

---

##  Usage

* The webcam will open automatically
* Stand or sit in front of the camera
* The system will display:

  * Knee angles
  * Posture classification
* Press **Q** to exit
* A graph will be displayed after closing

---

##  Output

* 🟢 Normal → Proper posture
* 🔴 Abnormal → Incorrect posture
* 🟡 Not Detected → Body not clearly visible

---

## 📸 Screenshots

*(Add your project screenshots here)*

---

##  Applications

* Healthcare monitoring
* Physiotherapy assistance
* Fitness posture correction
* Rehabilitation tracking

---

##  Future Enhancements

* Web app deployment (Streamlit)
* Video file input support
* Machine learning-based classification
* Data export (CSV reports)

---

## 🔗 Project Link

https://github.com/Keerthana-Kurmilla/Smart-Gait-Analysis
