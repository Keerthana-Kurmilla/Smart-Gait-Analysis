import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, deque

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Logs and smoothing buffer
posture_log = []
angle_buffer = deque(maxlen=5)

# -------- Angle Calculation -------- #
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return 0

    cosine_angle = np.dot(ba, bc) / denominator
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return angle


# -------- Knee Angle -------- #
def get_knee_angles(landmarks, image_shape):
    h, w, _ = image_shape

    left_hip = (landmarks[23].x * w, landmarks[23].y * h)
    left_knee = (landmarks[25].x * w, landmarks[25].y * h)
    left_ankle = (landmarks[27].x * w, landmarks[27].y * h)

    right_hip = (landmarks[24].x * w, landmarks[24].y * h)
    right_knee = (landmarks[26].x * w, landmarks[26].y * h)
    right_ankle = (landmarks[28].x * w, landmarks[28].y * h)

    left_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_angle = calculate_angle(right_hip, right_knee, right_ankle)

    # Convert to bent angle
    return 180 - left_angle, 180 - right_angle


# -------- Classification -------- #
def classify_knee(avg_angle):
    if avg_angle <= 20:
        return "Normal"
    else:
        return "Abnormal"


# -------- Camera -------- #
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

# -------- Main Loop -------- #
with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        posture = "Not Detected"
        left_angle, right_angle = 0, 0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            ids = [23, 24, 25, 26, 27, 28]
            visibility = [landmarks[i].visibility for i in ids]

            if min(visibility) > 0.4:
                left_angle, right_angle = get_knee_angles(landmarks, image.shape)

                avg_angle = (left_angle + right_angle) / 2
                angle_buffer.append(avg_angle)

                smooth_angle = sum(angle_buffer) / len(angle_buffer)
                posture = classify_knee(smooth_angle)

                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

        # Log
        posture_log.append(posture)

        # Color
        if posture == "Normal":
            color = (0, 255, 0)
        elif posture == "Abnormal":
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)

        # Display
        cv2.putText(image, f'Left Angle: {left_angle:.1f}', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        cv2.putText(image, f'Right Angle: {right_angle:.1f}', (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        cv2.putText(image, f'Posture: {posture}', (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Smart Gait Analysis", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# -------- Graph -------- #
counts = Counter(posture_log)

labels = ['Normal', 'Abnormal', 'Not Detected']
values = [
    counts.get('Normal', 0),
    counts.get('Abnormal', 0),
    counts.get('Not Detected', 0)
]

plt.bar(labels, values)
plt.title("Knee Posture Summary")
plt.xlabel("Posture")
plt.ylabel("Frames")
plt.show()