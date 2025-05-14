import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace


def analyze_emotions(frame, bbox):
    h_img, w_img, _ = frame.shape
    emotion_x = int(bbox.xmin * w_img)
    emotion_y = int(bbox.ymin * h_img)
    w_box = int(bbox.width * w_img)
    h_box = int(bbox.height * h_img)

    # Verifica se a região analisada está dentro dos limites da imagem
    if (
        emotion_x < 0
        or emotion_y < 0
        or emotion_x + w_box > w_img
        or emotion_y + h_box > h_img
    ):
        return None

    face_roi = frame[emotion_y : emotion_y + h_box, emotion_x : emotion_x + w_box]
    face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

    default_emotion = "N/A"
    try:
        analysis = DeepFace.analyze(
            face_roi_rgb, actions=["emotion"], enforce_detection=False
        )
        if isinstance(analysis, list):
            analysis = analysis[0]
        emotion = analysis["dominant_emotion"]
    except Exception:
        emotion = default_emotion

    return emotion, emotion_x, emotion_y


def is_hands_up(landmarks):
    mp_pose = mp.solutions.pose
    try:
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        # Verifica se os pontos necessários estão visíveis
        if (
            left_eye.visibility < 0.5
            or right_eye.visibility < 0.5
            or left_wrist.visibility < 0.5
            or right_wrist.visibility < 0.5
        ):
            return False

        left_arm_up = left_wrist.y < left_eye.y
        right_arm_up = right_wrist.y < right_eye.y

        return left_arm_up or right_arm_up
    except IndexError:
        return False


def is_mouth_movements(landmarks, previous_mouth_opening=[0]):
    mouth_threshold = 0.03

    upper_lip = landmarks[13]
    lower_lip = landmarks[14]
    current_mouth_opening = abs(upper_lip.y - lower_lip.y)

    # Detect movement only if the change in mouth opening exceeds the threshold
    mouth_movement_detected = (
        abs(current_mouth_opening - previous_mouth_opening[0]) > mouth_threshold
    )

    # Update the previous mouth opening value
    previous_mouth_opening[0] = current_mouth_opening

    return mouth_movement_detected


def _eye_aspect_ratio(landmarks, eye_indices):
    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
    ear = (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (
        2.0 * np.linalg.norm(p[0] - p[3])
    )
    return ear


def is_blink(landmarks, previous_ear=[0.3]):
    left_eye_indices = [33, 160, 158, 133, 153, 144]
    right_eye_indices = [362, 385, 387, 263, 373, 380]
    ear_threshold = 0.21
    blink_threshold = 0.05  # Threshold to detect a blink based on EAR change

    left_ear = _eye_aspect_ratio(landmarks, left_eye_indices)
    right_ear = _eye_aspect_ratio(landmarks, right_eye_indices)
    avg_ear = (left_ear + right_ear) / 2.0

    blink_detected = (
        previous_ear[0] > ear_threshold
        and avg_ear < ear_threshold
        and abs(previous_ear[0] - avg_ear) > blink_threshold
    )
    previous_ear[0] = avg_ear

    return blink_detected
