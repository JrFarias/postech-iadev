import cv2
import json
import numpy as np
import mediapipe as mp
from deepface import DeepFace 

# Instanciando o DeepFace
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Instanciando o detector de pose
mp_pose = mp.solutions.pose

# Caminho do vídeo a ser analisado
video_path = "facial_analysis.mp4"

# Captura o vídeo do arquivo especificado
cap = cv2.VideoCapture(video_path)

# Configura o tempo máximo em segundos do video a ser analisado
max_seconds = 30

# Lista para armazenar todas as emoções detectadas
emotions_list = []  

# Variável para contar o número de faces detectadas
face_count = 0

# Inicializa os módulos de face e pose
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True, 
                    enable_segmentation=False, min_detection_confidence=0.5)

def validate_video_time(frame_count: int) -> bool:
    fps = cap.get(cv2.CAP_PROP_FPS)

    if frame_count / fps > max_seconds:
         print("Tempo máximo de execução atingido.")
         return True 
         
    return False

def process_detections(frame):
    # Converter de BGR para RGB e processar
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_detection_process = face_detection.process(image_rgb)

    # Processa a pose usando a mesma imagem RGB
    pose_detection = pose.process(image_rgb)
    frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    return face_detection_process, pose_detection, frame

def analyze_emotions(frame, bbox):
    h_img, w_img, _ = frame.shape
    emotion_x = int(bbox.xmin * w_img)
    emotion_y = int(bbox.ymin * h_img)
    w_box = int(bbox.width * w_img)
    h_box = int(bbox.height * h_img)
    
    # Verifica se a região analisada está dentro dos limites da imagem
    if emotion_x < 0 or emotion_y < 0 or emotion_x + w_box > w_img or emotion_y + h_box > h_img:
        return None

    face_roi = frame[emotion_y:emotion_y+h_box, emotion_x:emotion_x+w_box]
    face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)                   
    
    default_emotion = "N/A"
    try:
        analysis = DeepFace.analyze(face_roi_rgb, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = analysis[0]
        emotion = analysis['dominant_emotion']
    except Exception:
        emotion = default_emotion
    
    return emotion, emotion_x, emotion_y 


def is_arm_up(landmarks):
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

    left_arm_up = left_elbow.y < left_eye.y
    right_arm_up = right_elbow.y < right_eye.y
 

    return left_arm_up or right_arm_up

# // TODO add new function to check if the hands are up
def is_hand_chacking(landmarks):
    letf_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    left_hand = landmarks[mp_pose.PoseLandmark.LEFT_HAND.value]
    right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_HAND.value]

    left_hand_up = left_hand.y < left_hand.x
    right_hand_up = right_hand.y < right_hand.x

    return left_hand_up or right_hand_up

def main():
    # Conta o número de frames processados
    frame_count = 0

    arm_movements_count = 0

    # Verifica se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Verifica se o tempo máximo foi atingido e interrompe o loop
        frame_count += 1
        if validate_video_time(frame_count):
            break

        face_detection_process, pose_detection, frame = process_detections(frame)
        if face_detection_process.detections:
            face_count = len(face_detection_process.detections)
            for idx, detection in enumerate(face_detection_process.detections):

                mp_drawing.draw_detection(frame, detection)
                bbox = detection.location_data.relative_bounding_box

            ############################## ANALISE DE EMOÇÕES #####################################
                result = analyze_emotions(frame, bbox)
                if result is not None:
                    emotion, emotion_x, emotion_y = result
                    if emotion not in emotions_list and emotion != "N/A":
                        emotions_list.append(emotion)
                    # Desenha o retângulo e a emoção detectada
                    cv2.putText(frame, emotion, (emotion_x, emotion_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            ##########################################################################################             

        # Desenha os landmarks da pose se houver
        if pose_detection.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_detection.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Verificar se o braço está levantado
            if is_arm_up(pose_detection.pose_landmarks.landmark):
                arm_movements_count += 1

        cv2.putText(frame, f'Movimentos dos bracos: {arm_movements_count}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Faces: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Análise Facial", frame)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    result = {
        "Total de Frames analisados": frame_count,
        "Total de movimentos de braços levantados": arm_movements_count,
        "Emoções encontradas durante a análise": emotions_list,
    }

    print(json.dumps(result, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()