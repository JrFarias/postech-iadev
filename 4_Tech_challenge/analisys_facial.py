import cv2
import json
import numpy as np
import mediapipe as mp

from utils import is_blink, analyze_emotions, is_hands_up, is_mouth_movements

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

video_path = "facial_analysis.mp4"
cap = cv2.VideoCapture(video_path)

# Configura o tempo máximo em segundos do video a ser analisado
max_seconds = 45

face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)
pose = mp_pose.Pose(
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)


def validate_video_time(frame_count: int) -> bool:
    fps = cap.get(cv2.CAP_PROP_FPS)

    if frame_count / fps > max_seconds:
        print("Tempo máximo de execução atingido.")
        return True

    return False


def init_detections(frame):
    # Converter de BGR para RGB e processar
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_detection_process = face_detection.process(image_rgb)

    # Processa a pose usando a mesma imagem RGB
    pose_detection = pose.process(image_rgb)
    frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    face_mesh_dection = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    return face_detection_process, pose_detection, face_mesh_dection, frame


def main():
    # Armazena o número de frames processados
    frame_count = 0

    emotions_list = []
    face_count = 0
    blink_count = 0
    mouth_movement_count = 0
    hands_up_count = 0
    unknown_positions_count = 0

    # Verifica se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    # fps = cap.get(cv2.CAP_PROP_FPS)
    # start_frame = int(fps * 40)  # Calcula o frame inicial para 10 segundos
    # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Define o frame inicial

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # if validate_video_time(frame_count):
        #     break

        ############################## Encontra rosto na imagem #####################################
        face_detection_process, pose_detection, face_mesh_detection, frame = (
            init_detections(frame)
        )
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
                    cv2.putText(
                        frame,
                        emotion,
                        (emotion_x, emotion_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 0, 0),
                        2,
                    )

        ############################## Encontra poses na imagem #####################################
        if pose_detection.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, pose_detection.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # Verificar se o braço está levantado
            if is_hands_up(pose_detection.pose_landmarks.landmark):
                hands_up_count += 1

        ############################## Analise detalhada dos rostos #####################################
        if face_mesh_detection.multi_face_landmarks:
            for face_landmarks in face_mesh_detection.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                if is_blink(landmarks):
                    blink_count += 1

                if is_mouth_movements(landmarks):
                    mouth_movement_count += 1

        cv2.putText(
            frame,
            f"Levantou as maos: {hands_up_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 189, 22),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Movimentos de boca: {mouth_movement_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 189, 22),
            2,
        )
        cv2.putText(
            frame,
            f"Piscou os olhos: {blink_count}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 189, 22),
            2,
        )
        # cv2.putText(frame, f"Movimentos de cabeça: {head_movement_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 189, 22), 2)
        cv2.putText(
            frame,
            f"Movimentos desconhecidos: {unknown_positions_count}",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 189, 22),
            2,
        )

        cv2.putText(
            frame,
            f"Faces: {face_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Análise Facial", frame)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    result = {
        "Total de Frames analisados": frame_count,
        "Levantou as mão acima do olhos": hands_up_count,
        "Emoções encontradas durante a análise": emotions_list,
        "Movimentos desconhecidos": unknown_positions_count,
        "Piscadas detectadas": blink_count,
        "Movimentos de boca detectados": mouth_movement_count,
    }

    print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
