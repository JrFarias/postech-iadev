import cv2
import mediapipe as mp
from deepface import DeepFace 

def main():
    # Instanciando o DeepFace
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # Instanciando o detector de pose
    mp_pose = mp.solutions.pose

    # Caminho do vídeo a ser analisado
    video_path = "facial_analysis.mp4"
    # Captura o vídeo do arquivo especificado
    cap = cv2.VideoCapture(video_path)

    # Verifica se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return
    
    # Configurar o tempo máximo em segundos
    max_seconds = 20
    # Obter FPS do vídeo para cálculo do tempo
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Conta o número de frames processados
    frame_count = 0

    # Inicializa os módulos de face e pose
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True, 
                        enable_segmentation=False, min_detection_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Verifica se o tempo máximo foi atingido e interrompe o loop
        if frame_count / fps > max_seconds:
            print("Tempo máximo de execução atingido.")
            break

        # Converter de BGR para RGB e processar
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        # Processa a pose usando a mesma imagem RGB
        results_pose = pose.process(image_rgb)
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        face_count = 0
        if results.detections:
            face_count = len(results.detections)
            for idx, detection in enumerate(results.detections):
                mp_drawing.draw_detection(frame, detection)
                bbox = detection.location_data.relative_bounding_box

                # EXTRAI A REGIÃO DO ROSTO E REALIZA A ANÁLISE DE EMOÇÃO
                h_img, w_img, _ = frame.shape
                x = int(bbox.xmin * w_img)
                y = int(bbox.ymin * h_img)
                w_box = int(bbox.width * w_img)
                h_box = int(bbox.height * h_img)
                
                # Verifica se a região é válida
                if x < 0 or y < 0 or x + w_box > w_img or y + h_box > h_img:
                    continue
                face_roi = frame[y:y+h_box, x:x+w_box]
                face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)                   

                
                try:
                    analysis = DeepFace.analyze(face_roi_rgb, actions=['emotion'], enforce_detection=False)
                    # Se o retorno for uma lista, pega o primeiro resultado
                    if isinstance(analysis, list):
                        analysis = analysis[0]
                    emotion = analysis['dominant_emotion']

                except Exception as e:
                    print(f"Erro ao analisar a face: {e}")
                    emotion = "N/A"
                
                    # Exibe a emoção acima do rosto
                cv2.putText(frame, emotion, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


         # Desenha os landmarks da pose se houver
        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Exibe a quantidade de faces detectadas no frame
        cv2.putText(frame, f"Faces: {face_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Exibe o frame processado
        cv2.imshow("Análise Facial", frame)
        # Adiciona um retângulo ao redor da face
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()