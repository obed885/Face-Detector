import os
import urllib.request

import cv2

VERT = (0, 255, 0)
BLEU = (255, 0, 0)
ROUGE = (0, 0, 255)


MODEL_PATH = "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

if not os.path.exists(MODEL_PATH):
    print("Téléchargement du modèle...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modèle téléchargé.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Impossible d'ouvrir la caméra.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, frame = cap.read()
if not ret:
    raise Exception("Impossible de lire la caméra.")
hauteur, largeur = frame.shape[:2]

detector = cv2.FaceDetectorYN.create(
    MODEL_PATH, "", (largeur, hauteur),
    score_threshold=0.85,
    nms_threshold=0.3,
    top_k=50
)

nombre_par_frame = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detector.setInputSize((frame.shape[1], frame.shape[0]))
    _, faces = detector.detect(frame)

    total = 0

    if faces is not None:
        total = len(faces)

        for i, f in enumerate(faces, start=1):
            x, y, w, h = f[:4].astype(int)
            score = float(f[14])


            cv2.rectangle(frame, (x, y), (x + w, y + h), VERT, 2)
            cv2.putText(frame, f"Face {i} ({score:.0%})", (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, VERT, 2)


            pts = f[4:14].reshape(5, 2).astype(int)
            for k, (px, py) in enumerate(pts):
                couleur = BLEU if k < 2 else ROUGE
                cv2.circle(frame, (int(px), int(py)), 3, couleur, -1)

    cv2.putText(frame, f"Total: {total}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, ROUGE, 2)

    nombre_par_frame.append(total)

    cv2.imshow("Détection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

with open("data.txt", "w") as fichier:
    maximum = max(nombre_par_frame) if nombre_par_frame else 0
    fichier.write(f"Nombre de personnes : {maximum}")