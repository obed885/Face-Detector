import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)


nombre_par_frame = []

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    for i, (x, y, w, h) in enumerate(faces, start=1):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f'Sujet {i}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    
    total = len(faces)
    cv2.putText(frame, f'Total: {total}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


    nombre_par_frame.append(total)

    cv2.imshow('Détection sujets', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

with open("data.txt", "w") as f:
    f.write(f"Nombre de personnes : {max(nombre_par_frame)}")