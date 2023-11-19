import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")


def gen_frames():
    while True:
        success, frame = cap.read()

        # Mengubah frame menjadi grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Mendeteksi wajah pada frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Menggambar persegi di sekitar wajah
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w + 10, y + h + 10), (255, 0, 0), 2)

        if not success:
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
