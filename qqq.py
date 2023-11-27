import keras
import cv2
import numpy as np
import time
import pandas as pd

# Load model
model = keras.models.load_model("static/models/model_cnn1_251123_5_gray.h5")
face_detector = cv2.CascadeClassifier("src/helper/haarcascade_frontalface_default.xml")
labels = ["Fara", "Fathan", "Indra", "Krisna", "Prof"]

detector = cv2.dnn.readNetFromCaffe(
    "src/helper/deploy.prototxt", "src/helper/res10_300x300_ssd_iter_140000.caffemodel"
)

# Membuat video capture
cap = cv2.VideoCapture(0)

while True:
    # membaca frame dari video
    ret, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    # Mendeteksi wajah
    faces = face_detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
    faces = faces[0:1]
    if faces is ():
        cv2.putText(
            frame,
            "Wajah Tidak Terdeteksi",
            (20, 40),
            cv2.FONT_HERSHEY_PLAIN,
            1.3,
            (0, 0, 255),
            2,
        )
    else:
        for x, y, w, h in faces:
            face = frame[y : y + h, x : x + w]
            face = cv2.resize(face, (224, 224))
            # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = face.reshape((1, 224, 224, 1))

            # Membuat kotak dan label untuk wajah
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            prediction = model.predict(face)
            identity = prediction.argmax()
            pred = np.max(prediction)

            # Menentukan apakah wajah dikenal atau tidak
            if pred > 0.95:
                label = labels[identity]
            else:
                label = "Tidak Diketahui"

            cv2.putText(
                frame,
                f"{label} {pred * 100:.2f}%",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    # Menampilkan frame
    cv2.imshow("Frame", frame)

    # Delay
    # time.sleep(0.1)

    # Exit
    if cv2.waitKey(1) == ord("q"):
        break

# Release camera
cap.release()

# Close all windows
cv2.destroyAllWindows()
