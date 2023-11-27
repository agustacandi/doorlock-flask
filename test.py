import keras
import cv2
import numpy as np
import time
import pandas as pd

# Load model
# Membuat face detector dari file haarcascade_frontalface_default.xml
# model = keras.models.load_model("static/models/model_vgg16_241123_5_rgb.h5")
model = keras.models.load_model("static/models/model_cnn1_251123_5_gray.h5")
face_detector = cv2.CascadeClassifier("src/helper/haarcascade_frontalface_default.xml")
# labels = ["Fara", "Fathan", "Indra", "Putri"]
labels = ["Fara", "Fathan", "Indra", "Krisna", "Prof"]

detector = cv2.dnn.readNetFromCaffe(
    "src/helper/deploy.prototxt", "src/helper/res10_300x300_ssd_iter_140000.caffemodel"
)

# Membuat video capture
cap = cv2.VideoCapture(0)
# original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # # Hitung dimensi persegi yang diinginkan
# min_side = min(original_width, original_height)
# desired_width = min_side
# desired_height = min_side

# # # Hitung ROI (Region of Interest) untuk dipotong dari tengah frame
# start_x = (original_width - min_side) // 2
# start_y = (original_height - min_side) // 2

while True:
    # membaca frame dari video
    ret, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    # crop_frame = cv2.resize(frame, (224, 224))
    # crop_frame = crop_frame.reshape((1, 224, 224, 3))
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
            # for i in prediction:
            #     for j in i:
            #         print(j * 100)
            print(pred)
            time.sleep(1)

            if pred > 0.95:
                cv2.putText(
                    frame,
                    f"{labels[identity]} {pred * 100:.2f}%",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Tidak Diketahui",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

    # imgblob = cv2.dnn.blobFromImage(frame)
    # detector.setInput(imgblob)
    # detections = detector.forward()
    # detections = detections[0][0]
    # df = pd.DataFrame(
    #   detections,
    #   columns=[
    #     "img_id",
    #     "is_face",
    #     "confidence",
    #     "left",
    #     "top",
    #     "right",
    #     "bottom",
    #   ],
    # )
    # df = df[df["is_face"] == 1]
    # df = df[df["confidence"] > 0.9][0:1]

    # if not df.empty:
    #   for i, instance in df.iterrows():
    #     x = int(instance["left"] * frame.shape[1])
    #     y = int(instance["top"] * frame.shape[0])
    #     w = int(instance["right"] * frame.shape[1])
    #     h = int(instance["bottom"] * frame.shape[0])
    # left = int(instance["left"] * frame.shape[1])
    # top = int(instance["top"] * frame.shape[0])
    # right = int(instance["right"] * frame.shape[1])
    # bottom = int(instance["bottom"] * frame.shape[0])
    # width = w - x
    # height = h - y

    # if width > height:
    #     diff = width - height
    #     y -= diff // 2
    #     h += diff - (diff // 2)
    # else:
    #     diff = height - width
    #     x -= diff // 2
    #     w += diff - (diff // 2)

    # face_crop = frame[y:h, x:w]

    # Membuat kotak dan label untuk wajah
    # (x, y), (x + w, y + h)
    # print(face_crop.shape)
    # if face_crop.shape[0] != face_crop.shape[1]:
    #   cv2.putText(
    #     frame,
    #     "Harap Letakkan Wajah di Tengah Frame",
    #     (10, 10),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.5,
    #     (0, 0, 255),
    #     2,
    #   )
    # elif face_crop.shape[0] < 160 or face_crop.shape[1] < 160:
    #   cv2.putText(
    #     frame,
    #     "Harap Dekatkan Wajah ke Kamera",
    #     (10, 10),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.5,
    #     (0, 0, 255),
    #     2,
    #   )
    #     else:
    #       cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
    #       cv2.putText(
    #           frame,
    #           "Ada Wajah",
    #           (x, y - 10),
    #           cv2.FONT_HERSHEY_SIMPLEX,
    #           0.5,
    #           (0, 255, 0),
    #           2,
    #       )
    # else:
    #   cv2.putText(
    #       frame,
    #       "Tidak Ada Wajah",
    #       (10, 10),
    #       cv2.FONT_HERSHEY_SIMPLEX,
    #       0.5,
    #       (0, 0, 255),
    #       2,
    #   )

    # Looping untuk setiap wajah yang terdeteksi
    # for x, y, w, h in faces:
    #     face = frame[y : y + h, x : x + w]
    #     face = cv2.resize(face, (224, 224))
    #     face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    #     face = face.reshape((1, 224, 224, 3))
    #     prediction = model.predict(face)
    #     identity = prediction.argmax()
    #     pred = np.max(prediction)
    #     for i in prediction:
    #       for j in i:
    #         print(j * 100)

    #     if pred > 0.9:
    # Membuat kotak dan label untuk wajah
    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.putText(
    #     frame,
    #     labels[identity],
    #     (x, y - 10),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.5,
    #     (0, 255, 0),
    #     2,
    # )
    #   cv2.putText(
    #     frame,
    #     labels[identity],
    #     (10, 10),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (0, 255, 0),
    #     2,
    #   )
    # else:
    #   cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #   cv2.putText(
    #       frame,
    #       "Tidak Diketahui",
    #       (10, 10),
    #       cv2.FONT_HERSHEY_SIMPLEX,
    #       0.5,
    #       (0, 0, 255),
    #       2,
    #   )

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
