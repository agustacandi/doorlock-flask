from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import os
import time
import keras
import numpy as np
import io
import serial
import pandas as pd
import json

app = Flask(__name__)
app.config["DEBUG"] = True


# serial_com = serial.Serial("/dev/ttyACM1", 9600, timeout=1)
# serial_com = serial.Serial("COM20", 9600, timeout=1)

dict = {}
face_detected = False
face_detected_name = None
# model = keras.models.load_model("static/models/model_4_rgb.h5")
model = keras.models.load_model("static\models\model_160x160_rgb (2).h5")

labels = []
# with io.open("static/labels/labels.txt", "r", encoding="utf-8") as f:
#     label = f.readlines()
#     label = [line.rstrip("\n") for line in label]
#     print(label)
#     for i in label:
#         split = i.split("-")
#         labels.append(split[0])

with open("static/labels/labels.json", "r") as json_file:
    data = json.load(json_file)

for i in data:
    labels.append(i)
    print(i)

# def capture_frame(param):
#     time.sleep(3)
#
#     cap = cv2.VideoCapture(0)
#
#     face_cascade = cv2.CascadeClassifier(
#         "./src/helper/haarcascade_frontalface_default.xml"
#     )
#
#     index = 0
#     while True:
#         index += 1
#
#         success, frame = cap.read()
#
#         if success:
#             # Mengubah frame menjadi grayscale
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#             # Mendeteksi wajah pada frame
#             faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#
#             # Menggambar persegi di sekitar wajah
#             for x, y, w, h in faces:
#                 os.mkdir("dataset/" + param.nama + "-" + param.profesi)
#                 cv2.imwrite(
#                     "dataset/" + param.nama + "-" + param.profesi + "/" + index, gray
#                 )
#                 cv2.rectangle(frame, (x, y), (x + w + 10, y + h + 10), (0, 255, 0), 2)
#             ret, buffer = cv2.imencode(".jpg", frame)
#             frame = buffer.tobytes()
#             yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
#         if index <= 10:
#             break
#         else:
#             break


def gen_frames(rdrct):
    global face_detected, face_detected_name
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(
        "./src/helper/haarcascade_frontalface_default.xml"
    )
    labellss = ""
    validate = 0
    fail = 0
    check = []
    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if fail > 30:
            face_detected = True
            face_detected_name = labels[identity]
            print(face_detected_name)
            validate = 0

        if success:
            # Mengubah frame menjadi grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Mendeteksi wajah pada frame
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
            faces = faces[0:1]
            # Mendeteksi persegi disekitar wajah
            if len(faces) > 0:
                for x, y, w, h in faces:
                    crop_face = frame[y : y + h, x : x + w]
                    crop_face = cv2.cvtColor(crop_face, cv2.COLOR_BGR2RGB)
                    # crop_face = cv2.cvtColor(crop_face, cv2.COLOR_BGR2GRAY)
                    # crop_face = crop_face / 255.0
                    crop_face = cv2.resize(crop_face, (160, 160))
                    crop_face = crop_face.reshape((1, 160, 160, 3))
                    cv2.rectangle(
                        frame, (x, y), (x + w + 10, y + h + 10), (0, 255, 0), 2
                    )
                    print(crop_face.shape)
                    prediction = model.predict(crop_face)
                    identity = prediction.argmax()
                    pred = np.max(prediction)
                    for i in prediction:
                        for j in i:
                            print(j * 100)
                    if pred > 0.9:
                        print(validate)
                        # if validate > 20:
                        # if labellss == "Unknown":
                        #     face_detected = True
                        #     face_detected_name = labels[identity]
                        #     print(face_detected_name)
                        #     validate = 0

                        if len(check) == 15:
                            print(check)
                            if all(elem == check[0] for elem in check):
                                if labellss == "Unknown":
                                    face_detected = True
                                    face_detected_name = "Unknown"
                                    # print(face_detected_name)
                                    # print(check)
                                    validate = 0
                                else:
                                    face_detected = True
                                    face_detected_name = labels[identity]
                                    print(face_detected_name)
                                    validate = 0
                            else:
                                face_detected = True
                                face_detected_name = str("Unknown")
                                print(face_detected_name)
                                validate = 0
                            # elif labellss == labels[identity]:
                            #     face_detected = True
                            #     face_detected_name = labels[identity]
                            #     print(face_detected_name)
                            #     validate = 0
                            # print("MASUK")
                            # with app.app_context():
                            #     url = url_for('greet', name='John')
                            #     return url
                        cv2.putText(
                            frame,
                            # f"{labels[identity]} {pred * 100:.2f}%",
                            f"Scanning...",
                            (25, 25),
                            2,
                            0.7,
                            (0, 255, 0),
                        )
                        labellss = labels[identity]
                        validate += 1
                        check.append(labels[identity])

                    else:
                        cv2.putText(
                            frame,
                            "Wajah Tidak Dikenali",
                            (25, 25),
                            2,
                            0.7,
                            (0, 0, 255),
                        )
                        labellss = ""
                        validate = 0
                        fail += 1
            # else:
            # print("Wajah Tidak Terdeteksi")

            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        else:
            break
        time.sleep(0.3)


import shutil


def generate_dataset(data):
    time.sleep(3)

    face_classifier = cv2.CascadeClassifier(
        "src/helper/haarcascade_frontalface_default.xml"
    )

    if os.path.exists("static/dataset/" + data):
        shutil.rmtree("static/dataset/" + data)
        os.mkdir("static/dataset/" + data)
    else:
        os.mkdir("static/dataset/" + data)

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor=1.3
        # Minimum neighbor = 5

        if not len(faces) > 0:
            return None
        for x, y, w, h in faces:
            cropped_face = img[y : y + h, x : x + w]
        return cropped_face

    cap = cv2.VideoCapture(0)

    count_img = 0

    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            face = cv2.resize(face_cropped(img), (224, 224))
            # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            file_name_path = "static/dataset/" + data + "/" + str(count_img) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(
                face,
                str(count_img),
                (50, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            frame = cv2.imencode(".jpg", face)[1].tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

            if cv2.waitKey(1) == 13 or count_img == 50:
                break
        time.sleep(0.3)


@app.route("/")
def index():
    return render_template("camera.html")


@app.route("/is-face-detected")
def is_face_detected():
    global face_detected, face_detected_name
    # print(is_face_detected, face_detected_name)
    if face_detected and face_detected_name is not None:
        # return jsonify({"status": True, "name": face_detected_name}), 200
        response = jsonify({"status": True, "name": face_detected_name}), 200
        face_detected = False
        face_detected_name = None
        return response
    else:
        # print("Tidak ada wajah terdeteksi")
        # return False
        # is_face_detected = False
        # face_detected_name = None
        return jsonify({"status": False, "name": str(face_detected_name)}), 200
    # return redirect(url_for("index"))


@app.route("/dashboard")
def dashboard():
    data_wajah = len(os.listdir("static/dataset"))
    logs = [
        # {"name": "Fathan", "job": "Teknisi", "time": "10:00", "status": "Masuk"},
        # {"name": "Candi", "job": "Mahasiswa", "time": "11:00", "status": "Masuk"},
        # {"name": "Fara", "job": "Mahasiswa", "time": "12:00", "status": "Masuk"},
        # {"name": "Fathan", "job": "Teknisi", "time": "11:00", "status": "Keluar"},
        # {"name": "Candi", "job": "Mahasiswa", "time": "12:00", "status": "Keluar"},
    ]
    return render_template("dashboard.html", logs=logs, data_wajah=data_wajah)


@app.route("/face-data")
def face_data():
    people = os.listdir("static/dataset")

    return render_template("face-data.html", people=people)


@app.route("/face-data/delete/<string:data>", methods=["DELETE"])
def delete_face_data(data):
    path = "static/dataset/" + data

    exists = os.path.exists(path)

    if not exists:
        return jsonify({"deskripsi": "Data tidak ditemukan"}), 404

    # Hapus semua file di directory
    for file in os.listdir(path):
        os.remove(os.path.join(path, file))

    # Hapus subdirektori
    os.rmdir(path)

    return jsonify({"deskripsi": "Data berhasil dihapus"}), 200


@app.route("/face-data/add", methods=["GET", "POST"])
def add_face_data():
    if request.method == "POST":
        nama = request.form["nama"]
        profesi = request.form["profesi"]
        param_value = nama + "-" + profesi
        return redirect(url_for("take_photo", data=param_value))
    else:
        return render_template("add-face.html")


@app.route("/face-data/detail/<string:data>", methods=["GET", "POST"])
def detail_face_data(data):
    path = "static/dataset/" + data
    if os.path.exists(path):
        list_image = os.listdir(path)
    else:
        return redirect(url_for("face_data"))
    return render_template("detail-face.html", data=data, list_image=list_image)


@app.route("/face-data/add/<string:data>")
def take_photo(data):
    return render_template("take-photo.html", data=data)


@app.route("/log-activity")
def log_activity():
    from src.lib.data import people

    return render_template("log-activity.html", people=people)


@app.route("/video-feed")
def video_feed():
    return Response(
        gen_frames(redirect), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/take-frame/<string:data>")
def take_frame(data):
    return Response(
        generate_dataset(data), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/success/<string:name>")
def greet(name):
    # serial_com.write(str("on").encode("utf-8"))
    from datetime import datetime
    import pytz

    def tambah_entri_log(log, nama, zona_waktu):
        # Mendapatkan waktu saat ini dengan zona waktu yang diinginkan
        waktu_sekarang = datetime.now(pytz.timezone(zona_waktu)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Menentukan ID otomatis berdasarkan jumlah entri yang sudah ada
        identitas = len(log) + 1

        # Membagi nama dan profesi
        name = nama.split("-")[0]
        job = nama.split("-")[1]

        # Membuat entri baru
        entri_baru = {
            "id": identitas,
            "name": name,
            "job": job,
            "created_at": waktu_sekarang,
        }

        # Menambahkan entri baru ke dalam log
        log.append(entri_baru)

        # Mengembalikan log yang diperbarui
        return log

    # Membaca log yang sudah ada (jika ada)
    path_log = "static/log/log.json"
    try:
        with open(path_log, "r") as file:
            log_data = json.load(file)
    except FileNotFoundError:
        log_data = []

    # Menambahkan entri pertama ke dalam log dengan zona waktu WIB
    log_data = tambah_entri_log(log_data, str(name), "Asia/Jakarta")

    # Menyimpan log ke dalam file JSON
    with open(path_log, "w") as file:
        json.dump(log_data, file, indent=2)

    return render_template("success.html", name=name)


@app.route("/failed/")
def fail():
    return render_template("fail.html")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
