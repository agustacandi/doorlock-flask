from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import os
import time

app = Flask(__name__)
app.config["DEBUG"] = True

dict = {}
face_detected = False


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


def gen_frames():
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(
        "./src/helper/haarcascade_frontalface_default.xml"
    )
    while True:
        success, frame = cap.read()

        if success:
            # Mengubah frame menjadi grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Mendeteksi wajah pada frame
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Menggambar persegi di sekitar wajah
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w + 10, y + h + 10), (0, 255, 0), 2)
                cv2.putText(frame, "Wajah Terdeteksi", (x, y - 10), 2, 0.7, (0, 255, 0))

            # Mendeteksi persegi disekitar wajah
            if len(faces) > 0:
                print("Wajah Terdeteksi")
            else:
                print("Wajah Tidak Terdeteksi")

            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        else:
            break


def generate_dataset(data):
    time.sleep(3)

    face_classifier = cv2.CascadeClassifier(
        "src/helper/haarcascade_frontalface_default.xml"
    )

    os.mkdir("static/dataset/" + data)

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor=1.3
        # Minimum neighbor = 5

        if faces == ():
            return None
        for x, y, w, h in faces:
            cropped_face = img[y: y + h, x: x + w]
        return cropped_face

    cap = cv2.VideoCapture(0)

    count_img = 0

    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

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

            if cv2.waitKey(1) == 13 or count_img == 100:
                break


@app.route("/")
def index():
    return render_template("camera.html")


@app.route("/dashboard")
def dashboard():
    data_wajah = len(os.listdir("static/dataset"))
    logs = [
        {"name": "Fathan", "job": "Teknisi", "time": "10:00", "status": "Masuk"},
        {"name": "Candi", "job": "Mahasiswa", "time": "11:00", "status": "Masuk"},
        {"name": "Fathan", "job": "Teknisi", "time": "11:00", "status": "Keluar"},
        {"name": "Candi", "job": "Mahasiswa", "time": "12:00", "status": "Keluar"},
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
    list_image = os.listdir(path)
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
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/take-frame/<string:data>")
def take_frame(data):
    return Response(
        generate_dataset(data), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run()
