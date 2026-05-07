from flask import Flask, render_template, request, redirect, session
import cv2, os, numpy as np

app = Flask(__name__)
app.secret_key = "key"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

KNOWN = "static/known.jpg"

@app.route("/")
def lock():
    return render_template("lock.html")

@app.route("/set_face")
def set_face():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Camera error"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No face"

    x, y, w, h = faces[0]
    cv2.imwrite(KNOWN, gray[y:y+h, x:x+w])

    return "Face saved"

@app.route("/verify_face")
def verify_face():
    if not os.path.exists(KNOWN):
        return redirect("/set_face")

    known = cv2.imread(KNOWN, 0)
    known = cv2.resize(known, (100, 100))

    cap = cv2.VideoCapture(0)
    ok = False

    for _ in range(20):
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            face = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
            if np.mean((face - known) ** 2) < 2000:
                ok = True
                break

        if ok:
            break

    cap.release()
    return redirect("/home") if ok else "Denied"


@app.route("/home", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        f = request.files["image"]
        path = "static/img.jpg"
        f.save(path)

        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            cv2.circle(img, (x+w//2, y+h//2), w//2, (0,255,0), 2)

        cv2.imwrite("static/result.jpg", img)

        return render_template("home.html",
                               count=len(faces),
                               img="result.jpg")

    return render_template("home.html", count=None, img=None)


app.run(debug=True)