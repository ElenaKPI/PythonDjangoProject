from django.shortcuts import render
import tensorflow as tf
import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "emotion_model.h5"),
    compile=False
)

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict_emotion(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, None

    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (64, 64))
    face = face / 255.0
    face = np.reshape(face, (1, 64, 64, 1))

    preds = model.predict(face)
    idx = np.argmax(preds)

    return EMOTIONS[idx], float(preds[0][idx] * 100)


def index(request):
    context = {}

    if request.method == "POST" and request.FILES.get("photo"):
        photo = request.FILES["photo"]

        save_path = os.path.join(BASE_DIR, "media", photo.name)
        with open(save_path, "wb+") as f:
            for chunk in photo.chunks():
                f.write(chunk)

        emotion, confidence = predict_emotion(save_path)

        context["emotion"] = emotion
        context["confidence"] = confidence
        context["image"] = photo.name

    return render(request, "emotion_recognition/index.html", context)