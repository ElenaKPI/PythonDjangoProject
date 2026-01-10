from django.shortcuts import render
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

# Загружаем модель один раз
model = MobileNetV2(weights="imagenet")

def index(request):
    context = {}

    if request.method == "POST" and request.FILES.get("video"):
        video_file = request.FILES["video"]

        # Открываем видео
        cap = cv2.VideoCapture(video_file.temporary_file_path())
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Преобразуем кадр
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frame_array = np.array(frame, dtype=np.float32)
            frame_array = np.expand_dims(frame_array, axis=0)
            frame_array = preprocess_input(frame_array)

            # Предсказание
            preds = model.predict(frame_array)
            decoded = decode_predictions(preds, top=1)[0][0]

            context["frame_shape"] = frame_array.shape
            context["frame_dtype"] = frame_array.dtype
            context["prediction_label"] = decoded[1]
            context["prediction_confidence"] = round(decoded[2] * 100, 2)

    return render(request, "video_recognition/index.html", context)