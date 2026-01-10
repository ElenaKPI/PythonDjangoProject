from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from django.shortcuts import render
from PIL import Image
import numpy as np

model = MobileNetV2(weights="imagenet")

def index(request):
    context = {}

    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]
        img = Image.open(image_file)
        img = img.convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)
        decoded = decode_predictions(predictions, top=1)[0][0]

        context["prediction_label"] = decoded[1]
        context["prediction_confidence"] = round(float(decoded[2]) * 100, 2)

        context["image_shape"] = img_array.shape
        context["image_dtype"] = img_array.dtype

    return render(request, "image_recognition/index.html", context)

def image_home(request):
    context = {}

    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]

        # 1. Открываем изображение
        img = Image.open(image_file)

        # 2. Приводим к RGB (важно для ML)
        img = img.convert("RGB")

        # 3. Меняем размер (стандарт для нейросетей)
        img = img.resize((224, 224))

        # 4. Преобразуем в numpy-массив
        img_array = np.array(img)

        # 5. Кладём информацию в контекст
        context["image_shape"] = img_array.shape
        context["image_dtype"] = img_array.dtype

    return render(request, "image_recognition/index.html", context)