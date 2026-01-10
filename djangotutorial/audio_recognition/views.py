import matplotlib
matplotlib.use("Agg")

from django.shortcuts import render
import tensorflow as tf
import numpy as np
import librosa
import csv
import os
import matplotlib.pyplot as plt
import uuid
from django.conf import settings


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAMNET_PATH = os.path.join(BASE_DIR, "yamnet-tensorflow2-yamnet-v1")

MODEL = tf.saved_model.load(YAMNET_PATH)

CLASS_MAP = {}

with open(os.path.join(YAMNET_PATH, "assets", "yamnet_class_map.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        CLASS_MAP[int(row["index"])] = row["display_name"]


def analyze_audio(file):
    # Загружаем аудио (в 16kHz, моно — так требует YAMNet)
    waveform, sr = librosa.load(file, sr=16000, mono=True)
    spectrogram_file = save_spectrogram(waveform, sr)

    # Преобразуем в Tensor
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)

    # Прогоняем через YAMNet
    scores, embeddings, spectrogram = MODEL(waveform)

    # Усредняем вероятности по времени
    mean_scores = tf.reduce_mean(scores, axis=0)

    # Берем класс с максимальной вероятностью
    top_index = tf.argmax(mean_scores).numpy()
    confidence = mean_scores[top_index].numpy()

    return CLASS_MAP[top_index], float(confidence * 100), spectrogram_file

def index(request):
    context = {}

    if request.method == "POST" and request.FILES.get("audio"):
        audio_file = request.FILES["audio"]

        label, confidence, spec = analyze_audio(audio_file)
        context["spectrogram"] = spec

        context["prediction_label"] = label
        context["prediction_confidence"] = f"{confidence:.2f}%"

    return render(request, "audio_recognition/index.html", context)

def save_spectrogram(waveform, sr):
    filename = f"{uuid.uuid4().hex}.png"
    path = os.path.join(settings.MEDIA_ROOT, "spectrograms", filename)

    plt.figure(figsize=(4, 4))
    plt.specgram(waveform, Fs=sr)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return filename