from django.urls import path
from . import views

app_name = "image_recognition"

urlpatterns = [
    path("", views.index, name="index"),
]