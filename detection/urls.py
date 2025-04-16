from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_video2, name='upload_video'),
]