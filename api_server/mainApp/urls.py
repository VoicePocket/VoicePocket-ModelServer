from django.urls import path, include
from rest_framework import routers
from .views import *

text_detail = TextViewSet.as_view({"post": "make_wav"})

urlpatterns = [
    path('texts/<str:email>/make_wav', text_detail),
]