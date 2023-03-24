from django.urls import path
from .views import *

text_detail = TextViewSet.as_view({"post": "make_wav"})

urlpatterns = [
    path('texts/<str:email>/make_wav', text_detail),
]