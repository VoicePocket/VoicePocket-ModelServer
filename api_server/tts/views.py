from .serializers import TextSerializer
from .models import Text
import time

from rest_framework import status, viewsets
from rest_framework.response import Response
from rest_framework.decorators import action

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from worker.wav_tasks import wav_process

class TextViewSet(viewsets.ModelViewSet):
    queryset = Text.objects.all()
    serializer_class = TextSerializer

    # /api/texts/{str:email}/make_wav
    @action(detail=True, methods=['post'])
    def make_wav(self, request, email=None):
        # TODO: 추후 User 모델이 추가되면 email을 api param으로 받아 회원 조회 후 user와 text 엔티티를 연관관계 매핑할 예정
        uuid = request.data.get("uuid")
        text = request.data.get("text")

        wav_worker = wav_process.delay(uuid, email, text)
        while (not wav_worker.ready()): # TODO: 이 부분 비동기식으로 바꾸기
            time.sleep(1)
        request.data["wav_url"] = wav_worker.result

        serializer = TextSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)