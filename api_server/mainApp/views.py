from .serializers import TextSerializer
from .models import Text

from rest_framework import status, viewsets
from rest_framework.response import Response
from rest_framework.decorators import action

from tts_process import add_synth, is_set, make_tts

class TextViewSet(viewsets.ModelViewSet):
    queryset = Text.objects.all()
    serializer_class = TextSerializer

    # /api/texts/{str:email}/make_wav
    @action(detail=True, methods=['post'])
    def make_wav(self, request, email=None):
        # TODO: 추후 User 모델이 추가되면 email을 api param으로 받아 회원 조회 후 user와 text 엔티티를 연관관계 매핑할 예정
        uuid = request.data.get("uuid")

        if not is_set(email):
            add_synth(email)

        make_tts(email, uuid, request.data.get("text"))
        request.data["wav_url"] = f"{email}/{uuid}.wav'"

        serializer = TextSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)