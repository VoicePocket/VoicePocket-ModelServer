from django.db import models
from django.utils import timezone

class Text(models.Model):
    id = models.AutoField(primary_key=True)  # pk
    # TODO: User 모델 만들어지면 외래키로 연관관계 설정하기
    # user_id = models.ForeignKey(User, on_delete=models.CASCADE, db_column='user_id')
    text = models.TextField()
    wav_url = models.CharField(max_length=100, null = False, default='')
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'Text'