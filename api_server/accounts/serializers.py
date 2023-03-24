from rest_framework import serializers
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .models import User


class JWTSignUpSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['email', 'password', 'nickname']

    def save(self, request):
        user = super(JWTSignUpSerializer, self).save()

        user.email = self.validated_data['email']
        user.nickname = self.validated_data['nickname']
        
        user.set_password(self.validated_data['password'])
        user.save()

        return user
    
    def validate(self, data):
        email = data.get('email', None)

        if User.objects.filter(email=email).exists():
            raise serializers.ValidationError("user already exists")
        
        return data


# jwt token 결과 커스텀 
class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    # response 커스텀 
    default_error_messages = {
        'no_active_account': {'message':'email or password is incorrect!',
                              'success': False,
                              'status' : 401}
    }

    # 유효성 검사
    def validate(self, attrs):
        data = super().validate(attrs)
        refresh = self.get_token(self.user)
        
        # response에 추가하고 싶은 key값들 추가
        data['email'] = self.user.email
        data['refresh'] = str(refresh)
        data['access'] = str(refresh.access_token)
        data['success'] = True
        
        return data