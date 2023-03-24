from django.utils import timezone
from django.db import models
from django.contrib.auth.models import AbstractBaseUser

from .managers import UserManager

class User(AbstractBaseUser):
    email = models.EmailField(verbose_name='email', max_length=255, unique=True)
    nickname = models.CharField(blank=True, max_length=10)

    is_active = models.BooleanField(default=True)
    is_admin = models.BooleanField(default=False)

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['nickname']

    class Meta:
        db_table = 'User'

    def __str__(self):
        return self.email

    def has_perm(self, perm, obj=None): # return True for having permission
        return True

    def has_module_perms(self, app_label):  # return True for accessing the app models
        return True

    @property
    def is_staff(self):
        return self.is_admin