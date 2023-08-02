import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from .celery_app import model_worker
from ..training_model import train_model
@model_worker.task(name="model_process")
def model_process(uuid, email):
    train_model(uuid, email)