from flask import Flask, request
from module import *
from trainer import Trainer
from requests import post
import json

DATA_PATH = "./data/"
OUTPUT_PATH = "./model_outputs/"
app = Flask()

def initialize_trainer(run_name:str, project_name:str, output_path:str, data_path:str)->tuple(Trainer, str):
    trainer:Trainer = train_vits(run_name, project_name, output_path, data_path)
    return (trainer, trainer.output_path)

@app.route("/", method=['POST'])
def train_model():
    '''
    JSON Format:
    {
        "uuid": "550e8400-e29b-41d4-a716-446655440000",
        "email": "test@gmail.com",
		"user_id": 3 // 학습할 TTS 음성 이름 id(확인 필요)
    }
    '''
    params = json.loads(request.get_data(), encoding='utf-8')
    # Download from google Cloud Storage
    audio_path = down_audio_from_bucket(params, DATA_PATH)
    # Build Trainer for learning Model
    trainer, output_path = initialize_trainer(params["email"], params["uuid"], OUTPUT_PATH, audio_path)
    # Ready response
    response = post(f"ENTER THE RESPONSE ADDRESS", data=json.dumps({"uuid": params["uuid"], "output_path": output_path, "status":"In Progress"}))
    if response.status_code != 200:
        del trainer, output_path
        return None
    trainer.fit()