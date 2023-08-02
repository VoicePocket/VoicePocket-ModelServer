from module import train_vits, build_dataset, upload_model_file
from ..api_server.bucket_process import down_audio_from_bucket
from trainer import Trainer

DATA_PATH = "./data/"
OUTPUT_PATH = "./model_outputs/"

def initialize_trainer(run_name:str, project_name:str, output_path:str, data_path:str)->tuple(Trainer, str):
    trainer:Trainer = train_vits(run_name, project_name, output_path, data_path)
    return (trainer, trainer.output_path)

def train_model(uuid, email):
    '''
    JSON Format:
    {
        "uuid": "550e8400-e29b-41d4-a716-446655440000",
        "email": "test@gmail.com",
		"user_id": 3 // 학습할 TTS 음성 이름 id(확인 필요)
    }
    '''
    params = {"uuid":uuid, "email":email}
    # Download from google Cloud Storage
    zip_path = down_audio_from_bucket(params, DATA_PATH)
    # Make Dataset + Resampling Audio File!
    audio_path = build_dataset(zip_path)
    # Build Trainer for learning Model
    trainer, output_path = initialize_trainer(params["email"], params["uuid"], OUTPUT_PATH, audio_path)
    
    # Fitting
    trainer.fit()
    
    upload_model_file(output_path, params)