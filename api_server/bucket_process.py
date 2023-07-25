import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="api_server/voicepocket-bucketKey.json"

from google.cloud import storage
from pathlib import Path
"""
bucket_name: 서비스 계정 생성한 bucket 이름 입력

source_blob_name: GCP에 저장되어 있는 파일 명
destination_file_name: 다운받을 파일을 저장할 경로 파일명까지("local/path/to/file")

source_file_name = GCP에 업로드할 파일 절대경로
destination_blob_name = 업로드할 파일을 GCP에 저장할 때의 이름
"""

def down_audio_from_bucket(params:dict, output_path:str):
    data_path = f"{Path(output_path).absolute().__str__}/{params['uuid']}"
    storage_client = storage.Client()
    bucket_name = 'voicepocket'
    source_blob_name = f'{params["email"]}.zip'
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    destination_file_name = f'{data_path}/{source_blob_name}'
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{params['email']}/{source_blob_name}")
    blob.download_to_filename(destination_file_name)
    return f"{data_path}/{source_blob_name}"

def down_model_from_bucket(email):
    storage_client = storage.Client()

    bucket_name = 'voice_pocket'
    source_blob_name = f'{email}_best_model.pth.tar'
    destination_file_name = f'./voice_model/{source_blob_name}'

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

def upload_wav_to_bucket(wav_path, email, uuid):
    storage_client = storage.Client()
    
    bucket_name = 'voice_pocket'
    source_file_name = wav_path
    destination_blob_name = f'{email}/{uuid}.wav'
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    
def upload_model_to_bucket(model_path, email):
    storage_client = storage.Client()
    
    bucket_name = 'voice_pocket'
    source_file_name = model_path
    destination_blob_name = f"{email}/model.pth"
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)