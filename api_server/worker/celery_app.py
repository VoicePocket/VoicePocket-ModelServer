from celery import Celery

wav_worker = Celery(
    'wav_worker',
    broker='amqp://pocket:pocket!@rabbit/voice_pocket_host',# '{transport}://{userid}:{password}@{hostname}:{port}/{virtual_host}'
    backend='rpc://',
    include=['api_server.worker.wav_tasks'],
)