from celery import Celery

model_worker = Celery(
    'model_worker',
    broker="amqp://pocket:pocket!@rabbit/voice_pocket_host",
    backend="rpc://",
    include=['model_server.worker.model_tasks'],
)