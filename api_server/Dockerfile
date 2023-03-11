FROM python:3.10
USER root 

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /voice_pocket
COPY requirements.txt /api_server/

RUN apt-get update && apt-get install -y python3-pip libffi-dev python3-dev && apt-get install -y default-libmysqlclient-dev && apt-get install -y ffmpeg

RUN pip install --upgrade pip
RUN pip install -r /api_server/requirements.txt

# wait-for 적용
ADD https://github.com/ufoscout/docker-compose-wait/releases/download/2.9.0/wait /wait
RUN chmod +x /wait
CMD /wait && /wait-for-mysql.sh

WORKDIR /voice_pocket