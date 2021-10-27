FROM tensorflow/tensorflow:2.4.1-gpu

WORKDIR /usr/src/app 
COPY requirements.txt ./ 
RUN pip install --no-cache-dir -r requirements.txt 

RUN set -eux; \
 	apt-get update; \
    apt-get install -y --no-install-recommends \
            libgl1-mesa-glx \
            libglib2.0-0\
            wget \
            nano \
    ; \
    rm -rf /var/lib/apt/lists/* 

COPY ./ /usr/src/app

CMD ["python3","/usr/src/app/main.py"]

## docker run  --gpus=all -d --restart=always  --name face-detect-gpu-container   -e RABBITMQ_HOST=192.168.10.94     -e REDIS_HOST=192.168.10.94    -e REDIS_PORT=6379   -e RABBIT_OUTPUT_QUEUE=face_queue  -e  RABBIT_INPUT_QUEUE=frame_queue -e LOG_LEVEL=INFO face-detect-gpu-image