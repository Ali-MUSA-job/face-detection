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