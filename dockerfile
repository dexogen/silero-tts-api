FROM python:3.10-slim

WORKDIR /usr/app

RUN apt-get update --fix-missing \
    && apt-get install -y --no-install-recommends sox libsox-fmt-mp3 opus-tools \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

COPY requirements_docker.txt .
ARG TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_PACKAGE=torch==2.3.1+cpu
RUN pip3 install --no-cache-dir --index-url ${TORCH_CPU_INDEX_URL} ${TORCH_PACKAGE} \
    && pip3 install --no-cache-dir -r requirements_docker.txt

COPY app ./app
COPY main.py ./main.py

EXPOSE 9898

CMD [ "python3", "-u", "./main.py" ]
