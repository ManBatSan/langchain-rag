FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl

COPY requirements.txt ./

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY vector_api.py ./

COPY index /data/index

RUN "uvicorn vector_api:app --port 8000 --host 0.0.0.0"