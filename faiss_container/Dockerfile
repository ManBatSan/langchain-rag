FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl

COPY requirements.txt ./

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY faiss_server.py ./

COPY index /data/index

EXPOSE 8000

CMD ["uvicorn", "faiss_server:app", "--port", "8000", "--host", "0.0.0.0"]
