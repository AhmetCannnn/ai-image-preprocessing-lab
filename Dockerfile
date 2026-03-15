# Görüntü İşleme Stüdyosu - Coolify / Hetzner deploy
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py image_processor.py ./

EXPOSE 7860

CMD ["python", "app.py"]
