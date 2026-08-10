FROM python:3.11-slim

# whisper ses işleme için ffmpeg'e ihtiyaç duyuyor
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# CPU sürümü torch kur (Mac'te GPU'ya container içinden erişilemez)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

# whisper paketini kendisini kur (bu repo openai/whisper kaynağı)
RUN pip install --no-cache-dir -e .

# server.py için gerekli ama requirements.txt'te olmayan paketler
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart

EXPOSE 8002

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]

