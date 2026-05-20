# Offizielles PyTorch-Image mit CUDA 12.1 und Ubuntu-Basis
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Arbeitsverzeichnis IM Container festlegen
WORKDIR /app

# System-Bibliotheken für Audio-Verarbeitung (sndfile für torchaudio, ffmpeg für Videos)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Anforderungen kopieren und installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .