# ----------------------------
# Base image
# ----------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ----------------------------
# System dependencies
# ----------------------------
RUN apt-get update && apt-get install -y \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    unzip \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Pre-download EasyOCR Models
# ----------------------------
RUN mkdir -p /root/.EasyOCR/model

# CRAFT Detection Model
RUN wget -O craft.zip https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip \
    && unzip craft.zip -d /root/.EasyOCR/model \
    && rm craft.zip

# English Recognition Model
RUN wget -O english.zip https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip \
    && unzip english.zip -d /root/.EasyOCR/model \
    && rm english.zip

# ----------------------------
# Working directory
# ----------------------------
WORKDIR /app

# Copy backend
COPY . /app

# Install dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# ----------------------------
# Expose and run FastAPI
# ----------------------------
EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
