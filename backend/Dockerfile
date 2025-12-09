# ----------------------------
# Base image
# ----------------------------
FROM python:3.11-slim

# ----------------------------
# Environment settings
# ----------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ----------------------------
# Install system dependencies
# ----------------------------
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ----------------------------
# EasyOCR Model Directory
# ----------------------------
RUN mkdir -p /root/.EasyOCR/model

# ----------------------------
# Download EasyOCR models (smallest stable versions)
# ----------------------------

# CRAFT Text Detection Model
RUN wget -O craft.zip https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip \
    && unzip craft.zip -d /root/.EasyOCR/model \
    && rm craft.zip

# English Text Recognition Model
RUN wget -O eng.zip https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip \
    && unzip eng.zip -d /root/.EasyOCR/model \
    && rm eng.zip

# ----------------------------
# Working directory
# ----------------------------
WORKDIR /app

# ----------------------------
# Install Python dependencies
# ----------------------------
COPY requirements.txt /app/
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ----------------------------
# Copy backend code
# ----------------------------
COPY . /app

# ----------------------------
# Expose port
# ----------------------------
EXPOSE 8000

# ----------------------------
# Start the application
# ----------------------------
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
