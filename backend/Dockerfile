# ----------------------------
# Base image
# ----------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ----------------------------
# Install system dependencies
# ----------------------------
RUN apt-get update && apt-get install -y \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Set working directory
# ----------------------------
WORKDIR /app

# ----------------------------
# Copy requirements & install
# ----------------------------
COPY requirements.txt /app/

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# -------------------------------------------------------------
# PRE-DOWNLOAD EASYOCR MODELS TO AVOID RUNTIME DOWNLOAD (Render Fix)
# -------------------------------------------------------------
RUN mkdir -p /root/.EasyOCR/model

# Detection model (CRAFT)
RUN wget -O /root/.EasyOCR/model/craft_mlt_25k.pth \
    https://github.com/JaidedAI/EasyOCR/releases/download/v1.6/craft_mlt_25k.pth

# Recognition model (English)
RUN wget -O /root/.EasyOCR/model/english_g2.pth \
    https://github.com/JaidedAI/EasyOCR/releases/download/v1.6/english_g2.pth

# Latin mapping
RUN wget -O /root/.EasyOCR/model/latin_g2.pth \
    https://github.com/JaidedAI/EasyOCR/releases/download/v1.6/latin_g2.pth

# ----------------------------
# Copy backend code
# ----------------------------
COPY . /app

# ----------------------------
# Expose port
# ----------------------------
EXPOSE 8000

# ----------------------------
# Run the app
# ----------------------------
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
