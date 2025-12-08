# ----------------------------
# Base Image
# ----------------------------
FROM python:3.11-slim

# ----------------------------
# Environment Config
# ----------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POPPLER_PATH="/usr/bin"

# ----------------------------
# System Dependencies
# ----------------------------
RUN apt-get update && apt-get install -y \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Working Directory
# ----------------------------
WORKDIR /app

# ----------------------------
# Install Python packages first (cache friendly)
# ----------------------------
COPY requirements.txt /app/
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ----------------------------
# Copy Rest of the App
# ----------------------------
COPY . /app

# ----------------------------
# Expose Port
# ----------------------------
EXPOSE 8000

# ----------------------------
# Run App
# ----------------------------
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
