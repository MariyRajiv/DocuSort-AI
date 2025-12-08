# ----------------------------
# Base image
# ----------------------------
FROM python:3.11-slim

# ----------------------------
# Environment variables
# ----------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PATH="/root/.local/bin:$PATH"

# ----------------------------
# System dependencies
# ----------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
        poppler-utils \
        tesseract-ocr \
        wget \
        unzip \
        git \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Set working directory
# ----------------------------
WORKDIR /app

# ----------------------------
# Copy project files
# ----------------------------
COPY . /app

# ----------------------------
# Install Python dependencies
# ----------------------------
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ----------------------------
# Expose port
# ----------------------------
EXPOSE 8000

# ----------------------------
# Run the app
# ----------------------------
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
