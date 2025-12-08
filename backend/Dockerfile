# ----------------------------
# Lightweight Base Image
# ----------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ----------------------------
# System dependencies (OCR + PDF)
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Working directory
# ----------------------------
WORKDIR /app

# ----------------------------
# Copy files
# ----------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# ----------------------------
# App port
# ----------------------------
EXPOSE 8000

# ----------------------------
# Start server
# ----------------------------
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
