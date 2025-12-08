# Use Python 3.10 (recommended for Torch + EasyOCR + Transformers)
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    tesseract-ocr \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Expose API port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
