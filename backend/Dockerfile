# Use Python 3.9 (EasyOCR + Torch compatible)
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    tesseract-ocr \
    && apt-get clean

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
