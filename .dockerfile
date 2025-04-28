# Start from a Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app/uploads /app/vector_db /app/vector_db/faiss_index

# Copy requirements file first for better caching
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python files
COPY *.py /app/

# Ensure directories are available (additional ones may be created at runtime)
RUN mkdir -p /app/__pycache__

# Set environment variables (customize as needed)
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_API_KEY='AIzaSyCopFldRoxw7xAL5fe3Rc2-RuvMMgIqntk'
ENV TESSERACT_CMD='/usr/bin/tesseract'

# Expose the port the app runs on
EXPOSE 10000

# Command to run the application
#CMD ["python", "rag_backend.py"]

# Use Gunicorn for production
CMD ["gunicorn", "app:rag_backend", "--bind", "0.0.0.0:10000"]