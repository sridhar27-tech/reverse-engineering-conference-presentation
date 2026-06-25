# NeuroGuard-XAI: Advanced Suicide Risk Prediction Framework
# Multi-stage Dockerfile for production deployment with MLOps, XAI, and inference

FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional ML dependencies if needed
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir onnxruntime

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p models data logs deployment

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=http://localhost:5000 \
    ZENML_REPO_PATH=/app \
    MODEL_NAME=mental-roberta-cbm

# Expose ports for services
EXPOSE 8501  # Streamlit dashboard
EXPOSE 8080  # FastAPI inference
EXPOSE 5000  # MLflow

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Stage 2: Development / Experimentation
FROM base AS dev
CMD ["bash"]

# Stage 3: MLOps Pipeline Runner
FROM base AS pipeline
CMD ["python", "run_pipeline.py"]

# Stage 4: FastAPI Inference Server
FROM base AS api
COPY deployment/api/main.py /app/deployment/api/
WORKDIR /app/deployment/api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

# Stage 5: Streamlit XAI Dashboard
FROM base AS dashboard
CMD ["streamlit", "run", "src/deployment/xai_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Default: Run inference API
FROM base AS production
CMD ["uvicorn", "deployment.api.main:app", "--host", "0.0.0.0", "--port", "8080"]