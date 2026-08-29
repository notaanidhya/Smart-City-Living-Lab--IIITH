# Multi-Stage Build for Full-Stack AI Quality Assessment System
# Stage 1: Build React 18 Frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Production Python Backend with PyTorch CPU
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU and backend dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and ML models
COPY backend /app/backend
COPY ml /app/ml

# Copy built frontend static files
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Set permissions for Hugging Face Spaces (runs as non-root user 1000)
RUN useradd -m -u 1000 user
RUN mkdir -p /app/backend/uploads/images /app/backend/uploads/heatmaps && \
    chown -R user:user /app
USER user

ENV PYTHONPATH=/app
ENV FRONTEND_DIST=/app/frontend/dist
ENV UPLOAD_DIR=/app/backend/uploads
ENV MODEL_DIR=/app/ml/models
ENV PORT=7860

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
