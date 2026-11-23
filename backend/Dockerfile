FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for compiling / runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first for compact image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy shared code and backend code
COPY shared/ /app/shared/
COPY backend/ /app/backend/

# Set environment variables
ENV PYTHONPATH=/app
ENV UPLOAD_DIR=/app/backend/uploads

EXPOSE 8000

# Entrypoint script: run migrations and start server
CMD ["sh", "-c", "cd /app/backend && alembic upgrade head && uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"]
