# Authentix Backend & AI Inference Production Dockerfile
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies required for OpenCV, FFmpeg, and PyTorch C++ extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-generate matplotlib font cache to prevent startup block/hangs
ENV MPLCONFIGDIR=/tmp/matplotlib
RUN python -c "import matplotlib.font_manager"

# Copy backend and AI code into the container
COPY . .

# Create persistent upload and temp directories
RUN mkdir -p backend/uploads backend/temp backend/reports

# Expose default port
EXPOSE 8000

# Start command: Apply database migrations and launch Uvicorn
CMD ["sh", "-c", "alembic upgrade head && uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
