# Authentix FastAPI Backend Production Dockerfile (Render Target)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV AUTHENTIX_DEVICE=cpu

# System dependencies required for PostgreSQL client, curl, git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install lightweight backend requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-generate matplotlib font cache
ENV MPLCONFIGDIR=/tmp/matplotlib
RUN python -c "import matplotlib.font_manager"

# Copy backend workspace into container
COPY . .

# Create runtime directories
RUN mkdir -p backend/uploads backend/temp backend/reports

EXPOSE 8000

# Start command: Apply database migrations and launch Uvicorn
CMD ["sh", "-c", "alembic upgrade head && uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
