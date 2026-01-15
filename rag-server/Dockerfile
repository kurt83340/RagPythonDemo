FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data
RUN mkdir -p /app/models /app/db

# Environment variables for local model storage
# Models are downloaded at runtime and persisted via volume mount
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV TORCH_HOME=/app/models

# NOTE: Models are NOT pre-downloaded in build anymore
# They are downloaded once at first run and persisted in ./rag-server/models volume
# This avoids re-downloading on every docker compose build

# Expose nothing - MCP uses stdio
CMD ["python", "server.py"]
