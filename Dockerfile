# LLMPerf Dockerfile
# Multi-stage build for optimized image with frontend and backend
# Configured for China mirror sources

# ============================================
# Stage 1: Frontend Build
# ============================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Configure npm to use Taobao mirror
RUN npm config set registry https://registry.npmmirror.com

# Copy frontend package files
COPY frontend/package*.json ./

# Install dependencies with Taobao mirror
RUN npm ci --legacy-peer-deps

# Copy frontend source
COPY frontend/ ./

# Build frontend
RUN npm run build

# ============================================
# Stage 2: Backend Builder
# ============================================
FROM python:3.12-slim AS backend-builder

WORKDIR /app

# Configure pip to use Tsinghua mirror
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies to a temporary location
RUN pip install --no-cache-dir --target=/app/deps .

# ============================================
# Stage 3: Runtime
# ============================================
FROM python:3.12-slim

# Labels
LABEL maintainer="LLMPerf Team"
LABEL description="LLMPerf - Unified benchmarking toolkit for LLM providers with Web UI"
LABEL version="0.1.0"

# Create non-root user
RUN useradd --create-home --shell /bin/bash llmperf

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Configure pip for runtime (in case of future installs)
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# Copy Python packages from builder
COPY --from=backend-builder /app/deps /usr/local/lib/python3.12/site-packages

# Copy backend source
COPY src/llmperf/ /app/llmperf/
COPY template/ /app/template/
COPY resource/ /app/resource/
COPY pricings/ /app/pricings/

# Copy frontend build from builder
COPY --from=frontend-builder /app/frontend/dist /app/static

# Create necessary directories
RUN mkdir -p /data /app/logs && \
    chown -R llmperf:llmperf /app /data

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LLMPerf_DB_PATH=/data/llmperf.db \
    LLMPerf_LOG_DIR=/app/logs \
    PYTHONPATH=/app \
    STATIC_DIR=/app/static

# Switch to non-root user
USER llmperf

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["python", "-m", "llmperf.web.main"]
