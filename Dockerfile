# =====================================================
# Stage 1 — Build Executable
# =====================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    build-essential \
    gcc \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Map Python to python3.10
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Copy project
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
# Ensure Torch uses CU121 for our base
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pyinstaller

# Build onedir executable
RUN pyinstaller \
    --onedir \
    --name lpr_app \
    --collect-all torch \
    --collect-all ultralytics \
    --hidden-import=cv2 \
    --hidden-import=cryptography \
    camera.py



# =====================================================
# Stage 2 — Runtime (Clean Image)
# =====================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install ONLY runtime libs (no build tools)
RUN apt-get update && apt-get install -y \
    python3.10 \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy built binary directory
COPY --from=builder /app/dist/lpr_app /app/lpr_app_dist

# Create runtime folders
RUN mkdir -p /app/license /app/logs /app/outputs /app/models

# Make executable safe (Only target the actual binary instead of recursive -R)
RUN chmod +x /app/lpr_app_dist/lpr_app

# Start application
CMD ["/app/lpr_app_dist/lpr_app", "--config_file", "/app/config.json"]
