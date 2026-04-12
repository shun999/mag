# ──────────────────────────────────────────────
# Base: CUDA 11.7 + cuDNN 8 on Ubuntu 22.04
# ──────────────────────────────────────────────
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:$PATH"

# ──────────────────────────────────────────────
# System: Python 3.11 + build tools
# ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-distutils \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

# ──────────────────────────────────────────────
# uv (fast Python package manager)
# ──────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# ──────────────────────────────────────────────
# Install Python dependencies
# Torch packages are pulled from the pytorch-cu117
# index as defined in pyproject.toml / uv.lock.
# ──────────────────────────────────────────────
WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

# ──────────────────────────────────────────────
# Copy project source
# ──────────────────────────────────────────────
COPY . .

# ──────────────────────────────────────────────
# Runtime: expose FastAPI port
# ──────────────────────────────────────────────
EXPOSE 8000

# Default: start the anomaly-detection API.
# Override CMD to run training scripts instead:
#   docker run mag uv run python AIbuild/scripts/train.py
CMD ["uv", "run", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
