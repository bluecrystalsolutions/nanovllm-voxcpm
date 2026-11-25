ARG BASE_IMAGE=pytorch/pytorch:2.14.1-cuda13.2-cudnn9-runtime
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

FROM ${BASE_IMAGE}

# Re-declare build args for this stage so they can be used in ENV/RUN.
ARG TORCH_CUDA_ARCH_LIST

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    UV_NO_PROGRESS=1 \
    UV_BREAK_SYSTEM_PACKAGES=1

WORKDIR /app

# Defensive build deps. Nothing in this dependency set is expected to
# compile from source any more - flash-attn now resolves as a prebuilt
# wheel from Astral's index pinned in pyproject.toml's [tool.uv.index]
# entry - but these stay as cheap insurance against an unexpected
# sdist-only package falling back to a plain (non-CUDA) C/C++ build.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      build-essential \
      git \
      pkg-config \
      ffmpeg \
      libsndfile1-dev \
  && rm -rf /var/lib/apt/lists/*

# Install uv. The pytorch/pytorch base marks its system Python as PEP 668
# "externally-managed" - pip needs --break-system-packages, and uv has its
# own separate guard (the UV_BREAK_SYSTEM_PACKAGES env var set above).
RUN pip install --no-cache-dir --break-system-packages uv

ENV NANOVLLM_CACHE_DIR=/var/cache/nanovllm
RUN useradd -m -u 10001 appuser \
 && mkdir -p "$NANOVLLM_CACHE_DIR" \
 && chown -R appuser:appuser "$NANOVLLM_CACHE_DIR" \
 && chown -R appuser:appuser /app

USER appuser

COPY --chown=appuser:appuser . .

# Create the project venv with visibility into the base image's system
# site-packages, so torch/triton/torchaudio (the [tool.uv] exclude-dependencies
# entries in pyproject.toml - together several GB of CUDA wheels already
# present in the base image) are resolved from there instead of being
# re-downloaded. --inexact is required because uv sync otherwise insists on
# removing anything not in its own lockfile, which would include every
# system-site-package it can see through this venv.
RUN uv venv --system-site-packages .venv --python /usr/bin/python3

# Install workspace packages from the frozen lockfile into that venv.
RUN uv sync --all-packages --no-dev -v --compile-bytecode --no-cache --no-editable --inexact

EXPOSE 8000

# k8s probes can use /health and /ready.
CMD ["uv", "run", "--no-sync", "fastapi", "run", "deployment/app/main.py", "--host", "0.0.0.0", "--port", "8000"]
