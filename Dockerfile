FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_TORCH_BACKEND=cu128 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=3.12 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/workspace/src \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /workspace
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra dev

CMD ["bash"]
