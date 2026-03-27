FROM python:3.10-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY memoryos-pypi/ ./memoryos-pypi/
RUN apt-get update && \
    apt-get install -y vim && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app/memoryos-pypi
RUN uv sync --frozen
WORKDIR /app
