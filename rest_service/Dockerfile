# Dockerfile (para grpc_service, graphql_service, rest_service, client)
FROM python:3.9-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY pyproject.toml ./
RUN poetry install --no-root --no-interaction

COPY . .

RUN mkdir -p /app/storage
