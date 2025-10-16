# Dockerfile (para grpc_service, graphql_service, rest_service, client)
FROM python:3.9-slim-bullseye

WORKDIR /app

# Paquetes del sistema (libglib2.0-0; tzdata para zoneinfo)
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy pyproject.toml and poetry.lock to the working directory
COPY pyproject.toml ./ 
# poetry.lock ./

# Install dependencies using Poetry
RUN poetry install --no-root --no-interaction

# Copia proto y genera código gRPC
COPY proto ./proto/
RUN poetry run python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. ./proto/img_service.proto

# Copy the rest of the application code
COPY . .

# Para gRPC, necesitamos generar el código proto dentro del contenedor
# Esto es específico para grpc_service, se puede omitir en los demás

RUN mkdir -p /app/storage

#RUN poetry run python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. ./proto/ml_service.proto
