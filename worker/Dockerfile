FROM python:3.10-slim

WORKDIR /app

# We need docker CLI inside the container to manage other containers (Docker-out-of-Docker)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8085", "--workers", "1"]
