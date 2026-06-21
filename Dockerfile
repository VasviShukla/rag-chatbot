# ---- builder stage: compile/collect wheels, including build tools ----
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ---- runtime stage: slim image, no compilers, just installed wheels ----
FROM python:3.11-slim

WORKDIR /app

# Keep CPU thread usage low — important on small/free-tier instances
ENV OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY app ./app

RUN mkdir -p data/uploads data/chroma_db

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
