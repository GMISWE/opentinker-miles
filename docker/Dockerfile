FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY training ./training

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "training.api:app", "--host", "0.0.0.0", "--port", "8000"]
