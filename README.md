# OpenTinker Miles Training API

Standalone extraction of kgateway's training service wired to the Miles backend. This repository focuses on the FastAPI application under `training/` and includes a Dockerfile for building a runnable image.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn training.api:app --reload --port 8000
```

## Docker

```bash
cd docker
docker build -t opentinker/miles-training:latest .
```
