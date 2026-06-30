# Installation

This guide covers installing dependencies and running the system locally or via Docker.

---

## Prerequisites

- **Python**: 3.14+ (see `requires-python` in `pyproject.toml`)
- **uv**: recommended package manager for this repo
- **GPU (optional but recommended)**: NVIDIA CUDA for training and fast ONNX inference
- **Docker (optional)**: for a reproducible inference + UI deployment
  - If you want GPU inside Docker, install **NVIDIA Container Toolkit**

---

## Install with `uv` (recommended)

From the repo root:

```powershell
uv sync
```

This installs runtime dependencies from `pyproject.toml` / `uv.lock`.

### Install docs dependencies

```powershell
uv sync --group dev
```

---

## Run locally (API + UI)

Start the FastAPI backend (port **8000**):

```powershell
uv run uvicorn api.api_server:app --host 0.0.0.0 --port 8000
```

Start the Streamlit UI (port **8501**) in a second terminal:

```powershell
uv run streamlit run app/main.py
```

### Environment variables

The Streamlit app uses `API_URL` to reach the backend. Default is `http://127.0.0.1:8000`.

```powershell
$env:API_URL="http://127.0.0.1:8000"
```

If you deploy the UI in a cloud environment, set `CLOUD_DEPLOYMENT=true` to hide the local-folder batch tab:

```powershell
$env:CLOUD_DEPLOYMENT="true"
```

---

## Run with Docker Compose (recommended for inference/UI)

This repo includes a compose setup:
- **backend**: FastAPI server on port **8000**
- **frontend**: Streamlit UI on port **8501**

```powershell
docker compose up --build
```

Notes:
- `docker-compose.yaml` reserves an NVIDIA GPU for the backend service.
- The frontend container is configured with `API_URL=http://backend:8000`.

---

## Troubleshooting

### Backend shows “Offline” in the sidebar

- Make sure the API is running on the URL the UI expects.
- If you’re running locally and changed the API host/port, set `API_URL` before starting Streamlit.

### GPU not detected

- ONNX Runtime detects GPU providers at runtime. If the backend is in CPU mode, confirm CUDA drivers/toolkit are installed (local) or that NVIDIA Container Toolkit is installed (Docker).