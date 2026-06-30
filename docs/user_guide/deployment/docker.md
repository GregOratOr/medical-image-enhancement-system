# Docker Deployment

The fastest way to run the full system (backend + UI) is Docker Compose.

This repo provides:
- **backend**: FastAPI server (port **8000**) using ONNX Runtime
- **frontend**: Streamlit UI (port **8501**) configured to talk to the backend

---

## Requirements

- Docker Desktop / Docker Engine
- For GPU acceleration: **NVIDIA Container Toolkit** + a CUDA-capable GPU

---

## Start the stack

From the repo root:

```powershell
docker compose up --build
```

Access:
- Backend API: `http://localhost:8000`
- Streamlit UI: `http://localhost:8501`

---

## What Compose configures

The compose file (`docker-compose.yaml`) does the following:

- Mounts `./onnx/models` into the container at `/workspace/onnx/models`
- Sets `CLOUD_DEPLOYMENT=True` for the backend (so the UI can hide local-only functionality when needed)
- Sets `API_URL=http://backend:8000` for the frontend so Streamlit can reach FastAPI on Docker’s internal network
- Reserves an NVIDIA GPU for the backend service

---

## Troubleshooting

### No GPU available in the backend container

- Ensure NVIDIA Container Toolkit is installed and configured.
- Confirm your Docker runtime can access the GPU.

### Streamlit cannot connect to the backend

- Verify both containers are running and ports are mapped.
- The frontend uses `API_URL=http://backend:8000` inside Docker; do not set it to `localhost` in the container.