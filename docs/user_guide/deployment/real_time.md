# Real-time Inference

This project supports real-time denoising via **ONNX Runtime** behind a **FastAPI** server. The Streamlit UI can connect to this server to provide an interactive “studio” experience.

---

## Start the API server

From the repo root:

```powershell
uv run uvicorn api.api_server:app --host 0.0.0.0 --port 8000
```

Health check:

- `GET /` returns `status` and `active_config`

---

## Inference endpoint

### `POST /predict`

Upload a PNG/JPG/TIF (it will be converted to grayscale internally) and receive the **denoised PNG** bytes.

Example:

```powershell
curl -X POST "http://127.0.0.1:8000/predict" `
  -F "file=@path\to\scan.png" `
  -o denoised.png
```

---

## Hot-swap models and compute backend

The server supports changing the active model without restarting:

### `POST /set_config`

Request body:

```json
{
  "model_name": "medical_denoiser_dyno",
  "enable_cuda": true,
  "wrapped_model": false
}
```

Key fields:
- `model_name`: base model name (without `.onnx`)
- `enable_cuda`: if true and CUDA providers are available, uses GPU inference
- `wrapped_model`: whether the ONNX model already contains wrapper logic (padding)

---

## Free VRAM / unload model

### `POST /unload`

Unloads the active model to free memory. The next `/predict` will auto-reload the configured model.

---

## Error maps

### `POST /error_map`

Upload two images (`image1`, `image2`) to generate a difference heatmap image.

---

## Notes on image sizes

This system is designed to handle large medical images:
- GPU inference uses tiling for throughput
- CPU inference can use padding/wrapping to satisfy model stride constraints

For the shipped models, see `onnx/models/` and the “Bundled ONNX models” section in the repository README.