# CT Scan Denoising

CT images often contain structured noise from acquisition constraints and reconstruction artifacts. This project focuses on denoising CT slices while preserving edges and fine anatomical detail.

---

## Data format in this repo

Training data is expected to come from NetCDF blocks (`.nc`) containing a tomography volume variable (default: `tomo`).

The dataset pipeline (`src/datasets/make_dataset.py`) performs:

- volume split into train/val/test blocks
- slice extraction
- optional intensity clipping/windowing
- saving grayscale PNG slices
- optional patch extraction (default patch size: 256)

Outputs are stored under `data/processed/` (gitignored).

---

## Recommended workflow

1. Prepare data: `uv run python src/datasets/make_dataset.py`
2. Train: `uv run python scripts/train.py --config_path configs/train.yaml`
3. Export ONNX for deployment: `uv run python onnx/export_onnx.py`
4. Run real-time inference: `uv run uvicorn api.api_server:app --port 8000`

---

## Inference on large images

Large CT slices can exceed typical model input sizes. ONNX inference supports:

- **GPU tiling** (splits a large image into overlapping tiles)
- **CPU padding/wrapping** (pads to satisfy stride constraints, then crops back)

These strategies allow inference on high-resolution inputs while keeping latency predictable.