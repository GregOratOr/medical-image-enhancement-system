# Training Your First Model

This tutorial walks through a complete training run:

1. Prepare CT data (NetCDF → PNG slices/patches)
2. Train a Noise2Noise U-Net using `configs/train.yaml`
3. Inspect logs and checkpoints

---

## 1) Prepare the dataset

Place your NetCDF files under:

- `data/raw/` (gitignored)

Then run the dataset pipeline:

```powershell
uv run python src/datasets/make_dataset.py
```

By default, the pipeline writes processed data to:

- `data/processed/<split>/images/`
- `data/processed/<split>/patches/` (if patch extraction is enabled)

---

## 2) Configure training

Training is driven by Draccus + YAML. The default config is:

- `configs/train.yaml`

Key fields to understand:

- `data.train_dir` / `data.val_dir`: where PNG patches/images live
- `data.preprocess_params.noise_params`: noise types and parameters
- `models[0]`: model name (`n2n-new`) and U-Net hyperparameters
- `logs`: enable TensorBoard / W&B
- `train`: epochs, AMP, checkpoint interval

---

## 3) Run training

From the repo root:

```powershell
uv run python scripts/train.py --config_path configs/train.yaml
```

The training script prints a run directory (created under `./experiments/` by default). Checkpoints are written under:

- `experiments/<run_name>/checkpoints/`

---

## 4) Monitor logs

TensorBoard is enabled by default in `configs/train.yaml`.

If you have TensorBoard installed (it is a runtime dependency in this repo), you can launch it and point it at your run directory:

```powershell
uv run tensorboard --logdir experiments
```

You can also enable Weights & Biases by setting `logs.use_wandb: true`.

For details on how logging is structured, see:

- `user_guide/concepts/logging.md`

---

## 5) Next steps

- Export a trained checkpoint to ONNX: `onnx/export_onnx.py`
- Run real-time inference: `user_guide/deployment/real_time.md`
- Deploy with Docker Compose: `user_guide/deployment/docker.md`

