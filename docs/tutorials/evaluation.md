# Evaluating Denoising Quality

This project supports both qualitative and quantitative evaluation workflows.

---

## Qualitative evaluation (recommended first)

### Use the Streamlit UI

Run the backend and UI:

```powershell
uv run uvicorn api.api_server:app --host 0.0.0.0 --port 8000
uv run streamlit run app/main.py
```

Then use:
- **Image Inspector** for side-by-side comparison and error-map visualization
- **Batch Processor** for denoising multiple images and downloading results

---

## Quantitative evaluation (metrics)

The training pipeline computes validation metrics via the evaluation utilities in:

- `src/evaluation/metrics.py`

Common metrics include:
- **Loss** (training objective)
- **PSNR** (Peak Signal-to-Noise Ratio)

If you want to compute metrics on a saved inference directory, a good pattern is:

1. Run inference to produce `denoised_*.png` outputs
2. Compare against reference images when available

---

## Simulated evaluation (synthetic noise)

If you don’t have paired references, you can still do controlled evaluations using the simulation inference script:

```powershell
uv run python scripts/simulate_inference.py --config_path configs/sim_inference.yaml
```

This path creates noisy inputs and produces side-by-side outputs (input / output / reference) for inspection.

---

## Next steps

- Tighten evaluation by saving a fixed test split and comparing runs across checkpoints.
- Add a lightweight metrics script for directory-to-directory comparisons if you want automated reporting.

