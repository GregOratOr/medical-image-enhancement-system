# Noise2Noise

Noise2Noise (N2N) is a training strategy that learns to denoise images **without clean targets**.

Instead of optimizing a model \(f_\theta\) to map noisy inputs to clean ground truth, we train on **pairs of independently noisy observations** of the same underlying signal:

\[
\hat{z} = s + n_1,\quad z = s + n_2
\]

Under mild assumptions (zero-mean noise, independent samples), minimizing an L2/L1 loss between \(f_\theta(\hat{z})\) and \(z\) encourages \(f_\theta\) to recover the underlying signal \(s\).

---

## How this repo applies Noise2Noise

- **Model**: a U-Net style architecture (`src/models/noise2noise.py`)
- **Noise injection**: configurable noise modules (`src/transforms/noise.py`)
- **Training entry point**: `scripts/train.py` using `configs/train.yaml`

The training config controls:
- which noise types are used (`data.preprocess_params.noise_params`)
- model architecture (`models[0].model_params`)
- logging and checkpoints

---

## Common pitfalls

- **Non-independent noise**: if both views share the same noise realization, the training signal collapses.
- **Mismatched preprocessing**: ensure inference uses the same grayscale normalization as training.
- **Cropping/patching**: training uses a crop size (default 256) — ensure your evaluation is consistent.

---

## Next steps

- Tutorial: `tutorials/training.md`
- Noise modules API: `api/noise_gen.md`