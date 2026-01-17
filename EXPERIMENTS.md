# Reproducing Experiments

This document provides exact commands to reproduce the results reported in our paper.

---

## Hardware Requirements

| Component   | Minimum                      | Recommended                    |
| :---------- | :--------------------------- | :----------------------------- |
| **GPU**     | NVIDIA RTX 3080 (10GB VRAM)  | NVIDIA RTX 3090 / A100 (24GB+) |
| **RAM**     | 32 GB                        | 64 GB                          |
| **Storage** | 50 GB (for datasets + cache) | 100 GB                         |
| **CUDA**    | 11.8                         | 12.x                           |

> **Note**: Training with `batch_size=192` requires ~20GB VRAM. Reduce to 64-128 for smaller GPUs.

---

## Dataset Preparation

1. Download datasets from PhysioNet (see [data/README.md](data/README.md))
2. Extract to the following structure:
   ```
   data/
   ├── chapman/          # ~45k records
   ├── cpsc2021/         # For zero-shot AF
   └── ptbxl/            # For zero-shot multi-class
   ```

---

## Interactive Experiments (Web App)

For real-time inference and clinical visualization, we provide a full-stack web application.

```bash
# Start the Clinical Dashboard
cd web_app
run_app.bat
```

See [web_app/README.md](web_app/README.md) for detailed usage.

---

## Table 1: Chapman-Shaoxing 5-Fold Cross-Validation

**Expected Results:** Macro F1 ≈ 0.31, ROC-AUC ≈ 0.85

### Training

```bash
# Full training (5 folds, 20 epochs each)
python scripts/train.py
```

**Output:**

- Checkpoints: `models/fold{1-5}_best.pt`
- Logs: `reports/logs/training_log_epochs.csv`

### Evaluation (OOF)

```bash
python scripts/eval_oof.py
```

**Expected Output:**

```
f1_macro           0.3115 ± 0.0132
precision_macro    0.3421 ± 0.0198
recall_macro       0.3012 ± 0.0156
auprc_macro        0.2845 ± 0.0187
```

---

## Table 2: Zero-Shot Transfer (CPSC-2021)

**Expected Results:** PR-AUC (AF) ≈ 0.708

### Prerequisites

1. Complete Table 1 training (need `fold*_best.pt`)
2. Prepare CPSC-2021 dataset in `data/cpsc2021/`

### Evaluation

```bash
python scripts/eval_zeroshot.py
```

---

## Table 3: Ablation Study

Ablations are controlled via the `ablation` dict passed to `ECGRambaV7Advanced`:

```python
from src.model import ECGRambaV7Advanced
from configs.config import CONFIG

# Full model (default)
model_full = ECGRambaV7Advanced(CONFIG, ablation={})

# Without MiniRocket
model_no_rocket = ECGRambaV7Advanced(CONFIG, ablation={"no_rocket": True})

# Without HRV
model_no_hrv = ECGRambaV7Advanced(CONFIG, ablation={"no_hrv": True})

# Without Multi-Scale Tokenizer
model_no_multiscale = ECGRambaV7Advanced(CONFIG, ablation={"no_multiscale": True})

# Without Cross-Attention Fusion
model_no_fusion = ECGRambaV7Advanced(CONFIG, ablation={"no_fusion": True})
```

To run ablation experiments, modify `scripts/train.py` to pass the desired ablation config.

---

## Inference-Time Ablation

For quick ablation during inference (without retraining):

```python
# Disable MiniRocket features at inference
logits = model(x, xh, xhr, use_rocket=False)

# Disable HRV features at inference
logits = model(x, xh, xhr, use_hrv=False)

# Disable Cross-Attention Fusion (use simple concat)
logits = model(x, xh, xhr, use_fusion=False)
```

---

## Troubleshooting

| Issue                          | Solution                                     |
| :----------------------------- | :------------------------------------------- |
| `CUDA out of memory`           | Reduce `batch_size` in `configs/config.py`   |
| `mamba-ssm` installation fails | Use pre-built wheels or install from source  |
| Different results              | Ensure `numpy==1.26.4` and same CUDA version |

---

## Citation

If you use these experiments, please cite:

```bibtex
@article{nguyen2025ecg,
  title={ECG-RAMBA: Zero-Shot ECG Generalization...},
  author={Nguyen, Hai Duong and Tran, Xuan-The},
  journal={arXiv preprint arXiv:2512.23347},
  year={2025}
}
```
