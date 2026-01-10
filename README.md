# ECG-RAMBA: Zero-Shot ECG Generalization by Morphology-Rhythm Disentanglement and Long-Range Modeling

[![arXiv](https://img.shields.io/badge/arXiv-2512.23347-b31b1b.svg)](https://arxiv.org/abs/2512.23347)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ecg-ramba-zero-shot-ecg-generalization-by/arrhythmia-detection-on-chapman-shaoxing)](https://paperswithcode.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This is the **Official PyTorch Implementation** of the paper:
**"ECG-RAMBA: Zero-Shot ECG Generalization by Morphology-Rhythm Disentanglement and Long-Range Modeling"**
_Hai Duong Nguyen, Xuan-The Tran (2025)_

---

## 📢 News

- **[2026-01-10]**: Code release for ECG-RAMBA.
- **[2025-12-30]**: Paper available on ArXiv.

---

## 📖 Abstract

Deep learning has achieved strong performance for electrocardiogram (ECG) classification within individual datasets, yet dependable generalization across heterogeneous acquisition settings remains a major obstacle. A key limitation of many model architectures is the implicit entanglement of morphological waveform patterns and rhythm dynamics, which can promote shortcut learning.

We propose **ECG-RAMBA**, a framework that separates morphology and rhythm and then re-integrates them through context-aware fusion.

<div align="center">
  <img src="reports/figures/architecture.png" alt="ECG-RAMBA Architecture" width="800"/>
</div>

**Key Contributions:**

1.  **Disentangled Architecture**: Combines deterministic morphological features (**MiniRocket**) with global rhythm descriptors (**HRV**) and long-range contextual modeling (**Bi-Mamba**).
2.  **Context-Aware Fusion**: Re-integrates independent streams via Cross-Attention to capture non-linear interactions suitable for complex arrhythmias.
3.  **Power Mean Pooling ($Q=3$)**: A numerically stable pooling operator that emphasizes high-evidence segments without the brittleness of max pooling.
4.  **Zero-Shot Robustness**: Achieves state-of-the-art zero-shot transfer performance on standard benchmarks (CPSC-2021, PTB-XL).

---

## 💡 Key Innovations

### 1. Morphology-Rhythm Disentanglement

Unlike traditional CNNs that entangle waveform shapes with rhythm, **ECG-RAMBA** explicitly separates them:

- **Morphology Stream**: Uses **MiniRocket**, a deterministic convolution kernel ensuring consistent feature extraction regardless of training distribution.
- **Rhythm Stream**: Computes global HRV descriptors (RMSSD, SDNN, Poincaré) to capture long-term autonomic nervous system dynamics.

### 2. Bi-Directional Mamba Backbone

Leverages **State Space Models (SSM)** to model long-range dependencies across 5000-timepoint whole-signal ECGs with linear computational complexity $O(N)$, overcoming the quadratic bottleneck of Transformers.

### 3. Power Mean Pooling ($Q=3$)

Introduces a numerically stable pooling operator that improves sensitivity to transient abnormalities (like Paroxysmal AF). Unlike Max Pooling (brittle) or Average Pooling (diluting), Power Mean with $Q=3$ emphasizes high-evidence segments while remaining robust to noise.

### 4. Zero-Shot Generalization

Designed for **clinical reliability**:

- **No Test-Time Adaptation**: Works out-of-the-box on unseen datasets.
- **Fixed Threshold ($\tau=0.5$)**: No dataset-specific threshold tuning required.
- **Subject-Aware Protocol**: Strict evaluation preventing identity leakage.

---

## 🛠️ Installation

```bash
# 1. Clone repository
git clone https://github.com/BrianNguyen29/ECG-RAMBA.git
cd ECG-RAMBA

# 2. Install dependencies
# Recommended: Python 3.10+, CUDA 11.8+
pip install -r requirements.txt
```

> **Note**: This project relies on `mamba-ssm` which requires CUDA. For CPU-only inference, performance will be significantly slower and the fallback path will be used.

---

## 🚀 Usage

### 1. Data Preparation

Download datasets from PhysioNet (detailed instructions in `data/README.md`) and organize them:

```text
data/
├── chapman/       # .mat and .hea files
├── cpsc2021/      # Extract CPSC-2021 here
└── ptbxl/         # Extract PTB-XL here (must contain ptbxl_database.csv)
```

### 2. Training

Train the model with 5-fold Cross-Validation (Subject-Aware):

```bash
python scripts/train.py
```

- **Configuration**: Modify hyperparameters in `configs/config.py`.
- **Logging**: Metrics are saved to `reports/logs/`.
- **Checkpoints**: Best models are saved to `models/`.

### 3. Evaluation (OOF)

Run Out-of-Fold (OOF) evaluation to verify internal performance:

```bash
python scripts/eval_oof.py
```

### 4. Zero-Shot Transfer

Test the trained model on unseen datasets (e.g., CPSC-2021, PTB-XL) without fine-tuning:

```bash
python scripts/eval_zeroshot.py
```

---

## 📂 Project Structure

This repository follows the **Clean Core** principle to ensure reproducibility:

```text
ECG-RAMBA/
├── configs/            # Centralized configuration (no hardcoded params).
├── data/               # Dataset storage (Git-ignored).
├── models/             # Pre-trained weights & checkpoints.
├── reports/            # Figures and experimental logs.
├── scripts/            # Executable recipes for training/evaluation.
├── src/                # Core Source Code (Model, Layers, Features).
└── web_app/            # Deployment Application (Backend/Frontend).
```

---

## 📜 Citation

If you use this code or model in your research, please cite our paper:

```bibtex
@article{nguyen2025ecg,
  title={ECG-RAMBA: Zero-Shot ECG Generalization by Morphology-Rhythm Disentanglement and Long-Range Modeling},
  author={Nguyen, Hai Duong and Tran, Xuan-The},
  journal={arXiv preprint arXiv:2512.23347},
  year={2025},
  url={https://arxiv.org/abs/2512.23347}
}
```

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🤝 Acknowledgements

We thank the PhysioNet team for hosting the Chapman-Shaoxing, CPSC-2021, and PTB-XL datasets.
