
# ==================================================================================
# 🧰 ECG RAMBA - Utilities Module
# ----------------------------------------------------------------------------------
# Design principles:
# - NO clinical heuristics
# - NO threshold optimization
# - NO validation peeking
# - NO label leakage
# - NO optimization tricks (GC, SWA, calibration)
# - Deterministic & reproducible
#
# Provided utilities:
# - Fixed-gamma FN-aware Asymmetric Loss
# - Honest evaluation metrics (fixed threshold)
# - EMA (evaluation-only)
# - Seed control
# ==================================================================================

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)

# ============================================================
# 🔢 PER-CLASS NEGATIVE WEIGHT (TRAIN-ONLY)
# ============================================================

def compute_negative_class_weights(y_train: np.ndarray) -> torch.Tensor:
    """
    Compute per-class negative weights using TRAIN labels only.
    Data-driven, normalized, no priors.

    NOTE:
    - Optional in Clean Core (can be disabled via config)
    - Never computed on val / test
    """
    assert y_train.ndim == 2, "y_train must be (N, C)"

    freq = y_train.mean(axis=0)
    neg_weight = 1.0 / (freq + 1e-6)
    neg_weight = neg_weight / neg_weight.mean()

    return torch.tensor(neg_weight, dtype=torch.float32)


# ============================================================
# 📉 ASYMMETRIC LOSS
# ============================================================

class AsymmetricLossMultiLabel(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification (.

    Properties:
    ✔ FN-aware
    ✔ FIXED gamma (no annealing)
    ✔ No thresholding
    ✔ TRAIN-only statistics

    IMPORTANT:
    - Gamma MUST be fixed from config
    - No dynamic updates allowed
    """

    def __init__(
        self,
        gamma_neg: float,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        neg_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = clip
        self.eps = eps

        if neg_weight is not None:
            self.register_buffer("neg_weight", neg_weight)
        else:
            self.neg_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Positive term
        loss_pos = -targets * ((1.0 - probs) ** self.gamma_pos) * \
                   torch.log(probs + self.eps)

        # Negative term (FN-aware)
        probs_neg = (probs - self.clip).clamp(min=0.0)

        if self.neg_weight is None:
            neg_w = 1.0
        else:
            neg_w = self.neg_weight.view(1, -1)

        loss_neg = -(1.0 - targets) * neg_w * \
                   (probs_neg ** self.gamma_neg) * \
                   torch.log(1.0 - probs_neg + self.eps)

        loss = loss_pos + loss_neg
        return loss.sum() / logits.size(0)


# ============================================================
# 📊 METRICS
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Honest record-level metrics.

    Rules:
    - Fixed threshold ONLY (default = 0.5)
    - No per-class thresholds
    - No optimization
    """
    assert y_true.shape == y_prob.shape

    y_pred = (y_prob > threshold).astype(np.float32)

    metrics = {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    auprcs = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() > 0:
            auprcs.append(
                average_precision_score(y_true[:, c], y_prob[:, c])
            )

    metrics["auprc_macro"] = float(np.mean(auprcs)) if len(auprcs) > 0 else 0.0
    return metrics


# ============================================================
# 🔁 EMA
# ============================================================

class EMA:
    """
    Exponential Moving Average (Clean Core).

    Rules:
    - Evaluation-only
    - NEVER mixed with SWA (SWA not allowed here)
    - Applied & restored explicitly in validation
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            n: p.data.clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.backup = {}

    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


# ============================================================
# 🎲 SEED CONTROL (DETERMINISTIC)
# ============================================================

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
