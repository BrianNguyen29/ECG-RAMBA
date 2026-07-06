"""Transformer ECG baseline under the frozen Chapman OOF protocol.

This runner intentionally reuses the audited raw-ECG baseline pipeline from
``14_resnet1d_cnn_baseline.py`` and swaps only the model/output contract. It is
optional reviewer evidence and must be interpreted as a comparator-specific
baseline, not as a route to broad superiority claims.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESNET_HELPER_PATH = PROJECT_ROOT / "scripts" / "revision" / "14_resnet1d_cnn_baseline.py"


def load_resnet_helpers():
    spec = importlib.util.spec_from_file_location("resnet1d_cnn_baseline_helpers", RESNET_HELPER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module: {RESNET_HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ECGPatchTransformer(nn.Module):
    """Compact patch transformer for 12-lead ECG slices."""

    def __init__(
        self,
        *,
        n_classes: int,
        embed_dim: int = 96,
        n_heads: int = 4,
        depth: int = 3,
        patch_size: int = 50,
        patch_stride: int = 25,
        dropout: float = 0.20,
        max_length: int = 2500,
    ) -> None:
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by n_heads={n_heads}")
        self.patch = nn.Conv1d(12, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=patch_size // 2)
        max_tokens = math.ceil(max_length / patch_stride) + 2
        self.positional = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        nn.init.trunc_normal_(self.positional, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(x).transpose(1, 2)
        tokens = tokens + self.positional[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens)
        pooled = self.norm(encoded.mean(dim=1))
        return self.head(pooled)


def main() -> None:
    helpers = load_resnet_helpers()
    helpers.PROTOCOL = "transformer_ecg_raw_same_folds_power_mean_v2_q3_threshold_0.5"
    helpers.RUNNER_DISPLAY_NAME = "Transformer ECG"
    helpers.ARCHITECTURE_NAME = "patch_transformer_raw_ecg"
    helpers.CHECKPOINT_STEM = "transformer_ecg"
    helpers.PREDICTION_PATH = helpers.PREDICTION_DIR / "transformer_ecg_oof_predictions.npz"
    helpers.SLICE_PREDICTION_PATH = helpers.PREDICTION_DIR / "transformer_ecg_slice_predictions.npz"
    helpers.SUMMARY_PATH = helpers.METRIC_DIR / "transformer_ecg_baseline_summary.json"
    helpers.MANIFEST_PATH = helpers.MANIFEST_DIR / "transformer_ecg_baseline_manifest.json"
    helpers.PER_CLASS_TABLE = helpers.TABLE_DIR / "table_transformer_ecg_class_metrics.csv"
    helpers.FOLD_TABLE = helpers.TABLE_DIR / "table_transformer_ecg_fold_summary.csv"

    original_parse_args = helpers.parse_args
    default_resnet_dir = (
        PROJECT_ROOT / "reports" / "revision" / "experimental" / "resnet1d_cnn_checkpoints"
    ).resolve()
    default_transformer_dir = PROJECT_ROOT / "reports" / "revision" / "experimental" / "transformer_ecg_checkpoints"

    def parse_args_with_transformer_defaults():
        args = original_parse_args()
        if Path(args.checkpoint_dir).resolve() == default_resnet_dir:
            args.checkpoint_dir = default_transformer_dir
        return args

    def build_transformer(args):
        embed_dim = int(args.base_channels)
        n_heads = 4 if embed_dim % 4 == 0 else 2
        return ECGPatchTransformer(
            n_classes=len(helpers.CLASSES),
            embed_dim=embed_dim,
            n_heads=n_heads,
            depth=3,
            dropout=float(args.dropout),
            max_length=int(helpers.CONFIG["slice_length"]),
        )

    helpers.parse_args = parse_args_with_transformer_defaults
    helpers.build_model = build_transformer
    helpers.main()


if __name__ == "__main__":
    main()
