"""Audit ECG-RAMBA revision protocol before running expensive Colab jobs.

Run from repo root:
    python scripts/revision/00_audit_protocol.py

The output is written to reports/revision/audit_protocol.json and should be
checked before using any result in the manuscript/rebuttal.
"""

from __future__ import annotations

import glob
import os
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    CURRENT_HRV36_SCHEMA,
    PTB_SUPERCLASS_MAPPING,
    REVISION_DIR,
    ensure_revision_dirs,
    save_csv,
    save_json,
)


def path_info(path: str | os.PathLike[str]) -> dict:
    p = Path(path)
    return {
        "path": str(p),
        "exists": p.exists(),
        "is_dir": p.is_dir() if p.exists() else False,
        "size_bytes": p.stat().st_size if p.exists() and p.is_file() else None,
    }


def main() -> None:
    ensure_revision_dirs()

    from configs.config import CLASSES, CONFIG, CONFIG_HASH, PATHS  # noqa: WPS433

    hrv_source = (PROJECT_ROOT / "src" / "features.py").read_text(encoding="utf-8")
    zero_filled_slots = [
        idx for name, idx in CURRENT_HRV36_SCHEMA if name.startswith("reserved_zero_")
    ]

    model_dir = Path(PATHS["model_dir"])
    fold_ckpts = sorted(glob.glob(str(model_dir / "fold*_best.pt")))
    global_pca_candidates = [
        model_dir / "global_pca_zeroshot.pkl",
        PROJECT_ROOT / "global_pca_zeroshot.pkl",
    ]
    dataset_keys = ["zip_path", "ptb_zip", "cpsc_zip", "georgia_zip"]
    dataset_paths = {key: path_info(PATHS[key]) for key in dataset_keys if key in PATHS}

    warnings = []
    if zero_filled_slots:
        warnings.append(
            "Current HRV36 schema has reserved zero-filled slots 5:24. "
            "Do not describe it as full HRV with RMSSD/SDNN/LF-HF unless implemented and retrained."
        )
    if not fold_ckpts:
        warnings.append("No fold*_best.pt checkpoints found under configured model_dir.")
    if not any(p.exists() for p in global_pca_candidates):
        warnings.append("No global_pca_zeroshot.pkl found for zero-shot evaluation.")
    if CONFIG.get("aggregation_method") == "mean":
        warnings.append(
            "CONFIG['aggregation_method'] is 'mean' while manuscript protocol uses Power Mean Q=3. "
            "Use a shared aggregation helper in revision scripts."
        )
    if "zip_path" in PATHS and not Path(PATHS["zip_path"]).exists():
        warnings.append(
            f"Chapman ZIP not found at configured zip_path: {PATHS['zip_path']}."
        )
    if "ptb_zip" in PATHS and not Path(PATHS["ptb_zip"]).exists():
        warnings.append(
            f"PTB-XL ZIP not found at configured ptb_zip: {PATHS['ptb_zip']}."
        )
    if "cpsc_zip" in PATHS and not Path(PATHS["cpsc_zip"]).exists():
        warnings.append(
            f"CPSC/Georgia ZIP not found at configured cpsc_zip: {PATHS['cpsc_zip']}."
        )

    payload = {
        "project_root": str(PROJECT_ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "config_hash": CONFIG_HASH,
        "num_classes": len(CLASSES),
        "classes": CLASSES,
        "config_core": {
            "d_model": CONFIG.get("d_model"),
            "n_layers": CONFIG.get("n_layers"),
            "hydra_dim": CONFIG.get("hydra_dim"),
            "hrv_dim": CONFIG.get("hrv_dim"),
            "slice_length": CONFIG.get("slice_length"),
            "slice_stride": CONFIG.get("slice_stride"),
            "max_slices_per_record": CONFIG.get("max_slices_per_record"),
            "default_threshold": CONFIG.get("default_threshold"),
            "aggregation_method": CONFIG.get("aggregation_method"),
        },
        "paths": {k: path_info(v) for k, v in PATHS.items()},
        "artifacts": {
            "fold_checkpoints": fold_ckpts,
            "global_pca_candidates": [path_info(p) for p in global_pca_candidates],
            "dataset_paths": dataset_paths,
            "reports_revision_dir": str(REVISION_DIR),
        },
        "hrv36_schema": [{"index": idx, "name": name} for name, idx in CURRENT_HRV36_SCHEMA],
        "hrv_extract_source_contains": {
            "rmssd": "rmssd" in hrv_source.lower(),
            "sdnn": "sdnn" in hrv_source.lower(),
            "lf_hf": ("lf" in hrv_source.lower() and "hf" in hrv_source.lower()),
            "reserved_zero_slots": zero_filled_slots,
        },
        "ptb_superclass_mapping": PTB_SUPERCLASS_MAPPING,
        "warnings": warnings,
    }

    save_json(REVISION_DIR / "audit_protocol.json", payload)
    save_csv(REVISION_DIR / "hrv36_schema.csv", payload["hrv36_schema"])

    print("=" * 80)
    print("ECG-RAMBA REVISION PROTOCOL AUDIT")
    print("=" * 80)
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Config hash        : {CONFIG_HASH}")
    print(f"Classes            : {len(CLASSES)}")
    print(f"Model dir          : {model_dir}")
    print(f"Fold checkpoints   : {len(fold_ckpts)}")
    print(f"Revision output    : {REVISION_DIR}")
    print("\nDataset paths:")
    for key, info in dataset_paths.items():
        print(f"  {key:<12}: {info['path']} | exists={info['exists']}")
    print("\nHRV36 schema:")
    print("  active RR slots   : 0:5")
    print("  reserved slots    : 5:25")
    print("  amplitude slots   : 25:30")
    print("  global stat slots : 30:36")
    print("\nPTB superclass mapping:")
    for target, spec in PTB_SUPERCLASS_MAPPING.items():
        print(f"  {target:<5} <- {spec['codes']}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    print("\nWrote:")
    print(f"  {REVISION_DIR / 'audit_protocol.json'}")
    print(f"  {REVISION_DIR / 'hrv36_schema.csv'}")


if __name__ == "__main__":
    main()
