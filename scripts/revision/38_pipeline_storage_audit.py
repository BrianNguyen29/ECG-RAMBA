"""Audit durable Colab cache/checkpoint/log state in the canonical Drive mirror."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import REVISION_DIR, save_json, sha256_file  # noqa: E402


COLAB_DRIVE_PREFIX = "/content/drive/MyDrive/ECG-Ramba/"


def resolve_declared_path(value: str | Path, canonical_root: Path) -> Path:
    """Resolve manifest paths in Colab and in a locally synced Drive tree."""

    raw = str(value).replace("\\", "/")
    if raw.startswith(COLAB_DRIVE_PREFIX):
        # canonical_root = <drive-root>/revision_artifacts/reports/revision
        drive_root = canonical_root.parents[2]
        return drive_root / raw[len(COLAB_DRIVE_PREFIX) :]
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@dataclass(frozen=True)
class StageStatus:
    stage: str
    pattern: str
    expected_count: int
    found_count: int
    manifest_covered_count: int
    missing_items: str
    unmanifested_items: str
    required_for_full_reviewer_run: bool
    status: str


STRESSES = (
    "snr20db",
    "snr10db",
    "snr5db",
    "random_3_lead_dropout",
    "precordial_dropout",
    "resample_250hz",
)
DEFAULT_OUT_JSON = REVISION_DIR / "metrics" / "pipeline_storage_audit.json"
DEFAULT_OUT_CSV = REVISION_DIR / "tables" / "table_pipeline_storage_audit.csv"


def fold_slots(pattern: str) -> tuple[tuple[str, str], ...]:
    return tuple((f"fold{fold}", pattern.format(fold=fold)) for fold in range(1, 6))


def stress_slots(model: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            stress,
            f"predictions/robustness_{model}_{stress}_predictions.npz",
        )
        for stress in STRESSES
    )


EXTERNAL_REPRESENTATION_MODELS = (
    ("full", "ecg_ramba_full"),
    ("resnet", "resnet1d_cnn"),
    ("raw_mamba", "raw_mamba"),
    ("transformer", "transformer_ecg"),
)
FEWSHOT_SEEDS = (42, 43, 44, 45, 46)
FEWSHOT_FRACTION_TAGS = ("0", "0p01", "0p05", "0p1")
FEWSHOT_METRICS = ("pr_auc_macro", "roc_auc_macro", "f1_macro", "brier_macro", "ece_macro")


def external_representation_slots(tag: str) -> tuple[tuple[str, str], ...]:
    prefix = f"_{tag}" if tag else ""
    return tuple(
        (
            f"{model}_fold{fold}",
            "predictions/external_representation_folds/"
            f"ptbxl{prefix}_{stem}_fold{fold}_record_embeddings.npz",
        )
        for model, stem in EXTERNAL_REPRESENTATION_MODELS
        for fold in range(1, 6)
    )


def true_fewshot_prediction_slots(extension: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            f"{model}_seed{seed}_frac{fraction}",
            "predictions/fewshot_head_adaptation_cache/ptbxl/"
            f"{model}_seed{seed}_frac{fraction}_*.{extension}",
        )
        for model, _stem in EXTERNAL_REPRESENTATION_MODELS
        for seed in FEWSHOT_SEEDS
        for fraction in FEWSHOT_FRACTION_TAGS
    )


def true_fewshot_metric_slots() -> tuple[tuple[str, str], ...]:
    adaptation = tuple(
        (
            f"{model}_seed{seed}_frac{fraction}_{metric}",
            "metrics/true_fewshot_head_metric_cache/ptbxl/"
            f"{model}_seed{seed}_frac{fraction}_{metric}_*.json",
        )
        for model, _stem in EXTERNAL_REPRESENTATION_MODELS
        for seed in FEWSHOT_SEEDS
        for fraction in FEWSHOT_FRACTION_TAGS
        for metric in FEWSHOT_METRICS
    )
    paired = tuple(
        (
            f"full_vs_{model}_seed{seed}_frac{fraction}_{metric}",
            "metrics/true_fewshot_head_metric_cache/ptbxl/"
            f"full_vs_{model}_seed{seed}_frac{fraction}_{metric}_*.json",
        )
        for model in ("resnet", "raw_mamba", "transformer")
        for seed in FEWSHOT_SEEDS
        for fraction in FEWSHOT_FRACTION_TAGS
        for metric in FEWSHOT_METRICS
    )
    return adaptation + paired


def group_safe_calibration_metric_slots() -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            f"seed{seed}_frac{fraction}_{metric}",
            "metrics/group_safe_score_calibration_metric_cache/ptbxl/"
            f"seed{seed}_frac{fraction}_{metric}_*.json",
        )
        for seed in FEWSHOT_SEEDS
        for fraction in FEWSHOT_FRACTION_TAGS
        for metric in FEWSHOT_METRICS
    )


STAGES = (
    (
        "oof_fold_cache",
        "predictions/folds/oof_fold{fold}_final_ema_*.npz",
        fold_slots("predictions/folds/oof_fold{fold}_final_ema_*.npz"),
        True,
    ),
    (
        "resnet_fold_cache",
        "predictions/folds/resnet1d_cnn_fold{fold}_predictions.npz",
        fold_slots("predictions/folds/resnet1d_cnn_fold{fold}_predictions.npz"),
        True,
    ),
    (
        "raw_mamba_fold_cache",
        "predictions/folds/raw_mamba_fold{fold}_predictions.npz",
        fold_slots("predictions/folds/raw_mamba_fold{fold}_predictions.npz"),
        True,
    ),
    (
        "transformer_fold_cache",
        "predictions/folds/transformer_ecg_fold{fold}_predictions.npz",
        fold_slots("predictions/folds/transformer_ecg_fold{fold}_predictions.npz"),
        True,
    ),
    (
        "hybrid_fold_cache",
        "predictions/folds/hybrid_morphology_fold{fold}_predictions.npz",
        fold_slots("predictions/folds/hybrid_morphology_fold{fold}_predictions.npz"),
        True,
    ),
    (
        "morphology_learnability_fold_cache",
        "predictions/folds/morphology_learnability_{variant}_fold{fold}_predictions.npz",
        tuple(
            (
                f"{variant}_fold{fold}",
                f"predictions/folds/morphology_learnability_{variant}_fold{fold}_predictions.npz",
            )
            for variant in ("frozen", "partial")
            for fold in range(1, 6)
        ),
        True,
    ),
    (
        "representation_fold_cache",
        "predictions/folds/representation_final_ema_fold{fold}_*.npz",
        fold_slots("predictions/folds/representation_final_ema_fold{fold}_*.npz"),
        True,
    ),
    (
        "external_ptbxl_test_representation_fold_cache",
        "PTB-XL fold-10 source-bound record representations (4 models x 5 folds)",
        external_representation_slots(""),
        True,
    ),
    (
        "external_ptbxl_fold9_representation_fold_cache",
        "PTB-XL fold-9 source-bound record representations (4 models x 5 folds)",
        external_representation_slots("fold9"),
        True,
    ),
    (
        "external_ptbxl_archive_hash_cache",
        "source-bound PTB-XL archive SHA256 sidecar",
        (("ptbxl", "predictions/external_representation_folds/ptbxl_archive_*_sha256.json"),),
        True,
    ),
    (
        "group_safe_score_calibration_metric_cache",
        "PTB-XL group-safe calibration bootstrap cache (5 seeds x 4 budgets x 5 metrics)",
        group_safe_calibration_metric_slots(),
        True,
    ),
    (
        "true_fewshot_prediction_cache",
        "PTB-XL true-head prediction cache (4 models x 5 seeds x 4 budgets)",
        true_fewshot_prediction_slots("npz"),
        True,
    ),
    (
        "true_fewshot_coefficient_cache",
        "PTB-XL true-head coefficient sidecars (4 models x 5 seeds x 4 budgets)",
        true_fewshot_prediction_slots("coefficients.json"),
        True,
    ),
    (
        "true_fewshot_metric_cache",
        "PTB-XL true-head group-bootstrap cache (adapted and paired metrics)",
        true_fewshot_metric_slots(),
        True,
    ),
    (
        "resnet_checkpoints",
        "experimental/resnet1d_cnn_checkpoints/fold{fold}_resnet1d_cnn_final.pt",
        fold_slots("experimental/resnet1d_cnn_checkpoints/fold{fold}_resnet1d_cnn_final.pt"),
        True,
    ),
    (
        "raw_mamba_checkpoints",
        "experimental/raw_mamba_checkpoints/fold{fold}_raw_mamba_final_ema.pt",
        fold_slots("experimental/raw_mamba_checkpoints/fold{fold}_raw_mamba_final_ema.pt"),
        True,
    ),
    (
        "transformer_checkpoints",
        "experimental/transformer_ecg_checkpoints/fold{fold}_transformer_ecg_final.pt",
        fold_slots("experimental/transformer_ecg_checkpoints/fold{fold}_transformer_ecg_final.pt"),
        True,
    ),
    (
        "hybrid_checkpoints",
        "experimental/hybrid_morphology_checkpoints/fold{fold}_hybrid_morphology_final.pt",
        fold_slots("experimental/hybrid_morphology_checkpoints/fold{fold}_hybrid_morphology_final.pt"),
        True,
    ),
    (
        "morphology_learnability_checkpoints",
        "experimental/morphology_learnability_checkpoints/{variant}/fold{fold}_morphology_learnability_{variant}_final.pt",
        tuple(
            (
                f"{variant}_fold{fold}",
                "experimental/morphology_learnability_checkpoints/"
                f"{variant}/fold{fold}_morphology_learnability_{variant}_final.pt",
            )
            for variant in ("frozen", "partial")
            for fold in range(1, 6)
        ),
        True,
    ),
    ("full_stress_predictions", "six frozen perturbations", stress_slots("full"), True),
    (
        "minirocket_stress_predictions",
        "six frozen perturbations (clean_ref excluded)",
        stress_slots("minirocket"),
        True,
    ),
    (
        "resnet_stress_predictions",
        "six frozen perturbations",
        stress_slots("resnet1d_cnn"),
        True,
    ),
    (
        "raw_mamba_stress_predictions",
        "six frozen perturbations",
        stress_slots("raw_mamba"),
        True,
    ),
    (
        "transformer_stress_predictions",
        "six frozen perturbations",
        stress_slots("transformer_ecg"),
        True,
    ),
)


LEARNED_BASELINE_CONTRACTS = {
    "resnet": {
        "manifest": "manifests/resnet1d_cnn_baseline_manifest.json",
        "predictions": (
            "predictions/resnet1d_cnn_oof_predictions.npz",
            "predictions/resnet1d_cnn_slice_predictions.npz",
        ),
        "checkpoint": "experimental/resnet1d_cnn_checkpoints/fold{fold}_resnet1d_cnn_final.pt",
    },
    "raw_mamba": {
        "manifest": "manifests/raw_mamba_baseline_manifest.json",
        "predictions": (
            "predictions/raw_mamba_oof_predictions.npz",
            "predictions/raw_mamba_slice_predictions.npz",
        ),
        "checkpoint": "experimental/raw_mamba_checkpoints/fold{fold}_raw_mamba_final_ema.pt",
    },
    "transformer": {
        "manifest": "manifests/transformer_ecg_baseline_manifest.json",
        "predictions": (
            "predictions/transformer_ecg_oof_predictions.npz",
            "predictions/transformer_ecg_slice_predictions.npz",
        ),
        "checkpoint": "experimental/transformer_ecg_checkpoints/fold{fold}_transformer_ecg_final.pt",
    },
    "hybrid": {
        "manifest": "manifests/hybrid_morphology_baseline_manifest.json",
        "predictions": ("predictions/hybrid_morphology_oof_predictions.npz",),
        "checkpoint": "experimental/hybrid_morphology_checkpoints/fold{fold}_hybrid_morphology_final.pt",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--full-sha", action="store_true")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_OUT_JSON,
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
    )
    return parser.parse_args()


def load_manifest(root: Path) -> tuple[Path, dict, dict[str, dict]]:
    path = root / "manifests" / "mirror_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Canonical mirror manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {
        Path(row["relative_path"]).as_posix(): row
        for row in payload.get("artifacts", [])
        if row.get("relative_path")
    }
    if not rows:
        raise ValueError(f"Canonical mirror manifest contains no artifacts: {path}")
    return path, payload, rows


def write_csv(path: Path, rows: list[StageStatus]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def audit_stages(root: Path, manifest_rows: dict[str, dict]) -> list[StageStatus]:
    stage_rows: list[StageStatus] = []
    available_files = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.stat().st_size > 0
    }
    for stage, pattern, slots, required in STAGES:
        found_labels: list[str] = []
        covered_labels: list[str] = []
        for label, slot_pattern in slots:
            candidates = sorted(
                relative
                for relative in available_files
                if fnmatch.fnmatchcase(relative, slot_pattern)
            )
            if not candidates:
                continue
            found_labels.append(label)
            if any(relative in manifest_rows for relative in candidates):
                covered_labels.append(label)
        expected_labels = [label for label, _ in slots]
        missing_labels = [label for label in expected_labels if label not in found_labels]
        unmanifested_labels = [
            label for label in found_labels if label not in covered_labels
        ]
        expected = len(slots)
        if len(found_labels) == expected and len(covered_labels) == expected:
            status = "complete_manifested"
        elif len(found_labels) == expected:
            status = "complete_needs_publish"
        elif found_labels:
            status = "partial"
        else:
            status = "absent"
        stage_rows.append(
            StageStatus(
                stage=stage,
                pattern=pattern,
                expected_count=expected,
                found_count=len(found_labels),
                manifest_covered_count=len(covered_labels),
                missing_items=";".join(missing_labels),
                unmanifested_items=";".join(unmanifested_labels),
                required_for_full_reviewer_run=required,
                status=status,
            )
        )
    return stage_rows


def audit_full_model_checkpoints(
    root: Path,
    *,
    full_sha: bool,
) -> tuple[StageStatus, list[str]]:
    manifest_path = root / "manifests" / "oof_final_ema_prediction_run_manifest.json"
    issues: list[str] = []
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        return StageStatus(
            stage="full_ecg_ramba_checkpoints",
            pattern="paths declared by oof_final_ema_prediction_run_manifest.json",
            expected_count=5,
            found_count=0,
            manifest_covered_count=0,
            missing_items="fold1;fold2;fold3;fold4;fold5",
            unmanifested_items="",
            required_for_full_reviewer_run=True,
            status="absent",
        ), ["missing_oof_prediction_run_manifest"]

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("checkpoints") or (payload.get("inputs") or {}).get(
        "checkpoints"
    ) or []
    by_fold = {int(row["fold"]): row for row in records if row.get("fold") is not None}
    found: list[str] = []
    covered: list[str] = []
    for fold in range(1, 6):
        label = f"fold{fold}"
        row = by_fold.get(fold)
        if not row or not row.get("path"):
            issues.append(f"{label}:missing_manifest_record")
            continue
        path = resolve_declared_path(str(row["path"]), root)
        if not path.exists() or not path.is_file() or path.stat().st_size == 0:
            issues.append(f"{label}:missing_checkpoint:{path}")
            continue
        found.append(label)
        expected_sha = str(row.get("sha256") or "")
        if not expected_sha:
            issues.append(f"{label}:missing_sha256")
            continue
        expected_size = row.get("size_bytes")
        if expected_size is not None and int(expected_size) != path.stat().st_size:
            issues.append(f"{label}:size_mismatch")
            continue
        if full_sha and sha256_file(path) != expected_sha:
            issues.append(f"{label}:sha256_mismatch")
            continue
        covered.append(label)

    expected_labels = [f"fold{fold}" for fold in range(1, 6)]
    missing = [label for label in expected_labels if label not in found]
    unverified = [label for label in found if label not in covered]
    if len(found) == 5 and len(covered) == 5:
        status = "complete_manifested"
    elif len(found) == 5:
        status = "complete_needs_publish"
    elif found:
        status = "partial"
    else:
        status = "absent"
    return StageStatus(
        stage="full_ecg_ramba_checkpoints",
        pattern="paths declared by oof_final_ema_prediction_run_manifest.json",
        expected_count=5,
        found_count=len(found),
        manifest_covered_count=len(covered),
        missing_items=";".join(missing),
        unmanifested_items=";".join(unverified),
        required_for_full_reviewer_run=True,
        status=status,
    ), issues


def audit_fold_pca_models(
    root: Path,
    *,
    full_sha: bool,
) -> tuple[StageStatus, list[str]]:
    manifest_path = root / "manifests" / "fold_pca_manifest.json"
    issues: list[str] = []
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        return StageStatus(
            stage="fold_pca_models",
            pattern="paths declared by fold_pca_manifest.json",
            expected_count=5,
            found_count=0,
            manifest_covered_count=0,
            missing_items="fold1;fold2;fold3;fold4;fold5",
            unmanifested_items="",
            required_for_full_reviewer_run=True,
            status="absent",
        ), ["missing_fold_pca_manifest"]

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True or payload.get("checkpoint_kind") != "final_ema":
        issues.append("fold_pca_manifest_not_complete_final_ema")
    rows = payload.get("fold_pca") or []
    by_fold = {
        int(row["fold"]): row
        for row in rows
        if isinstance(row, dict) and row.get("fold") is not None
    }
    found: list[str] = []
    covered: list[str] = []
    for fold in range(1, 6):
        label = f"fold{fold}"
        row = by_fold.get(fold)
        if not row or not row.get("path"):
            issues.append(f"{label}:missing_manifest_record")
            continue
        path = resolve_declared_path(str(row["path"]), root)
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"{label}:missing_pca:{path}")
            continue
        found.append(label)
        expected_size = int(row.get("size_bytes", -1))
        expected_sha = str(row.get("sha256") or "")
        if expected_size < 0 or path.stat().st_size != expected_size:
            issues.append(f"{label}:size_mismatch")
            continue
        if not expected_sha:
            issues.append(f"{label}:missing_sha256")
            continue
        if full_sha and sha256_file(path) != expected_sha:
            issues.append(f"{label}:sha256_mismatch")
            continue
        covered.append(label)

    expected_labels = [f"fold{fold}" for fold in range(1, 6)]
    missing = [label for label in expected_labels if label not in found]
    unverified = [label for label in found if label not in covered]
    if len(found) == 5 and len(covered) == 5 and not issues:
        status = "complete_manifested"
    elif found:
        status = "partial"
    else:
        status = "absent"
    return StageStatus(
        stage="fold_pca_models",
        pattern="paths declared by fold_pca_manifest.json",
        expected_count=5,
        found_count=len(found),
        manifest_covered_count=len(covered),
        missing_items=";".join(missing),
        unmanifested_items=";".join(unverified),
        required_for_full_reviewer_run=True,
        status=status,
    ), issues


def audit_registered_recomputable_caches(root: Path) -> tuple[list[StageStatus], list[str]]:
    """Report exact reusable cache paths recorded by evidence manifests."""

    issues: list[str] = []
    registered: list[tuple[str, list[str], int]] = []
    oof_manifest = root / "manifests" / "oof_final_ema_prediction_run_manifest.json"
    if oof_manifest.is_file() and oof_manifest.stat().st_size > 0:
        try:
            payload = json.loads(oof_manifest.read_text(encoding="utf-8"))
            paths = [
                str(row.get("pca_cache_path") or "")
                for row in payload.get("fold_summaries") or []
                if row.get("pca_cache_path")
            ]
            registered.append(("oof_hydra_feature_cache", paths, 5))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"oof_hydra_feature_cache:{oof_manifest}:{exc}")
            registered.append(("oof_hydra_feature_cache", [], 5))
    else:
        issues.append(f"oof_hydra_feature_cache:missing_manifest:{oof_manifest}")
        registered.append(("oof_hydra_feature_cache", [], 5))

    external_paths: list[str] = []
    external_root = root / "experimental" / "external"
    for manifest_path in sorted(external_root.glob("*/*_full_prediction_run_manifest.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            cache_path = str(payload.get("feature_cache") or "")
            if cache_path:
                external_paths.append(cache_path)
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"external_feature_cache:{manifest_path}:{exc}")
    registered.append(("external_feature_cache", external_paths, len(external_paths)))

    stages: list[StageStatus] = []
    for stage, paths, expected in registered:
        found_labels: list[str] = []
        for index, value in enumerate(paths, start=1):
            path = resolve_declared_path(value, root)
            if path.is_file() and path.stat().st_size > 0:
                found_labels.append(str(index))
            else:
                issues.append(f"{stage}:missing_cache:{path}")
        missing = [str(index) for index in range(1, expected + 1) if str(index) not in found_labels]
        status = (
            "complete_manifested"
            if expected > 0 and len(found_labels) == expected
            else "partial"
            if found_labels
            else "absent"
        )
        stages.append(
            StageStatus(
                stage=stage,
                pattern="exact recomputable cache paths declared by run manifests",
                expected_count=expected,
                found_count=len(found_labels),
                manifest_covered_count=len(found_labels),
                missing_items=";".join(missing),
                unmanifested_items="",
                required_for_full_reviewer_run=False,
                status=status,
            )
        )
    return stages, issues


def audit_learned_prediction_contracts(
    root: Path,
    manifest_rows: dict[str, dict],
    *,
    full_sha: bool,
) -> tuple[StageStatus, list[str]]:
    present_models: list[str] = []
    valid_models: list[str] = []
    issues: list[str] = []
    expected_folds = np.asarray([1, 2, 3, 4, 5], dtype=np.int16)

    for model, config in LEARNED_BASELINE_CONTRACTS.items():
        manifest_path = root / config["manifest"]
        prediction_paths = [root / relative for relative in config["predictions"]]
        checkpoint_paths = [
            root / config["checkpoint"].format(fold=fold) for fold in range(1, 6)
        ]
        required_paths = [manifest_path, *prediction_paths, *checkpoint_paths]
        if not all(path.is_file() and path.stat().st_size > 0 for path in required_paths):
            missing = [
                path.relative_to(root).as_posix()
                for path in required_paths
                if not path.is_file() or path.stat().st_size == 0
            ]
            issues.append(f"{model}:missing={','.join(missing)}")
            continue
        present_models.append(model)

        try:
            manifest_relative = manifest_path.relative_to(root).as_posix()
            if manifest_relative not in manifest_rows:
                raise ValueError(
                    f"baseline manifest is not covered by mirror manifest: {manifest_relative}"
                )
            baseline_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checkpoint_contract = baseline_manifest.get("checkpoint_contract") or {}
            checkpoint_rows = checkpoint_contract.get("checkpoints") or []
            by_fold = {
                int(row["fold"]): row
                for row in checkpoint_rows
                if isinstance(row, dict) and row.get("fold") is not None
            }
            if checkpoint_contract.get("status") != "complete" or sorted(by_fold) != [1, 2, 3, 4, 5]:
                raise ValueError("baseline manifest checkpoint contract is incomplete")

            expected_hashes: list[str] = []
            for fold, checkpoint_path in enumerate(checkpoint_paths, start=1):
                relative = checkpoint_path.relative_to(root).as_posix()
                mirror_row = manifest_rows.get(relative)
                if mirror_row is None:
                    raise ValueError(f"checkpoint is not covered by mirror manifest: {relative}")
                mirror_sha = str(mirror_row.get("sha256") or "")
                baseline_sha = str(by_fold[fold].get("sha256") or "")
                if not mirror_sha or baseline_sha != mirror_sha:
                    raise ValueError(f"fold{fold} checkpoint SHA differs between baseline and mirror manifests")
                if full_sha and sha256_file(checkpoint_path) != mirror_sha:
                    raise ValueError(f"fold{fold} checkpoint file SHA mismatch")
                expected_hashes.append(mirror_sha)

            expected_hashes_array = np.asarray(expected_hashes)
            for prediction_path in prediction_paths:
                relative = prediction_path.relative_to(root).as_posix()
                if relative not in manifest_rows:
                    raise ValueError(f"prediction is not covered by mirror manifest: {relative}")
                with np.load(prediction_path, allow_pickle=False) as data:
                    required = {"checkpoint_folds", "checkpoint_sha256"}
                    missing = required - set(data.files)
                    if missing:
                        raise ValueError(
                            f"{relative} lacks checkpoint provenance keys {sorted(missing)}"
                        )
                    if not np.array_equal(
                        np.asarray(data["checkpoint_folds"], dtype=np.int16),
                        expected_folds,
                    ):
                        raise ValueError(f"{relative} checkpoint fold order mismatch")
                    if not np.array_equal(
                        np.asarray(data["checkpoint_sha256"]).astype(str),
                        expected_hashes_array,
                    ):
                        raise ValueError(f"{relative} checkpoint SHA contract mismatch")
            valid_models.append(model)
        except Exception as exc:
            issues.append(f"{model}:{exc}")

    expected_count = len(LEARNED_BASELINE_CONTRACTS)
    if len(valid_models) == expected_count:
        status = "complete_manifested"
    elif present_models:
        status = "partial"
    else:
        status = "absent"
    missing_models = [model for model in LEARNED_BASELINE_CONTRACTS if model not in present_models]
    invalid_models = [model for model in present_models if model not in valid_models]
    return StageStatus(
        stage="learned_prediction_checkpoint_contracts",
        pattern="checkpoint SHA agreement across canonical files, baseline manifests, and prediction NPZs",
        expected_count=expected_count,
        found_count=len(present_models),
        manifest_covered_count=len(valid_models),
        missing_items=";".join(missing_models),
        unmanifested_items=";".join(invalid_models),
        required_for_full_reviewer_run=True,
        status=status,
    ), issues


def main() -> None:
    args = parse_args()
    root = args.canonical_root.expanduser().resolve()
    out_json = (
        root / "metrics" / "pipeline_storage_audit.json"
        if args.out_json == DEFAULT_OUT_JSON
        else args.out_json.expanduser().resolve()
    )
    out_csv = (
        root / "tables" / "table_pipeline_storage_audit.csv"
        if args.out_csv == DEFAULT_OUT_CSV
        else args.out_csv.expanduser().resolve()
    )
    manifest_path, manifest, manifest_rows = load_manifest(root)

    authority_path = root / "manifests" / "notebook_code_authority.json"
    authority_issues: list[str] = []
    authority: dict = {}
    try:
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
        current_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip().lower()
        expected_head = str(authority.get("git_commit", "")).strip().lower()
        if authority.get("capability") != "canonical_git_commit_pin_v1":
            authority_issues.append("code_authority:capability")
        if int(authority.get("schema_version", 0)) != 1:
            authority_issues.append("code_authority:schema")
        if len(expected_head) != 40 or expected_head != current_head:
            authority_issues.append(
                f"code_authority:git_commit expected={expected_head or 'missing'} observed={current_head}"
            )
    except Exception as exc:
        authority_issues.append(f"code_authority:{type(exc).__name__}:{exc}")

    missing_manifest_files: list[str] = []
    invalid_manifest_files: list[str] = []
    for relative, row in manifest_rows.items():
        path = root / relative
        if not path.exists() or not path.is_file():
            missing_manifest_files.append(relative)
            continue
        expected_size = int(row.get("size_bytes", -1))
        if expected_size >= 0 and path.stat().st_size != expected_size:
            invalid_manifest_files.append(f"{relative}:size")
            continue
        if args.full_sha and sha256_file(path) != str(row.get("sha256", "")):
            invalid_manifest_files.append(f"{relative}:sha256")

    stage_rows = audit_stages(root, manifest_rows)
    full_model_stage, full_model_issues = audit_full_model_checkpoints(
        root,
        full_sha=bool(args.full_sha),
    )
    stage_rows.insert(0, full_model_stage)
    invalid_manifest_files.extend(full_model_issues)
    fold_pca_stage, fold_pca_issues = audit_fold_pca_models(
        root,
        full_sha=bool(args.full_sha),
    )
    stage_rows.insert(1, fold_pca_stage)
    invalid_manifest_files.extend(fold_pca_issues)
    learned_contract_stage, learned_contract_issues = audit_learned_prediction_contracts(
        root,
        manifest_rows,
        full_sha=bool(args.full_sha),
    )
    stage_rows.insert(2, learned_contract_stage)
    invalid_manifest_files.extend(learned_contract_issues)
    recomputable_cache_stages, recomputable_cache_issues = audit_registered_recomputable_caches(root)
    stage_rows.extend(recomputable_cache_stages)

    log_count = sum(1 for path in (root / "logs").rglob("*.log") if path.is_file())
    incomplete_required = [
        row.stage
        for row in stage_rows
        if row.required_for_full_reviewer_run and row.status != "complete_manifested"
    ]
    payload = {
        "status": not missing_manifest_files
        and not invalid_manifest_files
        and not incomplete_required
        and not authority_issues,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_root": str(root),
        "canonical_is_authoritative": True,
        "mirror_manifest": str(manifest_path),
        "mirror_manifest_sha256": sha256_file(manifest_path),
        "mirror_manifest_artifact_count": int(manifest.get("artifact_count", len(manifest_rows))),
        "code_authority": {
            "path": str(authority_path),
            "sha256": sha256_file(authority_path) if authority_path.is_file() else None,
            "git_commit": authority.get("git_commit"),
            "issues": authority_issues,
        },
        "verification": "size_and_presence" if not args.full_sha else "full_sha256",
        "logs_are_durable_but_not_manifested": True,
        "durable_log_count": log_count,
        "storage_tiers": {
            "canonical_evidence_mirror": str(root),
            "full_model_checkpoints": "exact paths frozen by oof_final_ema_prediction_run_manifest.json",
            "fold_pca_models": "exact paths and SHA256 frozen by fold_pca_manifest.json",
            "recomputable_feature_caches": "exact paths registered by OOF/external run manifests",
            "legacy_drive_checkout": "audit_only_never_automatic_input",
            "final_evidence_tables": "output_only_convenience_snapshot",
        },
        "recomputable_cache_issues": recomputable_cache_issues,
        "missing_manifest_files": missing_manifest_files,
        "invalid_manifest_files": invalid_manifest_files,
        "authority_issues": authority_issues,
        "incomplete_required_stages": incomplete_required,
        "stages": [asdict(row) for row in stage_rows],
    }
    save_json(out_json, payload)
    write_csv(out_csv, stage_rows)

    print(json.dumps({
        "status": payload["status"],
        "manifest_artifacts": payload["mirror_manifest_artifact_count"],
        "durable_logs": log_count,
        "incomplete_required_stages": incomplete_required,
        "missing_manifest_files": len(missing_manifest_files),
        "invalid_manifest_files": len(invalid_manifest_files),
        "authority_issues": authority_issues,
    }, indent=2))
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")
    if args.strict and not payload["status"]:
        raise RuntimeError("Canonical pipeline storage audit is incomplete; inspect the output table.")


if __name__ == "__main__":
    main()
