"""Generate zero-target-label external predictions for learned comparators.

The comparator models are trained only on the frozen Chapman folds. This
runner applies their five saved fold models to the same mapped external tasks
and preprocessing used by ``03_generate_external_predictions.py``. Each
dataset/comparator/fold is cached immediately so interrupted Colab sessions can
resume without repeating completed inference.

PTB-XL and Georgia are record-level mapped tasks. CPSC2021 remains a separate
annotation-aligned 10-second AF/AFL-window task and is never pooled with the
record-level datasets.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CLASSES, CONFIG  # noqa: E402
from scripts.revision.common import (  # noqa: E402
    EXPERIMENTAL_DIR,
    MANIFEST_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    TABLE_DIR,
    calibration_summary,
    ensure_revision_dirs,
    git_commit,
    multilabel_metrics,
    save_csv,
    save_json,
    save_npz_compressed_atomic,
    sha256_file,
)
from src.aggregation import POWER_MEAN_IMPLEMENTATION, aggregate_record_probabilities  # noqa: E402
from src.training_data import build_slice_index  # noqa: E402


PROTOCOL_VERSION = 2
CACHE_ONLY_CPU_AGGREGATION_CAPABILITY = "validated_external_fold_cache_aggregation_v2_dataset_sidecar"
CPSC_DISK_BACKED_INFERENCE_CAPABILITY = "cpsc_disk_backed_comparator_inference_v1"
DATASET_CONTRACT_SCHEMA_VERSION = 1
COMPARATOR_STEMS = {
    "resnet": "resnet1d_cnn",
    "raw_mamba": "raw_mamba",
    "transformer": "transformer_ecg",
}
COMPARATOR_LABELS = {
    "resnet": "ResNet1D/CNN",
    "raw_mamba": "Raw Mamba",
    "transformer": "Transformer ECG",
}
CHECKPOINT_FILENAMES = {
    "resnet": "fold{fold}_resnet1d_cnn_final.pt",
    "raw_mamba": "fold{fold}_raw_mamba_final_ema.pt",
    "transformer": "fold{fold}_transformer_ecg_final.pt",
}
SUMMARY_FILENAMES = {
    "resnet": "resnet1d_cnn_baseline_summary.json",
    "raw_mamba": "raw_mamba_baseline_summary.json",
    "transformer": "transformer_ecg_baseline_summary.json",
}
MANIFEST_FILENAMES = {
    "resnet": "resnet1d_cnn_baseline_manifest.json",
    "raw_mamba": "raw_mamba_baseline_manifest.json",
    "transformer": "transformer_ecg_baseline_manifest.json",
}
PAIRED_FILENAMES = {
    "resnet": "paired_full_vs_resnet_comparison.json",
    "raw_mamba": "paired_full_vs_raw_mamba_comparison.json",
    "transformer": "paired_full_vs_transformer_comparison.json",
}


def load_revision_module(filename: str, module_name: str):
    path = PROJECT_ROOT / "scripts" / "revision" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


external_helpers = load_revision_module(
    "03_generate_external_predictions.py", "_external_comparator_dataset_helpers"
)
baseline_helpers = load_revision_module(
    "14_resnet1d_cnn_baseline.py", "_external_comparator_resnet_helpers"
)
raw_mamba_helpers = load_revision_module(
    "16_raw_mamba_baseline.py", "_external_comparator_raw_mamba_helpers"
)
model_loaders = load_revision_module(
    "23_generate_comparator_stress_predictions.py", "_external_comparator_model_loaders"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        choices=["ptbxl", "georgia", "cpsc2021", "all"],
        help="Repeat for multiple datasets. Defaults to PTB-XL.",
    )
    parser.add_argument("--comparators", default="resnet,raw_mamba")
    parser.add_argument("--ptbxl-folds", default="10")
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only-folds", default="")
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-experimental", action="store_true")
    parser.add_argument(
        "--extract-root",
        type=Path,
        default=Path("/content/ecg_ramba_runtime/external"),
    )
    parser.add_argument(
        "--georgia-mapping-review",
        type=Path,
        default=PROJECT_ROOT
        / "docs"
        / "revision_plan"
        / "georgia_label_mapping_review_20260703.csv",
    )
    parser.add_argument(
        "--georgia-code-inventory-out",
        type=Path,
        default=TABLE_DIR / "table_georgia_snomed_code_inventory.csv",
    )
    parser.add_argument(
        "--cpsc-annotation-audit-out",
        type=Path,
        default=TABLE_DIR / "table_cpsc2021_annotation_audit.csv",
    )
    parser.add_argument(
        "--resnet-checkpoint-dir",
        type=Path,
        default=EXPERIMENTAL_DIR / "resnet1d_cnn_checkpoints",
    )
    parser.add_argument(
        "--raw-mamba-checkpoint-dir",
        type=Path,
        default=EXPERIMENTAL_DIR / "raw_mamba_checkpoints",
    )
    parser.add_argument(
        "--transformer-checkpoint-dir",
        type=Path,
        default=EXPERIMENTAL_DIR / "transformer_ecg_checkpoints",
    )
    parser.add_argument(
        "--fold-cache-dir",
        type=Path,
        default=PREDICTION_DIR / "external_comparator_folds",
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=EXPERIMENTAL_DIR / "external",
    )
    parser.add_argument(
        "--cpsc-signal-memmap",
        type=Path,
        default=None,
        help=(
            "Local .npy path for disk-backed CPSC2021 float32 windows. Defaults under --extract-root; "
            "the file is intentionally ephemeral while fold prediction caches remain durable."
        ),
    )
    parser.add_argument(
        "--oof-predictions",
        type=Path,
        default=PREDICTION_DIR / "oof_final_ema_predictions.npz",
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=MANIFEST_DIR / "oof_final_ema_freeze_manifest.json",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_only_folds(value: str) -> set[int]:
    folds = {int(item) for item in parse_list(value)}
    invalid = sorted(fold for fold in folds if fold < 1 or fold > 5)
    if invalid:
        raise ValueError(f"Invalid --only-folds values: {invalid}")
    return folds


def selected_datasets(values: list[str] | None) -> list[str]:
    items = values or ["ptbxl"]
    if "all" in items:
        return ["ptbxl", "georgia", "cpsc2021"]
    return list(dict.fromkeys(items))


def record_fingerprint(record_ids: np.ndarray, groups: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(np.asarray(record_ids).astype(str)).encode())
    digest.update(b"\0")
    digest.update("\n".join(np.asarray(groups).astype(str)).encode())
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(labels.astype(np.float32)).tobytes())
    return digest.hexdigest()


def stable_json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def canonical_contract(args: argparse.Namespace) -> dict[str, str]:
    oof = resolve(args.oof_predictions)
    freeze = resolve(args.freeze_manifest)
    if not oof.exists() or not freeze.exists():
        raise FileNotFoundError("Canonical OOF/freeze artifacts are required before external comparator inference.")
    payload = read_json(freeze)
    if payload.get("status") != "frozen" or payload.get("manuscript_ready") is not True:
        raise RuntimeError("Freeze manifest is not frozen/manuscript_ready.")
    oof_sha = sha256_file(oof)
    expected = None
    for row in payload.get("artifacts", []):
        if str(row.get("path", "")).replace("\\", "/").endswith(oof.name):
            expected = row.get("sha256")
            break
    if expected != oof_sha:
        raise RuntimeError(f"Freeze OOF SHA mismatch: {expected} != {oof_sha}")
    return {"oof_sha256": oof_sha, "freeze_sha256": sha256_file(freeze)}


def validate_in_domain_comparator(
    comparator: str,
    contract: dict[str, str],
) -> dict[str, Any]:
    summary_path = METRIC_DIR / SUMMARY_FILENAMES[comparator]
    manifest_path = MANIFEST_DIR / MANIFEST_FILENAMES[comparator]
    paired_path = METRIC_DIR / PAIRED_FILENAMES[comparator]
    missing = [path for path in (summary_path, manifest_path, paired_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{COMPARATOR_LABELS[comparator]} external inference requires completed in-domain "
            "OOF and paired artifacts: " + "; ".join(str(path) for path in missing)
        )
    summary = read_json(summary_path)
    paired = read_json(paired_path)
    if summary.get("manuscript_ready") is not True:
        raise RuntimeError(f"{summary_path} is not manuscript_ready=true")
    inputs = paired.get("inputs") or {}
    full_sha = (inputs.get("full_predictions") or {}).get("sha256")
    freeze_sha = (inputs.get("freeze_manifest") or {}).get("sha256")
    if full_sha != contract["oof_sha256"] or freeze_sha != contract["freeze_sha256"]:
        raise RuntimeError(
            f"{COMPARATOR_LABELS[comparator]} paired artifact is stale for canonical OOF/freeze."
        )
    return {
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "paired": str(paired_path),
        "paired_sha256": sha256_file(paired_path),
    }


def checkpoint_dir(args: argparse.Namespace, comparator: str) -> Path:
    return resolve(
        {
            "resnet": args.resnet_checkpoint_dir,
            "raw_mamba": args.raw_mamba_checkpoint_dir,
            "transformer": args.transformer_checkpoint_dir,
        }[comparator]
    )


def checkpoint_path(args: argparse.Namespace, comparator: str, fold: int) -> Path:
    return checkpoint_dir(args, comparator) / CHECKPOINT_FILENAMES[comparator].format(fold=fold)


def load_model(args: argparse.Namespace, comparator: str, path: Path, device: torch.device):
    if comparator == "resnet":
        return model_loaders.load_resnet_model(args, path, device)
    if comparator == "raw_mamba":
        return model_loaders.load_raw_mamba_model(path, device)
    if comparator == "transformer":
        return model_loaders.load_transformer_model(path, device)
    raise ValueError(comparator)


def output_tag(args: argparse.Namespace, dataset: str) -> str:
    tag = str(args.output_tag).strip().replace(" ", "_")
    if dataset == "ptbxl":
        folds = external_helpers.parse_ptbxl_folds(args.ptbxl_folds)
        if folds != (10,) and not tag:
            tag = "folds" + "_".join(str(value) for value in folds)
    return tag


def final_output_paths(args: argparse.Namespace, dataset: str, comparator: str) -> dict[str, Path]:
    tag = output_tag(args, dataset)
    suffix = f"_{tag}" if tag else ""
    stem = f"{dataset}_{COMPARATOR_STEMS[comparator]}{suffix}"
    output_root = resolve(args.external_root) / dataset
    return {
        "predictions": output_root / f"{stem}_predictions.npz",
        "slice_predictions": output_root / f"{stem}_slice_predictions.npz",
        "summary": METRIC_DIR / f"external_{stem}_summary.json",
        "class_table": TABLE_DIR / f"table_external_{stem}_class_metrics.csv",
        "manifest": MANIFEST_DIR / f"external_{stem}_manifest.json",
    }


def full_model_source_paths(args: argparse.Namespace, dataset: str) -> dict[str, Path]:
    tag = output_tag(args, dataset)
    suffix = f"_{tag}" if tag else ""
    stem = f"{dataset}_full{suffix}"
    output_root = resolve(args.external_root) / dataset
    return {
        "predictions": output_root / f"{stem}_predictions.npz",
        "slice_predictions": output_root / f"{stem}_slice_predictions.npz",
        "summary": output_root / f"{stem}_prediction_summary.json",
        "manifest": output_root / f"{stem}_prediction_run_manifest.json",
    }


def dataset_contract_path(args: argparse.Namespace, dataset: str) -> Path:
    tag = output_tag(args, dataset)
    suffix = f"_{tag}" if tag else ""
    return resolve(args.fold_cache_dir) / f"{dataset}{suffix}_dataset_contract_v1.npz"


def expected_class_names(dataset: str) -> np.ndarray:
    if dataset == "ptbxl":
        return np.asarray(list(external_helpers.PTB_SUPERCLASS_MAPPING))
    if dataset == "cpsc2021":
        return np.asarray(["AF_or_AFL"])
    return np.asarray(CLASSES)


def dataset_loader_contract(sources: dict[str, Any]) -> dict[str, Any]:
    """Return the semantic input contract without the aggregation runner identity."""

    return {
        "archive_sha256": sources["archive_sha256"],
        "external_loader_sha256": sources["external_loader_sha256"],
        "loader_configuration": sources["loader_configuration"],
        "georgia_mapping_review": sources.get("georgia_mapping_review"),
        "slice_configuration": {
            "slice_length": int(CONFIG["slice_length"]),
            "slice_stride": int(CONFIG["slice_stride"]),
            "max_slices_per_record": int(CONFIG["max_slices_per_record"]),
        },
    }


def _manifest_output_sha(manifest: dict[str, Any], path: Path) -> str | None:
    row = (manifest.get("outputs") or {}).get(path.name) or {}
    return row.get("sha256")


def build_dataset_contract_from_full_artifact(
    args: argparse.Namespace,
    dataset: str,
    sources: dict[str, Any],
    canonical: dict[str, str],
) -> Path:
    """Create a small durable sidecar from the authenticated Full external export.

    This sidecar deliberately contains no ECG signal. It is sufficient to
    authenticate fold-cache record/slice order and to aggregate completed fold
    predictions on a fresh CPU runtime without loading CPSC2021 again.
    """

    paths = full_model_source_paths(args, dataset)
    missing = [path for path in paths.values() if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(
            f"{dataset}: current source-bound Full external export is required before comparator cache reuse: "
            + "; ".join(str(path) for path in missing)
        )
    manifest = read_json(paths["manifest"])
    expected_runner = sha256_file(PROJECT_ROOT / "scripts" / "revision" / "03_generate_external_predictions.py")
    if manifest.get("runner_sha256") != expected_runner:
        raise RuntimeError(f"{dataset}: Full external manifest was produced by a stale loader runner")
    if manifest.get("canonical_contract") != canonical:
        raise RuntimeError(f"{dataset}: Full external manifest is stale for the canonical OOF/freeze")
    if (manifest.get("archive") or {}).get("sha256") != sources["archive_sha256"]:
        raise RuntimeError(f"{dataset}: Full external manifest archive SHA does not match the active archive")
    for name in ("predictions", "slice_predictions", "summary"):
        observed = sha256_file(paths[name])
        if _manifest_output_sha(manifest, paths[name]) != observed:
            raise RuntimeError(f"{dataset}: Full external {name} failed manifest SHA validation")

    with np.load(paths["predictions"], allow_pickle=False) as pred:
        required = {"y_true", "record_id", "group_id", "split_id", "class_names", "dataset"}
        if required - set(pred.files):
            raise RuntimeError(f"{dataset}: Full external prediction contract is incomplete")
        y_true = np.asarray(pred["y_true"], dtype=np.float32)
        record_ids = np.asarray(pred["record_id"]).astype(str)
        group_ids = np.asarray(pred["group_id"]).astype(str)
        split_ids = np.asarray(pred["split_id"]).astype(str)
        class_names = np.asarray(pred["class_names"]).astype(str)
        if str(np.asarray(pred["dataset"]).item()) != dataset:
            raise RuntimeError(f"{dataset}: Full external prediction dataset identity mismatch")
    n_records = len(record_ids)
    if y_true.shape != (n_records, len(class_names)):
        raise RuntimeError(f"{dataset}: Full external label shape is invalid: {y_true.shape}")
    if len(group_ids) != n_records or len(split_ids) != n_records:
        raise RuntimeError(f"{dataset}: Full external record metadata lengths differ")
    if len(set(record_ids.tolist())) != n_records:
        raise RuntimeError(f"{dataset}: Full external record identifiers are not unique")
    if not np.isfinite(y_true).all() or not np.isin(y_true, [0.0, 1.0]).all():
        raise RuntimeError(f"{dataset}: Full external labels must be finite binary values")
    if not np.array_equal(class_names, expected_class_names(dataset).astype(str)):
        raise RuntimeError(f"{dataset}: Full external class order is not canonical")

    with np.load(paths["slice_predictions"], allow_pickle=False) as sliced:
        required = {"record_index", "slice_index", "record_id", "group_id", "split_id", "class_names"}
        if required - set(sliced.files):
            raise RuntimeError(f"{dataset}: Full external slice contract is incomplete")
        slice_record_index = np.asarray(sliced["record_index"], dtype=np.int64)
        slice_ordinal = np.asarray(sliced["slice_index"], dtype=np.int64)
        slice_classes = np.asarray(sliced["class_names"]).astype(str)
        slice_record_ids = np.asarray(sliced["record_id"]).astype(str)
        slice_group_ids = np.asarray(sliced["group_id"]).astype(str)
        slice_split_ids = np.asarray(sliced["split_id"]).astype(str)
    if slice_record_index.ndim != 1 or len(slice_record_index) == 0:
        raise RuntimeError(f"{dataset}: Full external slice index is empty or invalid")
    if np.any(slice_record_index < 0) or np.any(slice_record_index >= n_records):
        raise RuntimeError(f"{dataset}: Full external slice record index is out of bounds")
    if not np.array_equal(slice_classes, class_names):
        raise RuntimeError(f"{dataset}: Full external slice class order differs from record output")
    if not np.array_equal(slice_record_ids, record_ids[slice_record_index]):
        raise RuntimeError(f"{dataset}: Full external slice-to-record identity mapping changed")
    if not np.array_equal(slice_group_ids, group_ids[slice_record_index]):
        raise RuntimeError(f"{dataset}: Full external slice-to-group mapping changed")
    if not np.array_equal(slice_split_ids, split_ids[slice_record_index]):
        raise RuntimeError(f"{dataset}: Full external slice-to-split mapping changed")
    slice_start = (slice_ordinal * int(CONFIG["slice_stride"])).astype(np.int32)
    load_summary = manifest.get("load_summary") or read_json(paths["summary"]).get("load_summary") or {}
    loader_contract = dataset_loader_contract(sources)
    source_artifacts = {
        name: {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for name, path in paths.items()
    }
    destination = dataset_contract_path(args, dataset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_npz_compressed_atomic(
        destination,
        schema_version=np.asarray(DATASET_CONTRACT_SCHEMA_VERSION, dtype=np.int16),
        dataset=np.asarray(dataset),
        y_true=y_true,
        record_id=record_ids,
        group_id=group_ids,
        split_id=split_ids,
        class_names=class_names,
        slice_record_index=slice_record_index,
        slice_start=slice_start,
        input_fingerprint=np.asarray(record_fingerprint(record_ids, group_ids, y_true)),
        loader_contract_json=np.asarray(json.dumps(loader_contract, sort_keys=True)),
        loader_contract_sha256=np.asarray(stable_json_sha256(loader_contract)),
        source_artifacts_json=np.asarray(json.dumps(source_artifacts, sort_keys=True)),
        load_summary_json=np.asarray(json.dumps(load_summary, sort_keys=True)),
        created_utc=np.asarray(now_utc()),
    )
    print(f"Wrote authenticated dataset sidecar: {destination}", flush=True)
    return destination


def load_dataset_contract(
    args: argparse.Namespace,
    dataset: str,
    sources: dict[str, Any],
    canonical: dict[str, str],
) -> dict[str, Any]:
    path = dataset_contract_path(args, dataset)
    loader_contract = dataset_loader_contract(sources)
    expected_loader_sha = stable_json_sha256(loader_contract)

    def read_valid() -> dict[str, Any] | None:
        if not path.exists() or path.stat().st_size == 0:
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                required = {
                    "schema_version", "dataset", "y_true", "record_id", "group_id", "split_id",
                    "class_names", "slice_record_index", "slice_start", "input_fingerprint",
                    "loader_contract_sha256", "source_artifacts_json", "load_summary_json",
                }
                if required - set(data.files):
                    return None
                if int(data["schema_version"].item()) != DATASET_CONTRACT_SCHEMA_VERSION:
                    return None
                if str(data["dataset"].item()) != dataset:
                    return None
                if str(data["loader_contract_sha256"].item()) != expected_loader_sha:
                    return None
                payload = {name: np.asarray(data[name]) for name in (
                    "y_true", "record_id", "group_id", "split_id", "class_names",
                    "slice_record_index", "slice_start",
                )}
                payload["input_fingerprint"] = str(data["input_fingerprint"].item())
                payload["source_artifacts"] = json.loads(str(data["source_artifacts_json"].item()))
                payload["load_summary"] = json.loads(str(data["load_summary_json"].item()))
        except Exception:
            return None
        current_paths = full_model_source_paths(args, dataset)
        for name, row in payload["source_artifacts"].items():
            current = current_paths.get(name)
            if current is None or not current.exists() or sha256_file(current) != row.get("sha256"):
                return None
        y_true = np.asarray(payload["y_true"], dtype=np.float32)
        record_ids = np.asarray(payload["record_id"]).astype(str)
        group_ids = np.asarray(payload["group_id"]).astype(str)
        if payload["input_fingerprint"] != record_fingerprint(record_ids, group_ids, y_true):
            return None
        payload["dataset_contract_path"] = path
        payload["dataset_contract_sha256"] = sha256_file(path)
        return payload

    loaded = read_valid()
    if loaded is None:
        build_dataset_contract_from_full_artifact(args, dataset, sources, canonical)
        loaded = read_valid()
    if loaded is None:
        raise RuntimeError(f"{dataset}: failed to build or authenticate the comparator dataset sidecar")
    return loaded


def source_contract(args: argparse.Namespace, dataset: str, archive: Path) -> dict[str, Any]:
    mapping_path = resolve(args.georgia_mapping_review) if dataset == "georgia" else None
    loader_path = PROJECT_ROOT / "scripts" / "revision" / "03_generate_external_predictions.py"
    return {
        "archive_sha256": sha256_file(archive),
        "external_loader_sha256": sha256_file(loader_path),
        "loader_configuration": {
            "ptbxl_folds": list(external_helpers.parse_ptbxl_folds(args.ptbxl_folds))
            if dataset == "ptbxl"
            else None,
            "output_tag": output_tag(args, dataset),
            "limit_records": int(args.limit_records),
            "cpsc_signal_storage": (
                "disk_backed_float32_npy"
                if dataset == "cpsc2021"
                else None
            ),
        },
        "georgia_mapping_review": (
            {"path": str(mapping_path), "sha256": sha256_file(mapping_path)}
            if mapping_path is not None and mapping_path.exists()
            else None
        ),
        "runner_sha256": sha256_file(Path(__file__)),
    }


def final_artifacts_reusable(
    args: argparse.Namespace,
    dataset: str,
    comparator: str,
    contract: dict[str, str],
    checkpoint_hashes: list[str],
    sources: dict[str, Any],
) -> bool:
    """Return true only for a complete external artifact with current provenance."""

    if not args.reuse_existing or args.force_rerun or parse_only_folds(args.only_folds):
        return False
    paths = final_output_paths(args, dataset, comparator)
    if not all(path.exists() and path.stat().st_size > 0 for path in paths.values()):
        return False
    try:
        manifest = read_json(paths["manifest"])
        if manifest.get("status") != "complete_experimental_requires_external_comparator_gate":
            return False
        if manifest.get("canonical_contract") != contract:
            return False
        if manifest.get("dataset") != dataset or manifest.get("comparator") != comparator:
            return False
        if manifest.get("source_contract") != sources:
            return False
        checkpoints = manifest.get("checkpoints") or []
        if [row.get("sha256") for row in checkpoints] != checkpoint_hashes:
            return False
        artifacts = manifest.get("artifacts") or {}
        for name in ("predictions", "slice_predictions", "summary", "class_table"):
            if (artifacts.get(name) or {}).get("sha256") != sha256_file(paths[name]):
                return False
        with np.load(paths["predictions"], allow_pickle=False) as data:
            if str(np.asarray(data["dataset"]).item()) != dataset or str(
                np.asarray(data["comparator"]).item()
            ) != comparator:
                return False
            if int(np.asarray(data["adaptation_labels_used"]).item()) != 0:
                return False
            if [str(item) for item in np.asarray(data["checkpoint_sha256"]).tolist()] != checkpoint_hashes:
                return False
        return True
    except Exception:
        return False


def cache_path(
    args: argparse.Namespace,
    dataset: str,
    comparator: str,
    fold: int,
) -> Path:
    tag = output_tag(args, dataset)
    suffix = f"_{tag}" if tag else ""
    return resolve(args.fold_cache_dir) / (
        f"{dataset}{suffix}_{COMPARATOR_STEMS[comparator]}_fold{fold}_slice_predictions.npz"
    )


def cache_matches(
    path: Path,
    *,
    dataset: str,
    comparator: str,
    fold: int,
    checkpoint_sha: str,
    input_fingerprint: str,
    class_names: np.ndarray,
    dataset_contract_sha256: str,
    expected_record_index: np.ndarray,
    expected_starts: np.ndarray,
) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "slice_prob",
                "slice_record_index",
                "slice_start",
                "class_names",
                "dataset",
                "comparator",
                "fold",
                "checkpoint_sha256",
                "input_fingerprint",
                "dataset_contract_sha256",
                "protocol_version",
            }
            if required - set(data.files):
                return False
            return bool(
                str(data["dataset"].item()) == dataset
                and str(data["comparator"].item()) == comparator
                and int(data["fold"].item()) == fold
                and str(data["checkpoint_sha256"].item()) == checkpoint_sha
                and str(data["input_fingerprint"].item()) == input_fingerprint
                and str(data["dataset_contract_sha256"].item()) == dataset_contract_sha256
                and int(data["protocol_version"].item()) == PROTOCOL_VERSION
                and np.array_equal(np.asarray(data["class_names"]).astype(str), class_names.astype(str))
                and np.array_equal(
                    np.asarray(data["slice_record_index"], dtype=np.int64),
                    np.asarray(expected_record_index, dtype=np.int64),
                )
                and np.array_equal(
                    np.asarray(data["slice_start"], dtype=np.int32),
                    np.asarray(expected_starts, dtype=np.int32),
                )
                and np.asarray(data["slice_prob"]).shape
                == (len(expected_record_index), len(class_names))
                and np.isfinite(np.asarray(data["slice_prob"], dtype=np.float32)).all()
                and np.all(np.asarray(data["slice_prob"], dtype=np.float32) >= 0.0)
                and np.all(np.asarray(data["slice_prob"], dtype=np.float32) <= 1.0)
            )
    except Exception:
        return False


def infer_fold(
    *,
    comparator: str,
    model: torch.nn.Module,
    loader,
    dataset: str,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if comparator in {"resnet", "transformer"}:
        raw_prob, record_index, starts = baseline_helpers.predict_slice_probabilities(
            model,
            loader,
            device=device,
            use_amp=use_amp,
        )
    else:
        raw_args = SimpleNamespace(
            amp=use_amp,
            amp_dtype=(
                "bfloat16"
                if device.type == "cuda" and torch.cuda.is_bf16_supported()
                else "float16"
            ),
        )
        raw_prob, record_index, starts = raw_mamba_helpers.predict_slice_probabilities(
            model,
            loader,
            device=device,
            args=raw_args,
        )
    mapped, class_names = external_helpers.map_model_probabilities(dataset, raw_prob)
    return mapped, np.asarray(record_index, dtype=np.int64), np.asarray(starts, dtype=np.int32), class_names


def class_rows(dataset: str, comparator: str, y_true: np.ndarray, y_prob: np.ndarray, names: np.ndarray) -> list[dict]:
    rows = external_helpers.class_summary(dataset, y_true, y_prob, names)
    for row in rows:
        row["comparator"] = COMPARATOR_LABELS[comparator]
    return rows


def external_loader_args(args: argparse.Namespace, sources: dict[str, Any]) -> SimpleNamespace:
    cpsc_signal_memmap = args.cpsc_signal_memmap
    if cpsc_signal_memmap is None:
        cpsc_signal_memmap = (
            resolve(args.fold_cache_dir)
            / "cpsc2021_preprocessed_windows_source_bound_v2.npy"
        )
    return SimpleNamespace(
        ptbxl_folds=args.ptbxl_folds,
        georgia_mapping_review=resolve(args.georgia_mapping_review),
        georgia_code_inventory_out=resolve(args.georgia_code_inventory_out),
        cpsc_annotation_audit_out=resolve(args.cpsc_annotation_audit_out),
        cpsc_signal_memmap=resolve(cpsc_signal_memmap),
        source_archive_sha256=sources["archive_sha256"],
    )


def run_dataset(
    args: argparse.Namespace,
    dataset: str,
    comparators: list[str],
    selected_folds: set[int],
    contract: dict[str, str],
    device: torch.device,
) -> list[dict[str, Any]]:
    archive = external_helpers.archive_path(dataset)
    sources = source_contract(args, dataset, archive)
    result_rows: list[dict[str, Any]] = []
    pending: list[tuple[str, dict[str, Any], list[Path], list[str]]] = []
    for comparator in comparators:
        in_domain = validate_in_domain_comparator(comparator, contract)
        checkpoints = [checkpoint_path(args, comparator, fold) for fold in range(1, 6)]
        # Authenticate every trusted-pickle checkpoint against the completed
        # in-domain baseline manifest before any ``torch.load`` occurs.
        checkpoint_hashes = model_loaders.validate_checkpoint_set(comparator, checkpoints)
        if final_artifacts_reusable(args, dataset, comparator, contract, checkpoint_hashes, sources):
            print(f"Reusing verified final external artifact: {dataset}/{comparator}", flush=True)
            result_rows.append(
                {
                    "dataset": dataset,
                    "comparator": comparator,
                    "status": "reused_verified_final_artifact",
                    "predictions": str(final_output_paths(args, dataset, comparator)["predictions"]),
                    "manifest": str(final_output_paths(args, dataset, comparator)["manifest"]),
                }
            )
        else:
            pending.append((comparator, in_domain, checkpoints, checkpoint_hashes))
    if not pending:
        return result_rows
    metadata = load_dataset_contract(args, dataset, sources, contract)
    y_true = np.asarray(metadata["y_true"], dtype=np.float32)
    record_ids = np.asarray(metadata["record_id"]).astype(str)
    group_ids = np.asarray(metadata["group_id"]).astype(str)
    split_ids = np.asarray(metadata["split_id"]).astype(str)
    class_order = np.asarray(metadata["class_names"]).astype(str)
    slice_record_ids = np.asarray(metadata["slice_record_index"], dtype=np.int64)
    starts = np.asarray(metadata["slice_start"], dtype=np.int32)
    input_fingerprint = str(metadata["input_fingerprint"])
    dataset_contract_sha = str(metadata["dataset_contract_sha256"])
    load_summary = dict(metadata["load_summary"])
    folds_now = selected_folds or set(range(1, 6))

    def reusable_fold_cache(
        comparator: str,
        fold: int,
        checkpoint_hashes: list[str],
    ) -> bool:
        return bool(
            args.reuse_existing
            and not args.force_rerun
            and cache_matches(
                cache_path(args, dataset, comparator, fold),
                dataset=dataset,
                comparator=comparator,
                fold=fold,
                checkpoint_sha=checkpoint_hashes[fold - 1],
                input_fingerprint=input_fingerprint,
                class_names=class_order,
                dataset_contract_sha256=dataset_contract_sha,
                expected_record_index=slice_record_ids,
                expected_starts=starts,
            )
        )

    missing_requested = [
        (comparator, fold)
        for comparator, _in_domain, _checkpoints, checkpoint_hashes in pending
        for fold in sorted(folds_now)
        if not reusable_fold_cache(comparator, fold, checkpoint_hashes)
    ]
    loader = None
    dataset_obj = None
    signals = None
    if missing_requested:
        if device.type != "cuda":
            details = ", ".join(f"{name}/fold{fold}" for name, fold in missing_requested)
            raise RuntimeError(
                f"{dataset}: authenticated dataset metadata was restored without loading ECG signals, "
                f"but CUDA inference is required for missing/stale fold caches: {details}. "
                "No CPU model inference was started. Use a GPU runtime once; completed v2 fold "
                "caches can then be aggregated on CPU."
            )
        root = external_helpers.extract_archive(dataset, archive, resolve(args.extract_root))
        signals, loaded_y, loaded_records, loaded_groups, loaded_splits, loaded_summary = (
            external_helpers.load_records(
                dataset,
                root,
                int(args.limit_records),
                external_loader_args(args, sources),
            )
        )
        loaded_fingerprint = record_fingerprint(loaded_records, loaded_groups, loaded_y)
        if loaded_fingerprint != input_fingerprint:
            raise RuntimeError(f"{dataset}: live loader output differs from the authenticated dataset sidecar")
        if not np.array_equal(np.asarray(loaded_splits).astype(str), split_ids):
            raise RuntimeError(f"{dataset}: live loader split assignment differs from the dataset sidecar")
        if dict(loaded_summary).get("group_unit") != load_summary.get("group_unit"):
            raise RuntimeError(f"{dataset}: live loader group unit differs from the dataset sidecar")
        all_indices = np.arange(len(signals), dtype=np.int64)
        live_record_ids, live_starts, _positions, skipped = build_slice_index(
            all_indices,
            signals,
            slice_length=int(CONFIG["slice_length"]),
            slice_stride=int(CONFIG["slice_stride"]),
            max_slices_per_record=int(CONFIG["max_slices_per_record"]),
        )
        if skipped:
            raise RuntimeError(f"{dataset}: records without valid slices: {skipped[:10]}")
        if not np.array_equal(live_record_ids, slice_record_ids) or not np.array_equal(live_starts, starts):
            raise RuntimeError(f"{dataset}: live loader slice order differs from the authenticated sidecar")
        dataset_obj = baseline_helpers.RawECGSliceDataset(
            signals,
            loaded_y,
            slice_record_ids,
            starts,
            slice_length=int(CONFIG["slice_length"]),
        )
        loader = baseline_helpers.build_loader(
            dataset_obj,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            seed=42,
            device=device,
        )
    else:
        print(
            f"{dataset}: all requested fold caches passed the v2 dataset-sidecar contract; "
            "skipping archive extraction and signal loading.",
            flush=True,
        )
    for comparator, in_domain, checkpoints, checkpoint_hashes in pending:
        for fold in sorted(folds_now):
            fold_cache = cache_path(args, dataset, comparator, fold)
            if reusable_fold_cache(comparator, fold, checkpoint_hashes):
                print(f"Reusing {dataset}/{comparator}/fold{fold}: {fold_cache}", flush=True)
                continue
            print(
                f"Inference {dataset}/{comparator}/fold{fold} | records={len(record_ids)} "
                f"slices={len(slice_record_ids)} checkpoint={checkpoints[fold - 1]}",
                flush=True,
            )
            if device.type != "cuda" or loader is None:
                raise RuntimeError(f"{dataset}/{comparator}/fold{fold}: internal GPU-loader preflight failed")
            model = load_model(args, comparator, checkpoints[fold - 1], device)
            mapped, actual_record_index, actual_starts, class_names = infer_fold(
                comparator=comparator,
                model=model,
                loader=loader,
                dataset=dataset,
                device=device,
                use_amp=bool(args.amp),
            )
            if not np.array_equal(actual_record_index, slice_record_ids) or not np.array_equal(actual_starts, starts):
                raise RuntimeError(f"{dataset}/{comparator}/fold{fold}: slice order changed during inference")
            fold_cache.parent.mkdir(parents=True, exist_ok=True)
            save_npz_compressed_atomic(
                fold_cache,
                slice_prob=mapped.astype(np.float32),
                slice_record_index=actual_record_index.astype(np.int64),
                slice_start=actual_starts.astype(np.int32),
                class_names=np.asarray(class_names),
                dataset=np.asarray(dataset),
                comparator=np.asarray(comparator),
                fold=np.asarray(fold, dtype=np.int16),
                checkpoint_path=np.asarray(str(checkpoints[fold - 1])),
                checkpoint_sha256=np.asarray(checkpoint_hashes[fold - 1]),
                input_fingerprint=np.asarray(input_fingerprint),
                dataset_contract_sha256=np.asarray(dataset_contract_sha),
                protocol_version=np.asarray(PROTOCOL_VERSION, dtype=np.int16),
                created_utc=np.asarray(now_utc()),
            )
            print(f"Wrote fold cache: {fold_cache}", flush=True)
            del model, mapped
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        all_caches = [cache_path(args, dataset, comparator, fold) for fold in range(1, 6)]
        valid_all = all(
            cache_matches(
                path,
                dataset=dataset,
                comparator=comparator,
                fold=fold,
                checkpoint_sha=checkpoint_hashes[fold - 1],
                input_fingerprint=input_fingerprint,
                class_names=class_order,
                dataset_contract_sha256=dataset_contract_sha,
                expected_record_index=slice_record_ids,
                expected_starts=starts,
            )
            for fold, path in enumerate(all_caches, start=1)
        )
        if not valid_all:
            result_rows.append(
                {
                    "dataset": dataset,
                    "comparator": comparator,
                    "status": "fold_caches_incomplete",
                    "ready_folds": [fold for fold, path in enumerate(all_caches, 1) if path.exists()],
                }
            )
            print(
                f"{dataset}/{comparator}: fold caches incomplete; rerun missing folds, then aggregate.",
                flush=True,
            )
            continue

        probability_sum: np.ndarray | None = None
        reference_record_index: np.ndarray | None = None
        reference_starts: np.ndarray | None = None
        class_names: np.ndarray | None = None
        for fold, path in enumerate(all_caches, start=1):
            with np.load(path, allow_pickle=False) as cached:
                fold_prob = np.asarray(cached["slice_prob"], dtype=np.float32)
                fold_record_index = np.asarray(cached["slice_record_index"], dtype=np.int64)
                fold_starts = np.asarray(cached["slice_start"], dtype=np.int32)
                fold_classes = np.asarray(cached["class_names"]).astype(str)
            if reference_record_index is None:
                reference_record_index = fold_record_index
                reference_starts = fold_starts
                class_names = fold_classes
                probability_sum = np.zeros(fold_prob.shape, dtype=np.float64)
            elif (
                not np.array_equal(reference_record_index, fold_record_index)
                or not np.array_equal(reference_starts, fold_starts)
                or not np.array_equal(class_names, fold_classes)
            ):
                raise RuntimeError(f"{dataset}/{comparator}: fold-cache slice/class order mismatch")
            probability_sum += fold_prob
        assert probability_sum is not None
        assert reference_record_index is not None
        assert reference_starts is not None
        assert class_names is not None
        ensemble_slice_prob = (probability_sum / 5.0).astype(np.float32)
        y_prob, valid, slice_count = aggregate_record_probabilities(
            ensemble_slice_prob,
            reference_record_index,
            len(y_true),
            q=float(CONFIG["power_mean_q"]),
        )
        if not np.all(valid) or y_prob.shape != y_true.shape:
            raise RuntimeError(
                f"{dataset}/{comparator}: invalid record aggregation, shape={y_prob.shape}, missing={int(np.sum(~valid))}"
            )
        tag = output_tag(args, dataset)
        tag_suffix = f"_{tag}" if tag else ""
        stem = f"{dataset}_{COMPARATOR_STEMS[comparator]}{tag_suffix}"
        output_root = resolve(args.external_root) / dataset
        output_root.mkdir(parents=True, exist_ok=True)
        pred_path = output_root / f"{stem}_predictions.npz"
        slice_path = output_root / f"{stem}_slice_predictions.npz"
        summary_path = METRIC_DIR / f"external_{stem}_summary.json"
        class_path = TABLE_DIR / f"table_external_{stem}_class_metrics.csv"
        manifest_path = MANIFEST_DIR / f"external_{stem}_manifest.json"
        task_scope = (
            "annotation_aligned_10s_af_afl_mapped_windows"
            if dataset == "cpsc2021"
            else "record_level_mapped_external_task"
        )
        protocol = (
            f"zero_target_label_{COMPARATOR_STEMS[comparator]}_chapman_trained_ensemble5_"
            f"{POWER_MEAN_IMPLEMENTATION}_q{float(CONFIG['power_mean_q']):g}_{task_scope}_v{PROTOCOL_VERSION}"
        )
        save_npz_compressed_atomic(
            pred_path,
            y_true=y_true.astype(np.float32),
            y_prob=y_prob.astype(np.float32),
            record_id=np.asarray(record_ids),
            group_id=np.asarray(group_ids),
            group_unit=np.asarray(load_summary.get("group_unit", "group")),
            split_id=np.asarray(split_ids),
            class_names=class_names,
            dataset=np.asarray(dataset),
            comparator=np.asarray(comparator),
            protocol=np.asarray(protocol),
            task_scope=np.asarray(task_scope),
            adaptation_labels_used=np.asarray(0, dtype=np.int16),
            slice_count=slice_count.astype(np.int16),
            checkpoint_sha256=np.asarray(checkpoint_hashes),
            input_fingerprint=np.asarray(input_fingerprint),
            dataset_contract_sha256=np.asarray(dataset_contract_sha),
            aggregation_method=np.asarray("power_mean"),
            aggregation_implementation=np.asarray(POWER_MEAN_IMPLEMENTATION),
            aggregation_q=np.asarray(float(CONFIG["power_mean_q"])),
            manuscript_ready=np.asarray(False),
            created_utc=np.asarray(now_utc()),
        )
        save_npz_compressed_atomic(
            slice_path,
            slice_prob=ensemble_slice_prob,
            record_index=reference_record_index,
            record_id=np.asarray(record_ids)[reference_record_index],
            group_id=np.asarray(group_ids)[reference_record_index],
            split_id=np.asarray(split_ids)[reference_record_index],
            slice_start=reference_starts,
            class_names=class_names,
            dataset=np.asarray(dataset),
            comparator=np.asarray(comparator),
            protocol=np.asarray(protocol),
            task_scope=np.asarray(task_scope),
            created_utc=np.asarray(now_utc()),
        )
        point = {
            **multilabel_metrics(y_true, y_prob, threshold=0.5),
            **calibration_summary(y_true, y_prob, n_bins=15),
        }
        save_csv(class_path, class_rows(dataset, comparator, y_true, y_prob, class_names))
        summary = {
            "status": "complete_experimental_requires_external_comparator_gate",
            "created_utc": now_utc(),
            "dataset": dataset,
            "task_scope": task_scope,
            "comparator": comparator,
            "comparator_label": COMPARATOR_LABELS[comparator],
            "protocol": protocol,
            "adaptation_labels_used": 0,
            "n_records": int(len(y_true)),
            "n_groups": int(len(np.unique(np.asarray(group_ids).astype(str)))),
            "group_unit": load_summary.get("group_unit"),
            "class_names": class_names.tolist(),
            "metrics": point,
            "load_summary": load_summary,
            "in_domain_prerequisite": in_domain,
            "manuscript_ready": False,
            "safe_wording": (
                "Zero-target-label mapped-task comparator result; use only after paired external gate validation."
            ),
            "artifacts": {
                "predictions": str(pred_path),
                "slice_predictions": str(slice_path),
                "class_table": str(class_path),
                "manifest": str(manifest_path),
            },
        }
        save_json(summary_path, summary)
        artifacts = {
            "predictions": pred_path,
            "slice_predictions": slice_path,
            "summary": summary_path,
            "class_table": class_path,
        }
        save_json(
            manifest_path,
            {
                "status": "complete_experimental_requires_external_comparator_gate",
                "created_utc": now_utc(),
                "git_commit": git_commit(),
                "protocol_version": PROTOCOL_VERSION,
                "protocol": protocol,
                "dataset": dataset,
                "task_scope": task_scope,
                "comparator": comparator,
                "canonical_contract": contract,
                "source_contract": sources,
                "input_fingerprint": input_fingerprint,
                "dataset_contract": {
                    "path": str(metadata["dataset_contract_path"]),
                    "sha256": dataset_contract_sha,
                    "schema_version": DATASET_CONTRACT_SCHEMA_VERSION,
                    "capability": CACHE_ONLY_CPU_AGGREGATION_CAPABILITY,
                },
                "archive": {"path": str(archive), "sha256": sha256_file(archive)},
                "checkpoints": [
                    {"fold": fold, "path": str(path), "sha256": checkpoint_hashes[fold - 1]}
                    for fold, path in enumerate(checkpoints, start=1)
                ],
                "fold_caches": [
                    {"fold": fold, "path": str(path), "sha256": sha256_file(path)}
                    for fold, path in enumerate(all_caches, start=1)
                ],
                "in_domain_prerequisite": in_domain,
                "artifacts": {
                    name: {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                    for name, path in artifacts.items()
                },
            },
        )
        result_rows.append(
            {
                "dataset": dataset,
                "comparator": comparator,
                "status": "complete",
                "predictions": str(pred_path),
                "manifest": str(manifest_path),
            }
        )
        print(f"Wrote external comparator artifact: {pred_path}", flush=True)
    if loader is not None:
        del loader
    if dataset_obj is not None:
        del dataset_obj
    if signals is not None:
        del signals
    gc.collect()
    return result_rows


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    if not args.allow_experimental:
        raise RuntimeError(
            "External comparator outputs are experimental until their paired gate passes. "
            "Rerun with --allow-experimental to acknowledge this restriction."
        )
    comparators = parse_list(args.comparators)
    unknown = sorted(set(comparators) - set(COMPARATOR_STEMS))
    if unknown:
        raise ValueError(f"Unknown comparators: {unknown}")
    datasets = selected_datasets(args.dataset)
    if any(dataset != "ptbxl" for dataset in datasets) and args.ptbxl_folds != "10":
        if datasets != ["ptbxl"]:
            raise ValueError("Use --ptbxl-folds only in a PTB-XL-only invocation.")
    selected_folds = parse_only_folds(args.only_folds)
    contract = canonical_contract(args)
    device = model_loaders.select_device(args.device)
    if args.allow_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print("=" * 80, flush=True)
    print("LEARNED COMPARATOR EXTERNAL ZERO-TARGET-LABEL INFERENCE", flush=True)
    print("=" * 80, flush=True)
    print(
        f"datasets={datasets} comparators={comparators} folds={sorted(selected_folds) or 'all'} "
        f"device={device} batch_size={args.batch_size}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for dataset in datasets:
        try:
            rows.extend(run_dataset(args, dataset, comparators, selected_folds, contract, device))
        except Exception as exc:
            failures.append({"dataset": dataset, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{dataset} failed: {type(exc).__name__}: {exc}", flush=True)
            if args.strict:
                raise
    status_path = METRIC_DIR / "external_comparator_prediction_status.json"
    save_json(
        status_path,
        {
            "status": (
                "complete"
                if not failures
                and rows
                and all(
                    row.get("status") in {"complete", "reused_verified_final_artifact"}
                    for row in rows
                )
                else "incomplete"
            ),
            "created_utc": now_utc(),
            "canonical_contract": contract,
            "rows": rows,
            "failures": failures,
        },
    )
    print(json.dumps({"status": not failures, "rows": rows, "failures": failures}, indent=2), flush=True)


if __name__ == "__main__":
    main()
