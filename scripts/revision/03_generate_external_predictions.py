"""Export protocol-traceable PTB-XL, Georgia, and CPSC2021 predictions."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import resample_poly
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (  # noqa: E402
    CLASSES,
    CLASS_TO_IDX,
    CONFIG,
    DEVICE,
    EVALUATION_CONFIG_HASH,
    FS,
    PATHS,
    SNOMED_MAPPING,
)
from scripts.revision.common import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    EXPERIMENTAL_DIR,
    MANIFEST_DIR,
    POWER_MEAN_IMPLEMENTATION,
    PTB_SUPERCLASS_MAPPING,
    aggregate_record_probabilities,
    ensure_revision_dirs,
    multilabel_metrics,
    save_csv,
    save_json,
)
from src.data_loader import bandpass_filter, normalize_signal, pad_or_truncate  # noqa: E402
from src.features import (  # noqa: E402
    MiniRocketNative,
    extract_global_record_stats,
    extract_hrv_features,
)


STANDARD_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CHECKPOINT_KINDS = ["best", "final", "best_ema", "final_ema", "best_raw", "final_raw"]
AMP_DTYPE = (
    torch.bfloat16
    if DEVICE == "cuda" and torch.cuda.is_bf16_supported()
    else torch.float16
)
AMP_DTYPE_NAME = (
    str(AMP_DTYPE).replace("torch.", "")
    if DEVICE == "cuda"
    else "float32"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ptbxl", "georgia", "cpsc2021"], required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-kind", choices=CHECKPOINT_KINDS, default="best")
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument(
        "--allow-experimental",
        action="store_true",
        help="Required acknowledgement: external outputs are not manuscript-ready.",
    )
    parser.add_argument("--extract-root", type=Path, default=Path("/content/ecg_ramba_runtime/external"))
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return hashlib.sha256(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def archive_path(dataset: str) -> Path:
    return Path(
        {
            "ptbxl": PATHS["ptb_zip"],
            "georgia": PATHS["georgia_zip"],
            "cpsc2021": PATHS["cpsc_zip"],
        }[dataset]
    )


def extract_archive(dataset: str, archive: Path, extract_root: Path) -> Path:
    if not archive.exists():
        raise FileNotFoundError(f"{dataset} archive not found: {archive}")
    target = extract_root / f"{dataset}_{file_fingerprint(archive)}"
    marker = target / ".extract_complete"
    if marker.exists():
        return target
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} -> {target}", flush=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    return target


def normalize_lead_name(name: str) -> str:
    return name.strip().upper().replace(" ", "")


def align_leads(signal: np.ndarray, signal_names: list[str] | None) -> tuple[np.ndarray, int]:
    if signal.shape[0] == 12 and not signal_names:
        return signal.astype(np.float32), 0
    output = np.zeros((12, signal.shape[1]), dtype=np.float32)
    missing = 12
    if signal_names:
        name_to_idx = {normalize_lead_name(name): idx for idx, name in enumerate(signal_names)}
        found = 0
        for out_idx, lead in enumerate(STANDARD_LEADS):
            source_idx = name_to_idx.get(lead)
            if source_idx is not None:
                output[out_idx] = signal[source_idx]
                found += 1
        missing = 12 - found
    else:
        count = min(12, signal.shape[0])
        output[:count] = signal[:count]
        missing = 12 - count
    return output, missing


def preprocess_record(record: wfdb.Record) -> tuple[np.ndarray, int]:
    raw = record.p_signal.T if record.p_signal is not None else record.d_signal.T.astype(np.float32)
    raw, missing_leads = align_leads(raw, list(record.sig_name) if record.sig_name else None)
    source_fs = float(record.fs)
    if source_fs != FS:
        numerator = int(FS)
        denominator = int(round(source_fs))
        raw = resample_poly(raw, numerator, denominator, axis=-1).astype(np.float32)
    filtered = bandpass_filter(raw, fs=FS).astype(np.float32)
    filtered = pad_or_truncate(filtered, target_len=5000).astype(np.float32)
    normalized = normalize_signal(filtered).astype(np.float32)
    return normalized, missing_leads


def checkpoint_compatible_hrv36(normalized: np.ndarray) -> np.ndarray:
    """Match the feature semantics used by the current Chapman checkpoints.

    The training pipeline passed precomputed five-dimensional amplitude vectors
    back into extract_amplitude_features(), so amplitude slots 25:30 became zero.
    This exporter preserves that behavior until the model is retrained.
    """
    return np.concatenate(
        [
            extract_hrv_features(normalized, fs=FS),
            np.zeros(5, dtype=np.float32),
            extract_global_record_stats(normalized),
        ]
    ).astype(np.float32)


def ptb_metadata(root: Path, limit: int) -> tuple[list[dict], dict]:
    database_candidates = list(root.rglob("ptbxl_database.csv"))
    statement_candidates = list(root.rglob("scp_statements.csv"))
    if not database_candidates or not statement_candidates:
        raise FileNotFoundError("PTB-XL metadata files were not found after extraction")
    database = pd.read_csv(database_candidates[0], index_col="ecg_id")
    statements = pd.read_csv(statement_candidates[0], index_col=0)
    statements.index = statements.index.astype(str)
    test = database[database["strat_fold"] == 10].copy()
    if limit:
        test = test.iloc[:limit]
    data_root = database_candidates[0].parent
    rows = []
    unmapped_codes: dict[str, int] = {}
    unsupported_superclasses: dict[str, int] = {}
    for ecg_id, row in test.iterrows():
        codes = ast.literal_eval(row["scp_codes"]) if isinstance(row["scp_codes"], str) else row["scp_codes"]
        diagnostic_classes = set()
        for code, likelihood in codes.items():
            if float(likelihood) <= 0:
                continue
            code = str(code)
            if code not in statements.index:
                unmapped_codes[code] = unmapped_codes.get(code, 0) + 1
                continue
            statement = statements.loc[code]
            diagnostic = statement.get("diagnostic", 0)
            if not pd.notna(diagnostic) or int(diagnostic) != 1:
                continue
            value = statement.get("diagnostic_class")
            if isinstance(value, str) and value:
                value = value.upper()
                if value in PTB_SUPERCLASS_MAPPING:
                    diagnostic_classes.add(value)
                else:
                    unsupported_superclasses[value] = unsupported_superclasses.get(value, 0) + 1
        y = np.asarray(
            [float(name in diagnostic_classes) for name in PTB_SUPERCLASS_MAPPING],
            dtype=np.float32,
        )
        rows.append(
            {
                "record_id": str(ecg_id),
                "record_path": data_root / str(row["filename_hr"]),
                "y_true": y,
            }
        )
    return rows, {
        "label_protocol": "official_ptbxl_diagnostic_superclass_any_positive_likelihood",
        "supported_superclasses": list(PTB_SUPERCLASS_MAPPING),
        "unsupported_superclasses": unsupported_superclasses,
        "unmapped_scp_codes": unmapped_codes,
        "records_without_supported_superclass": int(sum(not np.any(row["y_true"]) for row in rows)),
    }


def dx_codes(header: Path) -> list[str]:
    for line in header.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("#Dx:"):
            return [code.strip() for code in line.split(":", 1)[1].split(",")]
    return []


def georgia_metadata(root: Path, limit: int) -> tuple[list[dict], dict]:
    headers = sorted(root.rglob("*.hea"))
    if limit:
        headers = headers[:limit]
    rows = []
    skipped_unmapped_records = 0
    unmapped_codes: dict[str, int] = {}
    for header in headers:
        y = np.zeros(len(CLASSES), dtype=np.float32)
        source_codes = dx_codes(header)
        mapped_count = 0
        for code in source_codes:
            mapped = SNOMED_MAPPING.get(code)
            if isinstance(mapped, int):
                y[mapped] = 1.0
                mapped_count += 1
            else:
                unmapped_codes[code] = unmapped_codes.get(code, 0) + 1
        if mapped_count == 0:
            skipped_unmapped_records += 1
            continue
        rows.append(
            {
                "record_id": header.stem,
                "record_path": header.with_suffix(""),
                "y_true": y,
            }
        )
    return rows, {
        "label_protocol": "chapman_27_class_snomed_intersection",
        "skipped_records_without_mapped_label": skipped_unmapped_records,
        "unmapped_snomed_codes": unmapped_codes,
    }


def cpsc_af_intervals(record_path: Path, signal_length: int) -> list[tuple[int, int]]:
    annotation = wfdb.rdann(str(record_path), "atr")
    events = []
    for sample, note in zip(annotation.sample, annotation.aux_note):
        normalized = str(note).strip("\x00").strip().upper()
        if "AFIB" in normalized or "AFL" in normalized:
            events.append((int(sample), True))
        elif normalized in {"(N", "N", "(NSR", "NSR"}:
            events.append((int(sample), False))
    if not events:
        raise ValueError("CPSC annotation has no recognized AF/AFL/normal rhythm boundaries")
    events.sort(key=lambda item: item[0])
    intervals = []
    for idx, (start, is_af) in enumerate(events):
        stop = events[idx + 1][0] if idx + 1 < len(events) else signal_length
        start = max(0, min(start, signal_length))
        stop = max(start, min(stop, signal_length))
        if is_af and stop > start:
            intervals.append((start, stop))
    return intervals


def cpsc_metadata(root: Path, limit: int) -> tuple[list[dict], dict]:
    headers = sorted(root.rglob("*.hea"))
    if limit:
        headers = headers[:limit]
    return (
        [
            {
                "record_id": header.stem,
                "record_path": header.with_suffix(""),
            }
            for header in headers
        ],
        {
            "label_protocol": "annotation_aligned_nonoverlapping_10s_windows_majority_af",
        },
    )


def interval_overlap(intervals: list[tuple[int, int]], start: int, stop: int) -> int:
    return sum(max(0, min(stop, right) - max(start, left)) for left, right in intervals)


def load_cpsc_windows(root: Path, limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    metadata, metadata_summary = cpsc_metadata(root, limit)
    signals, labels, record_ids = [], [], []
    skipped_annotation = 0
    skipped_signal = 0
    signal_skip_examples = []
    annotation_skip_examples = []
    preprocessing_skip_examples = []
    transition_windows = 0
    positive_windows = 0
    missing_leads = []
    for row in tqdm(metadata, desc="Loading cpsc2021 windows"):
        try:
            record = wfdb.rdrecord(str(row["record_path"]))
            raw = record.p_signal.T if record.p_signal is not None else record.d_signal.T.astype(np.float32)
            raw, missing = align_leads(raw, list(record.sig_name) if record.sig_name else None)
        except Exception as exc:
            skipped_signal += 1
            if len(signal_skip_examples) < 10:
                signal_skip_examples.append({"record_id": str(row["record_id"]), "error": str(exc)})
            if skipped_annotation + skipped_signal <= 10:
                print(f"Skipping signal {row['record_id']}: {exc}")
            continue

        try:
            intervals = cpsc_af_intervals(row["record_path"], raw.shape[-1])
        except Exception as exc:
            skipped_annotation += 1
            if len(annotation_skip_examples) < 10:
                annotation_skip_examples.append({"record_id": str(row["record_id"]), "error": str(exc)})
            if skipped_annotation + skipped_signal <= 10:
                print(f"Skipping annotation {row['record_id']}: {exc}")
            continue

        try:
            record_signals = []
            record_labels = []
            record_ids_for_windows = []
            record_transition_windows = 0
            record_positive_windows = 0
            source_fs = float(record.fs)
            if source_fs != FS:
                raw = resample_poly(raw, int(FS), int(round(source_fs)), axis=-1).astype(np.float32)
            filtered = bandpass_filter(raw, fs=FS).astype(np.float32)
            scale = FS / source_fs
            target_intervals = [
                (int(round(start * scale)), int(round(stop * scale)))
                for start, stop in intervals
            ]
            for start in range(0, filtered.shape[-1], 5000):
                stop = min(start + 5000, filtered.shape[-1])
                valid_length = stop - start
                if valid_length < 2500:
                    continue
                af_samples = interval_overlap(target_intervals, start, stop)
                af_fraction = af_samples / valid_length
                if 0.0 < af_fraction < 1.0:
                    record_transition_windows += 1
                label = float(af_fraction >= 0.5)
                record_positive_windows += int(label)
                segment = pad_or_truncate(filtered[:, start:stop], target_len=5000).astype(np.float32)
                normalized = normalize_signal(segment).astype(np.float32)
                if not np.isfinite(normalized).all():
                    raise ValueError("non-finite signal")
                record_signals.append(normalized)
                record_labels.append(np.asarray([label], dtype=np.float32))
                record_ids_for_windows.append(f"{row['record_id']}:{start}:{stop}")
            signals.extend(record_signals)
            labels.extend(record_labels)
            record_ids.extend(record_ids_for_windows)
            missing_leads.extend([missing] * len(record_signals))
            transition_windows += record_transition_windows
            positive_windows += record_positive_windows
        except Exception as exc:
            skipped_signal += 1
            if len(preprocessing_skip_examples) < 10:
                preprocessing_skip_examples.append({"record_id": str(row["record_id"]), "error": str(exc)})
            if skipped_annotation + skipped_signal <= 10:
                print(f"Skipping preprocessing {row['record_id']}: {exc}")
    if not signals:
        raise RuntimeError(
            "No usable CPSC2021 annotation-aligned windows were loaded. "
            f"headers={len(metadata)}; "
            f"skipped_signal_records={skipped_signal}; "
            f"skipped_annotation_records={skipped_annotation}; "
            f"signal_skip_examples={signal_skip_examples}; "
            f"annotation_skip_examples={annotation_skip_examples}; "
            f"preprocessing_skip_examples={preprocessing_skip_examples}. "
            "CPSC2021 manuscript-ready evaluation requires WFDB-readable signals plus "
            "annotation files with recognized AF/AFL/normal rhythm boundaries; otherwise "
            "leave CPSC2021 deferred."
        )
    return (
        np.asarray(signals, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(record_ids),
        {
            **metadata_summary,
            "metadata_records": len(metadata),
            "loaded_windows": len(signals),
            "positive_windows": positive_windows,
            "transition_windows": transition_windows,
            "skipped_annotation_records": skipped_annotation,
            "skipped_signal_records": skipped_signal,
            "missing_leads_mean": float(np.mean(missing_leads)),
            "windows_with_missing_leads": int(np.sum(np.asarray(missing_leads) > 0)),
        },
    )


def load_records(dataset: str, root: Path, limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if dataset == "cpsc2021":
        return load_cpsc_windows(root, limit)
    metadata, metadata_summary = {
        "ptbxl": ptb_metadata,
        "georgia": georgia_metadata,
    }[dataset](root, limit)
    signals, labels, record_ids = [], [], []
    skipped = 0
    skipped_examples = []
    missing_leads = []
    for row in tqdm(metadata, desc=f"Loading {dataset}"):
        try:
            record = wfdb.rdrecord(str(row["record_path"]))
            normalized, missing = preprocess_record(record)
            if not np.isfinite(normalized).all():
                raise ValueError("non-finite signal")
            signals.append(normalized)
            labels.append(row["y_true"])
            record_ids.append(row["record_id"])
            missing_leads.append(missing)
        except Exception as exc:
            skipped += 1
            if len(skipped_examples) < 10:
                skipped_examples.append({"record_id": str(row["record_id"]), "error": str(exc)})
            if skipped <= 10:
                print(f"Skipping {row['record_id']}: {exc}")
    if not signals:
        detail = ""
        if dataset == "georgia":
            unmapped = metadata_summary.get("unmapped_snomed_codes", {})
            top_unmapped = sorted(
                unmapped.items(),
                key=lambda item: (-int(item[1]), str(item[0])),
            )[:12]
            if metadata:
                detail = (
                    " Georgia label mapping produced mapped candidate records, but every candidate "
                    "was skipped while reading or preprocessing the ECG signal. "
                    f"mapped_candidate_records={len(metadata)}; signal_skipped_records={skipped}; "
                    f"skipped_without_mapped_label={metadata_summary.get('skipped_records_without_mapped_label')}; "
                    f"skip_examples={skipped_examples}. "
                    "Check whether the Georgia archive contains WFDB-compatible signal files next "
                    "to the headers, or leave Georgia deferred."
                )
            else:
                detail = (
                    " Georgia headers were readable, but every record was skipped because no "
                    "diagnosis code mapped to the frozen 27-class Chapman/SNOMED taxonomy. "
                    f"skipped_without_mapped_label={metadata_summary.get('skipped_records_without_mapped_label')}; "
                    f"top_unmapped_codes={top_unmapped}. "
                    "Do not coerce these records to negative labels; leave Georgia deferred or add a "
                    "reviewed label mapping before using it."
                )
        raise RuntimeError(f"No usable {dataset} records were loaded.{detail}")
    return (
        np.asarray(signals, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(record_ids),
        {
            **metadata_summary,
            "metadata_records": len(metadata),
            "loaded_records": len(signals),
            "skipped_records": skipped,
            "missing_leads_mean": float(np.mean(missing_leads)),
            "records_with_missing_leads": int(np.sum(np.asarray(missing_leads) > 0)),
        },
    )


def record_id_fingerprint(record_ids: np.ndarray) -> str:
    payload = "\n".join(str(value) for value in record_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def feature_cache_path(
    dataset: str,
    archive: Path,
    pca_paths: list[Path],
    record_ids: np.ndarray,
) -> Path:
    cache_dir = Path(PATHS["cache_dir"]) / "revision_external_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pca_fingerprint = hashlib.sha256(
        ":".join(file_fingerprint(path) for path in pca_paths).encode()
    ).hexdigest()[:16]
    return cache_dir / (
        f"{dataset}_features_v{CACHE_SCHEMA_VERSION}_{EVALUATION_CONFIG_HASH}_"
        f"{file_fingerprint(archive)}_{pca_fingerprint}_"
        f"N{len(record_ids)}_{record_id_fingerprint(record_ids)}.npz"
    )


def generate_features(
    dataset: str,
    archive: Path,
    signals: np.ndarray,
    record_ids: np.ndarray,
    pca_paths: list[Path],
    force: bool,
) -> tuple[list[np.ndarray], np.ndarray, Path, bool]:
    cache_path = feature_cache_path(dataset, archive, pca_paths, record_ids)
    hydra_keys = [f"X_hydra_fold{fold}" for fold in range(1, len(pca_paths) + 1)]
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as data:
            schema = int(data["cache_schema_version"]) if "cache_schema_version" in data.files else 0
            hydra = [data[key] for key in hydra_keys] if all(key in data.files for key in hydra_keys) else []
            hrv = data["X_hrv"]
            if (
                schema == CACHE_SCHEMA_VERSION
                and len(hydra) == len(pca_paths)
                and all(values.dtype == np.float32 for values in hydra)
                and hrv.dtype == np.float32
                and all(len(values) == len(signals) for values in hydra)
            ):
                print(f"Loaded float32 external feature cache: {cache_path}")
                return hydra, hrv, cache_path, True

    rocket = MiniRocketNative(c_in=12, seq_len=5000, seed=42).cpu().eval()
    raw_features = []
    with torch.no_grad():
        for start in tqdm(range(0, len(signals), 64), desc="MiniRocket"):
            batch = torch.from_numpy(signals[start : start + 64])
            raw_features.append(rocket(batch).numpy())
    raw_features = np.concatenate(raw_features, axis=0)
    hydra = [
        joblib.load(path).transform(raw_features).astype(np.float32)
        for path in pca_paths
    ]
    hrv = np.asarray(
        [checkpoint_compatible_hrv36(signal) for signal in tqdm(signals, desc="HRV36")],
        dtype=np.float32,
    )
    payload = {
        "X_hrv": hrv,
        "cache_schema_version": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
        "config_hash": np.asarray(EVALUATION_CONFIG_HASH),
        "pca_paths": np.asarray([str(path) for path in pca_paths]),
        "pca_sha256": np.asarray([sha256_file(path) for path in pca_paths]),
        "hrv_semantics": np.asarray("checkpoint_compatible_amplitude_slots_zero"),
        "record_id_fingerprint": np.asarray(record_id_fingerprint(record_ids)),
    }
    payload.update({key: values for key, values in zip(hydra_keys, hydra)})
    np.savez_compressed(cache_path, **payload)
    return hydra, hrv, cache_path, False


def build_slices(
    signals: np.ndarray,
    hrv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, xhr, record_index, slice_index = [], [], [], []
    for rid, signal in enumerate(signals):
        ordinal = 0
        for start in range(
            0,
            signal.shape[-1] - CONFIG["slice_length"] + 1,
            CONFIG["slice_stride"],
        ):
            xs.append(signal[:, start : start + CONFIG["slice_length"]])
            xhr.append(hrv[rid])
            record_index.append(rid)
            slice_index.append(ordinal)
            ordinal += 1
            if ordinal >= CONFIG["max_slices_per_record"]:
                break
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(xhr, dtype=np.float32),
        np.asarray(record_index, dtype=np.int64),
        np.asarray(slice_index, dtype=np.int16),
    )


def checkpoint_paths(kind: str) -> list[Path]:
    paths = []
    for fold in range(1, int(CONFIG["n_folds"]) + 1):
        path = Path(PATHS["model_dir"]) / f"fold{fold}_{kind}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing exact checkpoint for fold {fold}: {path}")
        paths.append(path)
    return paths


def checkpoint_provenance(paths: list[Path], kind: str) -> tuple[list[dict], str]:
    expected_weights_kind = (
        "ema" if kind.endswith("_ema")
        else "raw" if kind.endswith("_raw")
        else None
    )
    rows = []
    source_hashes = set()
    dataset_fingerprints = set()
    for fold, path in enumerate(paths, start=1):
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError(f"Checkpoint lacks explicit metadata: {path}")
        weights_kind = payload.get("weights_kind")
        if expected_weights_kind and weights_kind != expected_weights_kind:
            raise ValueError(
                f"{path} reports weights_kind={weights_kind}; expected {expected_weights_kind}"
            )
        source_hash = payload.get("config_hash")
        if not source_hash:
            raise ValueError(f"Checkpoint lacks config_hash provenance: {path}")
        source_hashes.add(str(source_hash))
        dataset_fingerprint = payload.get("dataset_record_order_fingerprint")
        if not dataset_fingerprint:
            raise ValueError(
                f"Checkpoint lacks dataset record-order provenance: {path}"
            )
        dataset_fingerprints.add(str(dataset_fingerprint))
        rows.append(
            {
                "fold": fold,
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "weights_kind": weights_kind,
                "epoch": payload.get("epoch"),
                "selection_rule": payload.get("selection_rule"),
                "source_config_hash": str(source_hash),
                "dataset_record_order_fingerprint": str(dataset_fingerprint),
            }
        )
        del payload
    if len(source_hashes) != 1:
        raise ValueError(
            f"External ensemble checkpoints use different config hashes: {sorted(source_hashes)}"
        )
    if len(dataset_fingerprints) != 1:
        raise ValueError(
            "External ensemble checkpoints use different dataset record-order "
            f"fingerprints: {sorted(dataset_fingerprints)}"
        )
    return rows, next(iter(source_hashes))


def fold_pca_paths(
    expected_folds: int,
    expected_source_config_hash: str,
    expected_checkpoint_kind: str,
    expected_dataset_record_order_fingerprint: str,
) -> list[Path]:
    manifest_path = MANIFEST_DIR / "fold_pca_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Fold PCA manifest not found: {manifest_path}. "
            "Run scripts/revision/08_build_fold_pca.py before external evaluation."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    folds_path = Path(PATHS["model_dir"]) / "folds.pkl"
    if not folds_path.exists():
        raise FileNotFoundError(folds_path)
    if payload.get("folds_path_sha256") != sha256_file(folds_path):
        raise RuntimeError("Fold PCA manifest was built from a different folds.pkl")
    if payload.get("source_config_hash") != expected_source_config_hash:
        raise RuntimeError(
            "Fold PCA manifest source_config_hash does not match the selected checkpoints"
        )
    if payload.get("checkpoint_kind") != expected_checkpoint_kind:
        raise RuntimeError(
            "Fold PCA manifest checkpoint_kind does not match the selected checkpoints"
        )
    if (
        payload.get("dataset_record_order_fingerprint")
        != expected_dataset_record_order_fingerprint
    ):
        raise RuntimeError(
            "Fold PCA manifest dataset record-order fingerprint does not match "
            "the selected checkpoints"
        )
    rows = {int(row["fold"]): row for row in payload.get("fold_pca", [])}
    if set(rows) != set(range(1, expected_folds + 1)):
        raise ValueError("Fold PCA manifest is incomplete")
    paths = []
    for fold in range(1, expected_folds + 1):
        path = Path(rows[fold]["path"])
        if not path.exists():
            raise FileNotFoundError(f"Fold {fold} PCA missing: {path}")
        if sha256_file(path) != rows[fold]["sha256"]:
            raise RuntimeError(f"Fold {fold} PCA checksum mismatch: {path}")
        paths.append(path)
    return paths


def infer_ensemble(
    xs: np.ndarray,
    xhr: np.ndarray,
    slice_record_index: np.ndarray,
    hydra_by_fold: list[np.ndarray],
    checkpoints: list[Path],
    batch_size: int,
) -> np.ndarray:
    from src.model import ECGRambaV7Advanced

    probability_sum = np.zeros((len(xs), len(CLASSES)), dtype=np.float64)
    if len(hydra_by_fold) != len(checkpoints):
        raise ValueError("One fold-specific PCA feature matrix is required per checkpoint")
    for fold, checkpoint in enumerate(checkpoints, start=1):
        print(f"Inference checkpoint {fold}/{len(checkpoints)}: {checkpoint}")
        state = torch.load(checkpoint, map_location=DEVICE)
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        model = ECGRambaV7Advanced(cfg=CONFIG).to(DEVICE)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        with torch.no_grad():
            for start in tqdm(range(0, len(xs), batch_size), desc=f"Fold {fold}", leave=False):
                stop = start + batch_size
                xb = torch.from_numpy(xs[start:stop]).to(DEVICE)
                record_batch = slice_record_index[start:stop]
                hb = torch.from_numpy(hydra_by_fold[fold - 1][record_batch]).to(DEVICE)
                rb = torch.from_numpy(xhr[start:stop]).to(DEVICE)
                with torch.amp.autocast(
                    "cuda",
                    enabled=DEVICE == "cuda",
                    dtype=AMP_DTYPE,
                ):
                    probability_sum[start:stop] += torch.sigmoid(model(xb, hb, rb)).float().cpu().numpy()
        del model, state, state_dict
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return (probability_sum / len(checkpoints)).astype(np.float32)


def map_model_probabilities(dataset: str, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "ptbxl":
        names = np.asarray(list(PTB_SUPERCLASS_MAPPING))
        mapped = np.column_stack(
            [
                np.max(probs[:, [CLASS_TO_IDX[code] for code in spec["codes"]]], axis=1)
                for spec in PTB_SUPERCLASS_MAPPING.values()
            ]
        )
        return mapped.astype(np.float32), names
    if dataset == "cpsc2021":
        indices = [CLASS_TO_IDX["AF"], CLASS_TO_IDX["AFL"]]
        return np.max(probs[:, indices], axis=1, keepdims=True).astype(np.float32), np.asarray(["AF_or_AFL"])
    return probs.astype(np.float32), np.asarray(CLASSES)


def class_summary(dataset: str, y_true: np.ndarray, y_prob: np.ndarray, names: np.ndarray) -> list[dict]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    for idx, name in enumerate(names):
        both = len(np.unique(y_true[:, idx])) >= 2
        rows.append(
            {
                "dataset": dataset,
                "class_index": idx,
                "class_name": str(name),
                "n_records": len(y_true),
                "n_positive": int(np.sum(y_true[:, idx])),
                "prevalence": float(np.mean(y_true[:, idx])),
                "roc_auc": float(roc_auc_score(y_true[:, idx], y_prob[:, idx])) if both else np.nan,
                "pr_auc": float(average_precision_score(y_true[:, idx], y_prob[:, idx])) if both else np.nan,
            }
        )
    return rows


def update_external_summary(dataset: str, protocol: str, n_records: int, metrics: dict) -> Path:
    path = EXPERIMENTAL_DIR / "external" / "external_summary_experimental.csv"
    rows = []
    if path.exists():
        rows = pd.read_csv(path).to_dict(orient="records")
        rows = [row for row in rows if str(row.get("dataset")) != dataset]
    rows.append(
        {
            "dataset": dataset,
            "protocol": protocol,
            "n_records": n_records,
            "evidence_status": "experimental",
            "manuscript_ready": False,
            **metrics,
        }
    )
    save_csv(path, sorted(rows, key=lambda row: str(row["dataset"])))
    return path


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    if not args.allow_experimental:
        raise RuntimeError(
            "External evaluation is intentionally blocked from manuscript use. "
            "Re-run with --allow-experimental only after acknowledging that outputs "
            "will be stored under reports/revision/experimental with manuscript_ready=false."
        )
    checkpoints = checkpoint_paths(args.checkpoint_kind)
    checkpoint_info, source_config_hash = checkpoint_provenance(
        checkpoints,
        args.checkpoint_kind,
    )
    pca_paths = fold_pca_paths(
        int(CONFIG["n_folds"]),
        source_config_hash,
        args.checkpoint_kind,
        checkpoint_info[0]["dataset_record_order_fingerprint"],
    )
    created_utc = datetime.now(timezone.utc).isoformat()
    archive = archive_path(args.dataset)
    root = extract_archive(args.dataset, archive, args.extract_root)
    signals, y_true, record_ids, load_summary = load_records(
        args.dataset,
        root,
        args.limit_records,
    )

    hydra_by_fold, hrv, feature_cache, feature_cache_hit = generate_features(
        args.dataset,
        archive,
        signals,
        record_ids,
        pca_paths,
        args.force_features,
    )
    xs, xhr, slice_record_index, slice_index = build_slices(signals, hrv)
    model_slice_probs = infer_ensemble(
        xs,
        xhr,
        slice_record_index,
        hydra_by_fold,
        checkpoints,
        args.batch_size,
    )
    mapped_slice_probs, class_names = map_model_probabilities(args.dataset, model_slice_probs)
    y_prob, valid_mask, slice_count = aggregate_record_probabilities(
        mapped_slice_probs,
        slice_record_index,
        len(signals),
        q=float(CONFIG["power_mean_q"]),
    )

    if y_true.shape != y_prob.shape:
        raise ValueError(f"Label/prediction shape mismatch: {y_true.shape} vs {y_prob.shape}")
    if not np.all(valid_mask):
        raise ValueError(f"{int(np.sum(~valid_mask))} external records have no slices")
    threshold = 0.5
    metrics = multilabel_metrics(y_true, y_prob, threshold=threshold)
    output_root = EXPERIMENTAL_DIR / "external" / args.dataset
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_path = output_root / f"{args.dataset}_full_predictions.npz"
    slice_path = output_root / f"{args.dataset}_full_slice_predictions.npz"
    summary_path = output_root / f"{args.dataset}_full_prediction_summary.json"
    class_path = output_root / f"{args.dataset}_full_class_summary.csv"
    manifest_path = output_root / f"{args.dataset}_full_prediction_run_manifest.json"
    pca_info = [
        {
            "fold": fold,
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "scope": "chapman_training_fold_only",
        }
        for fold, path in enumerate(pca_paths, start=1)
    ]
    aggregation_q = float(CONFIG["power_mean_q"])
    protocol = (
        f"experimental_external_{args.checkpoint_kind}_fold_pca_ensemble5_{POWER_MEAN_IMPLEMENTATION}_"
        f"q{aggregation_q:g}_threshold_0.5"
    )
    restrictions = [
        "External outputs remain experimental until fair baselines, bootstrap CI, and dataset-specific protocol review are complete.",
        "Current checkpoints use zero amplitude slots 25:30 because of the training feature bug.",
    ]
    if args.dataset == "ptbxl":
        restrictions.append(
            "PTB predictions are Chapman-class proxies for four supported PTB-XL diagnostic superclasses; HYP is unsupported."
        )
    if args.dataset == "cpsc2021":
        restrictions.append(
            "CPSC2021 is evaluated as annotation-aligned 10-second majority-rhythm windows, not as the official episode-boundary challenge score."
        )

    np.savez_compressed(
        prediction_path,
        y_true=y_true.astype(np.float32),
        y_prob=y_prob.astype(np.float32),
        record_id=record_ids,
        class_names=class_names,
        dataset=np.asarray(args.dataset),
        protocol=np.asarray(protocol),
        slice_count=slice_count,
        config_hash=np.asarray(EVALUATION_CONFIG_HASH),
        source_config_hash=np.asarray(source_config_hash),
        evaluation_config_hash=np.asarray(EVALUATION_CONFIG_HASH),
        git_commit=np.asarray(git_commit()),
        created_utc=np.asarray(created_utc),
        aggregation_method=np.asarray("power_mean"),
        aggregation_q=np.asarray(aggregation_q, dtype=np.float32),
        aggregation_implementation=np.asarray(POWER_MEAN_IMPLEMENTATION),
        cache_schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
        checkpoint_fingerprints_json=np.asarray(json.dumps(checkpoint_info, sort_keys=True)),
        pca_fingerprints_json=np.asarray(json.dumps(pca_info, sort_keys=True)),
        evidence_status=np.asarray("experimental"),
        manuscript_ready=np.asarray(False),
        restrictions_json=np.asarray(json.dumps(restrictions)),
        threshold=np.asarray(threshold, dtype=np.float32),
    )
    np.savez_compressed(
        slice_path,
        slice_prob=mapped_slice_probs.astype(np.float32),
        record_index=slice_record_index,
        record_id=record_ids[slice_record_index],
        slice_index=slice_index,
        class_names=class_names,
        dataset=np.asarray(args.dataset),
        protocol=np.asarray("external_slice_ensemble5"),
        cache_schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
        evidence_status=np.asarray("experimental"),
        manuscript_ready=np.asarray(False),
    )
    save_csv(class_path, class_summary(args.dataset, y_true, y_prob, class_names))
    external_summary_path = update_external_summary(
        args.dataset,
        protocol,
        len(y_true),
        metrics,
    )
    summary = {
        "dataset": args.dataset,
        "created_utc": created_utc,
        "evidence_status": "experimental",
        "manuscript_ready": False,
        "restrictions": restrictions,
        "protocol": protocol,
        "checkpoint_kind": args.checkpoint_kind,
        "inference_amp_dtype": AMP_DTYPE_NAME,
        "n_records": len(y_true),
        "n_classes": y_true.shape[1],
        "class_names": [str(x) for x in class_names],
        "metrics": metrics,
        "aggregation": {"method": "power_mean", "q": aggregation_q},
        "aggregation_implementation": POWER_MEAN_IMPLEMENTATION,
        "feature_cache": str(feature_cache),
        "feature_cache_hit": feature_cache_hit,
        "hrv_semantics": "checkpoint_compatible_amplitude_slots_zero",
        "load_summary": load_summary,
        "prediction_file": str(prediction_path),
        "slice_prediction_file": str(slice_path),
        "class_summary_csv": str(class_path),
        "external_summary_csv": str(external_summary_path),
    }
    save_json(summary_path, summary)
    output_paths = [
        prediction_path,
        slice_path,
        summary_path,
        class_path,
        external_summary_path,
    ]
    save_json(
        manifest_path,
        {
            **summary,
            "archive": {
                "path": str(archive),
                "size_bytes": archive.stat().st_size,
                "fingerprint": file_fingerprint(archive),
            },
            "pca": {
                "scope": "fold_specific_chapman_training_only",
                "folds": pca_info,
            },
            "checkpoints": checkpoint_info,
            "outputs": {
                path.name: {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in output_paths
            },
            "warnings": [
                *restrictions,
                "Amplitude slots 25:30 are zero to match the feature bug used by the current Chapman checkpoints.",
                "CPSC2021 records with fewer than 12 leads are zero-padded and reported in load_summary.",
            ],
        },
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote: {prediction_path}")
    print(f"Wrote: {slice_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {external_summary_path}")
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
