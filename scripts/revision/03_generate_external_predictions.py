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
    CONFIG_HASH,
    DEVICE,
    FS,
    PATHS,
    SNOMED_MAPPING,
)
from scripts.revision.common import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    MANIFEST_DIR,
    METRIC_DIR,
    POWER_MEAN_IMPLEMENTATION,
    PREDICTION_DIR,
    PTB_SUPERCLASS_MAPPING,
    TABLE_DIR,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ptbxl", "georgia", "cpsc2021"], required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-kind", choices=["best", "final"], default="best")
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--force-features", action="store_true")
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


def ptb_metadata(root: Path, limit: int) -> list[dict]:
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
    for ecg_id, row in test.iterrows():
        codes = ast.literal_eval(row["scp_codes"]) if isinstance(row["scp_codes"], str) else row["scp_codes"]
        diagnostic_classes = set()
        for code, likelihood in codes.items():
            if float(likelihood) < 100 or str(code) not in statements.index:
                continue
            value = statements.loc[str(code)].get("diagnostic_class")
            if isinstance(value, str):
                diagnostic_classes.add(value.upper())
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
    return rows


def dx_codes(header: Path) -> list[str]:
    for line in header.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("#Dx:"):
            return [code.strip() for code in line.split(":", 1)[1].split(",")]
    return []


def georgia_metadata(root: Path, limit: int) -> list[dict]:
    headers = sorted(root.rglob("*.hea"))
    if limit:
        headers = headers[:limit]
    rows = []
    for header in headers:
        y = np.zeros(len(CLASSES), dtype=np.float32)
        for code in dx_codes(header):
            mapped = SNOMED_MAPPING.get(code)
            if isinstance(mapped, int):
                y[mapped] = 1.0
        rows.append(
            {
                "record_id": header.stem,
                "record_path": header.with_suffix(""),
                "y_true": y,
            }
        )
    return rows


def cpsc_af_label(record_path: Path) -> float:
    try:
        annotation = wfdb.rdann(str(record_path), "atr")
        notes = [str(note).strip("\x00").upper() for note in annotation.aux_note]
        return float(any("AFIB" in note or "AFL" in note for note in notes))
    except Exception:
        return 0.0


def cpsc_metadata(root: Path, limit: int) -> list[dict]:
    headers = sorted(root.rglob("*.hea"))
    if limit:
        headers = headers[:limit]
    return [
        {
            "record_id": header.stem,
            "record_path": header.with_suffix(""),
            "y_true": np.asarray([cpsc_af_label(header.with_suffix(""))], dtype=np.float32),
        }
        for header in headers
    ]


def load_records(dataset: str, root: Path, limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    metadata = {
        "ptbxl": ptb_metadata,
        "georgia": georgia_metadata,
        "cpsc2021": cpsc_metadata,
    }[dataset](root, limit)
    signals, labels, record_ids = [], [], []
    skipped = 0
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
            if skipped <= 10:
                print(f"Skipping {row['record_id']}: {exc}")
    if not signals:
        raise RuntimeError(f"No usable {dataset} records were loaded")
    return (
        np.asarray(signals, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(record_ids),
        {
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
    pca_path: Path,
    record_ids: np.ndarray,
) -> Path:
    cache_dir = Path(PATHS["cache_dir"]) / "revision_external_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (
        f"{dataset}_features_v{CACHE_SCHEMA_VERSION}_{CONFIG_HASH}_"
        f"{file_fingerprint(archive)}_{file_fingerprint(pca_path)}_"
        f"N{len(record_ids)}_{record_id_fingerprint(record_ids)}.npz"
    )


def generate_features(
    dataset: str,
    archive: Path,
    signals: np.ndarray,
    record_ids: np.ndarray,
    pca_path: Path,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, Path, bool]:
    cache_path = feature_cache_path(dataset, archive, pca_path, record_ids)
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as data:
            schema = int(data["cache_schema_version"]) if "cache_schema_version" in data.files else 0
            hydra = data["X_hydra"]
            hrv = data["X_hrv"]
            if (
                schema == CACHE_SCHEMA_VERSION
                and hydra.dtype == np.float32
                and hrv.dtype == np.float32
                and len(hydra) == len(signals)
            ):
                print(f"Loaded float32 external feature cache: {cache_path}")
                return hydra, hrv, cache_path, True

    pca = joblib.load(pca_path)
    rocket = MiniRocketNative(c_in=12, seq_len=5000, seed=42).cpu().eval()
    raw_features = []
    with torch.no_grad():
        for start in tqdm(range(0, len(signals), 64), desc="MiniRocket"):
            batch = torch.from_numpy(signals[start : start + 64])
            raw_features.append(rocket(batch).numpy())
    hydra = pca.transform(np.concatenate(raw_features, axis=0)).astype(np.float32)
    hrv = np.asarray(
        [checkpoint_compatible_hrv36(signal) for signal in tqdm(signals, desc="HRV36")],
        dtype=np.float32,
    )
    np.savez_compressed(
        cache_path,
        X_hydra=hydra,
        X_hrv=hrv,
        cache_schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
        config_hash=np.asarray(CONFIG_HASH),
        pca_path=np.asarray(str(pca_path)),
        pca_sha256=np.asarray(sha256_file(pca_path)),
        hrv_semantics=np.asarray("checkpoint_compatible_amplitude_slots_zero"),
        record_id_fingerprint=np.asarray(record_id_fingerprint(record_ids)),
    )
    return hydra, hrv, cache_path, False


def build_slices(
    signals: np.ndarray,
    hydra: np.ndarray,
    hrv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, xh, xhr, record_index, slice_index = [], [], [], [], []
    for rid, signal in enumerate(signals):
        ordinal = 0
        for start in range(
            0,
            signal.shape[-1] - CONFIG["slice_length"] + 1,
            CONFIG["slice_stride"],
        ):
            xs.append(signal[:, start : start + CONFIG["slice_length"]])
            xh.append(hydra[rid])
            xhr.append(hrv[rid])
            record_index.append(rid)
            slice_index.append(ordinal)
            ordinal += 1
            if ordinal >= CONFIG["max_slices_per_record"]:
                break
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(xh, dtype=np.float32),
        np.asarray(xhr, dtype=np.float32),
        np.asarray(record_index, dtype=np.int64),
        np.asarray(slice_index, dtype=np.int16),
    )


def checkpoint_paths(kind: str) -> list[Path]:
    paths = []
    for fold in range(1, int(CONFIG["n_folds"]) + 1):
        preferred = Path(PATHS["model_dir"]) / f"fold{fold}_{kind}.pt"
        fallback = Path(PATHS["model_dir"]) / f"fold{fold}_{'final' if kind == 'best' else 'best'}.pt"
        path = preferred if preferred.exists() else fallback
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint for fold {fold}: {preferred} or {fallback}")
        paths.append(path)
    return paths


def infer_ensemble(
    xs: np.ndarray,
    xh: np.ndarray,
    xhr: np.ndarray,
    checkpoints: list[Path],
    batch_size: int,
) -> np.ndarray:
    from src.model import ECGRambaV7Advanced

    probability_sum = np.zeros((len(xs), len(CLASSES)), dtype=np.float64)
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
                hb = torch.from_numpy(xh[start:stop]).to(DEVICE)
                rb = torch.from_numpy(xhr[start:stop]).to(DEVICE)
                with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
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
    path = METRIC_DIR / "external_summary.csv"
    rows = []
    if path.exists():
        rows = pd.read_csv(path).to_dict(orient="records")
        rows = [row for row in rows if str(row.get("dataset")) != dataset]
    rows.append(
        {
            "dataset": dataset,
            "protocol": protocol,
            "n_records": n_records,
            **metrics,
        }
    )
    save_csv(path, sorted(rows, key=lambda row: str(row["dataset"])))
    return path


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    created_utc = datetime.now(timezone.utc).isoformat()
    archive = archive_path(args.dataset)
    root = extract_archive(args.dataset, archive, args.extract_root)
    signals, y_true, record_ids, load_summary = load_records(
        args.dataset,
        root,
        args.limit_records,
    )

    pca_path = Path(PATHS["model_dir"]) / "global_pca_zeroshot.pkl"
    if not pca_path.exists():
        raise FileNotFoundError(f"Chapman global PCA not found: {pca_path}")
    hydra, hrv, feature_cache, feature_cache_hit = generate_features(
        args.dataset,
        archive,
        signals,
        record_ids,
        pca_path,
        args.force_features,
    )
    xs, xh, xhr, slice_record_index, slice_index = build_slices(signals, hydra, hrv)
    checkpoints = checkpoint_paths(args.checkpoint_kind)
    model_slice_probs = infer_ensemble(xs, xh, xhr, checkpoints, args.batch_size)
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
    prediction_path = PREDICTION_DIR / f"{args.dataset}_full_predictions.npz"
    slice_path = PREDICTION_DIR / f"{args.dataset}_full_slice_predictions.npz"
    summary_path = METRIC_DIR / f"{args.dataset}_full_prediction_summary.json"
    class_path = TABLE_DIR / f"{args.dataset}_full_class_summary.csv"
    manifest_path = MANIFEST_DIR / f"{args.dataset}_full_prediction_run_manifest.json"
    checkpoint_info = [
        {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in checkpoints
    ]
    aggregation_q = float(CONFIG["power_mean_q"])
    protocol = (
        f"external_{args.checkpoint_kind}_ensemble5_{POWER_MEAN_IMPLEMENTATION}_"
        f"q{aggregation_q:g}_threshold_0.5"
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
        config_hash=np.asarray(CONFIG_HASH),
        git_commit=np.asarray(git_commit()),
        created_utc=np.asarray(created_utc),
        aggregation_method=np.asarray("power_mean"),
        aggregation_q=np.asarray(aggregation_q, dtype=np.float32),
        aggregation_implementation=np.asarray(POWER_MEAN_IMPLEMENTATION),
        cache_schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
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
        "protocol": protocol,
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
                "path": str(pca_path),
                "sha256": sha256_file(pca_path),
                "scope": "global_chapman_zero_shot",
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
                "Global Chapman PCA is shared across fold checkpoints because fold-specific PCA objects were not saved.",
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
