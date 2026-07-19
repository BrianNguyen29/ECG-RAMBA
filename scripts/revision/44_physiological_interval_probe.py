"""Gate and run fold-held-out probes for measured ECG physiological intervals.

The runner never derives interval targets from model inputs or predictions. It
requires an explicit record-aligned metadata table with measured HR, PR, QRS,
QT, and/or QTc values. If such metadata is absent, it emits a blocker artifact
instead of manufacturing proxy targets.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    MANIFEST_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    TABLE_DIR,
    ensure_revision_dirs,
    git_commit,
    save_csv,
    save_json,
    sha256_file,
)


SCHEMA_VERSION = 4
PROTOCOL = "fold_held_out_measured_physiological_interval_probe_v3"
RUNNER_SOURCE_PATH = Path(__file__).resolve()
TARGET_ALIASES = {
    "heart_rate_bpm": ["heart_rate_bpm", "heart_rate", "hr", "ventricular_rate"],
    "pr_ms": ["pr_ms", "pr_interval_ms", "pr_interval", "pr"],
    "qrs_ms": ["qrs_ms", "qrs_duration_ms", "qrs_duration", "qrs"],
    "qt_ms": ["qt_ms", "qt_interval_ms", "qt_interval", "qt"],
    "qtc_ms": ["qtc_ms", "qtc_interval_ms", "qtc_interval", "qtc", "qtc_bazett"],
}
PLAUSIBLE_RANGES = {
    "heart_rate_bpm": (20.0, 250.0),
    "pr_ms": (60.0, 400.0),
    "qrs_ms": (40.0, 250.0),
    "qt_ms": (150.0, 700.0),
    "qtc_ms": (200.0, 750.0),
}
VIEWS = ["morphology", "rhythm", "context", "fused"]
TARGET_UNITS = {
    "heart_rate_bpm": "bpm",
    "pr_ms": "ms",
    "qrs_ms": "ms",
    "qt_ms": "ms",
    "qtc_ms": "ms",
}
TARGET_LABELS = {
    "heart_rate_bpm": "Heart rate",
    "pr_ms": "PR interval",
    "qrs_ms": "QRS duration",
    "qt_ms": "QT interval",
    "qtc_ms": "QTc interval",
}
ACCEPTED_MEASUREMENT_KINDS = {"measured", "device_measured", "expert_annotated"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding-npz",
        type=Path,
        default=PREDICTION_DIR / "representation_embeddings_final_ema.npz",
    )
    parser.add_argument(
        "--embedding-manifest",
        type=Path,
        default=MANIFEST_DIR / "representation_embedding_manifest.json",
        help="Authenticated manifest emitted with the branch-embedding NPZ.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Record-aligned measured metadata. Must include record_id and at least one target.",
    )
    parser.add_argument(
        "--metadata-provenance-json",
        type=Path,
        default=None,
        help=(
            "Reviewed provenance sidecar declaring record alignment, target source columns, units, "
            "measurement kind, and independence from model outputs."
        ),
    )
    parser.add_argument("--record-id-column", default="record_id")
    parser.add_argument("--min-records", type=int, default=500)
    parser.add_argument("--min-records-per-fold", type=int, default=50)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-targets", action="store_true")
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=METRIC_DIR / "physiological_interval_probe_summary.json",
    )
    parser.add_argument(
        "--out-table",
        type=Path,
        default=TABLE_DIR / "table_physiological_interval_probe.csv",
    )
    parser.add_argument(
        "--out-audit",
        type=Path,
        default=TABLE_DIR / "table_physiological_interval_target_audit.csv",
    )
    parser.add_argument(
        "--out-contrast-table",
        type=Path,
        default=TABLE_DIR / "table_physiological_interval_probe_contrasts.csv",
    )
    parser.add_argument(
        "--out-tex-table",
        type=Path,
        default=TABLE_DIR / "table_physiological_interval_probe.tex",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=MANIFEST_DIR / "physiological_interval_probe_manifest.json",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path: Path) -> str:
    path = resolve(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def format_interval(point: object, low: object, high: object) -> str:
    values = [float(point), float(low), float(high)]
    if not all(math.isfinite(value) for value in values):
        return "--"
    return f"{values[0]:.3f} [{values[1]:.3f}, {values[2]:.3f}]"


def write_tex_table(rows: list[dict], path: Path) -> None:
    aggregate = {
        (str(row.get("target")), str(row.get("view"))): row
        for row in rows
        if row.get("row_type") == "aggregate"
    }
    targets = [target for target in TARGET_ALIASES if (target, VIEWS[0]) in aggregate]
    if not targets:
        raise RuntimeError("Cannot write physiological probe TeX table without aggregate rows")

    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table*}[t]",
        (
            r"\caption{Fold-held-out linear probes of reviewed measured physiological intervals. "
            r"Cells report Spearman $\rho$ [nominal pointwise 95\% record-bootstrap CI]. "
            r"Bootstrap resamples condition on the fitted probes, and no multiplicity adjustment is "
            r"applied. Differences indicate branch-associated linear information only and do not "
            r"establish causal or mechanistic morphology--rhythm separation.}"
        ),
        r"\label{tab:physiological_interval_probe}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Measured target ($n$) & Morphology & Rhythm & Context & Fused \\",
        r"\midrule",
    ]
    for target in targets:
        reference = aggregate[(target, VIEWS[0])]
        label = f"{TARGET_LABELS[target]} ({TARGET_UNITS[target]}), {int(reference['n_test'])}"
        cells = []
        for view in VIEWS:
            row = aggregate[(target, view)]
            cells.append(
                format_interval(
                    row.get("spearman"),
                    row.get("spearman_ci_low"),
                    row.get("spearman_ci_high"),
                )
            )
        lines.append(f"{latex_escape(label)} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_provenance_path(args: argparse.Namespace, metadata_path: Path) -> Path | None:
    candidates = [
        resolve(args.metadata_provenance_json) if args.metadata_provenance_json else None,
        metadata_path.with_suffix(metadata_path.suffix + ".provenance.json"),
        metadata_path.with_name(metadata_path.stem + "_provenance.json"),
    ]
    return next((path for path in candidates if path is not None and path.is_file()), None)


def validate_metadata_provenance(
    payload: dict,
    *,
    metadata_sha256: str,
    record_id_column: str,
    target_columns: dict[str, str],
) -> list[str]:
    issues = []
    if payload.get("status") != "reviewed":
        issues.append("provenance_status_not_reviewed")
    if payload.get("record_id_column") != record_id_column:
        issues.append("record_id_column_mismatch")
    if payload.get("record_alignment") != "one_row_per_record_id":
        issues.append("record_alignment_not_one_row_per_record_id")
    if payload.get("independent_of_model_outputs") is not True:
        issues.append("model_output_independence_not_declared")
    if payload.get("independent_of_ecg_ramba_feature_cache") is not True:
        issues.append("ecg_ramba_feature_cache_independence_not_declared")
    if payload.get("metadata_sha256") != metadata_sha256:
        issues.append("metadata_sha256_mismatch")
    if not str(payload.get("reviewed_by") or "").strip():
        issues.append("reviewer_identity_missing")
    reviewed_utc = str(payload.get("reviewed_utc") or "").strip()
    if not reviewed_utc:
        issues.append("review_timestamp_missing")
    else:
        try:
            reviewed_time = datetime.fromisoformat(reviewed_utc.replace("Z", "+00:00"))
            if reviewed_time.tzinfo is None:
                issues.append("review_timestamp_not_timezone_aware")
        except ValueError:
            issues.append("review_timestamp_invalid")
    source_description = str(payload.get("source_description") or "").strip()
    if not source_description or source_description.lower().startswith("replace_with"):
        issues.append("measurement_source_description_missing")
    declarations = payload.get("targets") or {}
    for target, source_column in target_columns.items():
        declaration = declarations.get(target) or {}
        if declaration.get("source_column") != source_column:
            issues.append(f"{target}:source_column_mismatch")
        if declaration.get("unit") != TARGET_UNITS[target]:
            issues.append(f"{target}:unit_mismatch")
        if declaration.get("measurement_kind") not in ACCEPTED_MEASUREMENT_KINDS:
            issues.append(f"{target}:measurement_kind_not_accepted")
    return issues


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_contract(path: Path) -> dict:
    resolved = resolve(path)
    return {
        "path": rel(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def array_sha256(values: np.ndarray, dtype: np.dtype | type | None = None) -> str:
    array = np.asarray(values, dtype=dtype)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def validate_embedding_provenance(
    *,
    embedding_path: Path,
    manifest_path: Path,
    record_id: np.ndarray,
    fold_id: np.ndarray,
    embeddings: dict[str, np.ndarray],
    source_oof_sha256: str,
    source_freeze_sha256: str,
) -> dict:
    """Bind probe inputs to the complete, current representation extraction."""

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing representation embedding manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues = []
    if manifest.get("status") != "complete" or int(manifest.get("missing_records", -1)) != 0:
        issues.append("embedding_manifest_not_complete")
    outputs = manifest.get("outputs") or {}
    if outputs.get("embedding_npz_sha256") != sha256_file(embedding_path):
        issues.append("embedding_npz_sha256_mismatch")
    if manifest.get("oof_predictions_sha256") != source_oof_sha256:
        issues.append("embedding_oof_sha256_mismatch")
    if str(manifest.get("freeze_manifest_sha256") or "") != source_freeze_sha256:
        issues.append("embedding_freeze_sha256_mismatch")
    split_contract = manifest.get("split_contract") or {}
    observed_fold_sha = array_sha256(fold_id, np.int16)
    if split_contract.get("fold_assignment_sha256") != observed_fold_sha:
        issues.append("embedding_fold_assignment_mismatch")
    if sorted(int(value) for value in np.unique(fold_id)) != [1, 2, 3, 4, 5]:
        issues.append("embedding_fold_grid_not_1_to_5")
    if len(np.unique(record_id)) != len(record_id):
        issues.append("embedding_record_id_not_unique")
    for view, matrix in embeddings.items():
        if matrix.ndim != 2 or matrix.shape[0] != len(record_id):
            issues.append(f"{view}:embedding_shape_mismatch")
        elif not np.isfinite(matrix).all():
            issues.append(f"{view}:embedding_nonfinite")
    if issues:
        raise RuntimeError("Representation embedding provenance failed: " + "; ".join(issues))
    return {
        "path": rel(manifest_path),
        "sha256": sha256_file(manifest_path),
        "status": manifest.get("status"),
        "oof_predictions_sha256": source_oof_sha256,
        "freeze_manifest_sha256": source_freeze_sha256,
        "fold_assignment_sha256": observed_fold_sha,
    }


def interval(values: list[float]) -> tuple[float | None, float | None]:
    valid = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(valid):
        return None, None
    low, high = np.quantile(valid, [0.025, 0.975])
    return float(low), float(high)


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return math.nan
    return float(spearmanr(y_true, y_pred).statistic)


def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int, seed: int) -> dict:
    from sklearn.metrics import mean_absolute_error, r2_score

    rng = np.random.default_rng(seed)
    values = {"mae": [], "r2": [], "spearman": []}
    for _ in range(n_boot):
        index = rng.integers(0, len(y_true), size=len(y_true))
        yt, yp = y_true[index], y_pred[index]
        mae = float(mean_absolute_error(yt, yp))
        if np.isfinite(mae):
            values["mae"].append(mae)
        if len(np.unique(yt)) > 1:
            r2 = float(r2_score(yt, yp))
            spearman = safe_spearman(yt, yp)
            if np.isfinite(r2):
                values["r2"].append(r2)
            if np.isfinite(spearman):
                values["spearman"].append(spearman)
    result = {}
    for metric, samples in values.items():
        low, high = interval(samples)
        result[metric] = {
            "mean": float(np.mean(samples)) if samples else None,
            "ci_low": low,
            "ci_high": high,
            "n_boot_valid": len(samples),
        }
    return result


def target_coverage_contract(
    values: np.ndarray,
    fold_id: np.ndarray,
    plausible: np.ndarray,
    *,
    min_records: int,
    min_records_per_fold: int,
) -> tuple[str, dict[int, int], dict[int, int]]:
    folds = sorted(int(fold) for fold in np.unique(fold_id))
    records_by_fold = {
        fold: int(np.sum(plausible & (fold_id == fold))) for fold in folds
    }
    unique_values_by_fold = {
        fold: int(len(np.unique(values[plausible & (fold_id == fold)]))) for fold in folds
    }
    if (
        int(np.sum(plausible)) < min_records
        or min(records_by_fold.values(), default=0) < min_records_per_fold
    ):
        status = "insufficient_fold_coverage"
    elif min(unique_values_by_fold.values(), default=0) < 2:
        status = "insufficient_target_variation"
    else:
        status = "usable"
    return status, records_by_fold, unique_values_by_fold


def paired_view_bootstrap(
    y_true: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, dict]:
    """Paired record bootstrap with positive values oriented toward view A."""

    from sklearn.metrics import mean_absolute_error, r2_score

    def values(yt: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> dict[str, float]:
        return {
            "mae": float(mean_absolute_error(yt, pb) - mean_absolute_error(yt, pa)),
            "r2": float(r2_score(yt, pa) - r2_score(yt, pb)),
            "spearman": float(safe_spearman(yt, pa) - safe_spearman(yt, pb)),
        }

    point = values(y_true, prediction_a, prediction_b)
    samples = {metric: [] for metric in point}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        index = rng.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[index])) < 2:
            continue
        current = values(y_true[index], prediction_a[index], prediction_b[index])
        for metric, value in current.items():
            if np.isfinite(value):
                samples[metric].append(value)

    result = {}
    for metric, point_value in point.items():
        low, high = interval(samples[metric])
        result[metric] = {
            "improvement_view_a_over_view_b": point_value,
            "ci_low": low,
            "ci_high": high,
            "n_boot_valid": len(samples[metric]),
            "inference_scope": "pointwise_percentile_ci_effect_size_only",
            "null_test": "not_run",
            "interpretation": (
                "view_a_nominal_95ci_better"
                if low is not None and low > 0
                else "view_b_nominal_95ci_better"
                if high is not None and high < 0
                else "inconclusive"
            ),
        }
    return result


def blocked_payload(
    args: argparse.Namespace,
    embedding_path: Path,
    embedding_provenance: dict,
    reason: str,
    audit_rows: list[dict],
    *,
    metadata_path: Path | None = None,
    provenance_path: Path | None = None,
) -> None:
    resolve(args.out_tex_table).unlink(missing_ok=True)
    save_csv(resolve(args.out_audit), audit_rows)
    save_csv(resolve(args.out_table), [])
    save_csv(resolve(args.out_contrast_table), [])
    inputs = {
        "runner": file_contract(RUNNER_SOURCE_PATH),
        "embedding": file_contract(embedding_path),
        "embedding_manifest": embedding_provenance,
    }
    if metadata_path is not None and metadata_path.is_file():
        inputs["metadata"] = file_contract(metadata_path)
    if provenance_path is not None and provenance_path.is_file():
        inputs["metadata_provenance"] = file_contract(provenance_path)
    summary = {
        "status": "blocked_missing_reliable_interval_metadata",
        "schema_version": SCHEMA_VERSION,
        "created_utc": now_utc(),
        "protocol": PROTOCOL,
        "reason": reason,
        "target_policy": (
            "Only measured, record-aligned HR/PR/QRS/QT/QTc metadata are accepted; reviewed provenance "
            "must bind the exact metadata SHA and confirm independence from ECG-RAMBA outputs, "
            "representations, and feature caches."
        ),
        "claim_boundary": (
            "Existing morphology/rhythm label probes remain an audit only. No physiological interval "
            "selectivity or morphology-rhythm disentanglement claim is supported."
        ),
        "embedding": inputs["embedding"],
        "inputs": inputs,
    }
    save_json(resolve(args.out_summary), summary)
    save_json(
        resolve(args.out_manifest),
        {
            "status": summary["status"],
            "schema_version": SCHEMA_VERSION,
            "created_utc": now_utc(),
            "protocol": PROTOCOL,
            "git_commit": git_commit(),
            "runner": inputs["runner"],
            "inputs": inputs,
            "reason": reason,
            "outputs": {
                rel(args.out_summary): sha256_file(resolve(args.out_summary)),
                rel(args.out_audit): sha256_file(resolve(args.out_audit)),
                rel(args.out_table): sha256_file(resolve(args.out_table)),
                rel(args.out_contrast_table): sha256_file(resolve(args.out_contrast_table)),
            },
        },
    )
    print(json.dumps({"status": summary["status"], "reason": reason}, indent=2))
    if args.require_targets:
        raise RuntimeError(reason)


def main() -> None:
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    args = parse_args()
    if args.n_boot <= 0:
        raise ValueError("--n-boot must be positive")
    if args.min_records <= 0 or args.min_records_per_fold <= 0:
        raise ValueError("minimum record counts must be positive")
    if args.ridge_alpha <= 0:
        raise ValueError("--ridge-alpha must be positive")
    ensure_revision_dirs()
    embedding_path = resolve(args.embedding_npz)
    embedding_manifest_path = resolve(args.embedding_manifest)
    with np.load(embedding_path, allow_pickle=False) as data:
        required = {"record_id", "fold_id"} | {f"{view}_embedding" for view in VIEWS}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"Embedding artifact missing {sorted(missing)}")
        record_id = np.asarray(data["record_id"]).astype(str)
        fold_id = np.asarray(data["fold_id"], dtype=np.int16)
        embeddings = {
            view: np.asarray(data[f"{view}_embedding"], dtype=np.float32) for view in VIEWS
        }
        source_oof_sha256 = (
            str(data["oof_predictions_sha256"].item())
            if "oof_predictions_sha256" in data.files
            else ""
        )
        source_freeze_sha256 = (
            str(data["freeze_manifest_sha256"].item())
            if "freeze_manifest_sha256" in data.files
            else ""
        )
    embedding_provenance = validate_embedding_provenance(
        embedding_path=embedding_path,
        manifest_path=embedding_manifest_path,
        record_id=record_id,
        fold_id=fold_id,
        embeddings=embeddings,
        source_oof_sha256=source_oof_sha256,
        source_freeze_sha256=source_freeze_sha256,
    )

    if args.metadata_csv is None:
        blocked_payload(
            args,
            embedding_path,
            embedding_provenance,
            "No measured physiological interval metadata table was supplied.",
            [
                {
                    "target": target,
                    "status": "missing_metadata_source",
                    "accepted_aliases": "|".join(aliases),
                }
                for target, aliases in TARGET_ALIASES.items()
            ],
        )
        return

    metadata_path = resolve(args.metadata_csv)
    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    if args.record_id_column not in metadata.columns:
        blocked_payload(
            args,
            embedding_path,
            embedding_provenance,
            f"Metadata lacks required record identifier column {args.record_id_column!r}.",
            [],
            metadata_path=metadata_path,
        )
        return
    metadata = metadata.copy()
    metadata[args.record_id_column] = metadata[args.record_id_column].astype(str)
    if metadata[args.record_id_column].duplicated().any():
        raise ValueError("Physiological metadata contains duplicate record identifiers")
    lookup = metadata.set_index(args.record_id_column)

    declared_target_columns = {
        target: source
        for target, aliases in TARGET_ALIASES.items()
        if (source := next((column for column in aliases if column in metadata.columns), None))
        is not None
    }
    provenance_path = find_provenance_path(args, metadata_path)
    if provenance_path is None:
        blocked_payload(
            args,
            embedding_path,
            embedding_provenance,
            (
                "Measured interval metadata lacks a reviewed provenance sidecar. Supply "
                "--metadata-provenance-json or <metadata>.provenance.json."
            ),
            [
                {
                    "target": target,
                    "source_column": source,
                    "status": "blocked_missing_reviewed_provenance",
                }
                for target, source in declared_target_columns.items()
            ],
            metadata_path=metadata_path,
        )
        return
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance_issues = validate_metadata_provenance(
        provenance,
        metadata_sha256=sha256_file(metadata_path),
        record_id_column=args.record_id_column,
        target_columns=declared_target_columns,
    )
    if provenance_issues:
        blocked_payload(
            args,
            embedding_path,
            embedding_provenance,
            "Physiological metadata provenance gate failed: " + "; ".join(provenance_issues),
            [
                {
                    "target": target,
                    "source_column": source,
                    "status": "blocked_provenance_contract_failed",
                    "issues": "|".join(provenance_issues),
                }
                for target, source in declared_target_columns.items()
            ],
            metadata_path=metadata_path,
            provenance_path=provenance_path,
        )
        return

    target_columns, audit_rows = {}, []
    for target, aliases in TARGET_ALIASES.items():
        source = declared_target_columns.get(target)
        if source is None:
            audit_rows.append(
                {
                    "target": target,
                    "source_column": "",
                    "status": "missing_target_column",
                    "matched_records": 0,
                    "valid_records": 0,
                    "out_of_range_records": 0,
                }
            )
            continue
        aligned = pd.to_numeric(lookup[source].reindex(record_id), errors="coerce").to_numpy(dtype=float)
        low, high = PLAUSIBLE_RANGES[target]
        matched = np.isfinite(aligned)
        plausible = matched & (aligned >= low) & (aligned <= high)
        status, per_fold, unique_values_by_fold = target_coverage_contract(
            aligned,
            fold_id,
            plausible,
            min_records=args.min_records,
            min_records_per_fold=args.min_records_per_fold,
        )
        audit_rows.append(
            {
                "target": target,
                "source_column": source,
                "status": status,
                "matched_records": int(np.sum(matched)),
                "valid_records": int(np.sum(plausible)),
                "out_of_range_records": int(np.sum(matched & ~plausible)),
                "plausible_min": low,
                "plausible_max": high,
                "records_by_fold": json.dumps(per_fold, sort_keys=True),
                "unique_values_by_fold": json.dumps(unique_values_by_fold, sort_keys=True),
            }
        )
        if status == "usable":
            target_columns[target] = aligned
    save_csv(resolve(args.out_audit), audit_rows)
    if not target_columns:
        blocked_payload(
            args,
            embedding_path,
            embedding_provenance,
            "No measured target passed plausibility and per-fold coverage gates.",
            audit_rows,
            metadata_path=metadata_path,
            provenance_path=provenance_path,
        )
        return

    rows = []
    contrast_rows = []
    for target_index, (target, values) in enumerate(target_columns.items()):
        low, high = PLAUSIBLE_RANGES[target]
        valid = np.isfinite(values) & (values >= low) & (values <= high)
        predictions_by_view = {}
        for view_index, (view, matrix) in enumerate(embeddings.items()):
            predictions = np.full(len(values), np.nan, dtype=np.float64)
            for fold in sorted(int(value) for value in np.unique(fold_id)):
                train = valid & (fold_id != fold)
                test = valid & (fold_id == fold)
                if len(np.unique(values[train])) < 2 or len(np.unique(values[test])) < 2:
                    raise RuntimeError(
                        f"{target}/fold{fold}: measured target lacks train/test variation"
                    )
                model = make_pipeline(
                    StandardScaler(),
                    Ridge(alpha=args.ridge_alpha, solver="lsqr"),
                )
                model.fit(matrix[train], values[train])
                predictions[test] = model.predict(matrix[test])
                rows.append(
                    {
                        "row_type": "fold",
                        "target": target,
                        "view": view,
                        "fold": fold,
                        "n_train": int(np.sum(train)),
                        "n_test": int(np.sum(test)),
                        "mae": float(mean_absolute_error(values[test], predictions[test])),
                        "r2": float(r2_score(values[test], predictions[test])),
                        "spearman": safe_spearman(values[test], predictions[test]),
                    }
                )
            evaluated = valid & np.isfinite(predictions)
            if int(np.sum(evaluated)) != int(np.sum(valid)):
                raise RuntimeError(f"{target}/{view}: incomplete fold-held-out predictions")
            predictions_by_view[view] = predictions
            bootstrap = bootstrap_metrics(
                values[evaluated],
                predictions[evaluated],
                args.n_boot,
                args.seed + target_index * 10 + view_index,
            )
            rows.append(
                {
                    "row_type": "aggregate",
                    "target": target,
                    "view": view,
                    "fold": "all",
                    "n_train": "",
                    "n_test": int(np.sum(evaluated)),
                    "mae": float(mean_absolute_error(values[evaluated], predictions[evaluated])),
                    "mae_ci_low": bootstrap["mae"]["ci_low"],
                    "mae_ci_high": bootstrap["mae"]["ci_high"],
                    "r2": float(r2_score(values[evaluated], predictions[evaluated])),
                    "r2_ci_low": bootstrap["r2"]["ci_low"],
                    "r2_ci_high": bootstrap["r2"]["ci_high"],
                    "spearman": safe_spearman(values[evaluated], predictions[evaluated]),
                    "spearman_ci_low": bootstrap["spearman"]["ci_low"],
                    "spearman_ci_high": bootstrap["spearman"]["ci_high"],
                    "mae_n_boot_valid": bootstrap["mae"]["n_boot_valid"],
                    "r2_n_boot_valid": bootstrap["r2"]["n_boot_valid"],
                    "spearman_n_boot_valid": bootstrap["spearman"]["n_boot_valid"],
                }
            )
        evaluated_index = np.flatnonzero(valid)
        for contrast_index, (view_a, view_b) in enumerate(itertools.combinations(VIEWS, 2)):
            contrasts = paired_view_bootstrap(
                values[evaluated_index],
                predictions_by_view[view_a][evaluated_index],
                predictions_by_view[view_b][evaluated_index],
                args.n_boot,
                args.seed + 10_000 + target_index * 100 + contrast_index,
            )
            for metric, result in contrasts.items():
                contrast_rows.append(
                    {
                        "target": target,
                        "view_a": view_a,
                        "view_b": view_b,
                        "metric": metric,
                        "positive_direction": "view_a_better",
                        **result,
                    }
                )
    aggregate_rows = [row for row in rows if row.get("row_type") == "aggregate"]
    expected_aggregate_rows = len(target_columns) * len(VIEWS)
    expected_contrast_rows = len(target_columns) * math.comb(len(VIEWS), 2) * 3
    aggregate_bootstrap_complete = (
        len(aggregate_rows) == expected_aggregate_rows
        and all(
            int(row.get(f"{metric}_n_boot_valid", 0)) == args.n_boot
            and row.get(f"{metric}_ci_low") is not None
            and row.get(f"{metric}_ci_high") is not None
            for row in aggregate_rows
            for metric in ("mae", "r2", "spearman")
        )
    )
    contrast_bootstrap_complete = (
        len(contrast_rows) == expected_contrast_rows
        and all(
            int(row.get("n_boot_valid", 0)) == args.n_boot
            and row.get("ci_low") is not None
            and row.get("ci_high") is not None
            and np.isfinite(float(row.get("improvement_view_a_over_view_b", math.nan)))
            for row in contrast_rows
        )
    )
    point_metrics_complete = all(
        np.isfinite(float(row.get(metric, math.nan)))
        for row in rows
        for metric in ("mae", "r2", "spearman")
    )
    completeness = {
        "point_metrics_complete": point_metrics_complete,
        "aggregate_bootstrap_complete": aggregate_bootstrap_complete,
        "contrast_bootstrap_complete": contrast_bootstrap_complete,
        "expected_aggregate_rows": expected_aggregate_rows,
        "observed_aggregate_rows": len(aggregate_rows),
        "expected_contrast_rows": expected_contrast_rows,
        "observed_contrast_rows": len(contrast_rows),
        "required_valid_bootstrap_replicates": int(args.n_boot),
    }
    probe_complete = all(
        completeness[key]
        for key in (
            "point_metrics_complete",
            "aggregate_bootstrap_complete",
            "contrast_bootstrap_complete",
        )
    )
    save_csv(resolve(args.out_table), rows)
    save_csv(resolve(args.out_contrast_table), contrast_rows)
    write_tex_table(rows, args.out_tex_table)
    summary = {
        "status": (
            "complete_measured_target_probe"
            if probe_complete
            else "incomplete_bootstrap_or_metric_contract"
        ),
        "schema_version": SCHEMA_VERSION,
        "created_utc": now_utc(),
        "protocol": PROTOCOL,
        "targets": list(target_columns),
        "views": VIEWS,
        "probe": {
            "model": "ridge regression",
            "alpha": args.ridge_alpha,
            "standardization": "training folds only",
            "evaluation": "five fold-held-out predictions",
            "uncertainty": "record bootstrap over aggregate held-out predictions",
            "view_contrasts": (
                "all six pre-specified view pairs; positive paired deltas favor view_a"
            ),
            "bootstrap_scope": (
                "pointwise record bootstrap conditions on fitted probes; probes are not refitted "
                "inside bootstrap resamples"
            ),
            "multiplicity_scope": (
                "nominal pointwise 95% intervals; no family-wise branch-selectivity claim"
            ),
        },
        "claim_boundary": (
            "Differences are evidence of branch-associated linear information only; they do not prove "
            "causal or mechanistic morphology-rhythm disentanglement."
        ),
        "completeness_contract": completeness,
        "inputs": {
            "runner": file_contract(RUNNER_SOURCE_PATH),
            "embedding": file_contract(embedding_path),
            "embedding_manifest": embedding_provenance,
            "metadata": file_contract(metadata_path),
            "metadata_provenance": {
                **file_contract(provenance_path),
                "status": provenance.get("status"),
            },
        },
    }
    save_json(resolve(args.out_summary), summary)
    save_json(
        resolve(args.out_manifest),
        {
            "status": summary["status"],
            "schema_version": SCHEMA_VERSION,
            "created_utc": now_utc(),
            "protocol": PROTOCOL,
            "git_commit": git_commit(),
            "runner": summary["inputs"]["runner"],
            "inputs": summary["inputs"],
            "outputs": {
                rel(path): sha256_file(resolve(path))
                for path in [
                    args.out_summary,
                    args.out_table,
                    args.out_contrast_table,
                    args.out_audit,
                    args.out_tex_table,
                ]
            },
        },
    )
    print(json.dumps({"status": summary["status"], "targets": list(target_columns)}, indent=2))
    if not probe_complete:
        raise RuntimeError(f"Physiological probe completeness contract failed: {completeness}")


if __name__ == "__main__":
    main()
