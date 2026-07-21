"""Independent, fail-closed forensic audit for the ECG-RAMBA notebook pipeline.

This runner intentionally does not import metric or aggregation helpers from the
project. It is the independent oracle used before Notebook 07 exports evidence.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVISION_ROOT = PROJECT_ROOT / "reports" / "revision"
AUDIT_SCHEMA_VERSION = 4
AUTHENTICATED_BOOTSTRAP_UNIT = "authenticated_source_patient_record"
GROUP_SEMANTICS = "physionet_ecg_arrhythmia_one_patient_per_record_v1"
GROUP_REFERENCE = "https://physionet.org/content/ecg-arrhythmia/1.0.0/"
GROUP_REFERENCE_COUNTS = {
    "chapman_shaoxing": {"patients": 10247, "recordings": 10247},
    "ningbo": {"patients": 34905, "recordings": 34905},
}
NOTEBOOKS = (
    "00_colab_bootstrap.ipynb",
    "01_a0_protocol_audit.ipynb",
    "02_predictions_and_external_eval.ipynb",
    "02a_retrain_best_ema.ipynb",
    "03_calibration_and_ci.ipynb",
    "04_baselines_and_component_checks.ipynb",
    "05_hrv_domain_and_robustness.ipynb",
    "06_pooling_and_representation.ipynb",
    "07_results_freeze.ipynb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-root", type=Path, default=REVISION_ROOT)
    parser.add_argument("--canonical-root", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--q-tolerance", type=float, default=1e-6)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def authority_utc() -> str:
    """Use the immutable authority commit timestamp for reproducible audit output."""

    try:
        return subprocess.check_output(
            ["git", "show", "-s", "--format=%cI", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    if not fields:
        fields = ["status"]
        rows = [{"status": "empty"}]
    lines: list[str] = []
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    lines.append(buffer.getvalue())
    atomic_text(path, "".join(lines))


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def git_dirty() -> bool:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except Exception:
        return True


def source_bundle_contract() -> tuple[str, int]:
    roots = (
        PROJECT_ROOT / "scripts" / "revision",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "notebooks",
    )
    paths = sorted(
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix.lower() in {".py", ".ipynb", ".json", ".yaml", ".yml"}
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(paths)


def scalar(data: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in data.files:
        return default
    value = data[key]
    return value.item() if np.ndim(value) == 0 else value


def validate_prediction_arrays(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    if y_true.ndim != 2 or y_true.shape != y_prob.shape:
        raise ValueError(f"prediction shape mismatch: {y_true.shape} vs {y_prob.shape}")
    if not np.isfinite(y_true).all() or not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("labels are not finite binary indicators")
    if not np.isfinite(y_prob).all() or np.min(y_prob) < 0 or np.max(y_prob) > 1:
        raise ValueError("probabilities are not finite values in [0,1]")


def ece_binary_independent(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
    total = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (y_prob >= lower) & (y_prob <= upper if index == n_bins - 1 else y_prob < upper)
        if np.any(mask):
            total += float(np.mean(mask)) * abs(float(np.mean(y_true[mask])) - float(np.mean(y_prob[mask])))
    return total


def independent_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, n_bins: int) -> dict[str, float]:
    validate_prediction_arrays(y_true, y_prob)
    predicted = (y_prob >= threshold).astype(np.int8)
    evaluable = [index for index in range(y_true.shape[1]) if len(np.unique(y_true[:, index])) == 2]
    if not evaluable:
        raise ValueError("no class has both labels")
    roc = [roc_auc_score(y_true[:, index], y_prob[:, index]) for index in evaluable]
    pr = [average_precision_score(y_true[:, index], y_prob[:, index]) for index in evaluable]
    brier = [float(np.mean((y_prob[:, index] - y_true[:, index]) ** 2)) for index in evaluable]
    ece = [ece_binary_independent(y_true[:, index], y_prob[:, index], n_bins) for index in evaluable]
    nll = [
        log_loss(y_true[:, index], np.clip(y_prob[:, index], 1e-15, 1 - 1e-15), labels=[0, 1])
        for index in evaluable
    ]
    return {
        "f1_macro": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "pr_auc_macro": float(np.mean(pr)),
        "roc_auc_macro": float(np.mean(roc)),
        "brier_macro": float(np.mean(brier)),
        "nll_macro": float(np.mean(nll)),
        "ece_macro": float(np.mean(ece)),
        "n_classes_evaluated": float(len(evaluable)),
    }


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    slopes: list[float] = []
    intercepts: list[float] = []
    clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1.0 - clipped))
    for index in range(y_true.shape[1]):
        if len(np.unique(y_true[:, index])) < 2:
            continue
        model = LogisticRegression(C=1e8, solver="lbfgs", max_iter=2000)
        model.fit(logits[:, [index]], y_true[:, index])
        slopes.append(float(model.coef_[0, 0]))
        intercepts.append(float(model.intercept_[0]))
    return float(np.mean(slopes)), float(np.mean(intercepts))


def q_reconstruction(
    record_path: Path,
    slice_path: Path,
    q: float,
) -> tuple[float, dict[str, Any]]:
    with np.load(record_path, allow_pickle=False) as record_data:
        record_prob = np.asarray(record_data["y_prob"], dtype=np.float64)
        record_ids = np.asarray(record_data["record_id"]).astype(str)
        record_folds = np.asarray(record_data["fold_id"], dtype=np.int64)
    if len(np.unique(record_ids)) != len(record_ids):
        raise ValueError("OOF record_id is not unique")
    positions = {value: index for index, value in enumerate(record_ids)}
    with np.load(slice_path, allow_pickle=False) as slice_data:
        slice_prob = np.asarray(slice_data["slice_prob"], dtype=np.float64)
        slice_ids = np.asarray(slice_data["record_id"]).astype(str)
        slice_folds = np.asarray(slice_data["fold_id"], dtype=np.int64)
    try:
        record_index = np.fromiter((positions[value] for value in slice_ids), dtype=np.int64, count=len(slice_ids))
    except KeyError as exc:
        raise ValueError(f"slice references unknown record_id: {exc}") from exc
    if not np.array_equal(slice_folds, record_folds[record_index]):
        raise ValueError("slice fold does not match parent record fold")
    counts = np.bincount(record_index, minlength=len(record_ids))
    if np.any(counts == 0):
        raise ValueError("at least one OOF record has no slice")
    reconstructed = np.zeros_like(record_prob, dtype=np.float32)
    clipped = np.clip(slice_prob, 1e-6, 1.0 - 1e-6)
    for index in range(len(record_ids)):
        values = clipped[record_index == index]
        scaled = q * np.log(values)
        maximum = np.max(scaled, axis=0, keepdims=True)
        log_mean_power = np.squeeze(maximum, axis=0) + np.log(np.mean(np.exp(scaled - maximum), axis=0))
        reconstructed[index] = np.exp(log_mean_power / q).astype(np.float32)
    return float(np.max(np.abs(reconstructed - record_prob))), {
        "n_records": int(len(record_ids)),
        "n_slices": int(len(slice_ids)),
        "fold_counts": {str(value): int(np.sum(record_folds == value)) for value in sorted(np.unique(record_folds))},
    }


def oracle_rows(revision_root: Path, tolerance: float, q_tolerance: float) -> tuple[list[dict[str, Any]], list[str]]:
    prediction = revision_root / "predictions" / "oof_final_ema_predictions.npz"
    slices = revision_root / "predictions" / "oof_final_ema_slice_predictions.npz"
    calibration = revision_root / "metrics" / "calibration_ci_oof_final_ema_predictions.json"
    failures: list[str] = []
    if not prediction.exists() or not slices.exists() or not calibration.exists():
        missing = [str(path) for path in (prediction, slices, calibration) if not path.exists()]
        return [], ["missing statistical-oracle inputs: " + ", ".join(missing)]
    with np.load(prediction, allow_pickle=False) as data:
        y_true = np.asarray(data["y_true"], dtype=np.float64)
        y_prob = np.asarray(data["y_prob"], dtype=np.float64)
        threshold = float(scalar(data, "threshold", 0.5))
        q = float(scalar(data, "aggregation_q", 3.0))
    observed = independent_metrics(y_true, y_prob, threshold=threshold, n_bins=15)
    expected_payload = json.loads(calibration.read_text(encoding="utf-8"))
    expected = {**expected_payload.get("metrics", {}), **expected_payload.get("calibration", {})}
    slope, intercept = calibration_slope_intercept(y_true, y_prob)
    observed["calibration_slope_macro"] = slope
    observed["calibration_intercept_macro"] = intercept
    rows: list[dict[str, Any]] = []
    for metric, value in observed.items():
        expected_value = expected.get(metric)
        if expected_value is None:
            status = "descriptive_only"
            difference = ""
        else:
            difference_value = abs(float(value) - float(expected_value))
            difference = difference_value
            status = "pass" if difference_value <= tolerance else "fail"
            if status == "fail":
                failures.append(f"oracle mismatch {metric}: {value} vs {expected_value}")
        rows.append(
            {
                "check": "independent_metric",
                "metric": metric,
                "expected": "" if expected_value is None else expected_value,
                "observed": value,
                "absolute_difference": difference,
                "tolerance": tolerance,
                "status": status,
                "statistical_unit": AUTHENTICATED_BOOTSTRAP_UNIT,
            }
        )
    try:
        q_error, q_details = q_reconstruction(prediction, slices, q)
        q_status = "pass" if q_error <= q_tolerance else "fail"
        if q_status == "fail":
            failures.append(f"Q={q:g} reconstruction error {q_error} exceeds {q_tolerance}")
    except Exception as exc:
        q_error = math.inf
        q_status = "fail"
        q_details = {"error": str(exc)}
        failures.append(f"Q reconstruction failed: {exc}")
    rows.append(
        {
            "check": "slice_to_record_reconstruction",
            "metric": f"power_mean_q{q:g}_max_abs",
            "expected": 0.0,
            "observed": q_error,
            "absolute_difference": q_error,
            "tolerance": q_tolerance,
            "status": q_status,
            "statistical_unit": AUTHENTICATED_BOOTSTRAP_UNIT,
            "details": json.dumps(q_details, sort_keys=True),
        }
    )
    return rows, failures


def notebook_cell_rows() -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for notebook_name in NOTEBOOKS:
        path = PROJECT_ROOT / "notebooks" / notebook_name
        if not path.exists():
            failures.append(f"missing notebook: {notebook_name}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        heading = ""
        for index, cell in enumerate(payload.get("cells", [])):
            source = "".join(cell.get("source", []))
            if cell.get("cell_type") == "markdown":
                matches = re.findall(r"^#{1,6}\s+(.+)$", source, flags=re.MULTILINE)
                if matches:
                    heading = matches[-1].strip()
                continue
            if cell.get("cell_type") != "code":
                continue
            compile_source = source
            if source.lstrip().startswith("%%"):
                compile_source = "pass\n"
            else:
                compile_source = "\n".join(
                    "pass  # IPython command" if line.lstrip().startswith(("!", "%")) else line
                    for line in source.splitlines()
                )
            try:
                compile(compile_source, f"{notebook_name}:cell{index}", "exec")
                status = "pass"
                error = ""
            except SyntaxError as exc:
                status = "fail"
                error = f"{exc.msg} line {exc.lineno}"
                failures.append(f"{notebook_name} cell {index} does not compile: {error}")
            runners = sorted(set(re.findall(r"scripts/revision/[A-Za-z0-9_./-]+\.py", source)))
            rows.append(
                {
                    "notebook": notebook_name,
                    "cell_index": index,
                    "heading": heading,
                    "compile_status": status,
                    "compile_error": error,
                    "runner_count": len(runners),
                    "runners": ";".join(runners),
                    "has_run_logging": bool("log_path=" in source or "Durable command log" in source),
                    "has_cache_or_resume_control": bool(re.search(r"REUSE|RESUME|CACHE|restore|mirror", source, re.I)),
                }
            )
    return rows, failures


def reviewer_traceability_rows(
    revision_root: Path | None = None,
) -> list[dict[str, Any]]:
    specifications = [
        ("AE", "Highlighted manuscript and point-by-point response", "07", "07_results_freeze.ipynb::cells 14,16", "36_build_marked_manuscript.py; forbidden-claim scan", "clean manuscript and final evidence tables", "clean and marked PDFs plus response", "document", "Changes highlighted; no claim expansion"),
        ("R1-C1", "Calibration and reliability", "03,07", "03_calibration_and_ci.ipynb::cells 8,10; 07_results_freeze.ipynb::cell 6", "04_calibration_ci.py;29_reviewer_presentation_assets.py;42_matched_oof_calibration.py", "frozen OOF predictions", "reliability figure; calibration tables", "record/subject", "Calibration behavior; no clinical safety interpretation"),
        ("R1-C2", "Determinism versus learned morphology", "04", "04_baselines_and_component_checks.ipynb::cells 20,36", "39_morphology_learnability_control.py;40_paired_morphology_learnability.py", "same-fold reduced-bank controls", "paired control table", "record/subject", "Reduced-control endpoint only; no causal explanation of full branch"),
        ("R1-C3", "HRV domain bias", "05", "05_hrv_domain_and_robustness.ipynb::cell 8", "09_hrv_domain_analysis.py", "record-fingerprinted HRV caches", "HRV-only and domain tables", "record/subject", "Domain sensitivity as limitation; no invariance"),
        ("R1-C4", "CNN, Mamba and Transformer comparators", "04", "04_baselines_and_component_checks.ipynb::cells 12,14,16,28,30,32", "14_resnet1d_cnn_baseline.py;16_raw_mamba_baseline.py;24_transformer_ecg_baseline.py", "same folds and raw ECG", "OOF and paired tables", "record/subject", "Same-fold comparators, not budget-matched unless contract proves parity"),
        ("R1-C5", "Confidence intervals and significance", "02,03,04,05", "02_predictions_and_external_eval.ipynb::cells 32,34,36,42; 03_calibration_and_ci.ipynb::cells 8,10; 04_baselines_and_component_checks.ipynb::cells 26-36; 05_hrv_domain_and_robustness.ipynb::cells 10,14", "11_paired_full_vs_minirocket.py;15_paired_full_vs_resnet.py;17_paired_full_vs_raw_mamba.py;21_robustness_multicomparator.py;25_paired_full_vs_transformer.py;27_paired_full_vs_hybrid_morphology.py;32_paired_external_comparators.py;35_true_fewshot_head_adaptation.py;40_paired_morphology_learnability.py;50_refresh_in_domain_paired_contracts.py;52_ptbxl_fold_protocol_audit.py", "paired predictions and authenticated group IDs", "pointwise effect-size estimates and percentile CIs", "authenticated patient/group", "No significance or p-value claim without a separate 10,000-permutation paired null test and a pre-declared multiplicity family"),
        ("R1-C6", "Power Mean Q=3 sensitivity", "06", "06_pooling_and_representation.ipynb::cells 8,10,12", "07_pooling_sensitivity.py;30_pooling_sensitivity_external.py", "slice probabilities", "within/external sensitivity table", "record/group", "Frozen operating point; not globally optimal"),
        ("R1-C7", "HRV/PCA/Mamba implementation detail", "04,07", "04_baselines_and_component_checks.ipynb::cells 4,24; 07_results_freeze.ipynb::cell 6", "29_reviewer_presentation_assets.py", "fold PCA and run manifests", "per-fold provenance table", "fold", "Only implemented HRV slots; exact fold variance"),
        ("R2-C1", "Stepwise mathematical pipeline", "07/manuscript", "07_results_freeze.ipynb::cells 6,8,10", "29_reviewer_presentation_assets.py", "frozen protocol", "algorithm and notation audit", "protocol", "Definition and reproducibility claim only"),
        ("R2-C2", "Morphology/rhythm representation evidence", "06", "06_pooling_and_representation.ipynb::cell 16", "22_extract_representations.py;20_representation_probe.py", "checkpoint-local train/validation embeddings", "fold probes, fold CKA, descriptive UMAP", "held-out fold", "Selectivity audit; no proven disentanglement"),
        ("R2-C3", "Noise, leads and sampling robustness", "05", "05_hrv_domain_and_robustness.ipynb::cells 12,14", "23_generate_comparator_stress_predictions.py;21_robustness_multicomparator.py", "clean and six stress predictions", "paired degradation ledger", "record/subject", "Named stress/metric/comparator only"),
        ("R2-C4", "Zero-target-label versus few-shot", "02,02a", "02_predictions_and_external_eval.ipynb::cells 28,30,32,34,38,42", "33_group_safe_score_calibration.py;35_true_fewshot_head_adaptation.py;51_ptbxl_adaptation_analysis_lock.py;52_ptbxl_fold_protocol_audit.py", "PTB-XL official fold 9 adaptation/fold 10 test with zero patient overlap; Georgia/CPSC label-independent hash splits; CPSC primary units are complete non-transition 10-second mapped windows", "dataset-specific learning curves with CI plus an unsupported-only exclusion sensitivity", "patient/group or audited CPSC window parent", "Score calibration is separate from frozen-encoder head adaptation; the analysis lock is post-initial-review and is not preregistration; Georgia/CPSC are dataset-specific sensitivity analyses; CPSC is not record-level 27-class or official episode-boundary scoring"),
    ]
    required_artifacts = {
        "AE": ["tables/table_final_evidence_matrix.csv", "tables/table_final_safe_wording.csv"],
        "R1-C1": ["metrics/calibration_ci_oof_final_ema_predictions.json", "figures/figure_calibration_audit.png"],
        "R1-C2": ["metrics/paired_morphology_learnability_comparison.json"],
        "R1-C3": ["metrics/hrv_domain_summary.csv"],
        "R1-C4": ["metrics/paired_full_vs_resnet_comparison.json", "metrics/paired_full_vs_raw_mamba_comparison.json", "metrics/paired_full_vs_transformer_comparison.json"],
        "R1-C5": [
            "metrics/paired_full_vs_minirocket_comparison.json",
            "metrics/paired_full_vs_resnet_comparison.json",
            "metrics/paired_full_vs_raw_mamba_comparison.json",
            "metrics/paired_full_vs_transformer_comparison.json",
            "metrics/external_comparator_paired_summary.json",
            "metrics/robustness_multicomparator_pairwise.json",
            "metrics/true_fewshot_head_ptbxl_summary.csv",
            "metrics/ptbxl_fold_protocol_audit.json",
            "tables/table_ptbxl_unsupported_only_sensitivity.csv",
        ],
        "R1-C6": ["metrics/pooling_sensitivity.json", "metrics/pooling_sensitivity_external.csv"],
        "R1-C7": ["tables/table_fold_pca_provenance.csv"],
        "R2-C1": ["tables/table_hypothesis_control_claim_boundary.csv"],
        "R2-C2": ["metrics/representation_probe_summary.json", "figures/figure_representation_audit.png"],
        "R2-C3": ["metrics/robustness_multicomparator_pairwise.json"],
        "R2-C4": [
            "manifests/ptbxl_adaptation_analysis_lock.json",
            "manifests/ptbxl_adaptation_analysis_lock_source_attestation.json",
            "metrics/ptbxl_fold_protocol_audit.json",
            "tables/table_ptbxl_unsupported_only_sensitivity.csv",
            "metrics/true_fewshot_head_ptbxl_summary.csv",
            "metrics/group_safe_score_calibration_ptbxl_summary.csv",
        ],
    }
    rows = []
    for item, requirement, notebook, notebook_cell, runner, inputs, outputs, unit, boundary in specifications:
        artifacts = required_artifacts[item]
        missing = (
            [relative for relative in artifacts if not (revision_root / relative).is_file()]
            if revision_root is not None
            else []
        )
        rows.append(
            {
                "reviewer_item": item,
                "requirement": requirement,
                "notebook": notebook,
                "notebook_cell_reference": notebook_cell,
                "runner": runner,
                "code_evidence": "; ".join(
                    f"scripts/revision/{token.strip()}"
                    for token in runner.split(";")
                    if token.strip().endswith(".py")
                ),
                "inputs": inputs,
                "outputs": outputs,
                "statistical_unit": unit,
                "allowed_claim_boundary": boundary,
                "required_artifacts": ";".join(artifacts),
                "evidence_readiness": (
                    "not_evaluated"
                    if revision_root is None
                    else ("present_requires_contract_validation" if not missing else "missing")
                ),
                "missing_required_artifacts": ";".join(missing),
            }
        )
    return rows


def traceability_contract_failures(rows: list[dict[str, Any]]) -> list[str]:
    expected_items = {"AE", *(f"R1-C{index}" for index in range(1, 8)), *(f"R2-C{index}" for index in range(1, 5))}
    observed_items = {str(row.get("reviewer_item", "")) for row in rows}
    failures: list[str] = []
    if observed_items != expected_items:
        failures.append(
            "reviewer traceability item set mismatch: "
            f"missing={sorted(expected_items - observed_items)} extra={sorted(observed_items - expected_items)}"
        )
    required_fields = (
        "requirement",
        "notebook_cell_reference",
        "runner",
        "inputs",
        "outputs",
        "statistical_unit",
        "allowed_claim_boundary",
        "required_artifacts",
    )
    for row in rows:
        item = str(row.get("reviewer_item", "missing_item"))
        missing = [field for field in required_fields if not str(row.get(field, "")).strip()]
        if missing:
            failures.append(f"reviewer traceability {item} lacks required fields: {','.join(missing)}")
        if item != "AE" and row.get("evidence_readiness") == "missing":
            failures.append(
                f"reviewer traceability {item} is missing required evidence: "
                f"{row.get('missing_required_artifacts', '')}"
            )
        for relative in str(row.get("code_evidence", "")).split(";"):
            relative = relative.strip()
            if relative and not (PROJECT_ROOT / relative).is_file():
                failures.append(f"reviewer traceability {item} references missing code evidence: {relative}")
        references = [
            reference.strip()
            for reference in str(row.get("notebook_cell_reference", "")).split(";")
            if reference.strip()
        ]
        referenced_cell_sources: list[str] = []
        for reference in references:
            match = re.fullmatch(
                r"(?P<notebook>[A-Za-z0-9_]+\.ipynb)::cells?\s+(?P<indices>[0-9,\-\s]+)",
                reference,
            )
            if match is None:
                failures.append(
                    f"reviewer traceability {item} has invalid notebook cell reference: {reference}"
                )
                continue
            notebook_path = PROJECT_ROOT / "notebooks" / match.group("notebook")
            if not notebook_path.is_file():
                failures.append(
                    f"reviewer traceability {item} references missing notebook: {notebook_path.name}"
                )
                continue
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            cells = notebook.get("cells", [])
            referenced_indices: set[int] = set()
            for token in match.group("indices").split(","):
                token = token.strip()
                if not token:
                    continue
                if "-" in token:
                    start_raw, end_raw = token.split("-", 1)
                    start, end = int(start_raw), int(end_raw)
                    if end < start:
                        failures.append(
                            f"reviewer traceability {item} has descending cell range: {reference}"
                        )
                        continue
                    referenced_indices.update(range(start, end + 1))
                else:
                    referenced_indices.add(int(token))
            for index in sorted(referenced_indices):
                if index < 0 or index >= len(cells):
                    failures.append(
                        f"reviewer traceability {item} references out-of-range cell "
                        f"{match.group('notebook')}::{index}"
                    )
                else:
                    referenced_cell_sources.append("".join(cells[index].get("source", [])))
        if item != "AE":
            combined_source = "\n".join(referenced_cell_sources)
            expected_runners = [
                Path(token.strip()).name
                for token in str(row.get("runner", "")).split(";")
                if token.strip().endswith(".py")
            ]
            for runner_name in expected_runners:
                if runner_name not in combined_source:
                    failures.append(
                        f"reviewer traceability {item} runner {runner_name} is not called by its referenced cells"
                    )
    return failures


def resolve_revision_artifact(value: str, revision_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project_candidate = (PROJECT_ROOT / path).resolve()
    if project_candidate.exists():
        return project_candidate
    return (revision_root / path).resolve()


def bootstrap_contract_failures(
    revision_root: Path,
    *,
    oof_sha: str,
    freeze_sha: str,
) -> list[str]:
    failures: list[str] = []
    freeze_path = revision_root / "manifests" / "oof_final_ema_freeze_manifest.json"
    calibration_path = revision_root / "metrics" / "calibration_ci_oof_final_ema_predictions.json"
    if not freeze_path.exists() or freeze_path.stat().st_size == 0:
        return ["authenticated bootstrap contract: freeze manifest is missing"]
    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"authenticated bootstrap contract: freeze manifest is unreadable: {exc}"]
    group = freeze.get("group_contract") or {}
    membership = freeze.get("membership_contract") or {}
    expected_records = int(freeze.get("validated_records", -1))
    freeze_checks = {
        "freeze_manuscript_ready": freeze.get("manuscript_ready") is True,
        "strict_manuscript_contract": freeze.get("strict_manuscript_contract") is True,
        "checkpoint_membership": membership.get("status") == "verified",
    }
    for field, passed in freeze_checks.items():
        if not passed:
            failures.append(f"authenticated bootstrap contract: freeze {field} is invalid")
    group_checks = {
        "status": group.get("status") == "verified",
        "group_semantics": group.get("group_semantics") == GROUP_SEMANTICS,
        "group_semantics_reference": group.get("group_semantics_reference") == GROUP_REFERENCE,
        "source_patient_record_counts": group.get("source_patient_record_counts") == GROUP_REFERENCE_COUNTS,
        "bootstrap_unit": group.get("bootstrap_unit") == AUTHENTICATED_BOOTSTRAP_UNIT,
        "one_record_per_group": group.get("one_record_per_group") is True,
        "n_records": int(group.get("n_records", -1)) == expected_records,
        "n_groups": int(group.get("n_groups", -1)) == expected_records,
    }
    for field, passed in group_checks.items():
        if not passed:
            failures.append(
                f"authenticated bootstrap contract: freeze group_contract.{field} is invalid"
            )
    sidecar = group.get("sidecar") or {}
    sidecar_value = str(sidecar.get("path", ""))
    sidecar_path = resolve_revision_artifact(sidecar_value, revision_root) if sidecar_value else None
    if sidecar_path is None or not sidecar_path.exists() or sidecar_path.stat().st_size == 0:
        failures.append("authenticated bootstrap contract: group sidecar is missing")
    elif sidecar.get("sha256") != sha256_file(sidecar_path):
        failures.append("authenticated bootstrap contract: group sidecar SHA256 mismatch")

    if not calibration_path.exists() or calibration_path.stat().st_size == 0:
        failures.append("authenticated bootstrap contract: calibration artifact is missing")
        return failures
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"authenticated bootstrap contract: calibration artifact is unreadable: {exc}")
        return failures
    bootstrap = calibration.get("bootstrap") or {}
    calibration_checks = {
        "predictions_sha256": calibration.get("predictions_sha256") == oof_sha,
        "freeze_manifest_sha256": calibration.get("freeze_manifest_sha256") == freeze_sha,
        "unit": bootstrap.get("unit") == AUTHENTICATED_BOOTSTRAP_UNIT,
        "independence_contract": bootstrap.get("independence_contract") == GROUP_SEMANTICS,
        "group_semantics_reference": bootstrap.get("group_semantics_reference") == GROUP_REFERENCE,
        "group_sidecar_sha256": bootstrap.get("group_sidecar_sha256") == sidecar.get("sha256"),
    }
    for field, passed in calibration_checks.items():
        if not passed:
            failures.append(f"authenticated bootstrap contract: calibration {field} is invalid")
    return failures


def training_loader_source_failures() -> list[str]:
    """Verify that the canonical trainer constructs disjoint fold-local loaders."""

    path = PROJECT_ROOT / "scripts" / "train.py"
    if not path.exists():
        return ["training loader source contract: scripts/train.py is missing"]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        return [f"training loader source contract: scripts/train.py is unreadable: {exc}"]

    slice_sources: dict[str, str] = {}
    loader_names: dict[str, set[str]] = {}
    trained_from_loader = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            function = node.value.func
            function_name = function.id if isinstance(function, ast.Name) else ""
            targets = {
                name.id
                for target in node.targets
                for name in ast.walk(target)
                if isinstance(name, ast.Name)
            }
            if function_name == "build_slice_index" and node.value.args:
                source = node.value.args[0]
                if isinstance(source, ast.Name):
                    for target in targets & {"rid_tr", "rid_va"}:
                        slice_sources[target] = source.id
            if function_name == "make_loader":
                referenced = {
                    name.id for name in ast.walk(node.value) if isinstance(name, ast.Name)
                }
                for target in targets & {"train_loader", "val_loader"}:
                    loader_names[target] = referenced
        if isinstance(node, (ast.For, ast.AsyncFor)):
            if isinstance(node.iter, ast.Name) and node.iter.id == "train_loader":
                trained_from_loader = True

    failures: list[str] = []
    if slice_sources.get("rid_tr") != "tr_idx":
        failures.append("training loader source contract: rid_tr is not derived from tr_idx")
    if slice_sources.get("rid_va") != "va_idx":
        failures.append("training loader source contract: rid_va is not derived from va_idx")
    train_refs = loader_names.get("train_loader", set())
    val_refs = loader_names.get("val_loader", set())
    if "rid_tr" not in train_refs or "rid_va" in train_refs:
        failures.append("training loader source contract: train_loader record membership is invalid")
    if "rid_va" not in val_refs or "rid_tr" in val_refs:
        failures.append("training loader source contract: val_loader record membership is invalid")
    if not trained_from_loader:
        failures.append("training loader source contract: optimizer loop does not consume train_loader")
    return failures


def _flatten_payload(payload: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_payload(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            rows.extend(_flatten_payload(value, f"{prefix}[{index}]"))
    else:
        rows.append((prefix, payload))
    return rows


def _finite_p_value(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed)


def _valid_permutation_null_contract(flattened: list[tuple[str, Any]]) -> tuple[bool, str]:
    methods = [
        str(value).lower()
        for path, value in flattened
        if any(token in path.lower() for token in ("test_method", "null_test", "randomization_test"))
    ]
    permutation_method = any(
        "permutation" in value
        and "paired" in value
        and any(token in value for token in ("sign_flip", "sign-flip", "label_swap", "label-swap"))
        for value in methods
    )
    permutation_counts = []
    for path, value in flattened:
        if any(token in path.lower() for token in ("n_perm", "n_randomization")):
            try:
                permutation_counts.append(int(float(value)))
            except (TypeError, ValueError):
                continue
    adjustments = [
        str(value).lower()
        for path, value in flattened
        if any(token in path.lower() for token in ("multiplicity", "adjustment", "correction"))
    ]
    families = [
        str(value).strip()
        for path, value in flattened
        if "family" in path.lower() and str(value).strip().lower() not in {"", "none", "not_declared"}
    ]
    null_centered = any(
        (
            "null_centered" in path.lower()
            and (
                value is True
                or str(value).strip().lower()
                in {"true", "yes", "paired_difference_under_exchangeability", "zero_centered"}
            )
        )
        for path, value in flattened
    )
    group_units = [
        str(value).lower()
        for path, value in flattened
        if any(
            token in path.lower()
            for token in ("sample_unit", "permutation_unit", "group_unit", "bootstrap_unit")
        )
    ]
    authenticated_group_unit = any(
        any(token in value for token in ("patient", "subject", "group"))
        for value in group_units
    )
    group_contract_shas = [
        str(value).strip().lower()
        for path, value in flattened
        if any(token in path.lower() for token in ("group_sidecar_sha256", "group_contract_sha256"))
    ]
    authenticated_group_sha = any(re.fullmatch(r"[0-9a-f]{64}", value) for value in group_contract_shas)
    valid = (
        permutation_method
        and bool(permutation_counts)
        and max(permutation_counts) >= 10_000
        and any("holm" in value for value in adjustments)
        and bool(families)
        and null_centered
        and authenticated_group_unit
        and authenticated_group_sha
    )
    detail = (
        f"paired_permutation={permutation_method}; n_perm={max(permutation_counts, default=0)}; "
        f"holm={any('holm' in value for value in adjustments)}; family_declared={bool(families)}; "
        f"null_centered={null_centered}; authenticated_group_unit={authenticated_group_unit}; "
        f"authenticated_group_sha={authenticated_group_sha}"
    )
    return valid, detail


def _iter_dict_nodes(payload: Any, prefix: str = "") -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(payload, dict):
        yield prefix or "root", payload
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_dict_nodes(value, child)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from _iter_dict_nodes(value, f"{prefix}[{index}]")


def _endpoint_contract_registry(payload: Any) -> dict[str, list[tuple[str, bool, str]]]:
    registry: dict[str, list[tuple[str, bool, str]]] = {}
    for path, node in _iter_dict_nodes(payload):
        contract_id = str(node.get("inference_contract_id") or "").strip()
        if not contract_id:
            continue
        flattened = _flatten_payload(node)
        has_contract_fields = any(
            any(
                token in key.lower()
                for token in (
                    "test_method",
                    "null_test",
                    "randomization_test",
                    "n_perm",
                    "multiplicity",
                    "group_sidecar_sha256",
                )
            )
            for key, _ in flattened
        )
        if not has_contract_fields:
            continue
        valid, detail = _valid_permutation_null_contract(flattened)
        registry.setdefault(contract_id, []).append((path, valid, detail))
    return registry


def _endpoint_inference_findings(payload: Any) -> list[dict[str, Any]]:
    registry = _endpoint_contract_registry(payload)
    findings: list[dict[str, Any]] = []
    for path, node in _iter_dict_nodes(payload):
        direct_scalars = [
            (str(key), value)
            for key, value in node.items()
            if not isinstance(value, (dict, list))
        ]
        significance_values = [
            f"{key}={value}"
            for key, value in direct_scalars
            if (
                isinstance(value, str)
                and "significant" in value.lower()
            )
            or (
                isinstance(value, bool)
                and value
                and "significant" in key.lower()
            )
        ]
        finite_p_values = [
            (key, value)
            for key, value in direct_scalars
            if ("p_value" in key.lower() or "pvalue" in key.lower())
            and not key.lower().endswith("_count")
            and _finite_p_value(value)
        ]
        if not significance_values and not finite_p_values:
            continue

        contract_id = str(node.get("inference_contract_id") or "").strip()
        candidates = registry.get(contract_id, []) if contract_id else []
        valid_candidates = [candidate for candidate in candidates if candidate[1]]
        valid_contract = bool(contract_id) and len(valid_candidates) == 1
        contract_detail = (
            valid_candidates[0][2]
            if valid_contract
            else (
                "missing_endpoint_local_inference_contract_id"
                if not contract_id
                else f"valid_contract_candidates={len(valid_candidates)} total_candidates={len(candidates)}"
            )
        )
        raw_p_values = [
            (key, value)
            for key, value in finite_p_values
            if "holm" not in key.lower() and "adjust" not in key.lower()
        ]
        holm_p_values = [
            (key, value)
            for key, value in finite_p_values
            if "holm" in key.lower()
        ]
        complete_p_value_pair = bool(raw_p_values) and bool(holm_p_values)
        issues: list[str] = []
        if not valid_contract:
            issues.append("missing_or_invalid_endpoint_local_permutation_contract")
        if not complete_p_value_pair:
            issues.append("missing_raw_and_holm_adjusted_p_value_pair")
        findings.append(
            {
                "endpoint": path,
                "inference_contract_id": contract_id,
                "significance_values": significance_values,
                "finite_p_values": finite_p_values,
                "valid_contract": valid_contract,
                "contract_detail": contract_detail,
                "issues": issues,
            }
        )
    return findings


def paired_inference_audit_rows(revision_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Reject significance language or finite p-values without a valid null test.

    Percentile bootstrap intervals remain valid effect-size uncertainty summaries,
    but bootstrap-tail proportions are not null-centered randomization tests.
    """

    excluded_names = {
        "artifact_provenance_audit.json",
        "table_paired_inference_audit.csv",
        "table_notebook_cell_audit.csv",
        "table_reviewer_traceability.csv",
        "table_statistical_oracle_check.csv",
        "table_forensic_rerun_dependencies.csv",
    }
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for parent in (revision_root / "metrics", revision_root / "tables"):
        if not parent.exists():
            continue
        for path in sorted(parent.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".json", ".csv"}:
                continue
            if path.name in excluded_names or "forbidden_claim_scan" in path.name:
                continue
            payload: Any = None
            parse_error = ""
            try:
                if path.suffix.lower() == ".json":
                    payload = json.loads(path.read_text(encoding="utf-8"))
                else:
                    with path.open("r", encoding="utf-8-sig", newline="") as handle:
                        payload = list(csv.DictReader(handle))
            except Exception as exc:
                parse_error = str(exc)

            endpoint_findings = _endpoint_inference_findings(payload) if parse_error == "" else []
            significance_values = [
                value
                for finding in endpoint_findings
                for value in finding["significance_values"]
            ]
            finite_p_values = [
                (f"{finding['endpoint']}.{key}", value)
                for finding in endpoint_findings
                for key, value in finding["finite_p_values"]
            ]
            unsafe_endpoints = [finding for finding in endpoint_findings if finding["issues"]]
            valid_null = bool(endpoint_findings) and not unsafe_endpoints
            null_detail = "; ".join(
                f"{finding['endpoint']}:{finding['contract_detail']}"
                for finding in endpoint_findings[:5]
            )
            status = "pass"
            issues: list[str] = []
            if parse_error:
                issues.append(f"parse_error={parse_error}")
            if significance_values and unsafe_endpoints:
                issues.append(
                    "significance_language_without_valid_paired_permutation_contract="
                    + ",".join(sorted(set(significance_values))[:5])
                )
            if finite_p_values and unsafe_endpoints:
                issues.append(
                    "finite_p_values_without_valid_paired_permutation_contract="
                    + ",".join(key for key, _ in finite_p_values[:5])
                )
            if unsafe_endpoints:
                issues.append(
                    "unsafe_endpoint_contracts="
                    + ",".join(
                        f"{finding['endpoint']}:{'|'.join(finding['issues'])}"
                        for finding in unsafe_endpoints[:5]
                    )
                )
            if issues:
                status = "fail"
                relative = str(path.relative_to(revision_root)).replace("\\", "/")
                failures.append(f"paired inference artifact is unsafe: {relative}: {'; '.join(issues)}")
            rows.append(
                {
                    "artifact": str(path.relative_to(revision_root)).replace("\\", "/"),
                    "status": status,
                    "interpretation_count": len(significance_values),
                    "significance_label_count": len(significance_values),
                    "finite_p_value_count": len(finite_p_values),
                    "endpoint_contract_count": len(endpoint_findings),
                    "unsafe_endpoint_count": len(unsafe_endpoints),
                    "valid_paired_permutation_contract": valid_null,
                    "null_contract_detail": null_detail,
                    "issues": "; ".join(issues),
                    "allowed_inference": (
                        "multiplicity_adjusted_null_test"
                        if valid_null
                        else "pointwise_effect_size_and_percentile_ci_only"
                    ),
                }
            )
    return rows, failures


def nested(payload: dict[str, Any], paths: Iterable[str], default: Any = "not_declared") -> Any:
    for candidate in paths:
        value: Any = payload
        valid = True
        for part in candidate.split("."):
            if not isinstance(value, dict) or part not in value:
                valid = False
                break
            value = value[part]
        if valid and value not in (None, "", [], {}):
            return value
    return default


def contract_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def comparator_contract_rows(revision_root: Path, oof_sha: str, freeze_sha: str) -> tuple[list[dict[str, Any]], list[str]]:
    definitions = {
        "Full ECG-RAMBA": ("oof_final_ema_prediction_summary.json", "oof_final_ema_prediction_run_manifest.json", "oof_final_ema_predictions.npz"),
        "Fixed-seed ROCKET-family MAX+PPV linear head": ("minirocket_only_baseline_summary.json", "minirocket_only_baseline_manifest.json", "minirocket_only_oof_predictions.npz"),
        "ResNet1D/CNN": ("resnet1d_cnn_baseline_summary.json", "resnet1d_cnn_baseline_manifest.json", "resnet1d_cnn_oof_predictions.npz"),
        "Raw Mamba": ("raw_mamba_baseline_summary.json", "raw_mamba_baseline_manifest.json", "raw_mamba_oof_predictions.npz"),
        "Compact Transformer ECG": ("transformer_ecg_baseline_summary.json", "transformer_ecg_baseline_manifest.json", "transformer_ecg_oof_predictions.npz"),
        "Frozen-transform morphology MLP": ("hybrid_morphology_baseline_summary.json", "hybrid_morphology_baseline_manifest.json", "hybrid_morphology_oof_predictions.npz"),
    }
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for name, (summary_name, manifest_name, prediction_name) in definitions.items():
        summary_path = revision_root / "metrics" / summary_name
        manifest_path = revision_root / "manifests" / manifest_name
        prediction_path = revision_root / "predictions" / prediction_name
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        training_contract = summary.get("comparator_contract", {})
        if not isinstance(training_contract, dict):
            training_contract = {}
        metadata: dict[str, Any] = {}
        n_records: Any = "not_declared"
        fold_count: Any = "not_declared"
        if prediction_path.exists():
            try:
                with np.load(prediction_path, allow_pickle=False) as data:
                    n_records = int(len(data["y_true"]))
                    fold_count = int(len(np.unique(data["fold_id"]))) if "fold_id" in data.files else "not_declared"
                    for key in ("oof_predictions_sha256", "freeze_manifest_sha256", "git_commit", "protocol"):
                        metadata[key] = scalar(data, key, "")
            except Exception as exc:
                failures.append(f"invalid comparator NPZ {name}: {exc}")
        declared_oof = str(
            nested(
                manifest,
                (
                    "inputs.oof_predictions.sha256",
                    "source_oof.sha256",
                    "load_info.oof_predictions_sha256",
                    "load_info.freeze_contract.oof_predictions_sha256",
                    "freeze_contract.oof_predictions_sha256",
                ),
                metadata.get("oof_predictions_sha256", ""),
            )
        )
        declared_freeze = str(
            nested(
                manifest,
                (
                    "inputs.freeze_manifest.sha256",
                    "source_freeze.sha256",
                    "load_info.freeze_manifest_sha256",
                    "load_info.freeze_contract.freeze_manifest_sha256",
                    "freeze_contract.freeze_manifest_sha256",
                ),
                metadata.get("freeze_manifest_sha256", ""),
            )
        )
        contract_current = name == "Full ECG-RAMBA" or (declared_oof == oof_sha and declared_freeze == freeze_sha)
        if name != "Full ECG-RAMBA" and summary_path.exists() and not contract_current:
            failures.append(f"{name} comparator contract is stale or does not bind current OOF/freeze")
        batch_paths = (
            "comparator_contract.batch_size",
            "training.batch_size",
            "model_params.batch_size",
            "classifier_params.batch_size",
        )
        if name != "Full ECG-RAMBA":
            batch_paths = (*batch_paths, "batch_size", "load_info.batch_size")
        contract_fields = {
            "preprocessing": nested(
                summary,
                ("comparator_contract.preprocessing", "preprocessing", "feature_preprocessing", "load_info.preprocessing", "protocol"),
            ),
            "training_unit": nested(summary, ("comparator_contract.training_unit",)),
            "optimizer_steps": nested(
                summary,
                ("comparator_contract.optimizer_steps", "training.optimizer_steps", "training_protocol.optimizer_steps", "model_params.optimizer_steps", "classifier_params.optimizer_steps"),
            ),
            "batch_size": nested(summary, batch_paths),
            "epochs": nested(summary, ("comparator_contract.epochs", "training.epochs", "model_params.epochs", "classifier_params.epochs")),
            "loss": nested(summary, ("comparator_contract.loss", "training.loss", "training_protocol.loss", "model_params.loss", "classifier_params.loss", "loss")),
            "regularization": nested(summary, ("comparator_contract.regularization", "training.regularization", "training_protocol.regularization", "model_params.weight_decay", "classifier_params.weight_decay", "weight_decay")),
            "amp": nested(summary, ("comparator_contract.amp", "training.amp", "model_params.amp", "runtime_config.amp", "runtime.amp")),
            "seed": nested(summary, ("comparator_contract.seed", "seed", "training.seed", "model_params.seed", "classifier_params.seed", "load_info.seed")),
            "checkpoint_rule": nested(summary, ("comparator_contract.checkpoint_rule", "checkpoint_rule", "training_protocol.model_selection", "model_params.selection_rule", "classifier_params.selection_rule", "model_selection")),
            "tuning_provenance": nested(summary, ("comparator_contract.tuning_provenance", "tuning_provenance", "training.tuning_provenance", "model_params.tuning_provenance", "classifier_params.tuning_provenance")),
        }
        missing_training_fields = sorted(
            field
            for field, value in contract_fields.items()
            if value == "not_declared" or "not_recorded" in str(value)
        )
        rows.append(
            {
                "comparator": name,
                "comparison_class": "primary_model" if name == "Full ECG-RAMBA" else "same-fold_comparator",
                "summary_exists": summary_path.exists(),
                "manifest_exists": manifest_path.exists(),
                "prediction_exists": prediction_path.exists(),
                "n_records": n_records,
                "fold_count": fold_count,
                "contract_schema_version": training_contract.get("schema_version", "legacy_or_missing"),
                "training_contract_status": "complete" if not missing_training_fields else "incomplete_explicit",
                "missing_training_fields": ";".join(missing_training_fields),
                "preprocessing": contract_cell(contract_fields["preprocessing"]),
                "training_unit": contract_cell(contract_fields["training_unit"]),
                "optimizer_steps": contract_cell(contract_fields["optimizer_steps"]),
                "batch_size": contract_cell(contract_fields["batch_size"]),
                "epochs": contract_cell(contract_fields["epochs"]),
                "loss": contract_cell(contract_fields["loss"]),
                "regularization": contract_cell(contract_fields["regularization"]),
                "amp": contract_cell(contract_fields["amp"]),
                "seed": contract_cell(contract_fields["seed"]),
                "checkpoint_rule": contract_cell(contract_fields["checkpoint_rule"]),
                "tuning_provenance": contract_cell(contract_fields["tuning_provenance"]),
                "current_oof_freeze_contract": contract_current,
                "fairness_boundary": training_contract.get(
                    "fairness_boundary",
                    "same folds; do not call budget-matched unless all training fields are declared and equal",
                ),
            }
        )
    return rows, failures


def canonical_manifest_rows(root: Path, revision_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    manifest_path = root / "manifests" / "mirror_manifest.json"
    if not manifest_path.exists():
        return [], [f"canonical mirror manifest missing: {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    trust_scope = str(manifest.get("trust_scope", ""))
    if "byte_integrity_only" not in trust_scope:
        failures.append(
            "canonical mirror manifest does not declare its byte-integrity-only trust boundary"
        )
    for row in manifest.get("artifacts", []):
        relative = str(row.get("relative_path", ""))
        path = root / relative
        expected_sha = str(row.get("sha256", ""))
        expected_size = int(row.get("size_bytes", -1))
        exists = path.exists() and path.is_file()
        actual_size = path.stat().st_size if exists else -1
        actual_sha = sha256_file(path) if exists else ""
        status = "pass" if exists and actual_size == expected_size and actual_sha == expected_sha else "fail"
        if status == "fail":
            failures.append(f"canonical manifest mismatch: {relative}")
        rows.append(
            {
                "relative_path": relative,
                "exists": exists,
                "expected_size": expected_size,
                "actual_size": actual_size,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "attestation_scope": str(
                    row.get("attestation_scope", "byte_integrity_only_legacy")
                ),
                "status": status,
            }
        )
    if not rows:
        failures.append("canonical manifest contains no artifacts")
    return rows, failures


def rerun_dependency_rows(revision_root: Path, current_head: str) -> list[dict[str, Any]]:
    stages = [
        ("02_oof_freeze", (), ("predictions/oof_final_ema_predictions.npz",), "A100 only if fold cache missing"),
        ("03_calibration", ("02_oof_freeze",), ("metrics/calibration_ci_oof_final_ema_predictions.json",), "CPU High-RAM"),
        ("04_comparators", ("02_oof_freeze",), ("manifests/baseline_component_input_contract.json",), "A100 missing folds; CPU paired inference"),
        ("02_external", ("02_oof_freeze",), (
            "metrics/external_protocol_gate_summary.csv",
            "manifests/external_ptbxl_protocol_gate_manifest.json",
            "manifests/external_georgia_protocol_gate_manifest.json",
            "manifests/external_cpsc2021_protocol_gate_manifest.json",
        ), "A100 missing exports; CPU gates"),
        ("05_robustness", ("02_oof_freeze", "04_comparators"), ("manifests/robustness_multicomparator_manifest.json",), "A100 missing stress predictions; CPU ledger"),
        ("06_representation", ("02_oof_freeze",), ("manifests/representation_probe_manifest.json",), "A100 missing fold-local embeddings; CPU probes"),
        ("07_final", ("03_calibration", "04_comparators", "02_external", "05_robustness", "06_representation"), ("manifests/final_evidence_matrix_manifest.json",), "CPU High-RAM"),
    ]
    rows = []
    upstream_state: dict[str, bool] = {}
    for stage, dependencies, artifact_rels, hardware in stages:
        paths = [revision_root / relative for relative in artifact_rels]
        present = all(path.exists() and path.stat().st_size > 0 for path in paths)
        producer_checks: list[bool] = []
        for path in paths:
            if not path.exists() or path.suffix.lower() == ".csv":
                continue
            try:
                if path.suffix.lower() == ".npz":
                    with np.load(path, allow_pickle=False) as data:
                        producer_checks.append(str(scalar(data, "git_commit", "")) == current_head)
                else:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    producer_checks.append(str(payload.get("git_commit", "")) == current_head)
            except Exception:
                producer_checks.append(False)
        producer_current = bool(producer_checks) and all(producer_checks)
        dependencies_ready = all(upstream_state.get(item, False) for item in dependencies)
        ready = present and producer_current and dependencies_ready
        upstream_state[stage] = ready
        rows.append(
            {
                "stage": stage,
                "dependencies": ";".join(dependencies),
                "artifacts": ";".join(artifact_rels),
                "artifact_present": present,
                "producer_current_head": producer_current,
                "dependencies_ready": dependencies_ready,
                "rerun_required": not ready,
                "hardware": hardware,
                "reason": "ready" if ready else "dependency/provenance contract failed; file presence alone is insufficient",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    revision_root = args.revision_root.resolve()
    current_head = git_head()
    working_tree_dirty = git_dirty()
    source_bundle_sha, source_bundle_files = source_bundle_contract()
    created_utc = authority_utc()
    audit_dir = revision_root / "audits"
    table_dir = revision_root / "tables"
    metric_dir = revision_root / "metrics"
    audit_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    notebook_rows, notebook_failures = notebook_cell_rows()
    oracle, oracle_failures = oracle_rows(revision_root, args.tolerance, args.q_tolerance)
    traceability = reviewer_traceability_rows(revision_root)
    traceability_failures = traceability_contract_failures(traceability)
    oof_path = revision_root / "predictions" / "oof_final_ema_predictions.npz"
    freeze_path = revision_root / "manifests" / "oof_final_ema_freeze_manifest.json"
    oof_sha = sha256_file(oof_path) if oof_path.exists() else ""
    freeze_sha = sha256_file(freeze_path) if freeze_path.exists() else ""
    bootstrap_failures = bootstrap_contract_failures(
        revision_root,
        oof_sha=oof_sha,
        freeze_sha=freeze_sha,
    )
    training_loader_failures = training_loader_source_failures()
    comparators, comparator_failures = comparator_contract_rows(revision_root, oof_sha, freeze_sha)
    paired_inference_rows, paired_inference_failures = paired_inference_audit_rows(revision_root)

    canonical_root = args.canonical_root.resolve() if args.canonical_root else revision_root
    provenance_rows, provenance_failures = canonical_manifest_rows(canonical_root, revision_root)
    rerun_rows = rerun_dependency_rows(revision_root, current_head)
    rerun_failures = [
        f"forensic rerun dependency is not ready: {row['stage']} ({row['reason']})"
        for row in rerun_rows
        if row.get("rerun_required")
    ]
    producer_commit = ""
    if oof_path.exists():
        with np.load(oof_path, allow_pickle=False) as data:
            producer_commit = str(scalar(data, "git_commit", ""))
    authority_failures = []
    if not current_head:
        authority_failures.append("current git HEAD is unavailable")
    if working_tree_dirty:
        authority_failures.append(
            "code authority is not immutable: git working tree contains tracked or untracked changes"
        )
    if producer_commit != current_head:
        authority_failures.append(f"OOF producer commit {producer_commit or 'missing'} != authority {current_head}")
    if len(traceability) != 12:
        authority_failures.append("traceability must contain Associate Editor plus 11 reviewer comments")

    failures = (
        notebook_failures
        + oracle_failures
        + traceability_failures
        + bootstrap_failures
        + training_loader_failures
        + comparator_failures
        + paired_inference_failures
        + provenance_failures
        + rerun_failures
        + authority_failures
    )
    p0 = [item for item in failures if "oracle" in item.lower() or "q=" in item.lower() or "prediction shape" in item.lower()]
    p1 = [item for item in failures if item not in p0]
    training_contract_fields = (
        "optimizer_steps",
        "batch_size",
        "loss",
        "regularization",
        "amp",
        "seed",
        "checkpoint_rule",
        "tuning_provenance",
    )
    p2 = []
    for row in comparators:
        if row.get("comparison_class") != "same-fold_comparator" or not row.get("summary_exists"):
            continue
        missing_fields = [field for field in training_contract_fields if row.get(field) == "not_declared"]
        if missing_fields:
            p2.append(
                "tables/table_comparator_contract.csv: "
                f"{row['comparator']} does not declare {','.join(missing_fields)}; "
                "retain the same-fold comparator boundary and do not call it budget-matched"
            )
    physiological_probe = revision_root / "metrics" / "physiological_interval_probe_summary.json"
    if not physiological_probe.exists():
        p2.append(
            "metrics/physiological_interval_probe_summary.json: independently provenance-bound interval targets "
            "are unavailable; do not create proxy HR/PR/QRS/QT/QTc targets"
        )
    if freeze_path.exists():
        freeze_payload = json.loads(freeze_path.read_text(encoding="utf-8"))
        record_order_hash = str(
            freeze_payload.get("dataset_record_order_fingerprint", "")
        )
        if len(record_order_hash) != 64:
            p2.append(
                "manifests/oof_final_ema_freeze_manifest.json: the legacy record-order "
                "fingerprint is truncated; full content and sidecar SHA contracts remain "
                "authoritative until a future checkpoint-compatible migration"
            )
    status = not p0 and not p1
    provenance_payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "created_utc": created_utc,
        "status": status,
        "audit_framework": {
            "reporting": "TRIPOD+AI",
            "reporting_reference": "https://www.bmj.com/content/385/bmj-2023-078378",
            "risk_of_bias": "PROBAST+AI",
            "risk_of_bias_reference": "https://www.bmj.com/content/388/bmj-2024-082505",
            "application": (
                "traceability, participant/data-flow reporting, outcome and statistical-unit contracts, "
                "analysis leakage controls, comparator fairness, calibration, and external-validation boundaries"
            ),
        },
        "code_authority_git_commit": current_head,
        "code_authority_worktree_dirty": working_tree_dirty,
        "source_bundle_sha256": source_bundle_sha,
        "source_bundle_file_count": source_bundle_files,
        "oof_producer_git_commit": producer_commit,
        "oof_sha256": oof_sha,
        "freeze_sha256": freeze_sha,
        "authenticated_bootstrap_contract_failures": bootstrap_failures,
        "training_loader_source_contract_failures": training_loader_failures,
        "canonical_root": str(canonical_root),
        "canonical_manifest_artifacts": len(provenance_rows),
        "canonical_manifest_failures": provenance_failures,
        "paired_inference_artifacts": len(paired_inference_rows),
        "paired_inference_failures": paired_inference_failures,
        "p0_findings": p0,
        "p1_findings": p1,
        "p2_findings": p2,
        "rerun_dependency_graph": rerun_rows,
        "claim_boundary": {
            "representation": "checkpoint-local held-out probes and fold CKA are an audit, not proof of disentanglement",
            "comparators": "same-fold comparator unless a complete budget-matching contract is present",
            "robustness": "named metric/stress/comparator only; pointwise CI is not family-wise significance",
            "adaptation": "score calibration and frozen-encoder head adaptation are distinct",
            "clinical": "no prospective validation or clinical-readiness claim",
        },
    }

    save_csv(table_dir / "table_notebook_cell_audit.csv", notebook_rows)
    save_csv(table_dir / "table_reviewer_traceability.csv", traceability)
    save_csv(table_dir / "table_statistical_oracle_check.csv", oracle)
    save_csv(table_dir / "table_paired_inference_audit.csv", paired_inference_rows)
    save_csv(table_dir / "table_comparator_contract.csv", comparators)
    save_csv(table_dir / "table_forensic_rerun_dependencies.csv", rerun_rows)
    save_json(metric_dir / "artifact_provenance_audit.json", provenance_payload)

    report = [
        "# ECG-RAMBA Notebook Forensic Audit",
        "",
        f"- Created UTC: `{created_utc}`",
        f"- Code authority: `{current_head}`",
        f"- Worktree dirty: `{working_tree_dirty}`",
        f"- Source bundle SHA256: `{source_bundle_sha}` ({source_bundle_files} files)",
        f"- OOF producer: `{producer_commit or 'missing'}`",
        f"- Result: **{'GO' if status else 'NO-GO'}**",
        "",
        "## Audit Framework",
        "",
        "- Reporting and traceability are reviewed against TRIPOD+AI.",
        "- Risk of bias, leakage, analysis, and applicability are reviewed against PROBAST+AI.",
        "- These frameworks organize the audit; they do not convert an automated gate into a clinical validation.",
        "",
        "## P0 Findings",
        "",
        *(f"- {item}" for item in p0),
        *( ["- None."] if not p0 else [] ),
        "",
        "## P1 Findings",
        "",
        *(f"- {item}" for item in p1),
        *( ["- None."] if not p1 else [] ),
        "",
        "## P2 Findings",
        "",
        *(f"- {item}" for item in p2),
        *( ["- None."] if not p2 else [] ),
        "",
        "## Claim Boundaries",
        "",
        "- The fixed morphology transform is a fixed-seed ROCKET-family MAX+PPV transform, not canonical MiniRocket.",
        "- Learned architectures are same-fold comparators unless the comparator contract proves matched budgets.",
        "- CKA and held-out probes audit representation behavior; they do not establish mechanistic disentanglement.",
        "- Robustness is reported by named stress, metric, and comparator. Pointwise intervals do not imply family-wise significance.",
        "- External adaptation separates score calibration from frozen-encoder head adaptation and never implies end-to-end fine-tuning.",
        "",
        "## Deliverables",
        "",
        "- `table_notebook_cell_audit.csv`",
        "- `table_reviewer_traceability.csv`",
        "- `table_statistical_oracle_check.csv`",
        "- `table_paired_inference_audit.csv`",
        "- `table_comparator_contract.csv`",
        "- `table_forensic_rerun_dependencies.csv`",
        "- `artifact_provenance_audit.json`",
        "",
    ]
    atomic_text(audit_dir / "notebook_forensic_audit.md", "\n".join(report))
    print(json.dumps({"status": status, "p0": len(p0), "p1": len(p1), "p2": len(p2), "outputs": 7}, indent=2))
    if args.strict and not status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
