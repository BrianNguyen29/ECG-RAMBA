"""Matched fold-excluded post-hoc calibration sensitivity on frozen OOF scores.

For each held-out outer fold, per-class Platt calibrators are fitted only on
OOF scores and labels from the other four folds. This exclusion applies to the
calibrator fit only: the base predictors are not nested-refitted, and models
behind the four calibration-fold score blocks may have trained on records in
the evaluated fold. The result is a conditional post-hoc sensitivity audit,
not a leakage-free estimate of a deploy-time calibration pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    AUTHENTICATED_RECORD_BOOTSTRAP_UNIT,
    CHAPMAN_GROUP_REFERENCE,
    CHAPMAN_GROUP_SEMANTICS,
    FIGURE_DIR,
    MANIFEST_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    TABLE_DIR,
    calibration_summary,
    ensure_revision_dirs,
    git_commit,
    macro_pr_auc,
    macro_roc_auc,
    multilabel_metrics,
    save_csv,
    save_json,
    save_npz_compressed_atomic,
    sha256_file,
)


SCHEMA_VERSION = 6
PROTOCOL = "matched_fold_excluded_platt_posthoc_sensitivity_v5"
PLATT_C = 1e6
PLATT_MIN_SLOPE = 1e-8
DEFAULT_MODELS = {
    "full": PREDICTION_DIR / "oof_final_ema_predictions.npz",
    "minirocket": PREDICTION_DIR / "minirocket_only_oof_predictions.npz",
    "resnet": PREDICTION_DIR / "resnet1d_cnn_oof_predictions.npz",
    "raw_mamba": PREDICTION_DIR / "raw_mamba_oof_predictions.npz",
    "transformer": PREDICTION_DIR / "transformer_ecg_oof_predictions.npz",
    "frozen_transform_mlp": PREDICTION_DIR / "hybrid_morphology_oof_predictions.npz",
}
MODEL_LABELS = {
    "full": "ECG-RAMBA",
    "minirocket": "Fixed-seed ROCKET-family MAX+PPV linear head",
    "resnet": "ResNet1D/CNN",
    "raw_mamba": "Raw Mamba",
    "transformer": "Transformer ECG",
    "frozen_transform_mlp": "Frozen-transform MLP",
}


@dataclass(frozen=True)
class PredictionSet:
    name: str
    path: Path
    sha256: str
    y_true: np.ndarray
    y_prob: np.ndarray
    record_id: np.ndarray
    fold_id: np.ndarray
    class_names: np.ndarray


@dataclass(frozen=True)
class MetricSpec:
    name: str
    higher_is_better: bool
    fn: Callable[[np.ndarray, np.ndarray], float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Prediction NPZ. Repeat to override defaults or select a subset.",
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=MANIFEST_DIR / "oof_final_ema_freeze_manifest.json",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--reuse-bootstrap", action="store_true")
    parser.add_argument(
        "--metric-cache-dir",
        type=Path,
        default=METRIC_DIR / "matched_calibration_metric_cache",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=METRIC_DIR / "matched_oof_calibration_summary.json",
    )
    parser.add_argument(
        "--out-table",
        type=Path,
        default=TABLE_DIR / "table_matched_oof_calibration.csv",
    )
    parser.add_argument(
        "--out-coefficients",
        type=Path,
        default=TABLE_DIR / "table_matched_oof_calibration_coefficients.csv",
    )
    parser.add_argument(
        "--out-tex-table",
        type=Path,
        default=TABLE_DIR / "table_matched_oof_calibration.tex",
    )
    parser.add_argument(
        "--out-paired-table",
        type=Path,
        default=TABLE_DIR / "table_paired_matched_oof_calibration.csv",
    )
    parser.add_argument(
        "--out-bootstrap",
        type=Path,
        default=METRIC_DIR / "matched_oof_calibration_bootstrap.json",
    )
    parser.add_argument(
        "--out-figure",
        type=Path,
        default=FIGURE_DIR / "figure_matched_calibration_audit.png",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=MANIFEST_DIR / "matched_oof_calibration_manifest.json",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path: Path) -> str:
    return resolve(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_tex_table(rows: list[dict], path: Path) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table*}[t]",
        (
            r"\caption{Matched post-hoc calibration audit on frozen Chapman OOF scores. "
            r"The calibrator fit excludes labels from the evaluated fold, but base predictors are not "
            r"nested-refitted. Lower is better for NLL, Brier, and ECE; ideal diagnostic "
            r"slope/intercept are 1/0. Results are a conditional post-hoc sensitivity, not a "
            r"deploy-time calibration estimate.}"
        ),
        r"\label{tab:matched_oof_calibration}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Model & State & NLL & Brier & ECE & Slope & Intercept \\",
        r"\midrule",
    ]
    for row in rows:
        state = "Raw" if row["state"] == "raw" else "Fold-excluded Platt"
        lines.append(
            "{} & {} & {:.4f} & {:.4f} & {:.4f} & {:.3f} & {:.3f} \\\\".format(
                latex_escape(row["model_label"]),
                state,
                float(row["nll_macro"]),
                float(row["brier_macro"]),
                float(row["ece_macro"]),
                float(row["calibration_slope_macro"]),
                float(row["calibration_intercept_macro"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_models(items: list[str]) -> dict[str, Path]:
    if not items:
        return dict(DEFAULT_MODELS)
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--model must be NAME=PATH, got {item!r}")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        if not name or name in parsed:
            raise ValueError(f"Invalid or duplicate model name: {name!r}")
        parsed[name] = Path(raw_path.strip())
    return parsed


def load_prediction(name: str, path: Path) -> PredictionSet:
    path = resolve(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"y_true", "y_prob", "record_id", "fold_id", "class_names"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{path} missing {sorted(missing)}")
        result = PredictionSet(
            name=name,
            path=path,
            sha256=sha256_file(path),
            y_true=np.asarray(data["y_true"], dtype=np.float32),
            y_prob=np.asarray(data["y_prob"], dtype=np.float32),
            record_id=np.asarray(data["record_id"], dtype=np.int64),
            fold_id=np.asarray(data["fold_id"], dtype=np.int16),
            class_names=np.asarray(data["class_names"]).astype(str),
        )
    if result.y_true.ndim != 2 or result.y_prob.shape != result.y_true.shape:
        raise ValueError(f"{name} shape mismatch: {result.y_true.shape} vs {result.y_prob.shape}")
    if len(result.record_id) != len(result.y_true) or len(result.fold_id) != len(result.y_true):
        raise ValueError(f"{name} record/fold length mismatch")
    if not np.isfinite(result.y_true).all():
        raise ValueError(f"{name} contains non-finite labels")
    if not np.isin(result.y_true, [0.0, 1.0]).all():
        raise ValueError(f"{name} labels must be binary")
    if not np.isfinite(result.y_prob).all():
        raise ValueError(f"{name} contains non-finite probabilities")
    if np.any(result.y_prob < 0.0) or np.any(result.y_prob > 1.0):
        raise ValueError(f"{name} probabilities must lie in [0, 1]")
    return result


def validate_contract(predictions: dict[str, PredictionSet], freeze_path: Path) -> dict:
    if "full" not in predictions:
        raise ValueError("The matched audit requires --model full=...")
    full = predictions["full"]
    for name, pred in predictions.items():
        if not np.array_equal(pred.y_true, full.y_true):
            raise ValueError(f"{name} y_true differs from Full")
        if not np.array_equal(pred.record_id, full.record_id):
            raise ValueError(f"{name} record_id differs from Full")
        if not np.array_equal(pred.fold_id, full.fold_id):
            raise ValueError(f"{name} fold_id differs from Full")
        if not np.array_equal(pred.class_names, full.class_names):
            raise ValueError(f"{name} class order differs from Full")
    folds = sorted(int(value) for value in np.unique(full.fold_id))
    if folds != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected frozen folds [1..5], got {folds}")
    if len(np.unique(full.record_id)) != len(full.record_id):
        raise ValueError("record_id is not unique; subject grouping must be supplied explicitly")

    freeze_path = resolve(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    artifacts = {row.get("path"): row for row in freeze.get("artifacts", [])}
    full_rel = rel(full.path)
    if freeze.get("status") != "frozen" or full_rel not in artifacts:
        raise RuntimeError("Full OOF freeze contract is not valid")
    if artifacts[full_rel].get("sha256") != full.sha256:
        raise RuntimeError("Full OOF checksum differs from freeze manifest")
    group = freeze.get("group_contract") or {}
    group_errors = []
    if group.get("status") != "verified":
        group_errors.append("status")
    if group.get("group_semantics") != CHAPMAN_GROUP_SEMANTICS:
        group_errors.append("group_semantics")
    if group.get("group_semantics_reference") != CHAPMAN_GROUP_REFERENCE:
        group_errors.append("group_semantics_reference")
    if group.get("bootstrap_unit") != AUTHENTICATED_RECORD_BOOTSTRAP_UNIT:
        group_errors.append("bootstrap_unit")
    if group.get("one_record_per_group") is not True:
        group_errors.append("one_record_per_group")
    if int(group.get("n_records", -1)) != len(full.record_id):
        group_errors.append("n_records")
    if int(group.get("n_groups", -1)) != len(full.record_id):
        group_errors.append("n_groups")
    sidecar = group.get("sidecar") or {}
    sidecar_path = Path(str(sidecar.get("path") or ""))
    if not str(sidecar.get("path") or ""):
        group_errors.append("sidecar_path")
    elif not sidecar_path.is_absolute():
        sidecar_path = PROJECT_ROOT / sidecar_path
    if not sidecar_path.is_file():
        group_errors.append("sidecar_missing")
    elif not sidecar.get("sha256") or sha256_file(sidecar_path) != sidecar.get("sha256"):
        group_errors.append("sidecar_sha256")
    if group_errors:
        raise RuntimeError(
            "Matched calibration requires the authenticated frozen patient-record contract: "
            + ", ".join(group_errors)
        )
    group_contract_sha256 = hashlib.sha256(
        json.dumps(group, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "path": rel(freeze_path),
        "sha256": sha256_file(freeze_path),
        "checkpoint_kind": freeze.get("checkpoint_kind"),
        "bootstrap_unit": AUTHENTICATED_RECORD_BOOTSTRAP_UNIT,
        "independence_contract": CHAPMAN_GROUP_SEMANTICS,
        "group_semantics_reference": CHAPMAN_GROUP_REFERENCE,
        "group_contract": group,
        "group_contract_sha256": group_contract_sha256,
        "group_sidecar": rel(sidecar_path),
        "group_sidecar_sha256": sidecar["sha256"],
        "n_records": int(group["n_records"]),
        "n_groups": int(group["n_groups"]),
        "one_record_per_subject": True,
    }


def validate_bootstrap_result(
    result: dict,
    *,
    n_boot: int,
    ci_fields: tuple[str, str],
) -> None:
    if int(result.get("n_boot_valid", -1)) != int(n_boot):
        raise RuntimeError(
            f"Bootstrap cache/result has n_boot_valid={result.get('n_boot_valid')}; "
            f"exactly {n_boot} are required."
        )
    for field in ci_fields:
        try:
            value = float(result[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Bootstrap cache/result lacks numeric {field}") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"Bootstrap cache/result contains non-finite {field}")
    for key, value in result.items():
        if "significant" in str(key).lower() or "significant" in str(value).lower():
            raise RuntimeError("Bootstrap cache/result contains prohibited legacy significance wording")


def clipped_logit(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(prob / (1.0 - prob))


def fit_platt(y: np.ndarray, prob: np.ndarray) -> tuple[float, float, str]:
    """Fit a monotone Platt map without allowing a ranking reversal."""

    from scipy.optimize import minimize
    from scipy.special import expit

    y = np.asarray(y, dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64)
    if not np.isfinite(y).all() or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("Platt labels must be finite and binary")
    if not np.isfinite(prob).all() or np.any(prob < 0.0) or np.any(prob > 1.0):
        raise ValueError("Platt probabilities must be finite and lie in [0, 1]")
    if len(np.unique(y)) < 2:
        return 0.0, 1.0, "identity_degenerate_training_label"
    x = clipped_logit(prob)
    prevalence = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
    initial_slope = 1.0
    initial_intercept = float(np.log(prevalence / (1.0 - prevalence)) - np.mean(x))
    n_records = float(len(y))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept, slope = parameters
        linear = intercept + slope * x
        loss = float(
            np.mean(np.logaddexp(0.0, linear) - y * linear)
            + 0.5 * slope * slope / (PLATT_C * n_records)
        )
        residual = expit(linear) - y
        gradient = np.asarray(
            [
                np.mean(residual),
                np.mean(residual * x) + slope / (PLATT_C * n_records),
            ],
            dtype=np.float64,
        )
        return loss, gradient

    fitted = minimize(
        objective,
        x0=np.asarray([initial_intercept, initial_slope], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=[(None, None), (PLATT_MIN_SLOPE, None)],
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"Monotone per-class Platt fit failed: {fitted.message}")
    intercept, slope = (float(value) for value in fitted.x)
    if slope < PLATT_MIN_SLOPE:
        raise RuntimeError("Monotone per-class Platt fit violated its positive-slope bound")
    status = "fitted_monotone_boundary" if slope <= PLATT_MIN_SLOPE * 1.01 else "fitted_monotone"
    return intercept, slope, status


def fit_calibration_diagnostic(y: np.ndarray, prob: np.ndarray) -> tuple[float, float, str]:
    """Fit the conventional unconstrained logistic calibration diagnostic."""

    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype=np.int8)
    if len(np.unique(y)) < 2:
        return 0.0, 1.0, "identity_degenerate_training_label"
    model = LogisticRegression(C=PLATT_C, solver="lbfgs", max_iter=5000)
    model.fit(clipped_logit(prob).reshape(-1, 1), y)
    if int(model.n_iter_[0]) >= int(model.max_iter):
        raise RuntimeError("Calibration diagnostic reached max_iter without convergence")
    return float(model.intercept_[0]), float(model.coef_[0, 0]), "fitted"


def apply_platt(prob: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    z = np.clip(intercept + slope * clipped_logit(prob), -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def cross_fitted_platt(pred: PredictionSet) -> tuple[np.ndarray, list[dict]]:
    calibrated = np.full_like(pred.y_prob, np.nan, dtype=np.float32)
    rows = []
    for fold in sorted(int(value) for value in np.unique(pred.fold_id)):
        train_mask = pred.fold_id != fold
        test_mask = pred.fold_id == fold
        for class_index, class_name in enumerate(pred.class_names):
            intercept, slope, status = fit_platt(
                pred.y_true[train_mask, class_index], pred.y_prob[train_mask, class_index]
            )
            calibrated[test_mask, class_index] = apply_platt(
                pred.y_prob[test_mask, class_index], intercept, slope
            )
            rows.append(
                {
                    "model": pred.name,
                    "model_label": MODEL_LABELS.get(pred.name, pred.name),
                    "evaluation_fold": fold,
                    "class_index": class_index,
                    "class_name": class_name,
                    "train_records": int(np.sum(train_mask)),
                    "evaluation_records": int(np.sum(test_mask)),
                    "train_positives": int(np.sum(pred.y_true[train_mask, class_index])),
                    "intercept": intercept,
                    "slope": slope,
                    "status": status,
                }
            )
    if not np.isfinite(calibrated).all():
        raise RuntimeError(f"{pred.name} calibration did not cover all OOF records")
    return calibrated, rows


def macro_nll(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    y = np.asarray(y_true, dtype=np.float64)
    return float(np.mean(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)), axis=0)))


def calibration_regression(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, int]:
    intercepts, slopes = [], []
    for class_index in range(y_true.shape[1]):
        if len(np.unique(y_true[:, class_index])) < 2:
            continue
        intercept, slope, status = fit_calibration_diagnostic(
            y_true[:, class_index], y_prob[:, class_index]
        )
        if status == "fitted":
            intercepts.append(intercept)
            slopes.append(slope)
    return (
        float(np.mean(intercepts)) if intercepts else math.nan,
        float(np.mean(slopes)) if slopes else math.nan,
        len(slopes),
    )


def metric_values(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, n_bins: int) -> dict:
    ranking = multilabel_metrics(y_true, y_prob, threshold=threshold)
    calibration = calibration_summary(y_true, y_prob, n_bins=n_bins)
    intercept, slope, count = calibration_regression(y_true, y_prob)
    return {
        "pr_auc_macro": ranking["pr_auc_macro"],
        "roc_auc_macro": ranking["roc_auc_macro"],
        "f1_macro": ranking["f1_macro"],
        "brier_macro": calibration["brier_macro"],
        "ece_macro": calibration["ece_macro"],
        "nll_macro": macro_nll(y_true, y_prob),
        "calibration_intercept_macro": intercept,
        "calibration_slope_macro": slope,
        "calibration_classes": count,
    }


def metric_specs(threshold: float, n_bins: int) -> list[MetricSpec]:
    return [
        MetricSpec("pr_auc_macro", True, macro_pr_auc),
        MetricSpec("roc_auc_macro", True, macro_roc_auc),
        MetricSpec("f1_macro", True, lambda y, p: multilabel_metrics(y, p, threshold)["f1_macro"]),
        MetricSpec("brier_macro", False, lambda y, p: calibration_summary(y, p, n_bins)["brier_macro"]),
        MetricSpec("ece_macro", False, lambda y, p: calibration_summary(y, p, n_bins)["ece_macro"]),
        MetricSpec("nll_macro", False, macro_nll),
    ]


def paired_bootstrap(
    y_true: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    spec: MetricSpec,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    raw_deltas = []
    improvements = []
    point_raw_delta = float(spec.fn(y_true, calibrated) - spec.fn(y_true, raw))
    point_improvement = point_raw_delta if spec.higher_is_better else -point_raw_delta
    for _ in range(n_boot):
        index = rng.integers(0, len(y_true), size=len(y_true))
        try:
            delta = spec.fn(y_true[index], calibrated[index]) - spec.fn(y_true[index], raw[index])
        except ValueError:
            continue
        if np.isfinite(delta):
            raw_deltas.append(float(delta))
            improvements.append(float(delta if spec.higher_is_better else -delta))
    if not improvements:
        return {
            "point_delta_calibrated_minus_raw": point_raw_delta,
            "point_improvement_calibrated_over_raw": point_improvement,
            "improvement_ci_low": None,
            "improvement_ci_high": None,
            "n_boot_valid": 0,
        }
    low, high = np.quantile(improvements, [0.025, 0.975])
    return {
        "point_delta_calibrated_minus_raw": point_raw_delta,
        "bootstrap_mean_delta_calibrated_minus_raw": float(np.mean(raw_deltas)),
        "point_improvement_calibrated_over_raw": point_improvement,
        "improvement_ci_low": float(low),
        "improvement_ci_high": float(high),
        "n_boot_valid": len(improvements),
        "inference_scope": "pointwise_percentile_ci_effect_size_only",
        "null_test": "not_run",
        "interpretation": (
            "calibrated_nominal_95ci_better"
            if low > 0
            else "raw_nominal_95ci_better"
            if high < 0
            else "pointwise_ci_overlaps_zero"
        ),
    }


def paired_model_bootstrap(
    y_true: np.ndarray,
    full_prob: np.ndarray,
    comparator_prob: np.ndarray,
    spec: MetricSpec,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    """Return an oriented paired delta where positive always favors Full."""

    full_value = float(spec.fn(y_true, full_prob))
    comparator_value = float(spec.fn(y_true, comparator_prob))
    raw_delta = full_value - comparator_value
    point = raw_delta if spec.higher_is_better else -raw_delta
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        index = rng.integers(0, len(y_true), size=len(y_true))
        try:
            delta = spec.fn(y_true[index], full_prob[index]) - spec.fn(
                y_true[index], comparator_prob[index]
            )
        except ValueError:
            continue
        oriented = delta if spec.higher_is_better else -delta
        if np.isfinite(oriented):
            values.append(float(oriented))
    if not values:
        return {
            "full_value": full_value,
            "comparator_value": comparator_value,
            "improvement_full_over_comparator": point,
            "ci_low": None,
            "ci_high": None,
            "n_boot_valid": 0,
            "interpretation": "unavailable",
        }
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "full_value": full_value,
        "comparator_value": comparator_value,
        "improvement_full_over_comparator": float(point),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_boot_valid": len(values),
        "inference_scope": "pointwise_percentile_ci_effect_size_only",
        "null_test": "not_run",
        "interpretation": (
            "full_nominal_95ci_better"
            if low > 0
            else "comparator_nominal_95ci_better"
            if high < 0
            else "inconclusive"
        ),
    }


def reliability_rows(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    y = y_true.ravel()
    p = y_prob.ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    confidence, observed = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (p >= low) & (p < high if high < 1.0 else p <= high)
        if np.any(mask):
            confidence.append(float(np.mean(p[mask])))
            observed.append(float(np.mean(y[mask])))
    return np.asarray(confidence), np.asarray(observed)


def write_figure(y_true: np.ndarray, raw: np.ndarray, calibrated: np.ndarray, path: Path, n_bins: int) -> None:
    import matplotlib.pyplot as plt

    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.0), dpi=180)
    ax.plot([0, 1], [0, 1], "--", color="0.45", linewidth=1, label="Ideal")
    for label, values, color in [
        ("Raw", raw, "#B34233"),
        ("Cross-fitted Platt", calibrated, "#226F54"),
    ]:
        confidence, observed = reliability_rows(y_true, values, n_bins)
        ax.plot(confidence, observed, marker="o", linewidth=1.4, markersize=3, label=label, color=color)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Predicted probability",
        ylabel="Observed frequency",
        title="ECG-RAMBA pooled label-instance reliability",
    )
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_boot <= 0:
        raise ValueError("--n-boot must be positive")
    if args.n_bins < 2:
        raise ValueError("--n-bins must be at least 2")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must lie strictly between 0 and 1")
    ensure_revision_dirs()
    model_paths = parse_models(args.model)
    missing = [name for name, path in model_paths.items() if not resolve(path).exists()]
    if missing and args.strict:
        raise FileNotFoundError(f"Missing requested prediction artifacts: {missing}")
    model_paths = {name: path for name, path in model_paths.items() if resolve(path).exists()}
    predictions = {name: load_prediction(name, path) for name, path in model_paths.items()}
    freeze_contract = validate_contract(predictions, args.freeze_manifest)
    runner_sha256 = sha256_file(Path(__file__).resolve())

    print("=" * 80)
    print("MATCHED CROSS-FITTED OOF-SCORE CALIBRATION AUDIT")
    print("=" * 80)
    print(f"models={list(predictions)} protocol={PROTOCOL} n_boot={args.n_boot}")

    table_rows, coefficient_rows, paired_rows = [], [], []
    audit_metric_specs = metric_specs(args.threshold, args.n_bins)
    bootstrap_payload = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "inference_scope": "pointwise_percentile_ci_effect_size_only",
        "null_test": "not_run",
        "multiplicity_adjustment": "not_applicable_no_null_test",
        "models": {},
        "matched_model_comparisons": {},
    }
    calibrated_by_model = {}
    calibrated_probability_sha256_by_model = {}
    calibrated_prediction_outputs = {}
    for model_index, (name, pred) in enumerate(predictions.items()):
        calibrated, coefficients = cross_fitted_platt(pred)
        coefficient_rows.extend(coefficients)
        calibrated_by_model[name] = calibrated
        raw_values = metric_values(pred.y_true, pred.y_prob, args.threshold, args.n_bins)
        calibrated_values = metric_values(pred.y_true, calibrated, args.threshold, args.n_bins)
        prediction_out = PREDICTION_DIR / f"{name}_cross_fitted_platt_oof_predictions.npz"
        save_npz_compressed_atomic(
            prediction_out,
            y_true=pred.y_true,
            y_prob=calibrated,
            y_prob_raw=pred.y_prob,
            record_id=pred.record_id,
            fold_id=pred.fold_id,
            class_names=pred.class_names,
            protocol=np.asarray(PROTOCOL),
            source_prediction_sha256=np.asarray(pred.sha256),
            freeze_manifest_sha256=np.asarray(freeze_contract["sha256"]),
            group_contract_sha256=np.asarray(freeze_contract["group_contract_sha256"]),
            group_sidecar_sha256=np.asarray(freeze_contract["group_sidecar_sha256"]),
        )
        calibrated_prediction_outputs[name] = prediction_out
        calibrated_probability_sha256 = array_sha256(calibrated)
        calibrated_probability_sha256_by_model[name] = calibrated_probability_sha256
        bootstrap_payload["models"][name] = {}
        for state, values in [("raw", raw_values), ("cross_fitted_platt", calibrated_values)]:
            table_rows.append(
                {
                    "model": name,
                    "model_label": MODEL_LABELS.get(name, name),
                    "state": state,
                    **values,
                    "prediction_path": rel(prediction_out) if state == "cross_fitted_platt" else rel(pred.path),
                }
            )
        for metric_index, spec in enumerate(audit_metric_specs):
            cache_path = resolve(args.metric_cache_dir) / f"{name}__{spec.name}.json"
            cache_contract = {
                "schema_version": SCHEMA_VERSION,
                "runner_sha256": runner_sha256,
                "source_sha256": pred.sha256,
                "freeze_manifest_sha256": freeze_contract["sha256"],
                "group_contract_sha256": freeze_contract["group_contract_sha256"],
                "group_sidecar_sha256": freeze_contract["group_sidecar_sha256"],
                "calibrated_probability_sha256": calibrated_probability_sha256,
                "metric": spec.name,
                "n_boot": args.n_boot,
                "seed": args.seed + model_index * 100 + metric_index,
                "protocol": PROTOCOL,
            }
            cached = None
            if args.reuse_bootstrap and cache_path.exists():
                candidate = json.loads(cache_path.read_text(encoding="utf-8"))
                if candidate.get("contract") == cache_contract:
                    candidate_result = candidate.get("result")
                    try:
                        validate_bootstrap_result(
                            candidate_result,
                            n_boot=args.n_boot,
                            ci_fields=("improvement_ci_low", "improvement_ci_high"),
                        )
                    except (AttributeError, TypeError, RuntimeError):
                        print(f"Rejecting invalid matched-calibration cache: {cache_path}", flush=True)
                    else:
                        cached = candidate_result
            if cached is None:
                result = paired_bootstrap(
                    pred.y_true,
                    pred.y_prob,
                    calibrated,
                    spec,
                    n_boot=args.n_boot,
                    seed=cache_contract["seed"],
                )
                validate_bootstrap_result(
                    result,
                    n_boot=args.n_boot,
                    ci_fields=("improvement_ci_low", "improvement_ci_high"),
                )
                save_json(cache_path, {"contract": cache_contract, "result": result})
            else:
                result = cached
            validate_bootstrap_result(
                result,
                n_boot=args.n_boot,
                ci_fields=("improvement_ci_low", "improvement_ci_high"),
            )
            bootstrap_payload["models"][name][spec.name] = result
            print(f"{name} {spec.name}: {result}")

    full = predictions["full"]
    for comparator_index, (name, pred) in enumerate(predictions.items()):
        if name == "full":
            continue
        bootstrap_payload["matched_model_comparisons"][name] = {}
        for state_index, state in enumerate(["raw", "cross_fitted_platt"]):
            full_prob = full.y_prob if state == "raw" else calibrated_by_model["full"]
            comparator_prob = pred.y_prob if state == "raw" else calibrated_by_model[name]
            state_payload = {}
            for metric_index, spec in enumerate(audit_metric_specs):
                seed = args.seed + 10_000 + comparator_index * 1000 + state_index * 100 + metric_index
                cache_path = resolve(args.metric_cache_dir) / (
                    f"full_vs_{name}__{state}__{spec.name}.json"
                )
                cache_contract = {
                    "schema_version": SCHEMA_VERSION,
                    "runner_sha256": runner_sha256,
                    "full_source_sha256": full.sha256,
                    "comparator_source_sha256": pred.sha256,
                    "freeze_manifest_sha256": freeze_contract["sha256"],
                    "group_contract_sha256": freeze_contract["group_contract_sha256"],
                    "group_sidecar_sha256": freeze_contract["group_sidecar_sha256"],
                    "full_state_prediction_sha256": (
                        full.sha256
                        if state == "raw"
                        else calibrated_probability_sha256_by_model["full"]
                    ),
                    "comparator_state_prediction_sha256": (
                        pred.sha256
                        if state == "raw"
                        else calibrated_probability_sha256_by_model[name]
                    ),
                    "comparator": name,
                    "state": state,
                    "metric": spec.name,
                    "n_boot": args.n_boot,
                    "seed": seed,
                    "protocol": PROTOCOL,
                }
                result = None
                if args.reuse_bootstrap and cache_path.exists():
                    candidate = json.loads(cache_path.read_text(encoding="utf-8"))
                    if candidate.get("contract") == cache_contract:
                        candidate_result = candidate.get("result")
                        try:
                            validate_bootstrap_result(
                                candidate_result,
                                n_boot=args.n_boot,
                                ci_fields=("ci_low", "ci_high"),
                            )
                        except (AttributeError, TypeError, RuntimeError):
                            print(f"Rejecting invalid paired-calibration cache: {cache_path}", flush=True)
                        else:
                            result = candidate_result
                if result is None:
                    result = paired_model_bootstrap(
                        full.y_true,
                        full_prob,
                        comparator_prob,
                        spec,
                        n_boot=args.n_boot,
                        seed=seed,
                    )
                    validate_bootstrap_result(
                        result,
                        n_boot=args.n_boot,
                        ci_fields=("ci_low", "ci_high"),
                    )
                    save_json(cache_path, {"contract": cache_contract, "result": result})
                validate_bootstrap_result(
                    result,
                    n_boot=args.n_boot,
                    ci_fields=("ci_low", "ci_high"),
                )
                paired_rows.append(
                    {
                        "comparison": f"full_vs_{name}",
                        "comparator": name,
                        "comparator_label": MODEL_LABELS.get(name, name),
                        "state": state,
                        "metric": spec.name,
                        "higher_is_better": spec.higher_is_better,
                        "bootstrap_unit": "Chapman record; one record per subject",
                        "uncertainty_scope": (
                            "paired record bootstrap conditional on fixed base models and fitted "
                            "cross-fitted calibrators; no model or calibrator refit per replicate"
                        ),
                        "inference_scope": "pointwise_percentile_ci_effect_size_only",
                        "null_test": "not_run",
                        **result,
                    }
                )
                state_payload[spec.name] = result
                print(f"full vs {name} {state} {spec.name}: {result}")
            bootstrap_payload["matched_model_comparisons"][name][state] = state_payload

    expected_coefficient_rows = len(predictions) * 5 * full.y_true.shape[1]
    model_bootstrap_complete = all(
        int(result.get("n_boot_valid", 0)) == int(args.n_boot)
        and result.get("improvement_ci_low") is not None
        and result.get("improvement_ci_high") is not None
        for model_payload in bootstrap_payload["models"].values()
        for result in model_payload.values()
    )
    expected_paired_rows = (len(predictions) - 1) * 2 * len(audit_metric_specs)
    paired_bootstrap_complete = (
        len(paired_rows) == expected_paired_rows
        and all(
            int(row.get("n_boot_valid", 0)) == int(args.n_boot)
            and row.get("ci_low") is not None
            and row.get("ci_high") is not None
            for row in paired_rows
        )
    )
    completeness = {
        "coefficient_grid_complete": len(coefficient_rows) == expected_coefficient_rows,
        "raw_vs_calibrated_bootstrap_complete": model_bootstrap_complete,
        "matched_model_bootstrap_complete": paired_bootstrap_complete,
        "expected_coefficient_rows": expected_coefficient_rows,
        "observed_coefficient_rows": len(coefficient_rows),
        "expected_paired_rows": expected_paired_rows,
        "observed_paired_rows": len(paired_rows),
        "required_valid_bootstrap_replicates": int(args.n_boot),
    }
    calibration_complete = all(
        completeness[key]
        for key in [
            "coefficient_grid_complete",
            "raw_vs_calibrated_bootstrap_complete",
            "matched_model_bootstrap_complete",
        ]
    )
    if args.strict and not calibration_complete:
        raise RuntimeError(f"Matched calibration completeness contract failed: {completeness}")

    write_figure(full.y_true, full.y_prob, calibrated_by_model["full"], args.out_figure, args.n_bins)
    save_csv(resolve(args.out_table), table_rows)
    save_csv(resolve(args.out_coefficients), coefficient_rows)
    save_csv(resolve(args.out_paired_table), paired_rows)
    write_tex_table(table_rows, args.out_tex_table)
    save_json(resolve(args.out_bootstrap), bootstrap_payload)
    summary = {
        "status": "complete" if calibration_complete else "incomplete_bootstrap_contract",
        "schema_version": SCHEMA_VERSION,
        "created_utc": now_utc(),
        "protocol": PROTOCOL,
        "runner_sha256": runner_sha256,
        "method": "per-class monotone Platt scaling",
        "platt_estimator": (
            "bounded logistic calibration on clipped logits; positive slope >=1e-8; C=1e6; "
            "L-BFGS-B"
        ),
        "ranking_contract": (
            "Each fold/class calibration map uses a positive fitted slope and is monotone "
            "non-decreasing after float32 serialization; it cannot reverse within-fold score ordering, "
            "although finite precision can introduce ties. Fold-specific mappings can still change "
            "pooled OOF ranking."
        ),
        "calibration_split_contract": (
            "For each evaluated OOF fold, only the calibrator fit excludes that fold's labels. "
            "The base predictors are not nested-refitted, so this exclusion must not be interpreted "
            "as an independent outer-fold calibration estimate."
        ),
        "nested_refit_scope": (
            "Base models are not refitted in a nested calibration loop. Models that generated the "
            "calibrator-training OOF scores may have used records from the evaluated fold in their "
            "own training sets. Therefore this is a post-hoc OOF-score sensitivity audit, not an "
            "unbiased nested estimate of a deploy-time calibration pipeline."
        ),
        "evaluation_contract": "same frozen records, folds, threshold, bins, and paired record bootstrap",
        "bootstrap_unit": AUTHENTICATED_RECORD_BOOTSTRAP_UNIT,
        "bootstrap_scope": (
            "Paired record bootstrap conditions on the already cross-fitted raw and calibrated OOF "
            "scores; calibrators and base models are not refitted within bootstrap resamples."
        ),
        "multiplicity_scope": (
            "Intervals are nominal pointwise 95% intervals over pre-specified metrics; no family-wise "
            "calibration-superiority claim is made."
        ),
        "null_test": "not_run",
        "calibration_slope_intercept_scope": (
            "Reported slope/intercept values are evaluation diagnostics fitted to held-out OOF scores; "
            "they are not the transformations used to score those same records."
        ),
        "reliability_figure_scope": (
            "The reliability figure pools record-class label instances to provide a visual audit. "
            "Macro Brier, ECE, NLL, slope, and intercept remain the quantitative endpoints in the table."
        ),
        "claim_boundary": (
            "Secondary post-hoc OOF-score calibration sensitivity audit. It does not establish clinical "
            "threshold safety, estimate a fully nested calibration pipeline, or compensate for lower "
            "discrimination. Pointwise bootstrap intervals condition on the fitted calibrators and do "
            "not include model- or calibrator-refit variability."
        ),
        "completeness_contract": completeness,
        "freeze_contract": freeze_contract,
        "models": {
            name: {
                "label": MODEL_LABELS.get(name, name),
                "source_path": rel(pred.path),
                "source_sha256": pred.sha256,
                "calibrated_prediction": {
                    "path": rel(calibrated_prediction_outputs[name]),
                    "sha256": sha256_file(calibrated_prediction_outputs[name]),
                    "probability_sha256": calibrated_probability_sha256_by_model[name],
                },
                "raw": metric_values(pred.y_true, pred.y_prob, args.threshold, args.n_bins),
                "cross_fitted_platt": metric_values(
                    pred.y_true, calibrated_by_model[name], args.threshold, args.n_bins
                ),
            }
            for name, pred in predictions.items()
        },
        "outputs": {
            "table": rel(args.out_table),
            "coefficients": rel(args.out_coefficients),
            "tex_table": rel(args.out_tex_table),
            "paired_table": rel(args.out_paired_table),
            "bootstrap": rel(args.out_bootstrap),
            "figure": rel(args.out_figure),
            "calibrated_predictions": {
                name: rel(path) for name, path in calibrated_prediction_outputs.items()
            },
        },
    }
    save_json(resolve(args.out_summary), summary)
    manifest = {
        "status": summary["status"],
        "schema_version": SCHEMA_VERSION,
        "created_utc": now_utc(),
        "protocol": PROTOCOL,
        "git_commit": git_commit(),
        "runner_sha256": runner_sha256,
        "inputs": {name: {"path": rel(pred.path), "sha256": pred.sha256} for name, pred in predictions.items()},
        "freeze_contract": freeze_contract,
        "outputs": {
            rel(path): sha256_file(resolve(path))
            for path in [
                args.out_summary,
                args.out_table,
                args.out_coefficients,
                args.out_tex_table,
                args.out_paired_table,
                args.out_bootstrap,
                args.out_figure,
            ]
        },
        "calibrated_prediction_outputs": {
            rel(path): sha256_file(path)
            for path in calibrated_prediction_outputs.values()
        },
    }
    save_json(resolve(args.out_manifest), manifest)
    print(json.dumps({"status": True, "models": list(predictions), "manifest": rel(args.out_manifest)}, indent=2))


if __name__ == "__main__":
    main()
