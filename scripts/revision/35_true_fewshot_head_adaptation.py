"""True few-shot classifier-head adaptation with frozen ECG encoders.

Unlike score calibration, this runner fits new multivariate linear classifier
weights on target-domain record representations. The five Chapman-trained
encoders remain frozen and one target head is fitted per encoder fold; their
probabilities are averaged. This is genuine parameter adaptation, but it is not
end-to-end encoder fine-tuning.

PTB-XL uses official fold 9 for adaptation and fold 10 for testing. Georgia and
CPSC2021 use repeated SHA256-seeded, label-independent group splits. All uncertainty resamples intact
patient/source-record groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    AUTHENTICATED_RECORD_BOOTSTRAP_UNIT,
    CHAPMAN_GROUP_REFERENCE,
    CHAPMAN_GROUP_SEMANTICS,
    EXPERIMENTAL_DIR,
    FIGURE_DIR,
    MANIFEST_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    TABLE_DIR,
    calibration_summary,
    cluster_bootstrap_ci,
    git_commit,
    hash_group_train_test_split,
    macro_pr_auc,
    macro_roc_auc,
    multilabel_metrics,
    paired_cluster_bootstrap_delta,
    save_csv,
    save_json,
    save_json_atomic,
    save_npz_compressed_atomic,
    sha256_file,
)


PROTOCOL = "frozen_encoder_true_linear_head_adaptation_v2_group_safe_gated"
REPRESENTATION_PROTOCOL = "frozen_encoder_external_record_representation_v2_source_bound"
REPRESENTATION_PROTOCOL_VERSION = 2
MODEL_STEMS = {
    "full": "ecg_ramba_full",
    "resnet": "resnet1d_cnn",
    "raw_mamba": "raw_mamba",
    "transformer": "transformer_ecg",
}
MODEL_LABELS = {
    "full": "ECG-RAMBA",
    "resnet": "ResNet1D/CNN",
    "raw_mamba": "Raw Mamba",
    "transformer": "Transformer ECG",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ptbxl", "georgia", "cpsc2021"])
    parser.add_argument("--models", default="full,resnet,raw_mamba,transformer")
    parser.add_argument("--fractions", default="0,0.01,0.05,0.10")
    parser.add_argument("--primary-fraction", type=float, default=0.10)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--test-fraction", type=float, default=0.50)
    parser.add_argument("--split-candidates", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--head-c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--analysis-lock",
        type=Path,
        default=MANIFEST_DIR / "ptbxl_adaptation_analysis_lock.json",
    )
    parser.add_argument("--external-root", type=Path, default=EXPERIMENTAL_DIR / "external")
    parser.add_argument("--embedding-root", type=Path, default=PREDICTION_DIR)
    parser.add_argument("--full-gate-json", type=Path, default=None)
    parser.add_argument(
        "--external-comparator-paired-manifest",
        type=Path,
        default=MANIFEST_DIR / "external_comparator_paired_manifest.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PREDICTION_DIR / "fewshot_head_adaptation_cache",
    )
    parser.add_argument(
        "--metric-cache-dir",
        type=Path,
        default=METRIC_DIR / "true_fewshot_head_metric_cache",
    )
    parser.add_argument("--out-summary", type=Path, default=None)
    parser.add_argument("--out-table", type=Path, default=None)
    parser.add_argument("--out-paired-table", type=Path, default=None)
    parser.add_argument("--out-bootstrap", type=Path, default=None)
    parser.add_argument("--out-primary-table", type=Path, default=None)
    parser.add_argument("--out-learning-curve-table", type=Path, default=None)
    parser.add_argument("--out-learning-curve-figure", type=Path, default=None)
    parser.add_argument("--out-coefficients", type=Path, default=None)
    parser.add_argument("--out-splits", type=Path, default=None)
    parser.add_argument("--out-manifest", type=Path, default=None)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def budget_role(fraction: float, primary_fraction: float) -> str:
    if fraction == 0:
        return "zero_target_label_reference"
    if math.isclose(fraction, primary_fraction, abs_tol=1e-12):
        return "analysis_locked_primary"
    return "sensitivity"


def interval_interpretation(
    low: float,
    high: float,
    *,
    positive: str,
    negative: str,
    inconclusive: str,
) -> str:
    if low > 0:
        return positive
    if high < 0:
        return negative
    return inconclusive


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_ptbxl_analysis_lock(
    path: Path,
    *,
    models: list[str],
    fractions: list[float],
    primary_fraction: float,
    seeds: list[int],
    threshold: float,
    n_bins: int,
    n_boot: int,
    head_c: float,
    max_iter: int,
) -> dict[str, str]:
    path = resolve(path)
    if not path.is_file():
        raise FileNotFoundError(f"PTB-XL adaptation analysis lock is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol") or {}
    head = protocol.get("frozen_encoder_head") or {}
    expected = {
        "status": payload.get("status") == "locked",
        "adaptation_split": protocol.get("adaptation_split") == "official_ptbxl_fold9",
        "test_split": protocol.get("test_split") == "official_ptbxl_fold10",
        "group_unit": protocol.get("group_unit") == "patient_id",
        "models": protocol.get("models") == models,
        "fractions": protocol.get("fractions") == fractions,
        "primary_fraction": math.isclose(float(protocol.get("primary_fraction", math.nan)), primary_fraction, abs_tol=1e-12),
        "seeds": protocol.get("seeds") == seeds,
        "threshold": math.isclose(float(protocol.get("threshold", math.nan)), threshold, abs_tol=1e-12),
        "n_bins": int(protocol.get("n_bins", -1)) == n_bins,
        "n_boot": int(protocol.get("n_boot", -1)) == n_boot,
        "head_c": math.isclose(float(head.get("regularization_C", math.nan)), head_c, abs_tol=1e-12),
        "max_iter": int(head.get("max_iter", -1)) == max_iter,
        "head_fit_split": head.get("fit_split") == "fold9_only",
    }
    protocol_sha = canonical_json_sha256(protocol)
    if payload.get("protocol_sha256") != protocol_sha:
        expected["protocol_sha256"] = False
    failed = [key for key, valid in expected.items() if not valid]
    if failed:
        raise RuntimeError(f"PTB-XL analysis lock mismatch: {failed}")
    return {"path": str(path), "sha256": sha256_file(path), "protocol_sha256": protocol_sha}


def resolve_contract_path(value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else PROJECT_ROOT / path


def canonical_contract() -> dict[str, Any]:
    oof = PREDICTION_DIR / "oof_final_ema_predictions.npz"
    freeze = MANIFEST_DIR / "oof_final_ema_freeze_manifest.json"
    if not oof.exists() or not freeze.exists():
        raise FileNotFoundError("Canonical frozen OOF artifacts are required before few-shot adaptation.")
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen" or payload.get("manuscript_ready") is not True:
        raise RuntimeError("Canonical freeze manifest is not frozen/manuscript_ready.")
    oof_sha = sha256_file(oof)
    expected = next(
        (
            row.get("sha256")
            for row in payload.get("artifacts", [])
            if str(row.get("path", "")).replace("\\", "/").endswith(oof.name)
        ),
        None,
    )
    if expected != oof_sha:
        raise RuntimeError(f"Freeze OOF SHA mismatch: {expected} != {oof_sha}")
    group = payload.get("group_contract") or {}
    sidecar = group.get("sidecar") or {}
    errors = []
    if group.get("status") != "verified":
        errors.append("status")
    if group.get("group_semantics") != CHAPMAN_GROUP_SEMANTICS:
        errors.append("group_semantics")
    if group.get("group_semantics_reference") != CHAPMAN_GROUP_REFERENCE:
        errors.append("group_semantics_reference")
    if group.get("bootstrap_unit") != AUTHENTICATED_RECORD_BOOTSTRAP_UNIT:
        errors.append("bootstrap_unit")
    if group.get("one_record_per_group") is not True:
        errors.append("one_record_per_group")
    if int(group.get("n_records", -1)) != int(payload.get("validated_records", -2)):
        errors.append("n_records")
    if int(group.get("n_groups", -1)) != int(payload.get("validated_records", -2)):
        errors.append("n_groups")
    sidecar_path = resolve_contract_path(sidecar.get("path"))
    if not str(sidecar.get("path") or "") or not sidecar_path.is_file():
        errors.append("group_sidecar_missing")
    elif not sidecar.get("sha256") or sha256_file(sidecar_path) != sidecar.get("sha256"):
        errors.append("group_sidecar_sha256")
    if errors:
        raise RuntimeError(
            "Canonical freeze lacks an authenticated live patient/group contract: "
            + ", ".join(errors)
        )
    return {
        "oof_sha256": oof_sha,
        "freeze_sha256": sha256_file(freeze),
        "group_contract_sha256": canonical_json_sha256(group),
        "group_sidecar_sha256": sidecar["sha256"],
        "bootstrap_unit": AUTHENTICATED_RECORD_BOOTSTRAP_UNIT,
    }


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_fractions(value: str) -> list[float]:
    fractions = sorted({float(item) for item in parse_list(value)})
    if not fractions or any(item < 0 or item > 1 for item in fractions):
        raise ValueError(f"Invalid fractions: {fractions}")
    return fractions


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item) for item in parse_list(value)]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def default_paths(args: argparse.Namespace) -> dict[str, Path]:
    dataset = args.dataset
    return {
        "summary": resolve(args.out_summary or METRIC_DIR / f"true_fewshot_head_{dataset}_summary.csv"),
        "table": resolve(args.out_table or TABLE_DIR / f"table_true_fewshot_head_{dataset}.csv"),
        "paired": resolve(args.out_paired_table or TABLE_DIR / f"table_true_fewshot_head_{dataset}_paired.csv"),
        "bootstrap": resolve(args.out_bootstrap or METRIC_DIR / f"true_fewshot_head_{dataset}_bootstrap.json"),
        "primary": resolve(args.out_primary_table or TABLE_DIR / f"table_true_fewshot_head_{dataset}_primary.csv"),
        "learning_curve": resolve(
            args.out_learning_curve_table
            or TABLE_DIR / f"table_true_fewshot_head_{dataset}_learning_curve.csv"
        ),
        "learning_curve_figure": resolve(
            args.out_learning_curve_figure
            or FIGURE_DIR / f"figure_true_fewshot_head_{dataset}_learning_curve.png"
        ),
        "coefficients": resolve(args.out_coefficients or TABLE_DIR / f"table_true_fewshot_head_{dataset}_coefficients.csv"),
        "splits": resolve(args.out_splits or MANIFEST_DIR / f"true_fewshot_head_{dataset}_splits.npz"),
        "manifest": resolve(args.out_manifest or MANIFEST_DIR / f"true_fewshot_head_{dataset}_manifest.json"),
    }


def source_prediction_path(args: argparse.Namespace, model: str, adaptation: bool) -> Path:
    root = resolve(args.external_root) / args.dataset
    stem = "full" if model == "full" else MODEL_STEMS[model]
    suffix = "_fold9" if adaptation and args.dataset == "ptbxl" else ""
    return root / f"{args.dataset}_{stem}{suffix}_predictions.npz"


def embedding_path(args: argparse.Namespace, model: str, adaptation: bool) -> Path:
    suffix = "_fold9" if adaptation and args.dataset == "ptbxl" else ""
    return resolve(args.embedding_root) / f"external_{args.dataset}_{MODEL_STEMS[model]}{suffix}_record_embeddings.npz"


def embedding_manifest_path(args: argparse.Namespace, model: str, adaptation: bool) -> Path:
    suffix = "_fold9" if adaptation and args.dataset == "ptbxl" else ""
    return MANIFEST_DIR / f"external_{args.dataset}_{MODEL_STEMS[model]}{suffix}_embedding_manifest.json"


def scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    value = data[key]
    return value.item() if np.ndim(value) == 0 else value


def load_prediction(path: Path, dataset: str) -> dict[str, Any]:
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"y_true", "y_prob", "record_id", "group_id", "split_id", "class_names", "dataset"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{path} missing keys: {sorted(missing)}")
        out = {
            "y_true": np.asarray(data["y_true"], dtype=np.float32),
            "y_prob": np.asarray(data["y_prob"], dtype=np.float32),
            "record_id": np.asarray(data["record_id"]).astype(str),
            "group_id": np.asarray(data["group_id"]).astype(str),
            "split_id": np.asarray(data["split_id"]).astype(str),
            "class_names": np.asarray(data["class_names"]).astype(str),
            "group_unit": str(scalar(data, "group_unit", "group")),
            "dataset": str(scalar(data, "dataset", "")),
        }
    if out["dataset"] != dataset:
        raise ValueError(f"{path}: dataset metadata mismatch {out['dataset']} != {dataset}")
    if out["y_true"].ndim != 2 or out["y_true"].shape != out["y_prob"].shape:
        raise ValueError(f"{path}: prediction shape mismatch")
    n_records, n_classes = out["y_true"].shape
    for key in ("record_id", "group_id", "split_id"):
        if len(out[key]) != n_records:
            raise ValueError(f"{path}: {key} length mismatch")
        if np.any(np.char.str_len(out[key].astype(str)) == 0):
            raise ValueError(f"{path}: {key} contains empty identifiers")
    if len(np.unique(out["record_id"])) != n_records:
        raise ValueError(f"{path}: record_id values are not unique")
    if len(out["class_names"]) != n_classes:
        raise ValueError(f"{path}: class_names length mismatch")
    if not np.isfinite(out["y_true"]).all() or not np.isfinite(out["y_prob"]).all():
        raise ValueError(f"{path}: labels or probabilities contain NaN/Inf")
    if not np.isin(out["y_true"], [0.0, 1.0]).all():
        raise ValueError(f"{path}: labels must be binary")
    if float(out["y_prob"].min()) < 0.0 or float(out["y_prob"].max()) > 1.0:
        raise ValueError(f"{path}: probabilities must be in [0,1]")
    if len(np.unique(out["group_id"])) < 2:
        raise ValueError(f"{path}: fewer than two independent groups")
    group_rows = [
        f"{record_id}\x1e{group_id}\x1e{split_id}"
        for record_id, group_id, split_id in zip(
            out["record_id"], out["group_id"], out["split_id"]
        )
    ]
    out.update(
        {
            "path": path,
            "sha256": sha256_file(path),
            "group_assignment_sha256": hashlib.sha256(
                "\x1f".join(group_rows).encode("utf-8")
            ).hexdigest(),
        }
    )
    return out


def load_embeddings(
    path: Path,
    manifest_path: Path,
    prediction: dict[str, Any],
    model: str,
    canonical: dict[str, str],
) -> dict[str, Any]:
    path = resolve(path)
    manifest_path = resolve(manifest_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        raise FileNotFoundError(manifest_path)
    with np.load(path, allow_pickle=False) as data:
        required = {
            "fold_embeddings",
            "y_true",
            "record_id",
            "group_id",
            "split_id",
            "class_names",
            "model",
            "source_prediction_sha256",
            "input_fingerprint",
            "protocol_version",
            "representation",
        }
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{path} missing keys: {sorted(missing)}")
        embeddings = np.asarray(data["fold_embeddings"], dtype=np.float32)
        if str(data["model"].item()) != model:
            raise ValueError(f"{path}: model metadata mismatch")
        if str(data["source_prediction_sha256"].item()) != prediction["sha256"]:
            raise RuntimeError(f"{path}: stale for source predictions")
        if int(data["protocol_version"].item()) != REPRESENTATION_PROTOCOL_VERSION:
            raise RuntimeError(f"{path}: stale representation protocol version")
        if str(data["representation"].item()) != "mean_of_preclassifier_slice_embeddings_per_fold":
            raise RuntimeError(f"{path}: unsupported representation pooling semantics")
        input_fingerprint = str(data["input_fingerprint"].item())
        for key in ("y_true", "record_id", "group_id", "split_id", "class_names"):
            actual = np.asarray(data[key])
            expected = prediction[key]
            if key != "y_true":
                actual = actual.astype(str)
            else:
                actual = actual.astype(np.float32)
            if not np.array_equal(actual, expected):
                raise ValueError(f"{path}: embedding/prediction {key} mismatch")
    if embeddings.ndim != 3 or embeddings.shape[0] != 5 or embeddings.shape[1] != len(prediction["y_true"]):
        raise ValueError(f"{path}: invalid fold_embeddings shape {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{path}: non-finite embeddings")
    output_sha = sha256_file(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    extractor = PROJECT_ROOT / "scripts" / "revision" / "34_extract_external_representations.py"
    checkpoint_rows = manifest.get("checkpoints") or []
    source_provenance = manifest.get("source_provenance") or {}
    source_archive = source_provenance.get("archive") or {}
    checkpoint_folds = [int(row.get("fold", -1)) for row in checkpoint_rows if isinstance(row, dict)]
    checkpoint_hashes = [str(row.get("sha256") or "") for row in checkpoint_rows if isinstance(row, dict)]
    if (
        manifest.get("status") != "complete"
        or manifest.get("protocol") != REPRESENTATION_PROTOCOL
        or manifest.get("runner_sha256") != sha256_file(extractor)
        or manifest.get("canonical_contract") != canonical
        or manifest.get("input_fingerprint") != input_fingerprint
        or manifest.get("representation") != "mean_of_preclassifier_slice_embeddings_per_fold"
        or (manifest.get("source_prediction") or {}).get("sha256") != prediction["sha256"]
        or (manifest.get("output") or {}).get("sha256") != output_sha
        or checkpoint_folds != [1, 2, 3, 4, 5]
        or any(len(value) != 64 for value in checkpoint_hashes)
        or not manifest.get("checkpoint_source_contract")
        or len(str(source_archive.get("sha256") or "")) != 64
    ):
        raise RuntimeError(f"Embedding manifest is stale or incomplete: {manifest_path}")
    return {
        "embedding": embeddings,
        "path": path,
        "sha256": output_sha,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_sha256": checkpoint_hashes,
    }


def validate_external_evidence_gates(
    args: argparse.Namespace,
    model_data: dict[str, dict[str, Any]],
    models: list[str],
    canonical: dict[str, str],
) -> dict[str, Any]:
    gate_path = resolve(args.full_gate_json or METRIC_DIR / f"external_{args.dataset}_protocol_gate.json")
    paired_manifest_path = resolve(args.external_comparator_paired_manifest)
    if not args.strict:
        return {
            "status": "development_gate_checks_disabled",
            "full_gate": str(gate_path),
            "paired_manifest": str(paired_manifest_path),
        }
    if not gate_path.exists() or not paired_manifest_path.exists():
        raise FileNotFoundError(
            "True few-shot adaptation requires the passed Full external gate and paired external comparator "
            f"manifest: {gate_path}; {paired_manifest_path}"
        )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        int(gate.get("gate_schema_version", 0)) < 4
        or gate.get("protocol_gate_passed") is not True
        or gate.get("manuscript_ready") is not True
    ):
        raise RuntimeError(f"Full external gate is not group-safe/manuscript-ready: {gate_path}")
    expected_full_sha = ((gate.get("artifacts") or {}).get("prediction") or {}).get("sha256")
    actual_full_sha = model_data["full"]["test_pred"]["sha256"]
    if expected_full_sha != actual_full_sha:
        raise RuntimeError("Full external gate is stale for the test prediction artifact")

    paired_manifest = json.loads(paired_manifest_path.read_text(encoding="utf-8"))
    if paired_manifest.get("status") != "complete" or paired_manifest.get("failures"):
        raise RuntimeError(f"External comparator paired manifest is not complete: {paired_manifest_path}")
    paired_runner = PROJECT_ROOT / "scripts" / "revision" / "32_paired_external_comparators.py"
    if (
        paired_manifest.get("canonical_contract") != canonical
        or not paired_runner.exists()
        or paired_manifest.get("runner_sha256") != sha256_file(paired_runner)
    ):
        raise RuntimeError("External comparator paired manifest has a stale canonical/runner contract")
    indexed_inputs = {
        (str(item.get("dataset")), str(item.get("model")), str(item.get("sha256")))
        for item in paired_manifest.get("inputs", [])
    }
    missing = []
    for model in models:
        expected = (args.dataset, model, model_data[model]["test_pred"]["sha256"])
        if expected not in indexed_inputs:
            missing.append(expected)
    if missing:
        raise RuntimeError(f"External comparator paired manifest lacks current inputs: {missing}")
    return {
        "status": "passed",
        "full_gate": {"path": str(gate_path), "sha256": sha256_file(gate_path)},
        "paired_manifest": {
            "path": str(paired_manifest_path),
            "sha256": sha256_file(paired_manifest_path),
        },
    }


def group_indices(groups: np.ndarray, selected: np.ndarray) -> np.ndarray:
    return np.where(np.isin(groups, selected))[0].astype(np.int64)


def nested_groups(pool: np.ndarray, fraction: float) -> np.ndarray:
    if fraction <= 0:
        return np.asarray([], dtype=pool.dtype)
    n = min(max(int(round(len(pool) * fraction)), 1), len(pool))
    return pool[:n]


def fit_fold_heads(
    train_embeddings: np.ndarray,
    y_train: np.ndarray,
    test_embeddings: np.ndarray,
    zero_prob: np.ndarray,
    class_names: np.ndarray,
    seed: int,
    c_value: float,
    max_iter: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    fold_probabilities: list[np.ndarray] = []
    coefficient_rows: list[dict[str, Any]] = []
    n_folds, _n_train, dimension = train_embeddings.shape
    for fold in range(n_folds):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_embeddings[fold])
        x_test = scaler.transform(test_embeddings[fold])
        fold_prob = zero_prob.copy().astype(np.float64)
        for class_index, class_name in enumerate(class_names):
            labels = y_train[:, class_index].astype(int)
            if len(np.unique(labels)) < 2:
                coefficient_rows.append(
                    {
                        "fold": fold + 1,
                        "class_index": class_index,
                        "class_name": class_name,
                        "status": "fallback_zero_shot_single_label",
                        "coefficient_l2": np.nan,
                        "intercept": np.nan,
                        "n_iter": 0,
                    }
                )
                continue
            classifier = LogisticRegression(
                C=c_value,
                solver="lbfgs",
                class_weight="balanced",
                max_iter=max_iter,
                random_state=seed + fold,
            )
            classifier.fit(x_train, labels)
            if int(classifier.n_iter_[0]) >= int(classifier.max_iter):
                raise RuntimeError(
                    f"Linear head did not converge for fold {fold + 1}, class {class_name}"
                )
            fold_prob[:, class_index] = classifier.predict_proba(x_test)[:, 1]
            coefficient_rows.append(
                {
                    "fold": fold + 1,
                    "class_index": class_index,
                    "class_name": class_name,
                    "status": "adapted",
                    "coefficient_l2": float(np.linalg.norm(classifier.coef_)),
                    "intercept": float(classifier.intercept_[0]),
                    "n_iter": int(classifier.n_iter_[0]),
                }
            )
        fold_probabilities.append(np.clip(fold_prob, 0.0, 1.0).astype(np.float32))
    return np.mean(np.stack(fold_probabilities), axis=0).astype(np.float32), coefficient_rows


def metric_functions(threshold: float, n_bins: int) -> dict[str, tuple[Callable, bool]]:
    return {
        "pr_auc_macro": (macro_pr_auc, True),
        "roc_auc_macro": (macro_roc_auc, True),
        "f1_macro": (lambda y, p: multilabel_metrics(y, p, threshold=threshold)["f1_macro"], True),
        "brier_macro": (lambda y, p: calibration_summary(y, p, n_bins=n_bins)["brier_macro"], False),
        "ece_macro": (lambda y, p: calibration_summary(y, p, n_bins=n_bins)["ece_macro"], False),
    }


def point_metrics(y: np.ndarray, p: np.ndarray, threshold: float, n_bins: int) -> dict[str, float]:
    return {**multilabel_metrics(y, p, threshold=threshold), **calibration_summary(y, p, n_bins=n_bins)}


def cache_key(
    args: argparse.Namespace,
    model: str,
    seed: int,
    fraction: float,
    test_pred_sha: str,
    test_embedding_sha: str,
    adaptation_embedding_sha: str,
    train_groups: np.ndarray,
    test_groups: np.ndarray,
    canonical: dict[str, Any],
    test_group_assignment_sha256: str,
    adaptation_group_assignment_sha256: str,
    analysis_lock_sha256: str | None,
) -> str:
    payload = {
        "protocol": PROTOCOL,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "dataset": args.dataset,
        "model": model,
        "seed": seed,
        "fraction": fraction,
        "head_c": args.head_c,
        "max_iter": args.max_iter,
        "test_pred_sha": test_pred_sha,
        "test_embedding_sha": test_embedding_sha,
        "adaptation_embedding_sha": adaptation_embedding_sha,
        "canonical_oof_sha256": canonical["oof_sha256"],
        "canonical_freeze_sha256": canonical["freeze_sha256"],
        "canonical_group_contract_sha256": canonical["group_contract_sha256"],
        "canonical_group_sidecar_sha256": canonical["group_sidecar_sha256"],
        "test_group_assignment_sha256": test_group_assignment_sha256,
        "adaptation_group_assignment_sha256": adaptation_group_assignment_sha256,
        "analysis_lock_sha256": analysis_lock_sha256,
        "train_groups_sha256": hashlib.sha256(
            np.asarray(train_groups).astype(str).tobytes()
        ).hexdigest(),
        "test_groups_sha256": hashlib.sha256(
            np.asarray(test_groups).astype(str).tobytes()
        ).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def metric_cache_contract(
    args: argparse.Namespace,
    *,
    comparison: str,
    metric: str,
    seed: int,
    fraction: float,
    prediction_keys: dict[str, str],
    train_groups: np.ndarray,
    test_groups: np.ndarray,
    canonical: dict[str, Any],
    analysis_lock_sha256: str | None,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "dataset": args.dataset,
        "comparison": comparison,
        "metric": metric,
        "seed": seed,
        "fraction": fraction,
        "prediction_keys": prediction_keys,
        "train_groups_sha256": hashlib.sha256(np.asarray(train_groups).astype(str).tobytes()).hexdigest(),
        "test_groups_sha256": hashlib.sha256(np.asarray(test_groups).astype(str).tobytes()).hexdigest(),
        "threshold": args.threshold,
        "n_bins": args.n_bins,
        "n_boot": args.n_boot,
        "canonical_oof_sha256": canonical["oof_sha256"],
        "canonical_freeze_sha256": canonical["freeze_sha256"],
        "canonical_group_contract_sha256": canonical["group_contract_sha256"],
        "canonical_group_sidecar_sha256": canonical["group_sidecar_sha256"],
        "analysis_lock_sha256": analysis_lock_sha256,
    }


def metric_cache_key(contract: dict[str, Any]) -> str:
    return canonical_json_sha256(contract)


def validate_interval_payload(
    payload: dict[str, Any],
    *,
    n_boot: int,
    low_field: str,
    high_field: str,
) -> None:
    if int(payload.get("n_boot_valid", -1)) != int(n_boot):
        raise RuntimeError(
            f"Bootstrap payload has n_boot_valid={payload.get('n_boot_valid')}; exactly {n_boot} are required"
        )
    for field in (low_field, high_field):
        try:
            value = float(payload[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Bootstrap payload lacks numeric {field}") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"Bootstrap payload contains non-finite {field}")
    for key, value in payload.items():
        if "significant" in str(key).lower() or "significant" in str(value).lower():
            raise RuntimeError("Bootstrap payload contains prohibited legacy significance wording")


def exact_zero_delta(n_boot: int, n_groups: int) -> dict[str, Any]:
    return {
        "point_delta_a_minus_b": 0.0,
        "mean": 0.0,
        "lo": 0.0,
        "hi": 0.0,
        "n_boot_valid": int(n_boot),
        "n_groups": int(n_groups),
        "sample_unit": "group",
    }


def shared_group_bootstrap_indices(groups: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    groups = np.asarray(groups).astype(str)
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("Primary adaptation bootstrap requires at least two independent groups")
    members = [np.where(inverse == index)[0] for index in range(len(unique))]
    rng = np.random.default_rng(seed)
    return [
        np.concatenate(
            [members[int(group_index)] for group_index in rng.integers(0, len(unique), size=len(unique))]
        )
        for _ in range(int(n_boot))
    ]


def metric_over_seed_ensemble(
    y_true: np.ndarray,
    probabilities_by_seed: dict[int, np.ndarray],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    bootstrap_indices: list[np.ndarray],
) -> tuple[float, np.ndarray]:
    ordered = [probabilities_by_seed[seed] for seed in sorted(probabilities_by_seed)]
    if not ordered:
        raise ValueError("No seed predictions were provided for the primary endpoint")
    point = float(np.mean([metric_fn(y_true, probability) for probability in ordered]))
    values = np.full(len(bootstrap_indices), np.nan, dtype=np.float64)
    for bootstrap_index, row_index in enumerate(bootstrap_indices):
        seed_values = [float(metric_fn(y_true[row_index], probability[row_index])) for probability in ordered]
        finite = [value for value in seed_values if np.isfinite(value)]
        if finite:
            values[bootstrap_index] = float(np.mean(finite))
    return point, values


def metric_single_prediction(
    y_true: np.ndarray,
    probability: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    bootstrap_indices: list[np.ndarray],
) -> tuple[float, np.ndarray]:
    point = float(metric_fn(y_true, probability))
    values = np.full(len(bootstrap_indices), np.nan, dtype=np.float64)
    for bootstrap_index, row_index in enumerate(bootstrap_indices):
        value = float(metric_fn(y_true[row_index], probability[row_index]))
        if np.isfinite(value):
            values[bootstrap_index] = value
    return point, values


def interval(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return math.nan, math.nan, 0
    return (
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
        int(len(finite)),
    )


def primary_endpoint_rows(
    *,
    dataset: str,
    y_true: np.ndarray,
    groups: np.ndarray,
    predictions: dict[str, dict[int, np.ndarray]],
    zero_probabilities: dict[str, np.ndarray],
    threshold: float,
    n_bins: int,
    n_boot: int,
    primary_fraction: float,
    bootstrap_seed: int = 20260712,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indices = shared_group_bootstrap_indices(groups, n_boot, bootstrap_seed)
    metric_rows: list[dict[str, Any]] = []
    distributions: dict[tuple[str, str], tuple[float, np.ndarray]] = {}
    bootstrap_payload: dict[str, Any] = {}
    specs = metric_functions(threshold, n_bins)
    for model, model_predictions in predictions.items():
        if not model_predictions:
            raise RuntimeError(f"{model}: invalid primary seed prediction map")
        for metric_name, (metric_fn, higher_is_better) in specs.items():
            adapted_point, adapted_values = metric_over_seed_ensemble(
                y_true, model_predictions, metric_fn, indices
            )
            zero_point, zero_values = metric_single_prediction(
                y_true, zero_probabilities[model], metric_fn, indices
            )
            distributions[(model, metric_name)] = (adapted_point, adapted_values)
            adapted_low, adapted_high, adapted_valid = interval(adapted_values)
            oriented_point = (
                adapted_point - zero_point if higher_is_better else zero_point - adapted_point
            )
            oriented_values = (
                adapted_values - zero_values if higher_is_better else zero_values - adapted_values
            )
            improvement_low, improvement_high, improvement_valid = interval(oriented_values)
            if adapted_valid != int(n_boot) or improvement_valid != int(n_boot):
                raise RuntimeError(
                    f"Primary endpoint bootstrap for {model}/{metric_name} did not produce "
                    f"the exact requested {n_boot} finite replicates"
                )
            if not all(
                math.isfinite(value)
                for value in (adapted_low, adapted_high, improvement_low, improvement_high)
            ):
                raise RuntimeError(
                    f"Primary endpoint bootstrap for {model}/{metric_name} contains non-finite CI bounds"
                )
            row = {
                "dataset": dataset,
                "comparison_type": "adapted_vs_zero_target_label",
                "model": model,
                "model_label": MODEL_LABELS[model],
                "metric": metric_name,
                "higher_is_better": higher_is_better,
                "primary_fraction": primary_fraction,
                "primary_value_mean_across_seeds": adapted_point,
                "primary_value_ci_low": adapted_low,
                "primary_value_ci_high": adapted_high,
                "zero_target_label_value": zero_point,
                "improvement_primary_over_zero": oriented_point,
                "improvement_ci_low": improvement_low,
                "improvement_ci_high": improvement_high,
                "n_seeds": len(model_predictions),
                "n_groups": len(np.unique(groups)),
                "n_boot_valid": min(adapted_valid, improvement_valid),
                "bootstrap_seed": bootstrap_seed,
                "bootstrap_unit": "patient/source-record group",
                "uncertainty_scope": (
                    "shared patient-group bootstrap conditional on the fixed encoder/head fits and "
                    "the analysis-locked adaptation seed grid; training-seed and encoder-refit variability "
                    "are not included"
                ),
                "interpretation": interval_interpretation(
                    improvement_low,
                    improvement_high,
                    positive="primary_ci_favors_adaptation",
                    negative="primary_ci_favors_zero_target_label",
                    inconclusive="paired_primary_difference_inconclusive",
                ),
            }
            metric_rows.append(row)
            bootstrap_payload[f"{model}_{metric_name}"] = {
                "adapted_mean_across_seeds": adapted_values.tolist(),
                "zero_target_label": zero_values.tolist(),
                "oriented_improvement": oriented_values.tolist(),
            }

    full_predictions = predictions["full"]
    for comparator in [model for model in predictions if model != "full"]:
        if set(predictions[comparator]) != set(full_predictions):
            raise RuntimeError(f"Primary seed grid differs for full vs {comparator}")
        for metric_name, (_metric_fn, higher_is_better) in specs.items():
            full_point, full_values = distributions[("full", metric_name)]
            comparator_point, comparator_values = distributions[(comparator, metric_name)]
            oriented_point = (
                full_point - comparator_point
                if higher_is_better
                else comparator_point - full_point
            )
            oriented_values = (
                full_values - comparator_values
                if higher_is_better
                else comparator_values - full_values
            )
            low, high, valid = interval(oriented_values)
            if valid != int(n_boot) or not math.isfinite(low) or not math.isfinite(high):
                raise RuntimeError(
                    f"Primary paired bootstrap for full vs {comparator}/{metric_name} "
                    f"did not produce {n_boot} finite replicates"
                )
            metric_rows.append(
                {
                    "dataset": dataset,
                    "comparison_type": "full_vs_comparator_at_primary_fraction",
                    "model": "full",
                    "model_label": MODEL_LABELS["full"],
                    "comparator": comparator,
                    "comparator_label": MODEL_LABELS[comparator],
                    "metric": metric_name,
                    "higher_is_better": higher_is_better,
                    "primary_fraction": primary_fraction,
                    "primary_value_mean_across_seeds": full_point,
                    "comparator_value_mean_across_seeds": comparator_point,
                    "improvement_full_over_comparator": oriented_point,
                    "improvement_ci_low": low,
                    "improvement_ci_high": high,
                    "n_seeds": len(full_predictions),
                    "n_groups": len(np.unique(groups)),
                    "n_boot_valid": valid,
                    "bootstrap_seed": bootstrap_seed,
                    "bootstrap_unit": "patient/source-record group",
                    "uncertainty_scope": (
                        "shared patient-group bootstrap conditional on the fixed encoder/head fits and "
                        "the analysis-locked adaptation seed grid; training-seed and encoder-refit variability "
                        "are not included"
                    ),
                    "interpretation": interval_interpretation(
                        low,
                        high,
                        positive="primary_ci_favors_full",
                        negative="primary_ci_favors_comparator",
                        inconclusive="paired_primary_difference_inconclusive",
                    ),
                }
            )
            bootstrap_payload[f"full_vs_{comparator}_{metric_name}"] = {
                "oriented_improvement": oriented_values.tolist()
            }
    return metric_rows, {
        "primary_fraction": primary_fraction,
        "bootstrap_seed": bootstrap_seed,
        "n_boot": n_boot,
        "n_groups": len(np.unique(groups)),
        "n_seeds": len(next(iter(predictions.values()))),
        "uncertainty_scope": (
            "shared patient-group bootstrap conditional on fixed encoder/head fits and the "
            "analysis-locked adaptation seed grid; no encoder or head refit within bootstrap replicates"
        ),
        "items": bootstrap_payload,
    }


def write_learning_curve_figure(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot fold-safe patient-bootstrap learning curves for F1 and PR-AUC."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), dpi=180, sharex=True)
    if not rows:
        for axis in axes:
            axis.text(0.5, 0.5, "Shared-test-set learning curve unavailable", ha="center", va="center")
            axis.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return
    colors = {
        "full": "#226F54",
        "resnet": "#3366AA",
        "raw_mamba": "#B34233",
        "transformer": "#8A5A9E",
    }
    for axis, metric, title in zip(axes, ["f1_macro", "pr_auc_macro"], ["Macro F1", "Macro PR-AUC"]):
        metric_rows = [
            row
            for row in rows
            if row.get("comparison_type") == "adapted_vs_zero_target_label"
            and row.get("metric") == metric
        ]
        for model in sorted({str(row["model"]) for row in metric_rows}):
            selected = sorted(
                [row for row in metric_rows if row["model"] == model],
                key=lambda row: float(row["fraction"]),
            )
            x = np.asarray([100.0 * float(row["fraction"]) for row in selected])
            y = np.asarray([float(row["value_mean_across_seeds"]) for row in selected])
            low = np.asarray([float(row["value_ci_low"]) for row in selected])
            high = np.asarray([float(row["value_ci_high"]) for row in selected])
            axis.errorbar(
                x,
                y,
                yerr=np.vstack([
                    np.maximum(0.0, y - low),
                    np.maximum(0.0, high - y),
                ]),
                marker="o",
                linewidth=1.4,
                capsize=2,
                label=MODEL_LABELS[model],
                color=colors.get(model),
            )
        axis.set_title(title)
        axis.set_xlabel("Target labels (%)")
        axis.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Metric value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    canonical = canonical_contract()
    models = parse_list(args.models)
    unknown = sorted(set(models) - set(MODEL_STEMS))
    if unknown or "full" not in models:
        raise ValueError(f"Models must include full and valid names; unknown={unknown}")
    fractions = parse_fractions(args.fractions)
    if not any(math.isclose(item, args.primary_fraction, abs_tol=1e-12) for item in fractions):
        raise ValueError("--primary-fraction must be included in --fractions")
    seeds = parse_seeds(args.seeds)
    if not 0 < args.test_fraction < 1:
        raise ValueError("--test-fraction must be in (0,1)")
    analysis_lock = (
        validate_ptbxl_analysis_lock(
            args.analysis_lock,
            models=models,
            fractions=fractions,
            primary_fraction=args.primary_fraction,
            seeds=seeds,
            threshold=args.threshold,
            n_bins=args.n_bins,
            n_boot=args.n_boot,
            head_c=args.head_c,
            max_iter=args.max_iter,
        )
        if args.dataset == "ptbxl"
        else None
    )
    paths = default_paths(args)
    cache_dir = resolve(args.cache_dir) / args.dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    metric_cache_dir = resolve(args.metric_cache_dir) / args.dataset
    metric_cache_dir.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 80, flush=True)
    print("TRUE FEW-SHOT FROZEN-ENCODER LINEAR-HEAD ADAPTATION", flush=True)
    print("=" * 80, flush=True)
    print(f"dataset={args.dataset} models={models} fractions={fractions} seeds={seeds}", flush=True)

    model_data: dict[str, dict[str, Any]] = {}
    for model in models:
        test_pred = load_prediction(source_prediction_path(args, model, False), args.dataset)
        test_emb = load_embeddings(
            embedding_path(args, model, False),
            embedding_manifest_path(args, model, False),
            test_pred,
            model,
            canonical,
        )
        if args.dataset == "ptbxl":
            adapt_pred = load_prediction(source_prediction_path(args, model, True), args.dataset)
            adapt_emb = load_embeddings(
                embedding_path(args, model, True),
                embedding_manifest_path(args, model, True),
                adapt_pred,
                model,
                canonical,
            )
            if set(test_pred["split_id"]) != {"ptbxl_fold10"} or set(adapt_pred["split_id"]) != {"ptbxl_fold9"}:
                raise RuntimeError("PTB-XL requires fold9 adaptation and fold10 test")
            if set(test_pred["group_id"]) & set(adapt_pred["group_id"]):
                raise RuntimeError("PTB-XL patient overlap between adaptation and test")
        else:
            adapt_pred = test_pred
            adapt_emb = test_emb
        model_data[model] = {
            "test_pred": test_pred,
            "test_emb": test_emb,
            "adapt_pred": adapt_pred,
            "adapt_emb": adapt_emb,
        }
    evidence_gate_contract = validate_external_evidence_gates(args, model_data, models, canonical)
    reference = model_data["full"]["test_pred"]
    adaptation_reference = model_data["full"]["adapt_pred"]
    for model, data in model_data.items():
        for key in ("y_true", "record_id", "group_id", "split_id", "class_names"):
            if not np.array_equal(data["test_pred"][key], reference[key]):
                raise RuntimeError(f"{model}: test {key} differs from Full")
            if not np.array_equal(data["adapt_pred"][key], adaptation_reference[key]):
                raise RuntimeError(f"{model}: adaptation {key} differs from Full")

    rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    bootstrap: dict[str, Any] = {}
    split_arrays: dict[str, np.ndarray] = {}
    split_audits: dict[str, Any] = {}
    primary_predictions: dict[str, dict[int, np.ndarray]] = {model: {} for model in models}
    primary_zero_probabilities: dict[str, np.ndarray] = {}
    primary_y_true: np.ndarray | None = None
    primary_groups: np.ndarray | None = None
    curve_predictions: dict[float, dict[str, dict[int, np.ndarray]]] = {
        fraction: {model: {} for model in models} for fraction in fractions
    }
    curve_zero_probabilities: dict[str, np.ndarray] = {}
    curve_y_true: np.ndarray | None = None
    curve_groups: np.ndarray | None = None
    for seed in seeds:
        if args.dataset == "ptbxl":
            pool_groups = np.random.default_rng(seed).permutation(
                np.unique(model_data["full"]["adapt_pred"]["group_id"])
            )
            test_groups = np.unique(reference["group_id"])
            test_idx = np.arange(len(reference["y_true"]), dtype=np.int64)
            split_audits[f"seed{seed}"] = {
                "split_policy": "official_ptbxl_fold9_adaptation_fold10_test",
                "train_groups": int(len(pool_groups)),
                "test_groups": int(len(test_groups)),
                "group_overlap": 0,
            }
        else:
            pool_groups, test_groups, split_audit = hash_group_train_test_split(
                reference["group_id"],
                args.test_fraction,
                seed,
            )
            split_audits[f"seed{seed}"] = split_audit
            test_idx = group_indices(reference["group_id"], test_groups)
        split_arrays[f"seed{seed}_test_group_id"] = test_groups
        split_arrays[f"seed{seed}_test_index"] = test_idx
        for fraction in fractions:
            train_groups = nested_groups(pool_groups, fraction)
            split_key = f"seed{seed}_frac{fraction:g}".replace(".", "p")
            split_arrays[f"{split_key}_train_group_id"] = train_groups
            if set(train_groups) & set(test_groups):
                raise RuntimeError(f"Group leakage in {split_key}")
            predictions_by_model: dict[str, np.ndarray] = {}
            prediction_keys_by_model: dict[str, str] = {}
            y_test = reference["y_true"][test_idx]
            groups_test = reference["group_id"][test_idx]
            for model in models:
                data = model_data[model]
                train_idx = group_indices(data["adapt_pred"]["group_id"], train_groups)
                zero_prob = data["test_pred"]["y_prob"][test_idx]
                key = cache_key(
                    args,
                    model,
                    seed,
                    fraction,
                    data["test_pred"]["sha256"],
                    data["test_emb"]["sha256"],
                    data["adapt_emb"]["sha256"],
                    train_groups,
                    test_groups,
                    canonical,
                    data["test_pred"]["group_assignment_sha256"],
                    data["adapt_pred"]["group_assignment_sha256"],
                    (analysis_lock or {}).get("sha256"),
                )
                cache = cache_dir / f"{model}_{split_key}_{key[:16]}.npz"
                coefficient_cache = cache.with_suffix(".coefficients.json")
                if args.reuse_existing and not args.force_rerun and cache.exists() and coefficient_cache.exists():
                    with np.load(cache, allow_pickle=False) as saved:
                        adapted_prob = np.asarray(saved["y_prob"], dtype=np.float32)
                        cached_test = np.asarray(saved["test_index"], dtype=np.int64)
                        cached_train_groups = np.asarray(saved["train_group_id"]).astype(str)
                        cached_key = str(saved["cache_key"].item())
                        cached_runner_sha256 = str(saved["runner_sha256"].item())
                        cached_group_contract_sha256 = str(
                            saved["canonical_group_contract_sha256"].item()
                        )
                        cached_group_sidecar_sha256 = str(
                            saved["canonical_group_sidecar_sha256"].item()
                        )
                        cached_analysis_lock_sha256 = str(
                            saved["analysis_lock_sha256"].item()
                        )
                    if (
                        cached_key != key
                        or cached_runner_sha256 != sha256_file(Path(__file__).resolve())
                        or cached_group_contract_sha256 != canonical["group_contract_sha256"]
                        or cached_group_sidecar_sha256 != canonical["group_sidecar_sha256"]
                        or cached_analysis_lock_sha256 != str((analysis_lock or {}).get("sha256") or "")
                        or adapted_prob.shape != zero_prob.shape
                        or not np.isfinite(adapted_prob).all()
                        or not np.array_equal(cached_test, test_idx)
                        or not np.array_equal(cached_train_groups, train_groups.astype(str))
                    ):
                        raise RuntimeError(f"Adaptation cache contract mismatch: {cache}")
                    coefficient_payload = json.loads(coefficient_cache.read_text(encoding="utf-8"))
                    if (
                        not isinstance(coefficient_payload, dict)
                        or coefficient_payload.get("cache_key") != key
                        or coefficient_payload.get("runner_sha256")
                        != sha256_file(Path(__file__).resolve())
                        or coefficient_payload.get("canonical_group_contract_sha256")
                        != canonical["group_contract_sha256"]
                        or coefficient_payload.get("canonical_group_sidecar_sha256")
                        != canonical["group_sidecar_sha256"]
                        or coefficient_payload.get("analysis_lock_sha256")
                        != str((analysis_lock or {}).get("sha256") or "")
                    ):
                        raise RuntimeError(f"Coefficient cache contract mismatch: {coefficient_cache}")
                    coefficients = (
                        coefficient_payload.get("coefficients", [])
                    )
                    print(f"Reusing {model}/{split_key}: {cache}", flush=True)
                elif len(train_idx) == 0:
                    adapted_prob = zero_prob.astype(np.float32)
                    coefficients = []
                    save_npz_compressed_atomic(
                        cache,
                        y_prob=adapted_prob,
                        test_index=test_idx,
                        train_group_id=train_groups,
                        cache_key=np.asarray(key),
                        runner_sha256=np.asarray(sha256_file(Path(__file__).resolve())),
                        canonical_group_contract_sha256=np.asarray(
                            canonical["group_contract_sha256"]
                        ),
                        canonical_group_sidecar_sha256=np.asarray(
                            canonical["group_sidecar_sha256"]
                        ),
                        analysis_lock_sha256=np.asarray(str((analysis_lock or {}).get("sha256") or "")),
                    )
                    save_json_atomic(
                        coefficient_cache,
                        {
                            "cache_key": key,
                            "runner_sha256": sha256_file(Path(__file__).resolve()),
                            "canonical_group_contract_sha256": canonical[
                                "group_contract_sha256"
                            ],
                            "canonical_group_sidecar_sha256": canonical[
                                "group_sidecar_sha256"
                            ],
                            "analysis_lock_sha256": str((analysis_lock or {}).get("sha256") or ""),
                            "coefficients": [],
                        },
                    )
                else:
                    adapted_prob, coefficients = fit_fold_heads(
                        data["adapt_emb"]["embedding"][:, train_idx, :],
                        data["adapt_pred"]["y_true"][train_idx],
                        data["test_emb"]["embedding"][:, test_idx, :],
                        zero_prob,
                        reference["class_names"],
                        seed,
                        args.head_c,
                        args.max_iter,
                    )
                    save_npz_compressed_atomic(
                        cache,
                        y_prob=adapted_prob,
                        test_index=test_idx,
                        train_group_id=train_groups,
                        cache_key=np.asarray(key),
                        runner_sha256=np.asarray(sha256_file(Path(__file__).resolve())),
                        canonical_group_contract_sha256=np.asarray(
                            canonical["group_contract_sha256"]
                        ),
                        canonical_group_sidecar_sha256=np.asarray(
                            canonical["group_sidecar_sha256"]
                        ),
                        analysis_lock_sha256=np.asarray(str((analysis_lock or {}).get("sha256") or "")),
                    )
                    save_json_atomic(
                        coefficient_cache,
                        {
                            "cache_key": key,
                            "runner_sha256": sha256_file(Path(__file__).resolve()),
                            "canonical_group_contract_sha256": canonical[
                                "group_contract_sha256"
                            ],
                            "canonical_group_sidecar_sha256": canonical[
                                "group_sidecar_sha256"
                            ],
                            "analysis_lock_sha256": str((analysis_lock or {}).get("sha256") or ""),
                            "coefficients": coefficients,
                        },
                    )
                    print(f"Wrote {model}/{split_key}: {cache}", flush=True)
                predictions_by_model[model] = adapted_prob
                prediction_keys_by_model[model] = key
                for coefficient in coefficients:
                    coefficient_rows.append(
                        {
                            "dataset": args.dataset,
                            "model": model,
                            "seed": seed,
                            "fraction": fraction,
                            **coefficient,
                        }
                    )
                point = point_metrics(y_test, adapted_prob, args.threshold, args.n_bins)
                row = {
                    "dataset": args.dataset,
                    "task_scope": "cpsc2021_10s_af_afl_mapped_windows" if args.dataset == "cpsc2021" else "record_level_mapped_external_task",
                    "protocol": PROTOCOL,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "mode": "zero_target_label" if len(train_idx) == 0 else "frozen_encoder_linear_head_adaptation",
                    "seed": seed,
                    "fraction": fraction,
                    "budget_role": budget_role(fraction, args.primary_fraction),
                    "fraction_unit": "independent_target_groups_from_adaptation_pool",
                    "fraction_sampling": "nested_seeded_label_independent_group_prefix",
                    "embedding_dimension": int(data["test_emb"]["embedding"].shape[2]),
                    "representation_pooling": "mean_of_preclassifier_slice_embeddings_per_fold",
                    "fold_heads": 0 if len(train_idx) == 0 else 5,
                    "train_groups": int(len(train_groups)),
                    "train_records_or_windows": int(len(train_idx)),
                    "test_groups": int(len(np.unique(groups_test))),
                    "test_records_or_windows": int(len(test_idx)),
                    "group_overlap": 0,
                    **point,
                }
                rows.append(row)
                metric_item: dict[str, Any] = {}
                for metric_name, (metric_fn, higher) in metric_functions(args.threshold, args.n_bins).items():
                    zero = data["test_pred"]["y_prob"][test_idx]
                    metric_contract = metric_cache_contract(
                        args,
                        comparison=f"{model}_adapted_vs_zero",
                        metric=metric_name,
                        seed=seed,
                        fraction=fraction,
                        prediction_keys={model: key},
                        train_groups=train_groups,
                        test_groups=test_groups,
                        canonical=canonical,
                        analysis_lock_sha256=(analysis_lock or {}).get("sha256"),
                    )
                    metric_key = metric_cache_key(metric_contract)
                    metric_cache = metric_cache_dir / (
                        f"{model}_{split_key}_{metric_name}_{metric_key[:16]}.json"
                    )
                    metric_cache_needs_write = False
                    if args.reuse_existing and not args.force_rerun and metric_cache.exists():
                        cached_metric = json.loads(metric_cache.read_text(encoding="utf-8"))
                        if (
                            cached_metric.get("cache_key") != metric_key
                            or cached_metric.get("contract") != metric_contract
                        ):
                            raise RuntimeError(f"Metric cache key mismatch: {metric_cache}")
                        ci = cached_metric["cluster_ci"]
                        paired = cached_metric["paired_adapted_minus_zero"]
                        print(f"Reusing metric cache: {metric_cache}", flush=True)
                    else:
                        ci = cluster_bootstrap_ci(
                            y_test,
                            adapted_prob,
                            groups_test,
                            metric_fn,
                            n_boot=args.n_boot,
                            seed=seed,
                        )
                        paired = (
                            exact_zero_delta(args.n_boot, len(np.unique(groups_test)))
                            if np.array_equal(adapted_prob, zero)
                            else paired_cluster_bootstrap_delta(
                                y_test,
                                adapted_prob,
                                zero,
                                groups_test,
                                metric_fn,
                                n_boot=args.n_boot,
                                seed=seed,
                            )
                        )
                        metric_cache_needs_write = True
                    validate_interval_payload(
                        ci,
                        n_boot=args.n_boot,
                        low_field="lo",
                        high_field="hi",
                    )
                    validate_interval_payload(
                        paired,
                        n_boot=args.n_boot,
                        low_field="lo",
                        high_field="hi",
                    )
                    if metric_cache_needs_write:
                        save_json_atomic(
                            metric_cache,
                            {
                                "cache_key": metric_key,
                                "contract": metric_contract,
                                "cluster_ci": ci,
                                "paired_adapted_minus_zero": paired,
                            },
                        )
                    metric_item[metric_name] = {
                        "cluster_ci": ci,
                        "paired_adapted_minus_zero": paired,
                        "improvement_orientation": "higher" if higher else "lower",
                    }
                bootstrap[f"{model}_{split_key}"] = metric_item

            if args.dataset == "ptbxl":
                if curve_y_true is None:
                    curve_y_true = y_test.copy()
                    curve_groups = groups_test.copy()
                elif not np.array_equal(curve_y_true, y_test) or not np.array_equal(
                    curve_groups, groups_test
                ):
                    raise RuntimeError("PTB-XL learning-curve test groups differ across seeds")
                for model in models:
                    curve_predictions[fraction][model][seed] = predictions_by_model[model].copy()
                    zero = model_data[model]["test_pred"]["y_prob"][test_idx]
                    if model in curve_zero_probabilities and not np.array_equal(
                        curve_zero_probabilities[model], zero
                    ):
                        raise RuntimeError(
                            f"{model}: learning-curve zero-target probabilities changed across seeds"
                        )
                    curve_zero_probabilities[model] = zero.copy()

            if args.dataset == "ptbxl" and math.isclose(
                fraction, args.primary_fraction, abs_tol=1e-12
            ):
                if primary_y_true is None:
                    primary_y_true = y_test.copy()
                    primary_groups = groups_test.copy()
                elif not np.array_equal(primary_y_true, y_test) or not np.array_equal(
                    primary_groups, groups_test
                ):
                    raise RuntimeError("PTB-XL primary endpoint test groups differ across seeds")
                for model in models:
                    primary_predictions[model][seed] = predictions_by_model[model].copy()
                    zero = model_data[model]["test_pred"]["y_prob"][test_idx]
                    if model in primary_zero_probabilities and not np.array_equal(
                        primary_zero_probabilities[model], zero
                    ):
                        raise RuntimeError(f"{model}: zero-target-label test probabilities changed across seeds")
                    primary_zero_probabilities[model] = zero.copy()

            full_prob = predictions_by_model["full"]
            for comparator in [model for model in models if model != "full"]:
                for metric_name, (metric_fn, higher) in metric_functions(args.threshold, args.n_bins).items():
                    metric_contract = metric_cache_contract(
                        args,
                        comparison=f"full_vs_{comparator}",
                        metric=metric_name,
                        seed=seed,
                        fraction=fraction,
                        prediction_keys={
                            "full": prediction_keys_by_model["full"],
                            comparator: prediction_keys_by_model[comparator],
                        },
                        train_groups=train_groups,
                        test_groups=test_groups,
                        canonical=canonical,
                        analysis_lock_sha256=(analysis_lock or {}).get("sha256"),
                    )
                    metric_key = metric_cache_key(metric_contract)
                    metric_cache = metric_cache_dir / (
                        f"full_vs_{comparator}_{split_key}_{metric_name}_{metric_key[:16]}.json"
                    )
                    if args.reuse_existing and not args.force_rerun and metric_cache.exists():
                        cached_metric = json.loads(metric_cache.read_text(encoding="utf-8"))
                        if (
                            cached_metric.get("cache_key") != metric_key
                            or cached_metric.get("contract") != metric_contract
                        ):
                            raise RuntimeError(f"Metric cache key mismatch: {metric_cache}")
                        paired = cached_metric["paired"]
                        print(f"Reusing paired metric cache: {metric_cache}", flush=True)
                    else:
                        paired = paired_cluster_bootstrap_delta(
                            y_test,
                            full_prob,
                            predictions_by_model[comparator],
                            groups_test,
                            metric_fn,
                            n_boot=args.n_boot,
                            seed=seed,
                        )
                        save_json_atomic(
                            metric_cache,
                            {
                                "cache_key": metric_key,
                                "contract": metric_contract,
                                "paired": paired,
                            },
                        )
                    validate_interval_payload(
                        paired,
                        n_boot=args.n_boot,
                        low_field="lo",
                        high_field="hi",
                    )
                    point = paired["point_delta_a_minus_b"] if higher else -paired["point_delta_a_minus_b"]
                    low, high = (paired["lo"], paired["hi"]) if higher else (-paired["hi"], -paired["lo"])
                    paired_rows.append(
                        {
                            "dataset": args.dataset,
                            "seed": seed,
                            "fraction": fraction,
                            "comparison": f"full_vs_{comparator}",
                            "comparator": comparator,
                            "metric": metric_name,
                            "improvement_full_over_comparator": point,
                            "improvement_ci_low": low,
                            "improvement_ci_high": high,
                            "n_boot_valid": paired["n_boot_valid"],
                            "inference_scope": "pointwise_percentile_ci_effect_size_only",
                            "null_test": "not_run",
                            "interpretation": (
                                "full_nominal_95ci_better"
                                if low > 0
                                else "comparator_nominal_95ci_better" if high < 0 else "inconclusive"
                            ),
                        }
                    )
            print(
                f"{split_key}: train_groups={len(train_groups)} test_groups={len(np.unique(groups_test))} "
                + " ".join(f"{model}:F1={point_metrics(y_test, prob, args.threshold, args.n_bins)['f1_macro']:.4f}" for model, prob in predictions_by_model.items()),
                flush=True,
            )

    primary_rows: list[dict[str, Any]] = []
    primary_bootstrap: dict[str, Any] = {}
    if args.dataset == "ptbxl":
        if primary_y_true is None or primary_groups is None:
            raise RuntimeError("PTB-XL primary endpoint predictions were not collected")
        expected_seed_set = set(seeds)
        incomplete = {
            model: sorted(expected_seed_set - set(model_predictions))
            for model, model_predictions in primary_predictions.items()
            if set(model_predictions) != expected_seed_set
        }
        if incomplete:
            raise RuntimeError(f"Primary endpoint seed grid is incomplete: {incomplete}")
        primary_rows, primary_bootstrap = primary_endpoint_rows(
            dataset=args.dataset,
            y_true=primary_y_true,
            groups=primary_groups,
            predictions=primary_predictions,
            zero_probabilities=primary_zero_probabilities,
            threshold=args.threshold,
            n_bins=args.n_bins,
            n_boot=args.n_boot,
            primary_fraction=args.primary_fraction,
        )

    learning_curve_rows: list[dict[str, Any]] = []
    learning_curve_bootstrap: dict[str, Any] = {}
    if args.dataset == "ptbxl":
        if curve_y_true is None or curve_groups is None:
            raise RuntimeError("PTB-XL learning-curve predictions were not collected")
        expected_seed_set = set(seeds)
        for fraction in fractions:
            incomplete = {
                model: sorted(expected_seed_set - set(model_predictions))
                for model, model_predictions in curve_predictions[fraction].items()
                if set(model_predictions) != expected_seed_set
            }
            if incomplete:
                raise RuntimeError(
                    f"Learning-curve seed grid is incomplete at fraction={fraction}: {incomplete}"
                )
            fraction_rows, fraction_bootstrap = primary_endpoint_rows(
                dataset=args.dataset,
                y_true=curve_y_true,
                groups=curve_groups,
                predictions=curve_predictions[fraction],
                zero_probabilities=curve_zero_probabilities,
                threshold=args.threshold,
                n_bins=args.n_bins,
                n_boot=args.n_boot,
                primary_fraction=fraction,
                bootstrap_seed=20260712 + int(round(fraction * 10000)),
            )
            for row in fraction_rows:
                row["fraction"] = fraction
                row["value_mean_across_seeds"] = row.pop("primary_value_mean_across_seeds", None)
                row["value_ci_low"] = row.pop("primary_value_ci_low", None)
                row["value_ci_high"] = row.pop("primary_value_ci_high", None)
                row["improvement_over_zero"] = row.pop("improvement_primary_over_zero", None)
            learning_curve_rows.extend(fraction_rows)
            learning_curve_bootstrap[f"fraction_{fraction:g}"] = fraction_bootstrap

    save_npz_compressed_atomic(
        paths["splits"],
        protocol=np.asarray(PROTOCOL),
        dataset=np.asarray(args.dataset),
        **split_arrays,
    )
    save_csv(paths["summary"], rows)
    save_csv(paths["table"], rows)
    save_csv(paths["paired"], paired_rows)
    save_csv(paths["primary"], primary_rows)
    save_csv(paths["learning_curve"], learning_curve_rows)
    write_learning_curve_figure(learning_curve_rows, paths["learning_curve_figure"])
    save_csv(paths["coefficients"], coefficient_rows)
    save_json(
        paths["bootstrap"],
        {
            "status": "complete",
            "protocol": PROTOCOL,
            "dataset": args.dataset,
            "n_boot": args.n_boot,
            "bootstrap_unit": "patient/source-record group",
            "canonical_group_contract": {
                "group_contract_sha256": canonical["group_contract_sha256"],
                "group_sidecar_sha256": canonical["group_sidecar_sha256"],
                "bootstrap_unit": canonical["bootstrap_unit"],
            },
            "inference_scope": "pointwise_percentile_ci_effect_size_only",
            "null_test": "not_run",
            "multiplicity_adjustment": "not_applicable_no_null_test",
            "items": bootstrap,
            "primary_endpoint": primary_bootstrap,
            "learning_curve": learning_curve_bootstrap,
        },
    )
    outputs = list(paths.values())[:-1]
    save_json(
        paths["manifest"],
        {
            "status": "complete_true_classifier_head_adaptation",
            "created_utc": now_utc(),
            "git_commit": git_commit(),
            "protocol": PROTOCOL,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "canonical_contract": canonical,
            "analysis_lock": analysis_lock,
            "dataset": args.dataset,
            "models": models,
            "adaptation_kind": "new_linear_classifier_parameters_on_frozen_encoder_representations",
            "encoder_weights_updated": False,
            "classifier_head_weights_updated": True,
            "head_protocol": {
                "five_fold_specific_heads": True,
                "probability_ensemble": "arithmetic_mean",
                "standardization": "adaptation_train_only_per_fold",
                "solver": "lbfgs_logistic_regression",
                "class_weight": "balanced",
                "C": args.head_c,
                "max_iter": args.max_iter,
                "hyperparameter_selection": "none_fixed_before_evaluation",
            },
            "fractions": fractions,
            "primary_fraction": args.primary_fraction,
            "primary_fraction_policy": "fixed_by_post_initial_review_analysis_lock_before_current_rerun",
            "primary_endpoint_inference": {
                "point_estimate": "mean_metric_across_analysis_locked_adaptation_seeds",
                "uncertainty": "shared_patient_group_bootstrap_applied_to_all_seeds_and_models",
                "bootstrap_seed": 20260712,
                "n_boot": args.n_boot,
                "inference_scope": "pointwise_percentile_ci_effect_size_only",
                "null_test": "not_run",
            },
            "learning_curve_inference": {
                "fractions": fractions,
                "point_estimate": "mean_metric_across_analysis_locked_adaptation_seeds",
                "uncertainty": "shared_patient_group_bootstrap_within_each_fraction",
                "uncertainty_scope": (
                    "Intervals condition on the five analysis-locked adaptation seeds and fixed fitted "
                    "heads; they do not estimate encoder-retraining or adaptation-seed population variance."
                ),
                "claim_boundary": (
                    "Within-model adaptation trajectory; absolute paired model comparisons remain "
                    "separate and this curve does not establish ECG-RAMBA superiority."
                ),
            },
            "fraction_unit": "independent_target_groups_from_adaptation_pool",
            "fraction_sampling": "nested_seeded_label_independent_group_prefix",
            "seeds": seeds,
            "split_audits": split_audits,
            "cache_contract": {
                "schema_version": 2,
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "canonical_group_contract_sha256": canonical["group_contract_sha256"],
                "canonical_group_sidecar_sha256": canonical["group_sidecar_sha256"],
                "prediction_cache_dir": str(cache_dir),
                "metric_cache_dir": str(metric_cache_dir),
                "atomic_npz_writes": True,
                "per_metric_bootstrap_cache": True,
            },
            "zero_group_overlap_all_splits": True,
            "external_evidence_gates": evidence_gate_contract,
            "safe_wording": (
                "Few-shot linear-head adaptation on nested fractions of independent target groups using mean-pooled "
                "pre-classifier record embeddings; this updates new classifier parameters but does not fine-tune encoders."
            ),
            "inputs": {
                model: {
                    "test_prediction_sha256": data["test_pred"]["sha256"],
                    "test_embedding_sha256": data["test_emb"]["sha256"],
                    "test_embedding_manifest_sha256": data["test_emb"]["manifest_sha256"],
                    "adaptation_prediction_sha256": data["adapt_pred"]["sha256"],
                    "adaptation_embedding_sha256": data["adapt_emb"]["sha256"],
                    "adaptation_embedding_manifest_sha256": data["adapt_emb"]["manifest_sha256"],
                    "test_group_assignment_sha256": data["test_pred"][
                        "group_assignment_sha256"
                    ],
                    "adaptation_group_assignment_sha256": data["adapt_pred"][
                        "group_assignment_sha256"
                    ],
                }
                for model, data in model_data.items()
            },
            "outputs": [
                {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                for path in outputs
            ],
        },
    )
    print(json.dumps({"status": True, "rows": len(rows), "paired_rows": len(paired_rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
