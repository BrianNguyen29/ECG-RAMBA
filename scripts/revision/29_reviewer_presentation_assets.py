"""Build reviewer-facing tables/figures from validated revision artifacts.

This runner is intentionally CPU-only.  It is also the method-identity gate for
the morphology transform used by the frozen ECG-RAMBA checkpoints.  It never
recomputes model predictions and refuses to combine stale OOF/freeze/calibration
contracts.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVISION_ROOT = PROJECT_ROOT / "reports" / "revision"
PAIR_FILES = {
    "Fixed-transform-only": "paired_full_vs_minirocket_comparison.json",
    "ResNet1D/CNN": "paired_full_vs_resnet_comparison.json",
    "Raw Mamba": "paired_full_vs_raw_mamba_comparison.json",
    "Transformer ECG": "paired_full_vs_transformer_comparison.json",
    "Frozen-transform MLP head": "paired_full_vs_hybrid_morphology_comparison.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-root", type=Path, default=DEFAULT_REVISION_ROOT)
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--expected-checkpoint-kind", default="final_ema")
    parser.add_argument("--expected-records", type=int, default=44186)
    parser.add_argument("--expected-classes", type=int, default=27)
    parser.add_argument("--skip-figure", action="store_true", help="Validate/write tables without matplotlib output.")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with tmp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def tex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
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


def write_tex_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        write_text_atomic(path, "% No rows available.\n")
        return
    columns = list(rows[0])
    lines = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\toprule"]
    lines.append(" & ".join(tex_escape(column) for column in columns) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(tex_escape(row.get(column, "")) for column in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_text_atomic(path, "\n".join(lines) + "\n")


def artifact_row(label: str, path: Path, required: bool) -> dict[str, Any]:
    exists = path.exists() and path.is_file() and path.stat().st_size > 0
    return {
        "label": label,
        "path": str(path),
        "required": bool(required),
        "exists": bool(exists),
        "size_bytes": int(path.stat().st_size) if exists else None,
        "sha256": sha256_file(path) if exists else None,
    }


def freeze_prediction_sha(freeze: dict[str, Any], prediction_path: Path) -> str | None:
    candidates = {
        prediction_path.as_posix(),
        f"reports/revision/predictions/{prediction_path.name}",
        f"predictions/{prediction_path.name}",
    }
    for row in freeze.get("artifacts", []):
        if str(row.get("path", "")).replace("\\", "/") in candidates:
            return row.get("sha256")
    return None


def validate_primary_contract(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    missing = [name for name, path in paths.items() if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing reviewer-presentation inputs: " + ", ".join(missing))

    freeze = read_json(paths["freeze"])
    calibration = read_json(paths["calibration"])
    oof_sha = sha256_file(paths["oof"])
    freeze_sha = sha256_file(paths["freeze"])
    expected_oof_sha = freeze_prediction_sha(freeze, paths["oof"])
    errors: list[str] = []
    if freeze.get("status") != "frozen" or freeze.get("manuscript_ready") is not True:
        errors.append("freeze manifest is not frozen/manuscript_ready")
    if freeze.get("checkpoint_kind") != args.expected_checkpoint_kind:
        errors.append(
            f"checkpoint_kind={freeze.get('checkpoint_kind')} expected={args.expected_checkpoint_kind}"
        )
    if int(freeze.get("validated_records", -1)) != args.expected_records:
        errors.append(f"validated_records={freeze.get('validated_records')} expected={args.expected_records}")
    if int(freeze.get("n_classes", -1)) != args.expected_classes:
        errors.append(f"n_classes={freeze.get('n_classes')} expected={args.expected_classes}")
    if expected_oof_sha != oof_sha:
        errors.append(f"freeze OOF SHA mismatch: {expected_oof_sha} != {oof_sha}")
    if calibration.get("predictions_sha256") != oof_sha:
        errors.append("calibration predictions_sha256 does not match canonical OOF")
    if calibration.get("freeze_manifest_sha256") != freeze_sha:
        errors.append("calibration freeze_manifest_sha256 does not match canonical freeze")
    if int(calibration.get("n_boot", -1)) < 1000:
        errors.append("calibration bootstrap uses fewer than 1000 resamples")
    bootstrap_contract = calibration.get("bootstrap") or {}
    if bootstrap_contract.get("unit") != "chapman_record_subject":
        errors.append("calibration bootstrap unit is not declared as chapman_record_subject")
    if bootstrap_contract.get("independence_contract") != "one_chapman_record_per_subject":
        errors.append("calibration does not declare the one-record-per-subject independence contract")
    if errors:
        raise RuntimeError("Primary evidence contract failed: " + "; ".join(errors))
    return {
        "oof_sha256": oof_sha,
        "freeze_sha256": freeze_sha,
        "dataset_record_order_fingerprint": freeze.get("dataset_record_order_fingerprint"),
        "checkpoint_kind": freeze.get("checkpoint_kind"),
        "validated_records": int(freeze.get("validated_records")),
        "n_classes": int(freeze.get("n_classes")),
    }


def morphology_contract(source_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = source_root / "src" / "features.py"
    legacy = source_root / "notebooks" / "archive" / "01_exploratory_legacy.ipynb"
    text = source.read_text(encoding="utf-8")
    required_tokens = {
        "fixed_seed_generator": 'manual_seed(seed)',
        "random_ternary_weights": 'torch.randint(-1, 2',
        "gaussian_random_biases": 'torch.randn(kernels_per_dilation',
        "max_statistic": 'maxv, _ = out.max(dim=-1)',
        "ppv_statistic": '(out > bias.view(1, -1, 1)).float().mean(dim=-1)',
        "two_features_per_kernel": 'self.num_features = num_kernels * 2',
        "twenty_thousand_output_contract": 'expected_shape = (len(X), 20000)',
    }
    missing = [name for name, token in required_tokens.items() if token not in text]
    if missing:
        raise RuntimeError("Morphology transform source no longer matches the evaluated contract: " + ", ".join(missing))
    legacy_text = legacy.read_text(encoding="utf-8") if legacy.exists() else ""
    legacy_match = all(
        token in legacy_text
        for token in ["class MiniRocketNative", "num_kernels: int = 10000", "self.num_features = num_kernels * 2"]
    )
    rows = [
        {"field": "evaluated_transform_name", "value": "fixed_seed_rocket_family_random_convolution_max_ppv"},
        {"field": "canonical_minirocket", "value": False},
        {"field": "kernel_length", "value": 9},
        {"field": "requested_kernel_count", "value": 10000},
        {"field": "weight_distribution", "value": "seeded randint{-1,0,1}"},
        {"field": "bias_generation", "value": "seeded Gaussian std=0.1; not train-quantile fitted"},
        {"field": "pooling_statistics", "value": "MAX+PPV"},
        {"field": "raw_output_dimension", "value": 20000},
        {"field": "pca_output_dimension", "value": 3072},
        {"field": "legacy_training_source_matches", "value": legacy_match},
    ]
    payload = {
        "status": "complete" if legacy_match else "blocked_legacy_source_not_verified",
        "manuscript_ready": bool(legacy_match),
        "evaluated_transform_name": rows[0]["value"],
        "safe_wording": (
            "The evaluated morphology branch is a fixed-seed ROCKET-family random-convolution "
            "transform with MAX and PPV statistics; it is not canonical MiniRocket."
        ),
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "legacy_training_source": {
            "path": str(legacy),
            "exists": legacy.exists(),
            "sha256": sha256_file(legacy) if legacy.exists() else None,
            "contract_tokens_match": legacy_match,
        },
        "contract": {row["field"]: row["value"] for row in rows},
    }
    return rows, payload


def calibration_rows(calibration: dict[str, Any]) -> list[dict[str, Any]]:
    point_sources = {
        "macro_pr_auc": ("metrics", "pr_auc_macro"),
        "macro_roc_auc": ("metrics", "roc_auc_macro"),
        "f1_macro": ("metrics", "f1_macro"),
        "brier_macro": ("calibration", "brier_macro"),
        "ece_macro": ("calibration", "ece_macro"),
    }
    rows = []
    bootstrap_contract = calibration.get("bootstrap") or {}
    bootstrap_unit = bootstrap_contract.get("unit", "undeclared")
    for metric, (section, point_key) in point_sources.items():
        ci = calibration.get("bootstrap_ci", {}).get(metric, {})
        rows.append(
            {
                "metric": metric,
                "point_estimate": calibration.get(section, {}).get(point_key),
                "ci_low": ci.get("lo"),
                "ci_high": ci.get("hi"),
                "bootstrap_mean": ci.get("mean"),
                "n_boot_valid": ci.get("n_boot_valid"),
                "bootstrap_unit": bootstrap_unit,
                "threshold": calibration.get("threshold"),
                "n_bins": calibration.get("n_bins"),
            }
        )
    return rows


def paired_rows(metric_dir: Path, contract: dict[str, Any], strict: bool) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for label, filename in PAIR_FILES.items():
        path = metric_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            skipped.append(label)
            continue
        payload = read_json(path)
        full_sha = ((payload.get("inputs") or {}).get("full_predictions") or {}).get("sha256")
        freeze_sha = ((payload.get("inputs") or {}).get("freeze_manifest") or {}).get("sha256")
        if full_sha != contract["oof_sha256"] or freeze_sha != contract["freeze_sha256"]:
            message = f"{label} paired artifact is stale for the canonical OOF/freeze contract"
            if strict and label in {"Fixed-transform-only", "ResNet1D/CNN", "Raw Mamba"}:
                raise RuntimeError(message)
            skipped.append(message)
            continue
        for metric_name, metric in (payload.get("metrics") or {}).items():
            rows.append(
                {
                    "comparator": label,
                    "metric": metric_name,
                    "full_value": metric.get("full_value"),
                    "comparator_value": metric.get("comparator_value"),
                    "improvement_full_over_comparator": metric.get("improvement_full_over_comparator"),
                    "improvement_ci_low": metric.get("improvement_ci_low"),
                    "improvement_ci_high": metric.get("improvement_ci_high"),
                    "holm_p_value_two_sided": metric.get("holm_p_value_two_sided"),
                    "n_boot_valid": metric.get("n_boot_valid"),
                    "interpretation": metric.get("interpretation"),
                    "safe_wording": metric.get("safe_wording"),
                }
            )
    return rows, skipped


def pca_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in payload.get("fold_pca", []):
        rows.append(
            {
                "fold": int(item["fold"]),
                "train_records": int(item["train_records"]),
                "raw_transform_outputs": 20000,
                "pca_components": int(item["n_components"]),
                "explained_variance_fraction": float(item["explained_variance"]),
                "explained_variance_percent": round(100.0 * float(item["explained_variance"]), 4),
                "train_index_hash": item.get("train_index_hash"),
                "pca_sha256": item.get("sha256"),
            }
        )
    return sorted(rows, key=lambda row: row["fold"])


def training_rows(source_root: Path, oof_path: Path) -> list[dict[str, Any]]:
    config_path = source_root / "configs" / "config.py"
    module = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))
    config: dict[str, Any] | None = None
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "BASE_CONFIG" for target in node.targets
        ):
            config = ast.literal_eval(node.value)
            break
    if config is None:
        raise RuntimeError(f"Could not parse BASE_CONFIG from {config_path}")
    with np.load(oof_path, allow_pickle=False) as payload:
        config_hash = str(payload["config_hash"].item()) if "config_hash" in payload.files else ""
    keys = [
        "d_model", "n_layers", "hydra_dim", "n_latents", "hrv_dim", "drop_path_rate",
        "fusion_heads", "slice_length", "slice_stride", "max_slices_per_record", "power_mean_q",
        "epochs", "lr_max", "lr_min", "weight_decay", "grad_clip", "loss_type", "asym_start_epoch",
        "asym_gamma_neg", "asym_gamma_pos", "asym_clip", "ema_decay", "default_threshold",
        "cv_strategy", "group_key", "n_folds",
    ]
    rows = [{"field": "evaluation_config_hash", "value": config_hash, "source": str(oof_path)}]
    rows.append({"field": "config_source_sha256", "value": sha256_file(config_path), "source": str(config_path)})
    rows.append(
        {
            "field": "signal_normalization",
            "value": "per-record per-lead z-score (epsilon=1e-8)",
            "source": "src/data_loader.py:normalize_signal",
        }
    )
    for key in keys:
        if key in config:
            rows.append({"field": key, "value": json.dumps(config[key]) if isinstance(config[key], (list, dict)) else config[key], "source": "configs/config.py"})
    return rows


def write_calibration_figure(reliability_csv: Path, class_csv: Path, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reliability = pd.read_csv(reliability_csv)
    classes = pd.read_csv(class_csv)
    classes = classes[classes["evaluated"].astype(str).str.lower().isin(["true", "1"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1, label="Perfect calibration")
    axes[0].plot(
        reliability["confidence"], reliability["empirical_rate"], marker="o", linewidth=1.8,
        color="#006d77", label="Micro reliability",
    )
    axes[0].set(xlabel="Mean predicted probability", ylabel="Empirical positive rate", xlim=(0, 1), ylim=(0, 1))
    axes[0].set_title("All record-class pairs (15 bins)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].hist(classes["ece"].astype(float), bins=min(12, max(5, len(classes) // 2)), color="#e29578", edgecolor="white")
    axes[1].axvline(classes["ece"].astype(float).mean(), color="#9b2226", linestyle="--", linewidth=1.5, label="Class mean")
    axes[1].set(xlabel="Per-class ECE", ylabel="Number of classes")
    axes[1].set_title("Per-class calibration distribution")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.stem}.{os.getpid()}.partial{out_path.suffix}")
    try:
        fig.savefig(tmp_path, dpi=220, bbox_inches="tight")
        os.replace(tmp_path, out_path)
    finally:
        plt.close(fig)
        if tmp_path.exists():
            tmp_path.unlink()


def main() -> None:
    args = parse_args()
    revision = resolve(args.revision_root)
    source_root = resolve(args.source_root)
    paths = {
        "oof": revision / "predictions" / "oof_final_ema_predictions.npz",
        "freeze": revision / "manifests" / "oof_final_ema_freeze_manifest.json",
        "calibration": revision / "metrics" / "calibration_ci_oof_final_ema_predictions.json",
        "reliability_bins": revision / "tables" / "reliability_bins_oof_final_ema_predictions.csv",
        "class_calibration": revision / "tables" / "calibration_by_class_oof_final_ema_predictions.csv",
        "fold_pca": revision / "manifests" / "fold_pca_manifest.json",
        "pooling": revision / "metrics" / "pooling_sensitivity.csv",
    }
    print("=" * 80)
    print("REVIEWER PRESENTATION ASSETS AND METHOD-IDENTITY GATE")
    print("=" * 80)
    print(f"revision_root={revision}")
    input_rows = [artifact_row(name, path, required=True) for name, path in paths.items()]
    contract = validate_primary_contract(args, paths)

    tables = revision / "tables"
    metrics = revision / "metrics"
    manifests = revision / "manifests"
    figures = revision / "figures"
    output_paths: list[Path] = []

    transform_rows, transform_payload = morphology_contract(source_root)
    transform_csv = tables / "table_morphology_transform_contract.csv"
    transform_tex = tables / "table_morphology_transform_contract.tex"
    transform_json = manifests / "morphology_transform_identity_gate.json"
    write_csv(transform_csv, transform_rows)
    write_tex_table(transform_tex, transform_rows)
    write_json(transform_json, {**transform_payload, "created_utc": now_utc(), "canonical_contract": contract})
    output_paths.extend([transform_csv, transform_tex, transform_json])

    calibration = read_json(paths["calibration"])
    cal_rows = calibration_rows(calibration)
    cal_csv = tables / "table_calibration_ci_compact.csv"
    cal_tex = tables / "table_calibration_ci_compact.tex"
    write_csv(cal_csv, cal_rows)
    write_tex_table(cal_tex, cal_rows)
    output_paths.extend([cal_csv, cal_tex])

    pair_rows, skipped_pairs = paired_rows(metrics, contract, args.strict)
    pair_csv = tables / "table_paired_baseline_ci_compact.csv"
    pair_tex = tables / "table_paired_baseline_ci_compact.tex"
    write_csv(pair_csv, pair_rows)
    write_tex_table(pair_tex, pair_rows)
    output_paths.extend([pair_csv, pair_tex])

    pca_payload = read_json(paths["fold_pca"])
    rows_pca = pca_rows(pca_payload)
    if len(rows_pca) != 5:
        raise RuntimeError(f"Fold PCA manifest is incomplete: expected 5 rows, got {len(rows_pca)}")
    pca_csv = tables / "table_fold_pca_provenance.csv"
    pca_tex = tables / "table_fold_pca_provenance.tex"
    write_csv(pca_csv, rows_pca)
    write_tex_table(pca_tex, rows_pca)
    output_paths.extend([pca_csv, pca_tex])

    train_rows = training_rows(source_root, paths["oof"])
    train_csv = tables / "table_training_configuration.csv"
    train_tex = tables / "table_training_configuration.tex"
    write_csv(train_csv, train_rows)
    write_tex_table(train_tex, train_rows)
    output_paths.extend([train_csv, train_tex])

    pooling = pd.read_csv(paths["pooling"])
    pooling_rows = pooling.to_dict(orient="records")
    pooling_csv = tables / "table_pooling_sensitivity_compact.csv"
    pooling_tex = tables / "table_pooling_sensitivity_compact.tex"
    write_csv(pooling_csv, pooling_rows)
    write_tex_table(pooling_tex, pooling_rows)
    output_paths.extend([pooling_csv, pooling_tex])

    if not args.skip_figure:
        calibration_figure = figures / "figure_calibration_audit.png"
        write_calibration_figure(paths["reliability_bins"], paths["class_calibration"], calibration_figure)
        output_paths.append(calibration_figure)

    preflight_csv = tables / "table_reviewer_completion_input_contract.csv"
    preflight_json = manifests / "reviewer_completion_input_contract.json"
    write_csv(preflight_csv, input_rows)
    output_paths.append(preflight_csv)
    manifest = {
        "status": True,
        "created_utc": now_utc(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "canonical_contract": contract,
        "method_identity_status": transform_payload["status"],
        "paired_comparators_included": sorted({row["comparator"] for row in pair_rows}),
        "paired_comparators_skipped": skipped_pairs,
        "calibration_scope": calibration.get("reliability"),
        "bootstrap_contract": calibration.get("bootstrap"),
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in output_paths
        ],
    }
    write_json(preflight_json, manifest)
    print(json.dumps({"status": True, "outputs": len(output_paths) + 1, "skipped_pairs": skipped_pairs}, indent=2))
    print(f"Wrote: {preflight_json}")


if __name__ == "__main__":
    main()
