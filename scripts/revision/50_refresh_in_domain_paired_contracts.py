"""Refresh stale in-domain paired artifacts without retraining comparators.

The external learned-comparator runner is intentionally strict: its in-domain
paired artifact must reference the exact canonical OOF and freeze manifest.
When a metadata-only freeze refresh changes only the freeze SHA, rerunning the
CPU paired comparison is sufficient. This helper detects that case, invokes
the existing paired runners, and verifies their complete output contracts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    MANIFEST_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    TABLE_DIR,
    save_json,
    sha256_file,
)


IN_DOMAIN_PAIRED_REFRESH_CAPABILITY = "exact_oof_freeze_paired_refresh_v1"
IN_DOMAIN_PAIRED_REFRESH_SCHEMA_VERSION = 1

MODEL_CONFIG = {
    "resnet": {
        "runner": "15_paired_full_vs_resnet.py",
        "prediction": PREDICTION_DIR / "resnet1d_cnn_oof_predictions.npz",
        "summary": METRIC_DIR / "resnet1d_cnn_baseline_summary.json",
        "baseline_manifest": MANIFEST_DIR / "resnet1d_cnn_baseline_manifest.json",
        "paired": METRIC_DIR / "paired_full_vs_resnet_comparison.json",
        "table": TABLE_DIR / "table_paired_full_vs_resnet.csv",
        "samples": METRIC_DIR / "paired_full_vs_resnet_bootstrap_samples.csv",
        "paired_manifest": MANIFEST_DIR / "paired_full_vs_resnet_manifest.json",
    },
    "raw_mamba": {
        "runner": "17_paired_full_vs_raw_mamba.py",
        "prediction": PREDICTION_DIR / "raw_mamba_oof_predictions.npz",
        "summary": METRIC_DIR / "raw_mamba_baseline_summary.json",
        "baseline_manifest": MANIFEST_DIR / "raw_mamba_baseline_manifest.json",
        "paired": METRIC_DIR / "paired_full_vs_raw_mamba_comparison.json",
        "table": TABLE_DIR / "table_paired_full_vs_raw_mamba.csv",
        "samples": METRIC_DIR / "paired_full_vs_raw_mamba_bootstrap_samples.csv",
        "paired_manifest": MANIFEST_DIR / "paired_full_vs_raw_mamba_manifest.json",
    },
    "transformer": {
        "runner": "25_paired_full_vs_transformer.py",
        "prediction": PREDICTION_DIR / "transformer_ecg_oof_predictions.npz",
        "summary": METRIC_DIR / "transformer_ecg_baseline_summary.json",
        "baseline_manifest": MANIFEST_DIR / "transformer_ecg_baseline_manifest.json",
        "paired": METRIC_DIR / "paired_full_vs_transformer_comparison.json",
        "table": TABLE_DIR / "table_paired_full_vs_transformer.csv",
        "samples": METRIC_DIR / "paired_full_vs_transformer_bootstrap_samples.csv",
        "paired_manifest": MANIFEST_DIR / "paired_full_vs_transformer_manifest.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="resnet,raw_mamba,transformer")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--out-status",
        type=Path,
        default=METRIC_DIR / "in_domain_paired_contract_refresh.json",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def canonical_contract() -> dict[str, str]:
    oof = PREDICTION_DIR / "oof_final_ema_predictions.npz"
    freeze = MANIFEST_DIR / "oof_final_ema_freeze_manifest.json"
    if not oof.is_file() or not freeze.is_file():
        raise FileNotFoundError("Canonical OOF/freeze artifacts are missing")
    payload = read_json(freeze)
    if payload.get("status") != "frozen" or payload.get("manuscript_ready") is not True:
        raise RuntimeError("Canonical freeze is not frozen/manuscript_ready")
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
        raise RuntimeError(f"Canonical freeze OOF SHA mismatch: {expected} != {oof_sha}")
    return {"oof_sha256": oof_sha, "freeze_sha256": sha256_file(freeze)}


def prerequisites(config: dict[str, Any]) -> list[Path]:
    return [
        PREDICTION_DIR / "oof_final_ema_predictions.npz",
        MANIFEST_DIR / "oof_final_ema_freeze_manifest.json",
        config["prediction"],
        config["summary"],
        config["baseline_manifest"],
        PROJECT_ROOT / "scripts" / "revision" / config["runner"],
    ]


def validate_paired(
    model: str,
    config: dict[str, Any],
    canonical: dict[str, str],
    n_boot: int,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    for key in ("paired", "table", "samples", "paired_manifest"):
        path = resolve(config[key])
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"missing_or_empty:{key}")
    if issues:
        return False, issues
    try:
        payload = read_json(config["paired"])
        inputs = payload.get("inputs") or {}
        if (inputs.get("full_predictions") or {}).get("sha256") != canonical["oof_sha256"]:
            issues.append("canonical_oof_sha256")
        if (inputs.get("freeze_manifest") or {}).get("sha256") != canonical["freeze_sha256"]:
            issues.append("canonical_freeze_sha256")
        metrics = payload.get("metrics") or {}
        expected_metrics = {"pr_auc_macro", "roc_auc_macro", "f1_macro", "brier_macro", "ece_macro"}
        if not expected_metrics.issubset(metrics):
            issues.append("metric_coverage")
        for metric in expected_metrics & set(metrics):
            if int((metrics[metric] or {}).get("n_boot_valid", -1)) != int(n_boot):
                issues.append(f"{metric}:n_boot_valid")
        manifest = read_json(config["paired_manifest"])
        if str(manifest.get("comparison") or "").strip() == "":
            issues.append("paired_manifest_comparison")
        input_sha = manifest.get("input_sha256") or {}
        if input_sha.get("full_predictions") != canonical["oof_sha256"]:
            issues.append("paired_manifest_oof_sha256")
        if input_sha.get("freeze_manifest") != canonical["freeze_sha256"]:
            issues.append("paired_manifest_freeze_sha256")
        if int((manifest.get("paired_bootstrap") or {}).get("n_boot", -1)) != int(n_boot):
            issues.append("paired_manifest_n_boot")
        artifact_sha = manifest.get("artifact_sha256") or {}
        for artifact_key, config_key in (
            ("json", "paired"),
            ("table", "table"),
            ("bootstrap_samples", "samples"),
        ):
            if artifact_sha.get(artifact_key) != sha256_file(config[config_key]):
                issues.append(f"paired_manifest_{artifact_key}_sha256")
    except Exception as exc:
        issues.append(f"unreadable:{type(exc).__name__}:{exc}")
    return not issues, issues


def run_pair(model: str, config: dict[str, Any], n_boot: int) -> None:
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "revision" / config["runner"]),
        "--n-boot",
        str(n_boot),
        "--require-manuscript-ready",
    ]
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = sorted(set(models) - set(MODEL_CONFIG))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    canonical = canonical_contract()
    rows: list[dict[str, Any]] = []
    print("=" * 80, flush=True)
    print("IN-DOMAIN PAIRED CONTRACT REFRESH", flush=True)
    print("=" * 80, flush=True)
    for model in models:
        config = MODEL_CONFIG[model]
        missing = [str(path) for path in prerequisites(config) if not resolve(path).is_file()]
        if missing:
            row = {"model": model, "status": "deferred_missing_prerequisites", "missing": missing}
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)
            if not args.allow_incomplete:
                raise FileNotFoundError("; ".join(missing))
            continue
        current, issues = validate_paired(model, config, canonical, args.n_boot)
        if current:
            action = "reused_current"
        else:
            print(f"{model}: refreshing stale paired artifact; issues={issues}", flush=True)
            run_pair(model, config, args.n_boot)
            current, issues = validate_paired(model, config, canonical, args.n_boot)
            if not current:
                raise RuntimeError(f"{model}: paired refresh did not satisfy the exact contract: {issues}")
            action = "regenerated_cpu"
        rows.append(
            {
                "model": model,
                "status": "complete",
                "action": action,
                "paired_path": str(config["paired"]),
                "paired_sha256": sha256_file(config["paired"]),
                "paired_manifest_path": str(config["paired_manifest"]),
                "paired_manifest_sha256": sha256_file(config["paired_manifest"]),
            }
        )
        print(f"{model}: {action}", flush=True)
    complete = all(row.get("status") == "complete" for row in rows)
    out = resolve(args.out_status)
    save_json(
        out,
        {
            "status": "complete" if complete else "incomplete",
            "created_utc": now_utc(),
            "capability": IN_DOMAIN_PAIRED_REFRESH_CAPABILITY,
            "schema_version": IN_DOMAIN_PAIRED_REFRESH_SCHEMA_VERSION,
            "canonical_contract": canonical,
            "n_boot": args.n_boot,
            "models": rows,
        },
    )
    print(f"Wrote: {out}", flush=True)
    if not complete and not args.allow_incomplete:
        raise RuntimeError("At least one paired comparator contract is incomplete")


if __name__ == "__main__":
    main()
