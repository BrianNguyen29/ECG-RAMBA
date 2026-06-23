"""Audit reviewer-facing baseline training pipelines.

This script performs static and artifact-level checks for the MiniRocket-only,
ResNet1D/CNN, and Raw Mamba fair comparator pipelines. It is intentionally
lightweight: it does not train models or import Mamba runtime code.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import METRIC_DIR, TABLE_DIR, save_csv, save_json  # noqa: E402
from src.aggregation import POWER_MEAN_IMPLEMENTATION, aggregate_record_probabilities, power_mean  # noqa: E402


EXPECTED = {
    "minirocket_protocol": "minirocket_raw_standardized_torch_linear_same_folds_threshold_0.5",
    "resnet_protocol": "resnet1d_cnn_raw_same_folds_power_mean_v2_q3_threshold_0.5",
    "raw_mamba_protocol": "raw_mamba_retrained_weighted_bce_same_folds_power_mean_v2_q3_threshold_0.5",
    "oof_sha256": "375e5f7b5b312ae4101de9a4a98ba3484d02c46ea560e80a9f5862763294ea31",
    "record_fingerprint": "965c332ad7bf95ed",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_constant(source: str, name: str) -> str | None:
    match = re.search(rf'^{name}\s*=\s*["\']([^"\']+)["\']', source, flags=re.MULTILINE)
    return match.group(1) if match else None


def _add(rows: list[dict], component: str, check: str, status: str, detail: str) -> None:
    if status not in {"pass", "warn", "fail"}:
        raise ValueError(f"Invalid audit status: {status}")
    rows.append(
        {
            "component": component,
            "check": check,
            "status": status,
            "detail": detail,
        }
    )


def _contains_all(source: str, needles: list[str]) -> tuple[bool, list[str]]:
    missing = [needle for needle in needles if needle not in source]
    return not missing, missing


def _audit_power_mean(rows: list[dict]) -> None:
    probs = np.asarray([[0.2, 0.8], [0.4, 0.6], [0.9, 0.1]], dtype=np.float32)
    mean_q1 = power_mean(probs, q=1.0, axis=0)
    expected_q1 = np.mean(np.clip(probs, 1e-6, 1 - 1e-6), axis=0)
    q3 = power_mean(probs, q=3.0, axis=0)
    if np.allclose(mean_q1, expected_q1, atol=1e-6) and q3.shape == (2,) and np.all((q3 > 0) & (q3 < 1)):
        _add(rows, "aggregation", "power_mean_q1_and_q3", "pass", f"{POWER_MEAN_IMPLEMENTATION} numeric sanity passed")
    else:
        _add(rows, "aggregation", "power_mean_q1_and_q3", "fail", "power_mean sanity check failed")

    slice_prob = np.asarray([[0.2, 0.4], [0.8, 0.6], [0.3, 0.9]], dtype=np.float32)
    slice_record_id = np.asarray([0, 0, 1], dtype=np.int64)
    y_prob, valid, counts = aggregate_record_probabilities(slice_prob, slice_record_id, n_records=2, q=3.0)
    if y_prob.shape == (2, 2) and valid.tolist() == [True, True] and counts.tolist() == [2, 1]:
        _add(rows, "aggregation", "record_aggregation_counts", "pass", "slice counts and validity mask are correct")
    else:
        _add(rows, "aggregation", "record_aggregation_counts", "fail", "record aggregation count/mask mismatch")


def _audit_source(rows: list[dict]) -> dict[str, str]:
    paths = {
        "minirocket": PROJECT_ROOT / "scripts" / "revision" / "10_minirocket_only_baseline.py",
        "resnet": PROJECT_ROOT / "scripts" / "revision" / "14_resnet1d_cnn_baseline.py",
        "raw_mamba": PROJECT_ROOT / "scripts" / "revision" / "16_raw_mamba_baseline.py",
        "paired_raw": PROJECT_ROOT / "scripts" / "revision" / "17_paired_full_vs_raw_mamba.py",
        "notebook04": PROJECT_ROOT / "notebooks" / "04_baselines_and_component_checks.ipynb",
    }
    sources = {}
    for key, path in paths.items():
        if path.exists():
            sources[key] = _read(path)
            _add(rows, key, "source_exists", "pass", str(path.relative_to(PROJECT_ROOT)))
        else:
            sources[key] = ""
            _add(rows, key, "source_exists", "fail", f"Missing {path.relative_to(PROJECT_ROOT)}")

    mini_protocol = _extract_constant(sources["minirocket"], "PROTOCOL")
    _add(
        rows,
        "minirocket",
        "protocol_constant",
        "pass" if mini_protocol == EXPECTED["minirocket_protocol"] else "fail",
        str(mini_protocol),
    )
    ok, missing = _contains_all(
        sources["minirocket"],
        [
            "validate_oof_freeze_contract",
            "dataset_record_order_fingerprint",
            "allow_legacy_shape_cache",
            "compute_train_standardization",
            "BCEWithLogitsLoss",
            "pos_weight",
            "fold_train_standardization",
            "fold_id_out[va_idx]",
        ],
    )
    _add(
        rows,
        "minirocket",
        "fold_safe_contract_markers",
        "pass" if ok else "fail",
        "all required markers present" if ok else "missing: " + ", ".join(missing),
    )
    if "np.clip(pos_weight" in sources["minirocket"]:
        _add(rows, "minirocket", "pos_weight_clipping", "pass", "MiniRocket pos_weight is clipped")
    else:
        _add(
            rows,
            "minirocket",
            "pos_weight_clipping",
            "warn",
            "Torch-linear MiniRocket uses un-clipped fold pos_weight; report its calibration tradeoff explicitly",
        )

    resnet_protocol = _extract_constant(sources["resnet"], "PROTOCOL")
    _add(
        rows,
        "resnet1d_cnn",
        "protocol_constant",
        "pass" if resnet_protocol == EXPECTED["resnet_protocol"] else "fail",
        str(resnet_protocol),
    )
    ok, missing = _contains_all(
        sources["resnet"],
        [
            "load_raw_cache",
            "expected_record_fingerprint",
            "raw-cache labels do not exactly match frozen OOF labels",
            "pos_weight_from_labels(y[tr_idx])",
            "selection_rule",
            "fixed_final_epoch",
            "aggregate_record_probabilities",
            "SLICE_PREDICTION_PATH",
            "slice_count does not match slice artifact",
        ],
    )
    _add(
        rows,
        "resnet1d_cnn",
        "raw_fold_contract_markers",
        "pass" if ok else "fail",
        "all required markers present" if ok else "missing: " + ", ".join(missing),
    )

    raw_protocol = _extract_constant(sources["raw_mamba"], "PROTOCOL")
    _add(
        rows,
        "raw_mamba",
        "protocol_constant",
        "pass" if raw_protocol == EXPECTED["raw_mamba_protocol"] else "fail",
        str(raw_protocol),
    )
    ok, missing = _contains_all(
        sources["raw_mamba"],
        [
            "--bce-pos-weight",
            "resnet_helpers.pos_weight_from_labels(y[tr_idx])",
            "BCE pos_weight enabled",
            "Pstd=",
            "P>=thr=",
            "no_rocket",
            "no_hrv",
            "no_fusion",
            "AsymmetricLossMultiLabel",
            "EMA(model",
            "uses_ecg_ramba_checkpoints",
            "training_from_scratch",
            "slice_count does not match slice artifact",
        ],
    )
    _add(
        rows,
        "raw_mamba",
        "weighted_bce_structural_ablation_markers",
        "pass" if ok else "fail",
        "all required markers present" if ok else "missing: " + ", ".join(missing),
    )
    ok, missing = _contains_all(
        sources["paired_raw"],
        [EXPECTED["raw_mamba_protocol"], "raw_mamba_oof_predictions.npz", "paired_full_vs_raw_mamba"],
    )
    _add(
        rows,
        "raw_mamba",
        "paired_comparison_protocol",
        "pass" if ok else "fail",
        "paired raw comparison expects weighted protocol" if ok else "missing: " + ", ".join(missing),
    )

    ok, missing = _contains_all(
        sources["notebook04"],
        [
            "10_minirocket_only_baseline.py",
            "14_resnet1d_cnn_baseline.py",
            "16_raw_mamba_baseline.py",
            "--bce-pos-weight fold",
            EXPECTED["raw_mamba_protocol"],
        ],
    )
    _add(
        rows,
        "notebook04",
        "runner_commands",
        "pass" if ok else "fail",
        "Notebook 04 calls all baseline runners with current Raw Mamba protocol" if ok else "missing: " + ", ".join(missing),
    )
    return sources


def _artifact_roots() -> list[Path]:
    roots = [PROJECT_ROOT / "reports" / "revision"]
    drive_root = PROJECT_ROOT.parent / "drive" / "revision_artifacts" / "reports" / "revision"
    if drive_root.exists():
        roots.append(drive_root)
    return roots


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_check(rows: list[dict], root: Path, name: str, expected_protocol: str) -> None:
    path = root / "metrics" / f"{name}_baseline_summary.json"
    summary = _load_json(path)
    component = f"artifact:{name}"
    root_label = str(root)
    if summary is None:
        _add(rows, component, "summary_present", "warn", f"missing under {root_label}")
        return
    _add(rows, component, "summary_present", "pass", f"{path}")
    protocol = summary.get("protocol")
    _add(
        rows,
        component,
        "protocol",
        "pass" if protocol == expected_protocol else "fail",
        str(protocol),
    )
    load_info = summary.get("load_info") or {}
    oof_sha = load_info.get("oof_predictions_sha256")
    _add(
        rows,
        component,
        "oof_sha",
        "pass" if oof_sha == EXPECTED["oof_sha256"] else "fail",
        str(oof_sha),
    )
    fingerprint = load_info.get("dataset_record_order_fingerprint")
    _add(
        rows,
        component,
        "record_fingerprint",
        "pass" if fingerprint == EXPECTED["record_fingerprint"] else "fail",
        str(fingerprint),
    )
    metrics = summary.get("metrics") or {}
    calibration = summary.get("calibration") or {}
    metric_detail = {
        "pr_auc_macro": metrics.get("pr_auc_macro"),
        "roc_auc_macro": metrics.get("roc_auc_macro"),
        "f1_macro": metrics.get("f1_macro"),
        "brier_macro": calibration.get("brier_macro"),
        "ece_macro": calibration.get("ece_macro"),
    }
    if all(value is not None and math.isfinite(float(value)) for value in metric_detail.values()):
        _add(rows, component, "finite_metrics", "pass", json.dumps(metric_detail, sort_keys=True))
    else:
        _add(rows, component, "finite_metrics", "fail", json.dumps(metric_detail, sort_keys=True))


def _audit_artifacts(rows: list[dict]) -> None:
    for root in _artifact_roots():
        _add(rows, "artifacts", "root_checked", "pass", str(root))
        _summary_check(rows, root, "minirocket_only", EXPECTED["minirocket_protocol"])
        _summary_check(rows, root, "resnet1d_cnn", EXPECTED["resnet_protocol"])
        _summary_check(rows, root, "raw_mamba", EXPECTED["raw_mamba_protocol"])


def main() -> None:
    rows: list[dict] = []
    _audit_power_mean(rows)
    _audit_source(rows)
    _audit_artifacts(rows)

    status_counts = {status: sum(row["status"] == status for row in rows) for status in ["pass", "warn", "fail"]}
    overall_status = "fail" if status_counts["fail"] else ("warn" if status_counts["warn"] else "pass")
    payload = {
        "created_utc": _now_utc(),
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "overall_status": overall_status,
        "status_counts": status_counts,
        "expected": EXPECTED,
        "checks": rows,
    }

    out_json = METRIC_DIR / "baseline_pipeline_audit.json"
    out_csv = TABLE_DIR / "table_baseline_pipeline_audit.csv"
    save_json(out_json, payload)
    save_csv(out_csv, rows)

    print(json.dumps({"overall_status": overall_status, "status_counts": status_counts}, indent=2))
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")
    if overall_status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
