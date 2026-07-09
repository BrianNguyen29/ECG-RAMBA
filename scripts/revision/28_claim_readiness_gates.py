"""Claim readiness gates for reviewer-sensitive ECG-RAMBA claims.

This lightweight audit writes explicit blocker/status artifacts for claims that
cannot be inferred from the main frozen OOF evaluation. It is intentionally
conservative: a row is marked complete only when the required downstream
artifacts already exist. Otherwise it records the missing evidence and the
manuscript-safe wording.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    MANIFEST_DIR,
    METRIC_DIR,
    TABLE_DIR,
    ensure_revision_dirs,
    git_commit,
    save_json,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-json",
        type=Path,
        default=METRIC_DIR / "claim_readiness_gates.json",
    )
    parser.add_argument(
        "--out-table",
        type=Path,
        default=TABLE_DIR / "table_claim_readiness_gates.csv",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=MANIFEST_DIR / "claim_readiness_gates_manifest.json",
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path: Path) -> str:
    return resolve(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def present(path: Path) -> bool:
    candidate = resolve(path)
    return candidate.exists() and candidate.stat().st_size > 0


def status_from_required(required: list[Path], complete_status: str, blocked_status: str) -> tuple[str, list[str]]:
    missing = [p.as_posix() for p in required if not present(p)]
    return (complete_status, missing) if not missing else (blocked_status, missing)


def row(
    *,
    claim_id: str,
    claim_area: str,
    status: str,
    manuscript_ready: bool,
    evidence_status: str,
    required_artifacts: list[Path],
    missing_artifacts: list[str] | None,
    safe_wording: str,
    blocker: str,
    next_action: str,
) -> dict:
    existing = [p.as_posix() for p in required_artifacts if present(p)]
    return {
        "claim_id": claim_id,
        "claim_area": claim_area,
        "status": status,
        "manuscript_ready": bool(manuscript_ready),
        "evidence_status": evidence_status,
        "required_artifacts": ";".join(p.as_posix() for p in required_artifacts),
        "existing_artifacts": ";".join(existing),
        "missing_artifacts": ";".join(missing_artifacts or []),
        "safe_wording": safe_wording,
        "blocker": blocker,
        "next_action": next_action,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "claim_id",
        "claim_area",
        "status",
        "manuscript_ready",
        "evidence_status",
        "required_artifacts",
        "existing_artifacts",
        "missing_artifacts",
        "safe_wording",
        "blocker",
        "next_action",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            writer.writerow(item)


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    print("=" * 80, flush=True)
    print("CLAIM READINESS GATES", flush=True)
    print("=" * 80, flush=True)

    transformer_required = [
        METRIC_DIR / "transformer_ecg_baseline_summary.json",
        Path("reports/revision/predictions/transformer_ecg_oof_predictions.npz"),
        MANIFEST_DIR / "transformer_ecg_baseline_manifest.json",
        METRIC_DIR / "paired_full_vs_transformer_comparison.json",
        TABLE_DIR / "table_paired_full_vs_transformer.csv",
        MANIFEST_DIR / "paired_full_vs_transformer_manifest.json",
    ]
    transformer_status, transformer_missing = status_from_required(
        transformer_required,
        "complete_optional_comparator_available",
        "not_run_optional_comparator",
    )

    hybrid_required = [
        METRIC_DIR / "hybrid_morphology_baseline_summary.json",
        Path("reports/revision/predictions/hybrid_morphology_oof_predictions.npz"),
        MANIFEST_DIR / "hybrid_morphology_baseline_manifest.json",
        METRIC_DIR / "paired_full_vs_hybrid_morphology_comparison.json",
        TABLE_DIR / "table_paired_full_vs_hybrid_morphology.csv",
        MANIFEST_DIR / "paired_full_vs_hybrid_morphology_manifest.json",
    ]
    hybrid_status, hybrid_missing = status_from_required(
        hybrid_required,
        "complete_optional_morphology_sensitivity_available",
        "not_run_optional_morphology_sensitivity",
    )

    robustness_required = [
        METRIC_DIR / "robustness_full_vs_resnet_comparison.json",
        METRIC_DIR / "robustness_full_vs_raw_mamba_comparison.json",
        METRIC_DIR / "robustness_multicomparator_pairwise.json",
        METRIC_DIR / "robustness_multicomparator_summary.csv",
        TABLE_DIR / "table_robustness_multicomparator.csv",
        MANIFEST_DIR / "robustness_multicomparator_manifest.json",
    ]
    robustness_status, robustness_missing = status_from_required(
        robustness_required,
        "complete_multicomparator_robustness_available",
        "blocked_missing_learned_comparator_stress_evidence",
    )

    representation_required = [
        METRIC_DIR / "representation_evidence_status.json",
        METRIC_DIR / "representation_probe_summary.json",
        TABLE_DIR / "table_representation_probe.csv",
        TABLE_DIR / "table_representation_cka.csv",
        MANIFEST_DIR / "representation_probe_manifest.json",
    ]
    representation_status, representation_missing = status_from_required(
        representation_required,
        "audit_available_not_mechanistic_proof",
        "blocked_missing_representation_audit",
    )

    rows = [
        row(
            claim_id="transformer_foundation_baseline",
            claim_area="Transformer/foundation ECG comparator",
            status=transformer_status,
            manuscript_ready=transformer_status.startswith("complete"),
            evidence_status="optional_comparator_specific",
            required_artifacts=transformer_required,
            missing_artifacts=transformer_missing,
            safe_wording=(
                "Use only comparator-specific wording if the Transformer ECG baseline and paired bootstrap "
                "artifacts are complete; otherwise state that this optional comparator was not run."
            ),
            blocker="Missing Transformer ECG baseline and/or paired Full-vs-Transformer artifacts."
            if transformer_missing
            else "",
            next_action=(
                "Run scripts/revision/24_transformer_ecg_baseline.py, then "
                "scripts/revision/25_paired_full_vs_transformer.py under the frozen OOF contract."
            ),
        ),
        row(
            claim_id="hybrid_morphology_mlp",
            claim_area="Hybrid/partially learnable MiniRocket morphology",
            status=hybrid_status,
            manuscript_ready=hybrid_status.startswith("complete"),
            evidence_status="optional_mechanism_sensitivity",
            required_artifacts=hybrid_required,
            missing_artifacts=hybrid_missing,
            safe_wording=(
                "Use only as morphology-head sensitivity evidence. Do not claim that deterministic "
                "MiniRocket morphology is causally superior or that the branch is mechanistically isolated."
            ),
            blocker="Missing Hybrid MiniRocket-MLP baseline and/or paired comparison artifacts."
            if hybrid_missing
            else "",
            next_action=(
                "Run scripts/revision/26_hybrid_morphology_baseline.py, then "
                "scripts/revision/27_paired_full_vs_hybrid_morphology.py."
            ),
        ),
        row(
            claim_id="robustness_learned_comparators",
            claim_area="Robustness vs ResNet1D/CNN and Raw Mamba",
            status=robustness_status,
            manuscript_ready=robustness_status.startswith("complete"),
            evidence_status="metric_specific_if_complete",
            required_artifacts=robustness_required,
            missing_artifacts=robustness_missing,
            safe_wording=(
                "Without learned-comparator stress artifacts, robustness claims remain limited to "
                "metric-specific Full-vs-MiniRocket evidence."
            ),
            blocker="Missing ResNet1D/CNN and/or Raw Mamba stress prediction comparisons."
            if robustness_missing
            else "",
            next_action=(
                "Save/reuse ResNet and Raw Mamba checkpoints, run comparator stress predictions, then run "
                "scripts/revision/21_robustness_multicomparator.py."
            ),
        ),
        row(
            claim_id="full_hrv_feature_set",
            claim_area="Full HRV feature set",
            status="blocked_retrain_required",
            manuscript_ready=False,
            evidence_status="not_available_for_current_checkpoint",
            required_artifacts=[],
            missing_artifacts=["true_RMSSD_SDNN_LFHF_checkpoint_training_contract"],
            safe_wording=(
                "Do not claim RMSSD, SDNN, LF/HF, or a complete HRV feature set for the current final-EMA "
                "checkpoint. The current HRV slots are checkpoint-compatible and partly reserved."
            ),
            blocker="Full HRV semantics require a new feature schema and full retraining; they cannot be retrofitted.",
            next_action="Define a true HRV schema, retrain all folds, regenerate OOF/calibration/baselines, and re-freeze evidence.",
        ),
        row(
            claim_id="mechanistic_disentanglement",
            claim_area="Mechanistic morphology-rhythm disentanglement",
            status=representation_status,
            manuscript_ready=representation_status.startswith("audit_available"),
            evidence_status="audit_only",
            required_artifacts=representation_required,
            missing_artifacts=representation_missing,
            safe_wording=(
                "Representation/CKA results may be described only as a conservative branch-embedding audit. "
                "They do not prove morphology-rhythm disentanglement."
            ),
            blocker="" if not representation_missing else "Missing representation probe/CKA artifacts.",
            next_action=(
                "If stronger mechanism evidence is required, add preregistered probes with leakage-safe splits and "
                "keep wording as suggestive unless tests support stronger claims."
            ),
        ),
        row(
            claim_id="clinical_deployment_readiness",
            claim_area="Clinical deployment/safety readiness",
            status="blocked_no_prospective_or_clinical_utility_validation",
            manuscript_ready=False,
            evidence_status="not_available",
            required_artifacts=[],
            missing_artifacts=["prospective_validation", "clinical_threshold_target", "decision_curve_or_utility_analysis"],
            safe_wording=(
                "Do not claim clinical deployment readiness, safety readiness, or prospective clinical utility. "
                "Current evidence is retrospective/model-evaluation evidence only."
            ),
            blocker="No prospective validation, clinical utility analysis, or prespecified deployment threshold target.",
            next_action="Plan a prospective or clinically curated external validation with utility/threshold analysis.",
        ),
        row(
            claim_id="broad_in_domain_global_superiority",
            claim_area="Broad in-domain/global superiority",
            status="contradicted_by_current_fair_baselines",
            manuscript_ready=False,
            evidence_status="contradicted",
            required_artifacts=[
                METRIC_DIR / "paired_full_vs_resnet_comparison.json",
                METRIC_DIR / "paired_full_vs_raw_mamba_comparison.json",
            ],
            missing_artifacts=[],
            safe_wording=(
                "Do not claim SOTA, best in-domain performance, or broad superiority over fair baselines. "
                "Use metric-specific and comparator-specific wording only."
            ),
            blocker="ResNet1D/CNN and Raw Mamba outperform ECG-RAMBA on multiple principal frozen OOF metrics.",
            next_action="Keep the manuscript framed as structured analysis and evidence-bounded tradeoffs, not global superiority.",
        ),
    ]

    out_table = resolve(args.out_table)
    out_json = resolve(args.out_json)
    out_manifest = resolve(args.out_manifest)
    write_csv(out_table, rows)
    payload = {
        "status": True,
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "rows": rows,
        "claim_guidance": {
            "use": "Use this table to keep manuscript and rebuttal wording bounded by completed evidence.",
            "do_not_use": "Do not convert blocked rows into positive claims.",
        },
    }
    save_json(out_json, payload)
    manifest = {
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "artifacts": {
            "json": rel(out_json),
            "table": rel(out_table),
        },
        "artifact_sha256": {
            "json": sha256_file(out_json),
            "table": sha256_file(out_table),
        },
    }
    save_json(out_manifest, manifest)
    print(json.dumps({"status": True, "rows": len(rows), "table": rel(out_table)}, indent=2), flush=True)
    print(f"Wrote: {out_json}", flush=True)
    print(f"Wrote: {out_table}", flush=True)
    print(f"Wrote: {out_manifest}", flush=True)


if __name__ == "__main__":
    main()
