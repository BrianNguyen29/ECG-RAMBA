import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.revision.common import sha256_file
from scripts.revision.robustness_profile_audit import (
    PROTOCOL,
    profile_paths,
    select_best_profile,
    validate_profile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "revision" / "21_robustness_multicomparator.py"


class RobustnessProfileAuditTests(unittest.TestCase):
    def _write_profile(
        self,
        revision_root: Path,
        profile: str,
        *,
        canonical: dict[str, str],
        n_boot: int,
        stresses: list[str],
        metrics: list[str],
        n_boot_valid: int | None = None,
    ) -> dict[str, Path]:
        paths = profile_paths(revision_root, profile)
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        comparators = ["full", "minirocket", "resnet", "raw_mamba", "transformer"]
        rows = [
            {
                "stress": stress,
                "comparator": comparator,
                "metric": metric,
                "status": "complete",
                "output_profile": profile,
                "n_boot": n_boot,
                "n_boot_valid": n_boot if n_boot_valid is None else n_boot_valid,
                "degradation_adv_ci_low": -0.01,
                "degradation_adv_ci_high": 0.01,
                "stressed_adv_ci_low": -0.02,
                "stressed_adv_ci_high": 0.02,
                "interpretation": "inconclusive_degradation_difference",
            }
            for stress in stresses
            for comparator in comparators[1:]
            for metric in metrics
        ]
        for key in ("summary", "table"):
            with paths[key].open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        runner_sha = sha256_file(RUNNER)
        pairwise = {
            "status": "complete",
            "protocol": PROTOCOL,
            "output_profile": profile,
            "canonical_contract": canonical,
            "runner_sha256": runner_sha,
            "comparators": comparators,
            "stress_tests": stresses,
            "metrics": metrics,
            "n_boot": n_boot,
            "completed_rows": len(rows),
            "blocked_rows": 0,
            "items": {
                f"{row['stress']}/{row['comparator']}/{row['metric']}": row
                for row in rows
            },
        }
        paths["pairwise"].write_text(json.dumps(pairwise), encoding="utf-8")
        pairwise_sha = sha256_file(paths["pairwise"])

        sidecar_paths = {}
        sidecar_hashes = {}
        for comparator in ("resnet", "raw_mamba", "transformer"):
            sidecar = revision_root / "metrics" / (
                f"robustness_full_vs_{comparator}_{profile}_comparison.json"
            )
            comparator_rows = [row for row in rows if row["comparator"] == comparator]
            sidecar.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "protocol": PROTOCOL,
                        "output_profile": profile,
                        "canonical_contract": canonical,
                        "runner_sha256": runner_sha,
                        "source_pairwise_sha256": pairwise_sha,
                        "rows": comparator_rows,
                    }
                ),
                encoding="utf-8",
            )
            sidecar_paths[comparator] = sidecar.relative_to(PROJECT_ROOT).as_posix()
            sidecar_hashes[comparator] = sha256_file(sidecar)

        manifest = {
            **{key: value for key, value in pairwise.items() if key != "items"},
            "outputs": {
                "summary": paths["summary"].relative_to(PROJECT_ROOT).as_posix(),
                "table": paths["table"].relative_to(PROJECT_ROOT).as_posix(),
                "pairwise": paths["pairwise"].relative_to(PROJECT_ROOT).as_posix(),
                "manifest": paths["manifest"].relative_to(PROJECT_ROOT).as_posix(),
                "comparator_sidecars": sidecar_paths,
            },
            "artifact_sha256": {
                "summary": sha256_file(paths["summary"]),
                "table": sha256_file(paths["table"]),
                "pairwise": pairwise_sha,
                "comparator_sidecars": sidecar_hashes,
            },
        }
        paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
        return paths

    def test_reviewer_minimal_is_valid_screening_not_final_ci(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmp:
            revision_root = Path(tmp) / "reports" / "revision"
            canonical = {"oof_sha256": "oof", "freeze_sha256": "freeze"}
            self._write_profile(
                revision_root,
                "reviewer_minimal",
                canonical=canonical,
                n_boot=200,
                stresses=["snr5db", "precordial_dropout"],
                metrics=["pr_auc_macro", "roc_auc_macro", "f1_macro"],
            )
            result = validate_profile(
                revision_root,
                "reviewer_minimal",
                canonical_contract=canonical,
                runner_path=RUNNER,
                project_root=PROJECT_ROOT,
            )
            self.assertTrue(result["valid"])
            self.assertEqual(result["evidence_tier"], "screening_subset")
            self.assertFalse(result["metric_specific_ci_ready"])
            self.assertFalse(result["canonical_gate_ready"])

            selected = select_best_profile(
                revision_root,
                canonical_contract=canonical,
                runner_path=RUNNER,
                project_root=PROJECT_ROOT,
            )
            self.assertEqual(selected["selected_profile"], "reviewer_minimal")
            self.assertIn("screening", selected["safe_wording"].lower())

    def test_contract_mismatch_rejects_profile(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmp:
            revision_root = Path(tmp) / "reports" / "revision"
            written_contract = {"oof_sha256": "old", "freeze_sha256": "old"}
            self._write_profile(
                revision_root,
                "core_final",
                canonical=written_contract,
                n_boot=1000,
                stresses=["snr5db", "precordial_dropout"],
                metrics=[
                    "pr_auc_macro",
                    "roc_auc_macro",
                    "f1_macro",
                    "brier_macro",
                    "ece_macro",
                ],
            )
            result = validate_profile(
                revision_root,
                "core_final",
                canonical_contract={"oof_sha256": "new", "freeze_sha256": "new"},
                runner_path=RUNNER,
                project_root=PROJECT_ROOT,
            )
            self.assertFalse(result["valid"])
            self.assertIn("manifest.canonical_contract_mismatch", result["issues"])

    def test_insufficient_valid_bootstraps_rejects_profile(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmp:
            revision_root = Path(tmp) / "reports" / "revision"
            canonical = {"oof_sha256": "oof", "freeze_sha256": "freeze"}
            self._write_profile(
                revision_root,
                "core_final",
                canonical=canonical,
                n_boot=1000,
                n_boot_valid=900,
                stresses=["snr5db"],
                metrics=["pr_auc_macro"],
            )
            result = validate_profile(
                revision_root,
                "core_final",
                canonical_contract=canonical,
                runner_path=RUNNER,
                project_root=PROJECT_ROOT,
            )
            self.assertFalse(result["valid"])
            self.assertTrue(
                any(issue.startswith("summary_invalid_bootstrap_count") for issue in result["issues"])
            )


if __name__ == "__main__":
    unittest.main()
