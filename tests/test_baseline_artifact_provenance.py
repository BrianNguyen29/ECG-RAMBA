from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.revision import baseline_artifact_provenance as provenance


ROOT = Path(__file__).resolve().parents[1]


class BaselineArtifactProvenanceTests(unittest.TestCase):
    def write_package(
        self,
        root: Path,
        *,
        commit: str,
        manifest_runner_sha256: str | None,
        summary_runner_sha256: str | None = None,
    ) -> tuple[Path, Path, dict, dict]:
        summary_path = root / "summary.json"
        manifest_path = root / "manifest.json"
        summary = {
            "git_commit": commit,
            "runner_sha256": summary_runner_sha256,
            "protocol": "synthetic",
        }
        summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
        manifest = {
            "git_commit": commit,
            "runner_sha256": manifest_runner_sha256,
            "artifact_sha256": {"summary": provenance.sha256_file(summary_path)},
        }
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return summary_path, manifest_path, summary, manifest

    def test_current_manifest_authenticates_legacy_summary_without_runner_field(self) -> None:
        producer = ROOT / "scripts" / "revision" / "24_transformer_ecg_baseline.py"
        current_sha = "1" * 64
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            paths = self.write_package(
                Path(temp_dir),
                commit="a" * 40,
                manifest_runner_sha256=current_sha,
            )
            with mock.patch.object(
                provenance,
                "current_authority_source_sha256",
                return_value=current_sha,
            ):
                result = provenance.validate_baseline_producer_provenance(
                    baseline_key="transformer",
                    producer_path=producer,
                    summary_path=paths[0],
                    manifest_path=paths[1],
                    summary=paths[2],
                    manifest=paths[3],
                )
        self.assertEqual(result["status"], "accepted_current_runner")
        self.assertTrue(result["summary_runner_sha256_inherited_from_manifest"])

    def test_reviewed_historical_runner_is_recovered_from_git(self) -> None:
        attestation = provenance.REVIEWED_RUNNER_COMPATIBILITY["resnet"]
        producer = ROOT / "scripts" / "revision" / "14_resnet1d_cnn_baseline.py"
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            paths = self.write_package(
                Path(temp_dir),
                commit=attestation["producer_commit"],
                manifest_runner_sha256=attestation["historical_runner_sha256"],
            )
            with mock.patch.object(
                provenance,
                "current_authority_source_sha256",
                return_value=attestation["compatible_current_runner_sha256"],
            ):
                result = provenance.validate_baseline_producer_provenance(
                    baseline_key="resnet",
                    producer_path=producer,
                    summary_path=paths[0],
                    manifest_path=paths[1],
                    summary=paths[2],
                    manifest=paths[3],
                )
        self.assertEqual(result["status"], "accepted_reviewed_runner_compatibility")
        self.assertEqual(
            result["historical_runner_sha256"],
            attestation["historical_runner_sha256"],
        )
        self.assertIn("does not claim", result["claim_boundary"])

    def test_only_reviewed_minirocket_release_may_omit_manifest_runner(self) -> None:
        attestation = provenance.REVIEWED_RUNNER_COMPATIBILITY["minirocket"]
        producer = ROOT / "scripts" / "revision" / "10_minirocket_only_baseline.py"
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            paths = self.write_package(
                Path(temp_dir),
                commit=attestation["producer_commit"],
                manifest_runner_sha256=None,
            )
            with mock.patch.object(
                provenance,
                "current_authority_source_sha256",
                return_value=attestation["compatible_current_runner_sha256"],
            ):
                result = provenance.validate_baseline_producer_provenance(
                    baseline_key="minirocket",
                    producer_path=producer,
                    summary_path=paths[0],
                    manifest_path=paths[1],
                    summary=paths[2],
                    manifest=paths[3],
                )
        self.assertEqual(result["status"], "accepted_reviewed_runner_compatibility")
        self.assertIsNone(result["observed_runner_sha256"])

    def test_unreviewed_runner_drift_is_rejected(self) -> None:
        producer = ROOT / "scripts" / "revision" / "24_transformer_ecg_baseline.py"
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            paths = self.write_package(
                Path(temp_dir),
                commit="b" * 40,
                manifest_runner_sha256="2" * 64,
            )
            with mock.patch.object(
                provenance,
                "current_authority_source_sha256",
                return_value="3" * 64,
            ):
                with self.assertRaisesRegex(RuntimeError, "no reviewed compatibility"):
                    provenance.validate_baseline_producer_provenance(
                        baseline_key="transformer",
                        producer_path=producer,
                        summary_path=paths[0],
                        manifest_path=paths[1],
                        summary=paths[2],
                        manifest=paths[3],
                    )

    def test_summary_mutation_breaks_manifest_binding(self) -> None:
        producer = ROOT / "scripts" / "revision" / "24_transformer_ecg_baseline.py"
        current_sha = "4" * 64
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            paths = self.write_package(
                Path(temp_dir),
                commit="c" * 40,
                manifest_runner_sha256=current_sha,
            )
            paths[0].write_text("{}", encoding="utf-8")
            with mock.patch.object(
                provenance,
                "current_authority_source_sha256",
                return_value=current_sha,
            ):
                with self.assertRaisesRegex(RuntimeError, "not hash-bound"):
                    provenance.validate_baseline_producer_provenance(
                        baseline_key="transformer",
                        producer_path=producer,
                        summary_path=paths[0],
                        manifest_path=paths[1],
                        summary=paths[2],
                        manifest=paths[3],
                    )


if __name__ == "__main__":
    unittest.main()
