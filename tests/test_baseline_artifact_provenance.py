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
        include_prediction_digest: bool = True,
    ) -> tuple[Path, Path, Path, dict, dict]:
        summary_path = root / "summary.json"
        manifest_path = root / "manifest.json"
        prediction_path = root / "predictions.npz"
        prediction_path.write_bytes(b"synthetic-predictions")
        summary = {
            "git_commit": commit,
            "runner_sha256": summary_runner_sha256,
            "protocol": "synthetic",
        }
        summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
        artifact_sha256 = {"summary": provenance.sha256_file(summary_path)}
        if include_prediction_digest:
            artifact_sha256["predictions"] = provenance.sha256_file(prediction_path)
        manifest = {
            "git_commit": commit,
            "runner_sha256": manifest_runner_sha256,
            "artifact_sha256": artifact_sha256,
        }
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return summary_path, manifest_path, prediction_path, summary, manifest

    def exact_attestation(
        self,
        baseline_key: str,
        paths: tuple[Path, Path, Path, dict, dict],
    ) -> dict:
        attestation = dict(provenance.REVIEWED_RUNNER_COMPATIBILITY[baseline_key])
        attestation.update(
            {
                "expected_summary_sha256": provenance.sha256_file(paths[0]),
                "expected_manifest_sha256": provenance.sha256_file(paths[1]),
                "expected_prediction_sha256": provenance.sha256_file(paths[2]),
            }
        )
        return attestation

    def validate(
        self,
        *,
        baseline_key: str,
        producer: Path,
        paths: tuple[Path, Path, Path, dict, dict],
    ) -> dict:
        return provenance.validate_baseline_producer_provenance(
            baseline_key=baseline_key,
            producer_path=producer,
            summary_path=paths[0],
            manifest_path=paths[1],
            prediction_path=paths[2],
            summary=paths[3],
            manifest=paths[4],
        )

    def test_current_authority_package_is_accepted(self) -> None:
        producer = ROOT / "scripts" / "revision" / "24_transformer_ecg_baseline.py"
        current_commit = "a" * 40
        current_sha = "1" * 64
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            paths = self.write_package(
                Path(temp_dir),
                commit=current_commit,
                manifest_runner_sha256=current_sha,
            )
            with (
                mock.patch.object(
                    provenance,
                    "current_authority_source_sha256",
                    return_value=current_sha,
                ),
                mock.patch.object(
                    provenance,
                    "current_authority_commit",
                    return_value=current_commit,
                ),
                mock.patch.object(
                    provenance,
                    "current_authority_source_bundle_sha256",
                    return_value="2" * 64,
                ),
            ):
                result = self.validate(
                    baseline_key="transformer",
                    producer=producer,
                    paths=paths,
                )
        self.assertEqual(result["status"], "accepted_current_authority_producer")
        self.assertTrue(result["summary_runner_sha256_inherited_from_manifest"])

    def test_noncurrent_commit_cannot_use_current_runner_shortcut(self) -> None:
        producer = ROOT / "scripts" / "revision" / "24_transformer_ecg_baseline.py"
        current_sha = provenance.REVIEWED_RUNNER_COMPATIBILITY["transformer"][
            "compatible_current_runner_sha256"
        ]
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            paths = self.write_package(
                Path(temp_dir),
                commit="b" * 40,
                manifest_runner_sha256=current_sha,
            )
            with (
                mock.patch.object(
                    provenance,
                    "current_authority_source_sha256",
                    return_value=current_sha,
                ),
                mock.patch.object(
                    provenance,
                    "current_authority_commit",
                    return_value="c" * 40,
                ),
                mock.patch.object(
                    provenance,
                    "current_authority_source_bundle_sha256",
                    return_value=provenance.REVIEWED_RUNNER_COMPATIBILITY[
                        "transformer"
                    ]["compatible_current_source_bundle_sha256"],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "outside the reviewed"):
                    self.validate(
                        baseline_key="transformer",
                        producer=producer,
                        paths=paths,
                    )

    def test_reviewed_historical_package_and_bundle_are_authenticated(self) -> None:
        baseline_key = "resnet"
        original = provenance.REVIEWED_RUNNER_COMPATIBILITY[baseline_key]
        producer = ROOT / "scripts" / "revision" / "14_resnet1d_cnn_baseline.py"
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            paths = self.write_package(
                Path(temp_dir),
                commit=original["producer_commit"],
                manifest_runner_sha256=original["historical_runner_sha256"],
            )
            attestation = self.exact_attestation(baseline_key, paths)
            with (
                mock.patch.dict(
                    provenance.REVIEWED_RUNNER_COMPATIBILITY,
                    {baseline_key: attestation},
                ),
                mock.patch.object(
                    provenance,
                    "current_authority_source_sha256",
                    return_value=attestation["compatible_current_runner_sha256"],
                ),
                mock.patch.object(
                    provenance,
                    "current_authority_commit",
                    return_value="d" * 40,
                ),
                mock.patch.object(
                    provenance,
                    "current_authority_source_bundle_sha256",
                    return_value=attestation[
                        "compatible_current_source_bundle_sha256"
                    ],
                ),
                mock.patch.object(
                    provenance,
                    "source_bundle_sha256",
                    return_value=attestation["historical_source_bundle_sha256"],
                ),
            ):
                result = self.validate(
                    baseline_key=baseline_key,
                    producer=producer,
                    paths=paths,
                )
        self.assertEqual(result["status"], "accepted_reviewed_runner_compatibility")
        self.assertEqual(
            result["historical_source_bundle_sha256"],
            attestation["historical_source_bundle_sha256"],
        )
        self.assertIn("does not claim", result["claim_boundary"])

    def test_only_exact_reviewed_minirocket_package_may_omit_runner(self) -> None:
        baseline_key = "minirocket"
        original = provenance.REVIEWED_RUNNER_COMPATIBILITY[baseline_key]
        producer = ROOT / "scripts" / "revision" / "10_minirocket_only_baseline.py"
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            paths = self.write_package(
                Path(temp_dir),
                commit=original["producer_commit"],
                manifest_runner_sha256=None,
            )
            attestation = self.exact_attestation(baseline_key, paths)
            with (
                mock.patch.dict(
                    provenance.REVIEWED_RUNNER_COMPATIBILITY,
                    {baseline_key: attestation},
                ),
                mock.patch.object(
                    provenance,
                    "current_authority_source_sha256",
                    return_value=attestation["compatible_current_runner_sha256"],
                ),
                mock.patch.object(
                    provenance,
                    "current_authority_commit",
                    return_value="e" * 40,
                ),
                mock.patch.object(
                    provenance,
                    "current_authority_source_bundle_sha256",
                    return_value=attestation[
                        "compatible_current_source_bundle_sha256"
                    ],
                ),
                mock.patch.object(
                    provenance,
                    "source_bundle_sha256",
                    return_value=attestation["historical_source_bundle_sha256"],
                ),
            ):
                result = self.validate(
                    baseline_key=baseline_key,
                    producer=producer,
                    paths=paths,
                )
        self.assertEqual(result["status"], "accepted_reviewed_runner_compatibility")
        self.assertIsNone(result["observed_runner_sha256"])

    def test_missing_prediction_digest_is_rejected(self) -> None:
        producer = ROOT / "scripts" / "revision" / "24_transformer_ecg_baseline.py"
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            paths = self.write_package(
                Path(temp_dir),
                commit="f" * 40,
                manifest_runner_sha256="1" * 64,
                include_prediction_digest=False,
            )
            with (
                mock.patch.object(
                    provenance,
                    "current_authority_source_sha256",
                    return_value="1" * 64,
                ),
                mock.patch.object(
                    provenance,
                    "current_authority_commit",
                    return_value="f" * 40,
                ),
                mock.patch.object(
                    provenance,
                    "current_authority_source_bundle_sha256",
                    return_value="2" * 64,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "prediction SHA"):
                    self.validate(
                        baseline_key="transformer",
                        producer=producer,
                        paths=paths,
                    )

    def test_prediction_mutation_breaks_manifest_binding(self) -> None:
        producer = ROOT / "scripts" / "revision" / "24_transformer_ecg_baseline.py"
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            paths = self.write_package(
                Path(temp_dir),
                commit="1" * 40,
                manifest_runner_sha256="3" * 64,
            )
            paths[2].write_bytes(b"mutated")
            with (
                mock.patch.object(
                    provenance,
                    "current_authority_source_sha256",
                    return_value="3" * 64,
                ),
                mock.patch.object(
                    provenance,
                    "current_authority_commit",
                    return_value="1" * 40,
                ),
                mock.patch.object(
                    provenance,
                    "current_authority_source_bundle_sha256",
                    return_value="4" * 64,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "prediction is not hash-bound"):
                    self.validate(
                        baseline_key="transformer",
                        producer=producer,
                        paths=paths,
                    )

    def test_dependency_bundle_drift_is_rejected(self) -> None:
        baseline_key = "resnet"
        original = provenance.REVIEWED_RUNNER_COMPATIBILITY[baseline_key]
        producer = ROOT / "scripts" / "revision" / "14_resnet1d_cnn_baseline.py"
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            paths = self.write_package(
                Path(temp_dir),
                commit=original["producer_commit"],
                manifest_runner_sha256=original["historical_runner_sha256"],
            )
            attestation = self.exact_attestation(baseline_key, paths)
            with (
                mock.patch.dict(
                    provenance.REVIEWED_RUNNER_COMPATIBILITY,
                    {baseline_key: attestation},
                ),
                mock.patch.object(
                    provenance,
                    "current_authority_source_sha256",
                    return_value=attestation["compatible_current_runner_sha256"],
                ),
                mock.patch.object(
                    provenance,
                    "current_authority_commit",
                    return_value="2" * 40,
                ),
                mock.patch.object(
                    provenance,
                    "current_authority_source_bundle_sha256",
                    return_value="0" * 64,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "source bundle changed"):
                    self.validate(
                        baseline_key=baseline_key,
                        producer=producer,
                        paths=paths,
                    )

    def test_summary_mutation_breaks_manifest_binding(self) -> None:
        producer = ROOT / "scripts" / "revision" / "24_transformer_ecg_baseline.py"
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            paths = self.write_package(
                Path(temp_dir),
                commit="3" * 40,
                manifest_runner_sha256="4" * 64,
            )
            paths[0].write_text("{}", encoding="utf-8")
            with (
                mock.patch.object(
                    provenance,
                    "current_authority_source_sha256",
                    return_value="4" * 64,
                ),
                mock.patch.object(
                    provenance,
                    "current_authority_commit",
                    return_value="3" * 40,
                ),
                mock.patch.object(
                    provenance,
                    "current_authority_source_bundle_sha256",
                    return_value="5" * 64,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "not hash-bound"):
                    self.validate(
                        baseline_key="transformer",
                        producer=producer,
                        paths=paths,
                    )


if __name__ == "__main__":
    unittest.main()
