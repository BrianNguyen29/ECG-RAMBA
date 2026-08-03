import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROBUSTNESS = importlib.import_module("scripts.revision.12_robustness_stress")
COMPARATOR_STRESS = importlib.import_module(
    "scripts.revision.23_generate_comparator_stress_predictions"
)


class RobustnessCacheContractTests(unittest.TestCase):
    def test_disk_backed_chapman_loader_preserves_cleaned_order_without_materializing_x(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "clean.npz"
            signals = np.arange(6 * 12 * ROBUSTNESS.SEQ_LEN, dtype=np.float32).reshape(
                6, 12, ROBUSTNESS.SEQ_LEN
            )
            labels = np.ones((6, len(ROBUSTNESS.CLASSES)), dtype=np.float32)
            raw_amp = np.arange(30, dtype=np.float32).reshape(6, 5)
            subjects = np.asarray([f"record-{index}" for index in range(6)])
            archive_sha = "a" * 64
            np.savez_compressed(
                cache_path,
                X=signals,
                y=labels,
                X_raw_amp=raw_amp,
                subjects=subjects,
                archive_sha256=np.asarray(archive_sha),
                cache_schema_version=np.asarray(3, dtype=np.int16),
                preprocessing_source_sha256=np.asarray("source"),
                preprocessing_config_sha256=np.asarray("config"),
                record_order_fingerprint=np.asarray(
                    ROBUSTNESS.record_order_fingerprint(subjects)
                ),
            )
            info = ROBUSTNESS.inspect_disk_backed_chapman_cache(
                expected_y=labels,
                expected_record_fingerprint=ROBUSTNESS.record_order_fingerprint(subjects),
                expected_archive_sha256=archive_sha,
                explicit_cache=cache_path,
                limit_records=0,
            )
            indexed = ROBUSTNESS.open_disk_backed_chapman_signals(info, root / "mmap")
            self.assertIsInstance(indexed.base, np.memmap)
            self.assertEqual(indexed.shape, signals.shape)
            np.testing.assert_array_equal(indexed[2:4], signals[2:4])
            self.assertEqual(info["cache_schema_version"], 3)
            self.assertEqual(info["preprocessing_source_sha256"], "source")
            ROBUSTNESS.close_memmap(indexed.base)

    def test_legacy_disk_backed_cache_requires_frozen_oof_identity_attestation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "legacy_clean.npz"
            signals = np.zeros((6, 12, ROBUSTNESS.SEQ_LEN), dtype=np.float32)
            labels = np.ones((6, len(ROBUSTNESS.CLASSES)), dtype=np.float32)
            raw_amp = np.zeros((6, 5), dtype=np.float32)
            subjects = np.asarray([f"record-{index}" for index in range(6)])
            record_fp = ROBUSTNESS.record_order_fingerprint(subjects)
            np.savez_compressed(
                cache_path,
                X=signals,
                y=labels,
                X_raw_amp=raw_amp,
                subjects=subjects,
                record_order_fingerprint=np.asarray(record_fp),
            )
            expected = {
                "path": str(cache_path.resolve()),
                "size_bytes": cache_path.stat().st_size,
                "record_order_fingerprint": record_fp,
                "source_config_hash": "config",
                "oof_run_manifest_sha256": "manifest",
            }
            info = ROBUSTNESS.inspect_disk_backed_chapman_cache(
                expected_y=labels,
                expected_record_fingerprint=record_fp,
                expected_archive_sha256="a" * 64,
                explicit_cache=cache_path,
                limit_records=0,
                expected_cache_contract=expected,
            )
            self.assertEqual(info["provenance_mode"], "frozen_oof_legacy_cache_content_bound")
            self.assertEqual(info["legacy_cache_attestation"], expected)

            rejected = dict(expected, size_bytes=cache_path.stat().st_size + 1)
            with self.assertRaises(FileNotFoundError):
                ROBUSTNESS.inspect_disk_backed_chapman_cache(
                    expected_y=labels,
                    expected_record_fingerprint=record_fp,
                    expected_archive_sha256="a" * 64,
                    explicit_cache=cache_path,
                    limit_records=0,
                    expected_cache_contract=rejected,
                )

    def test_disk_backed_perturbations_match_in_memory_reference(self):
        rng = np.random.default_rng(123)
        signals = rng.normal(size=(5, 12, 40)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(ROBUSTNESS, "EXPERIMENTAL_DIR", root / "experimental"):
                for spec in ROBUSTNESS.stress_specs(
                    [
                        "snr20db",
                        "random_3_lead_dropout",
                        "precordial_dropout",
                        "resample_250hz",
                    ],
                    42,
                ):
                    expected, _ = ROBUSTNESS.perturb_signals(signals, spec)
                    observed, _metadata, path = ROBUSTNESS.perturb_signals_disk_backed(
                        signals,
                        spec,
                        out_dir=root / "perturbed",
                        raw_cache_sha256="a" * 64,
                        source_bundle_sha256="b" * 64,
                    )
                    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
                    ROBUSTNESS.remove_local_perturbation(observed, path)

    def test_feature_cache_sidecar_rejects_content_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "features.npz"
            features = np.arange(24, dtype=np.float16).reshape(3, 8)
            np.savez_compressed(
                cache_path,
                X=features,
                stress_name=np.asarray("snr20db"),
                stress_hash=np.asarray("stress"),
                record_order_fingerprint=np.asarray("records"),
            )
            ROBUSTNESS.write_feature_cache_contract(
                cache_path,
                feature_kind="test_features",
                features=features,
                stress_name="snr20db",
                stress_hash="stress",
                record_fp="records",
            )
            self.assertIsNotNone(
                ROBUSTNESS.inspect_feature_cache(
                    cache_path,
                    feature_kind="test_features",
                    expected_shape=features.shape,
                    expected_dtype=np.dtype(np.float16),
                    stress_name="snr20db",
                    stress_hash="stress",
                    record_fp="records",
                )
            )
            cache_path.write_bytes(cache_path.read_bytes() + b"mutation")
            self.assertIsNone(
                ROBUSTNESS.inspect_feature_cache(
                    cache_path,
                    feature_kind="test_features",
                    expected_shape=features.shape,
                    expected_dtype=np.dtype(np.float16),
                    stress_name="snr20db",
                    stress_hash="stress",
                    record_fp="records",
                )
            )

    def test_features_only_contract_does_not_require_minirocket_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_path = root / "full.npz"
            full_path.write_bytes(b"full-prediction-contract")
            sidecar_path = root / "groups.npz"
            np.savez_compressed(sidecar_path, group_id=np.asarray(["a", "b"]))
            freeze_path = root / "freeze.json"
            relative_full = "reports/revision/predictions/oof_final_ema_predictions.npz"
            freeze_path.write_text(
                json.dumps(
                    {
                        "status": "frozen",
                        "manuscript_ready": True,
                        "checkpoint_kind": "final_ema",
                        "validated_records": 2,
                        "dataset_record_order_fingerprint": "records",
                        "artifacts": [
                            {
                                "path": relative_full,
                                "sha256": ROBUSTNESS.sha256_file(full_path),
                            }
                        ],
                        "group_contract": {
                            "status": "verified",
                            "one_record_per_group": True,
                            "n_records": 2,
                            "n_groups": 2,
                            "bootstrap_unit": "chapman_record_subject",
                            "group_semantics": "one_record_per_subject",
                            "source_archive": {"sha256": "archive"},
                            "sidecar": {
                                "path": str(sidecar_path),
                                "sha256": ROBUSTNESS.sha256_file(sidecar_path),
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            full = {
                "path": full_path,
                "sha256": ROBUSTNESS.sha256_file(full_path),
                "y_true": np.zeros((2, 2), dtype=np.float32),
                "record_id": np.asarray([0, 1], dtype=np.int64),
                "class_names": ["A", "B"],
                "fold_id": np.asarray([1, 2], dtype=np.int16),
            }
            args = SimpleNamespace(
                freeze_manifest=freeze_path,
                expected_checkpoint_kind="final_ema",
                minirocket_manifest=root / "missing_manifest.json",
                minirocket_summary=root / "missing_summary.json",
            )
            with patch.object(ROBUSTNESS, "project_relative", return_value=relative_full):
                contract = ROBUSTNESS.validate_clean_prediction_contract(
                    full,
                    None,
                    args,
                    require_minirocket=False,
                )
            self.assertEqual(contract["freeze_manifest"]["validated_records"], 2)
            self.assertNotIn("minirocket_manifest", contract)

    def test_stress_feature_hash_binds_archive_freeze_and_record_order(self):
        stress = ROBUSTNESS.stress_specs(["snr20db"], 42)[0]
        contract = {
            "freeze_manifest": {
                "sha256": "freeze-a",
                "source_archive_sha256": "archive-a",
            }
        }
        base = ROBUSTNESS.stress_feature_hash(stress, contract, "records-a")
        self.assertNotEqual(
            base,
            ROBUSTNESS.stress_feature_hash(
                stress,
                {
                    "freeze_manifest": {
                        "sha256": "freeze-a",
                        "source_archive_sha256": "archive-b",
                    }
                },
                "records-a",
            ),
        )
        self.assertNotEqual(
            base,
            ROBUSTNESS.stress_feature_hash(stress, contract, "records-b"),
        )

    def test_feature_cache_hash_binds_exact_raw_cache_and_preprocessing(self):
        base = {
            "sha256": "raw-a",
            "cache_schema_version": 3,
            "preprocessing_source_sha256": "source-a",
            "preprocessing_config_sha256": "config-a",
        }
        observed = ROBUSTNESS.feature_cache_hash("stress", base, feature_device="cuda")
        for key, changed in [
            ("sha256", "raw-b"),
            ("cache_schema_version", 4),
            ("preprocessing_source_sha256", "source-b"),
            ("preprocessing_config_sha256", "config-b"),
        ]:
            mutated = dict(base)
            mutated[key] = changed
            self.assertNotEqual(
                observed,
                ROBUSTNESS.feature_cache_hash("stress", mutated, feature_device="cuda"),
            )
        self.assertNotEqual(
            observed,
            ROBUSTNESS.feature_cache_hash("stress", base, feature_device="cpu"),
        )

    def test_inference_only_rejects_missing_feature_cache_before_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            signals = np.zeros((2, 12, 32), dtype=np.float32)
            with self.assertRaisesRegex(FileNotFoundError, "ROCKET-family cache"):
                ROBUSTNESS.generate_minirocket_features(
                    signals,
                    stress_name="snr20db",
                    stress_hash="stress-hash",
                    record_fp="records",
                    batch_size=2,
                    device_name="cpu",
                    save_cache=True,
                    cache_dir=Path(tmp),
                    require_existing=True,
                )

    def test_minirocket_cache_requires_matching_head_and_clean_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mini.npz"
            y = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
            fold_id = np.asarray([1, 2], dtype=np.int16)
            prob = np.asarray([[0.2, 0.8], [0.9, 0.1]], dtype=np.float32)
            metadata = {
                "minirocket_heads_manifest": {
                    "params_hash": "params-a",
                    "clean_prediction_sha256": "clean-a",
                }
            }
            expected_contract_hash = ROBUSTNESS.prediction_contract_hash(
                model_label="MiniRocket-only",
                minirocket_heads_manifest=metadata["minirocket_heads_manifest"],
            )
            metadata["prediction_contract_hash"] = expected_contract_hash
            metadata["source_bundle"] = ROBUSTNESS.source_bundle_contract()
            stress_spec = ROBUSTNESS.stress_specs(["snr20db"], 42)[0]
            np.savez_compressed(
                path,
                y_true=y,
                y_prob=prob,
                fold_id=fold_id,
                protocol=np.asarray(ROBUSTNESS.PROTOCOL),
                stress_name=np.asarray("snr20db"),
                stress_json=np.asarray(json.dumps(stress_spec, sort_keys=True)),
                model_label=np.asarray("MiniRocket-only"),
                metadata_json=np.asarray(json.dumps(metadata)),
            )

            accepted = ROBUSTNESS.load_existing_prediction(
                path,
                y=y,
                fold_id=fold_id,
                expected_stress="snr20db",
                expected_stress_spec=stress_spec,
                expected_model_label="MiniRocket-only",
                expected_contract_hash=expected_contract_hash,
                expected_minirocket_params_hash="params-a",
                expected_minirocket_clean_prediction_sha256="clean-a",
            )
            np.testing.assert_allclose(accepted, prob)

            rejected = ROBUSTNESS.load_existing_prediction(
                path,
                y=y,
                fold_id=fold_id,
                expected_stress="snr20db",
                expected_stress_spec=stress_spec,
                expected_model_label="MiniRocket-only",
                expected_contract_hash=ROBUSTNESS.prediction_contract_hash(
                    model_label="MiniRocket-only",
                    minirocket_heads_manifest={
                        **metadata["minirocket_heads_manifest"],
                        "clean_prediction_sha256": "clean-b",
                    },
                ),
                expected_minirocket_params_hash="params-a",
                expected_minirocket_clean_prediction_sha256="clean-b",
            )
            self.assertIsNone(rejected)

    def test_corrupt_npz_is_rejected_instead_of_crashing_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corrupt.npz"
            path.write_bytes(b"not-a-zip")
            y = np.zeros((2, 2), dtype=np.float32)
            fold_id = np.asarray([1, 2], dtype=np.int16)
            result = ROBUSTNESS.load_existing_prediction(
                path,
                y=y,
                fold_id=fold_id,
                expected_stress="snr20db",
                expected_stress_spec=ROBUSTNESS.stress_specs(["snr20db"], 42)[0],
                expected_model_label="Full ECG-RAMBA",
                expected_contract_hash="contract",
                expected_checkpoint_sha_by_fold={1: "a", 2: "b"},
            )
            self.assertIsNone(result)

    def test_learned_comparator_cache_requires_record_class_and_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resnet.npz"
            y = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
            probability = np.asarray([[0.2, 0.8], [0.9, 0.1]], dtype=np.float32)
            fold_id = np.asarray([1, 2], dtype=np.int16)
            record_id = np.asarray([0, 1], dtype=np.int64)
            class_names = ["A", "B"]
            hashes = ["fold-1", "fold-2"]
            freeze = {
                "oof_predictions_sha256": "oof",
                "freeze_manifest_sha256": "freeze",
            }
            stress_spec = COMPARATOR_STRESS.robust_helpers.stress_specs(["snr20db"], 42)[0]
            source_bundle = COMPARATOR_STRESS.source_bundle_contract()
            np.savez_compressed(
                path,
                y_true=y,
                y_prob=probability,
                fold_id=fold_id,
                record_id=record_id,
                class_names=np.asarray(class_names),
                protocol=np.asarray(COMPARATOR_STRESS.PROTOCOL),
                comparator=np.asarray("resnet"),
                stress_test=np.asarray("snr20db"),
                stress_metadata_json=np.asarray(json.dumps({"spec": stress_spec}, sort_keys=True)),
                aggregation_implementation=np.asarray(
                    COMPARATOR_STRESS.POWER_MEAN_IMPLEMENTATION
                ),
                power_mean_q=np.asarray(float(COMPARATOR_STRESS.CONFIG["power_mean_q"])),
                oof_predictions_sha256=np.asarray("oof"),
                freeze_manifest_sha256=np.asarray("freeze"),
                checkpoint_sha256=np.asarray(hashes),
                raw_cache_sha256=np.asarray("raw-cache"),
                source_bundle_sha256=np.asarray(source_bundle["sha256"]),
                producer_runner_sha256=np.asarray(
                    source_bundle["files"][
                        "scripts/revision/23_generate_comparator_stress_predictions.py"
                    ]
                ),
            )
            self.assertTrue(
                COMPARATOR_STRESS.validate_existing(
                    path,
                    y,
                    fold_id,
                    record_id,
                    class_names,
                    comparator="resnet",
                    stress="snr20db",
                    stress_spec=stress_spec,
                    freeze_contract=freeze,
                    checkpoint_hashes=hashes,
                    source_bundle_sha256=source_bundle["sha256"],
                    raw_cache_sha256="raw-cache",
                )
            )
            self.assertFalse(
                COMPARATOR_STRESS.validate_existing(
                    path,
                    y,
                    fold_id,
                    record_id,
                    class_names,
                    comparator="resnet",
                    stress="snr20db",
                    stress_spec={**stress_spec, "seed": 999},
                    freeze_contract=freeze,
                    checkpoint_hashes=hashes,
                    source_bundle_sha256=source_bundle["sha256"],
                    raw_cache_sha256="raw-cache",
                )
            )
            self.assertFalse(
                COMPARATOR_STRESS.validate_existing(
                    path,
                    y,
                    fold_id,
                    record_id,
                    class_names,
                    comparator="resnet",
                    stress="snr20db",
                    stress_spec=stress_spec,
                    freeze_contract=freeze,
                    checkpoint_hashes=hashes,
                    source_bundle_sha256=source_bundle["sha256"],
                    raw_cache_sha256="different-raw-cache",
                )
            )
            self.assertFalse(
                COMPARATOR_STRESS.validate_existing(
                    path,
                    y,
                    fold_id,
                    record_id[::-1],
                    class_names,
                    comparator="resnet",
                    stress="snr20db",
                    stress_spec=stress_spec,
                    freeze_contract=freeze,
                    checkpoint_hashes=hashes,
                    source_bundle_sha256=source_bundle["sha256"],
                )
            )

    def test_stress_inference_trusts_only_checkpoint_set_bound_by_baseline_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_dir = root / "manifests"
            checkpoint_dir = root / "checkpoints"
            manifest_dir.mkdir()
            checkpoint_dir.mkdir()
            rows = []
            paths = []
            for fold in range(1, 6):
                path = checkpoint_dir / f"fold{fold}.pt"
                path.write_bytes(f"checkpoint-{fold}".encode("ascii"))
                paths.append(path)
                rows.append(
                    {
                        "fold": fold,
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": COMPARATOR_STRESS.sha256_file(path),
                    }
                )
            (manifest_dir / "resnet1d_cnn_baseline_manifest.json").write_text(
                json.dumps(
                    {
                        "protocol": (
                            "resnet1d_cnn_raw_same_folds_power_mean_v2_q3_threshold_0.5"
                        ),
                        "checkpoint_contract": {
                            "status": "complete",
                            "checkpoints": rows,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(COMPARATOR_STRESS, "MANIFEST_DIR", manifest_dir):
                hashes = COMPARATOR_STRESS.validate_checkpoint_set("resnet", paths)
                self.assertEqual(hashes, [row["sha256"] for row in rows])
                paths[2].write_bytes(b"tampered")
                with self.assertRaisesRegex(RuntimeError, "size mismatch|SHA mismatch"):
                    COMPARATOR_STRESS.validate_checkpoint_set("resnet", paths)


if __name__ == "__main__":
    unittest.main()
