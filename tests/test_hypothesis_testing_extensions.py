from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (
    HRV36_CHECKPOINT_SEMANTICS,
    checkpoint_compatible_hrv36_contract,
    validate_checkpoint_compatible_hrv36,
)

def load_script(name: str, relative: str):
    path = PROJECT_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class HypothesisTestingExtensionTests(unittest.TestCase):
    def test_final_generator_accepts_authenticated_physiology_blocker_without_tex(self):
        generator = load_script(
            "final_generator_physiology_blocker_test_module",
            "scripts/revision/13_final_evidence_matrix.py",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "summary.json"
            table_path = root / "table.csv"
            contrast_path = root / "contrasts.csv"
            audit_path = root / "audit.csv"
            tex_path = root / "table.tex"
            manifest_path = root / "manifest.json"
            summary_path.write_text("{}", encoding="utf-8")
            table_path.write_text("", encoding="utf-8")
            contrast_path.write_text("", encoding="utf-8")
            audit_path.write_text("", encoding="utf-8")
            manifest_path.write_text("{}", encoding="utf-8")
            artifacts = (summary_path, table_path, contrast_path, audit_path)
            manifest = {
                "status": "blocked_missing_reliable_interval_metadata",
                "outputs": {
                    str(path): generator.sha256_file(path) for path in artifacts
                },
            }
            result = generator.summarize_physiological_probe(
                {
                    "status": "blocked_missing_reliable_interval_metadata",
                    "protocol": "fold_held_out_measured_physiological_interval_probe_v3",
                },
                manifest,
                required_paths=(*artifacts, tex_path, manifest_path),
            )

        self.assertFalse(result["complete"])
        self.assertEqual(result["issues"], [])
        self.assertEqual(
            result["status"], "blocked_missing_reliable_interval_metadata"
        )

    def test_canonical_full_oof_protocol_name_remains_backward_compatible(self):
        prediction_runner = load_script(
            "prediction_protocol_test_module",
            "scripts/revision/01_generate_predictions.py",
        )

        full_record, full_slice = prediction_runner.oof_protocol_names(
            "final_ema", "full", 3.0
        )
        removal_record, removal_slice = prediction_runner.oof_protocol_names(
            "final_ema", "no_morphology", 3.0
        )

        self.assertEqual(
            full_record,
            "fold_final_ema_power_mean_v2_q3_threshold_0.5",
        )
        self.assertEqual(full_slice, "slice_level_fold_final_ema")
        self.assertEqual(
            removal_record,
            "fold_final_ema_no_morphology_power_mean_v2_q3_threshold_0.5",
        )
        self.assertEqual(
            removal_slice,
            "slice_level_fold_final_ema_no_morphology",
        )

    def test_cross_fitted_platt_excludes_evaluation_fold(self):
        calibration = load_script(
            "matched_calibration_test_module",
            "scripts/revision/42_matched_oof_calibration.py",
        )
        folds = np.repeat(np.arange(1, 6, dtype=np.int16), 2)
        y_true = np.asarray([[0], [1]] * 5, dtype=np.float32)
        y_prob = np.linspace(0.1, 0.9, len(y_true), dtype=np.float32).reshape(-1, 1)
        observed_training_lengths = []

        def record_fit(y, prob):
            observed_training_lengths.append((len(y), len(prob)))
            return 0.0, 1.0, "fitted"

        calibration.fit_platt = record_fit
        prediction = calibration.PredictionSet(
            name="full",
            path=Path("unused.npz"),
            sha256="unused",
            y_true=y_true,
            y_prob=y_prob,
            record_id=np.arange(len(y_true)),
            fold_id=folds,
            class_names=np.asarray(["class_a"]),
        )
        calibrated, rows = calibration.cross_fitted_platt(prediction)

        self.assertEqual(calibrated.shape, y_prob.shape)
        self.assertEqual(len(rows), 5)
        self.assertEqual(observed_training_lengths, [(8, 8)] * 5)
        self.assertEqual({row["evaluation_records"] for row in rows}, {2})

    def test_monotone_platt_cannot_reverse_score_order(self):
        calibration = load_script(
            "matched_calibration_monotone_test_module",
            "scripts/revision/42_matched_oof_calibration.py",
        )
        probability = np.linspace(0.01, 0.99, 200)
        labels = (probability < 0.5).astype(np.float64)

        intercept, slope, status = calibration.fit_platt(labels, probability)
        calibrated = calibration.apply_platt(probability, intercept, slope)

        self.assertGreaterEqual(slope, calibration.PLATT_MIN_SLOPE)
        self.assertTrue(status.startswith("fitted_monotone"))
        self.assertTrue(np.all(np.diff(calibrated) >= 0.0))

    def test_paired_model_bootstrap_orients_lower_metrics_toward_full(self):
        calibration = load_script(
            "matched_calibration_orientation_test_module",
            "scripts/revision/42_matched_oof_calibration.py",
        )
        y_true = np.asarray([[0], [1]] * 100, dtype=np.float32)
        full_prob = np.where(y_true > 0, 0.9, 0.1).astype(np.float32)
        comparator_prob = np.full_like(y_true, 0.5)
        spec = calibration.MetricSpec(
            "brier",
            False,
            lambda y, p: float(np.mean((np.asarray(p) - np.asarray(y)) ** 2)),
        )

        result = calibration.paired_model_bootstrap(
            y_true,
            full_prob,
            comparator_prob,
            spec,
            n_boot=100,
            seed=42,
        )

        self.assertGreater(result["improvement_full_over_comparator"], 0)
        self.assertGreater(result["ci_low"], 0)
        self.assertEqual(result["interpretation"], "full_significantly_better")

    def test_structured_ablation_checkpoint_uses_trainer_fold_fingerprint(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        ablation = load_script(
            "structured_ablation_test_module",
            "scripts/revision/43_structured_ablation_5fold.py",
        )
        folds = []
        for fold in range(1, 6):
            val = np.asarray([fold * 2, fold * 2 + 1], dtype=np.int64)
            folds.append({"tr_idx": np.asarray([0, 1], dtype=np.int64), "va_idx": val})

        import hashlib

        def index_hash(values):
            return hashlib.sha256(
                np.ascontiguousarray(values, dtype=np.int64).tobytes()
            ).hexdigest()[:16]

        protocol = {
            "epochs": 20,
            "loss_switch_epoch": 4,
            "bce_reduction": "mean",
            "asymmetric_reduction": "mean",
            "scheduler": "cosine",
            "lr_max": 1e-3,
            "lr_min": 1e-5,
            "ema_decay": 0.999,
            "ema_scope": "trainable_parameters",
            "amp_dtype": "bfloat16",
            "model_selection": "fixed_final_epoch",
        }
        canonical_contracts = {}
        for fold, split in enumerate(folds, start=1):
            canonical_contracts[fold] = {
                "sha256": f"canonical-fold-{fold}",
                "config_hash": "config-v1",
                "dataset_record_order_fingerprint": "records-v1",
                "class_names": ["class_a"],
                "aggregation": {"method": "power_mean", "q": 3.0},
                "training_protocol": protocol,
                "feature_contract": {"hrv36": checkpoint_compatible_hrv36_contract()},
                "pca_explained_variance": 0.975,
                "train_index_hash": index_hash(split["tr_idx"]),
                "val_index_hash": index_hash(split["va_idx"]),
            }

        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            pca_path = model_dir / "fold1_pca.joblib"
            pca_path.write_bytes(b"fold-one-pca-fixture")
            pca_sha256 = hashlib.sha256(pca_path.read_bytes()).hexdigest()
            torch.save(
                {
                    "fold": 1,
                    "epoch": 20,
                    "weights_kind": "ema",
                    "selection_rule": "fixed_final_epoch",
                    "ablation_variant": "no_morphology",
                    "ablation_spec": ablation.STRUCTURED_ABLATION_SPECS["no_morphology"],
                    "architecture_contract": "ecg_ramba_structured_ablation_v1",
                    "training_seed": int(ablation.CONFIG["seeds"][0]) + 1,
                    "config_hash": "config-v1",
                    "dataset_record_order_fingerprint": "records-v1",
                    "class_names": ["class_a"],
                    "aggregation": {"method": "power_mean", "q": 3.0},
                    "training_protocol": protocol,
                    "feature_contract": {
                        "hrv36": checkpoint_compatible_hrv36_contract()
                    },
                    "initialization_contract": {
                        "policy": "fold_seeded_full_reference_overlap_v1",
                        "reference_variant": "full",
                        "reference_seed": int(ablation.CONFIG["seeds"][0]) + 1,
                        "variant_group_sha256": {"raw_tokenizer": "same-init"},
                        "reference_group_sha256": {"raw_tokenizer": "same-init"},
                    },
                    "pca_explained_variance": 0.975,
                    "pca_contract": {
                        "path": str(pca_path),
                        "sha256": pca_sha256,
                        "explained_variance_ratio_sum": 0.975,
                        "fit_scope": "training_records_of_this_outer_fold_only",
                        "train_index_hash": canonical_contracts[1]["train_index_hash"],
                        "output_dim": int(ablation.CONFIG["hydra_dim"]),
                    },
                    "split": {
                        "train_index_hash": canonical_contracts[1]["train_index_hash"],
                        "val_index_hash": canonical_contracts[1]["val_index_hash"],
                    },
                },
                model_dir / "fold1_final_ema.pt",
            )

            rows = ablation.checkpoint_status(
                "no_morphology", model_dir, canonical_contracts
            )

        self.assertTrue(rows[0]["contract_valid"])
        self.assertEqual(rows[0]["issue"], "")
        self.assertTrue(all(row["contract_valid"] is False for row in rows[1:]))

    def test_structured_ablation_rejects_nonmatched_common_initialization(self):
        ablation = load_script(
            "structured_ablation_initialization_test_module",
            "scripts/revision/43_structured_ablation_5fold.py",
        )
        rows = [
            {
                "variant": "full",
                "fold": 1,
                "contract_valid": True,
                "issue": "",
                "initialization_group_sha256": {
                    "raw_tokenizer": "full-tokenizer",
                    "normalization_head": "shared-head",
                },
            },
            {
                "variant": "no_morphology",
                "fold": 1,
                "contract_valid": True,
                "issue": "",
                "initialization_group_sha256": {
                    "raw_tokenizer": "different-tokenizer",
                    "normalization_head": "shared-head",
                },
            },
        ]

        ablation.enforce_shared_initialization_contract(rows)

        self.assertFalse(rows[0]["contract_valid"])
        self.assertFalse(rows[1]["contract_valid"])
        self.assertIn("shared_full_initialization", rows[1]["issue"])

    def test_structured_ablation_labels_match_the_actual_removed_interfaces(self):
        ablation = load_script(
            "structured_ablation_wording_test_module",
            "scripts/revision/43_structured_ablation_5fold.py",
        )

        self.assertIn(
            "cross-attention interaction",
            ablation.VARIANT_CONTROLS["no_morphology"],
        )
        self.assertIn(
            "five-RR-plus-six-global-statistics",
            ablation.VARIANT_CONTROLS["no_rhythm"],
        )

    def test_checkpoint_compatible_hrv36_contract_rejects_nonzero_reserved_inputs(self):
        values = np.zeros((3, 36), dtype=np.float32)
        values[:, :5] = 1.0
        values[:, 30:] = 2.0
        validate_checkpoint_compatible_hrv36(values)
        self.assertEqual(
            checkpoint_compatible_hrv36_contract()["semantics"],
            HRV36_CHECKPOINT_SEMANTICS,
        )

        values[0, 25] = 1.0
        with self.assertRaisesRegex(ValueError, "amplitude slots"):
            validate_checkpoint_compatible_hrv36(values)

        values[0, 25] = 0.0
        values[0, 5] = 1.0
        with self.assertRaisesRegex(ValueError, "reserved slots"):
            validate_checkpoint_compatible_hrv36(values)

    def test_physiological_metadata_requires_model_independence(self):
        probe = load_script(
            "physiological_probe_provenance_test_module",
            "scripts/revision/44_physiological_interval_probe.py",
        )
        provenance = {
            "status": "reviewed",
            "metadata_sha256": "metadata-sha",
            "record_id_column": "record_id",
            "record_alignment": "one_row_per_record_id",
            "independent_of_model_outputs": False,
            "independent_of_ecg_ramba_feature_cache": False,
            "reviewed_by": "reviewer",
            "reviewed_utc": "2026-07-18T00:00:00+00:00",
            "source_description": "Device-exported interval measurements",
            "targets": {
                "qrs_ms": {
                    "source_column": "qrs_ms",
                    "unit": "ms",
                    "measurement_kind": "device_measured",
                }
            },
        }

        issues = probe.validate_metadata_provenance(
            provenance,
            metadata_sha256="metadata-sha",
            record_id_column="record_id",
            target_columns={"qrs_ms": "qrs_ms"},
        )

        self.assertIn("model_output_independence_not_declared", issues)
        self.assertIn("ecg_ramba_feature_cache_independence_not_declared", issues)

    def test_physiological_metadata_provenance_binds_exact_csv_hash(self):
        probe = load_script(
            "physiological_probe_metadata_hash_test_module",
            "scripts/revision/44_physiological_interval_probe.py",
        )
        provenance = {
            "status": "reviewed",
            "metadata_sha256": "old-sha",
            "record_id_column": "record_id",
            "record_alignment": "one_row_per_record_id",
            "independent_of_model_outputs": True,
            "independent_of_ecg_ramba_feature_cache": True,
            "reviewed_by": "reviewer",
            "reviewed_utc": "2026-07-18T00:00:00+00:00",
            "source_description": "Device-exported interval measurements",
            "targets": {
                "qrs_ms": {
                    "source_column": "qrs_ms",
                    "unit": "ms",
                    "measurement_kind": "device_measured",
                }
            },
        }

        issues = probe.validate_metadata_provenance(
            provenance,
            metadata_sha256="current-sha",
            record_id_column="record_id",
            target_columns={"qrs_ms": "qrs_ms"},
        )

        self.assertIn("metadata_sha256_mismatch", issues)

    def test_physiological_bootstrap_counts_only_finite_replicates(self):
        probe = load_script(
            "physiological_probe_finite_bootstrap_test_module",
            "scripts/revision/44_physiological_interval_probe.py",
        )
        y_true = np.linspace(1.0, 100.0, 200)
        constant_prediction = np.ones(200)

        result = probe.bootstrap_metrics(y_true, constant_prediction, n_boot=50, seed=42)

        self.assertEqual(result["mae"]["n_boot_valid"], 50)
        self.assertEqual(result["r2"]["n_boot_valid"], 50)
        self.assertEqual(result["spearman"]["n_boot_valid"], 0)
        self.assertIsNone(result["spearman"]["ci_low"])

    def test_physiological_target_gate_rejects_constant_fold(self):
        probe = load_script(
            "physiological_probe_target_variation_test_module",
            "scripts/revision/44_physiological_interval_probe.py",
        )
        fold_id = np.repeat(np.arange(1, 6, dtype=np.int16), 4)
        values = np.arange(20, dtype=np.float64)
        values[fold_id == 3] = 80.0
        plausible = np.ones(20, dtype=bool)

        status, records, unique_values = probe.target_coverage_contract(
            values,
            fold_id,
            plausible,
            min_records=20,
            min_records_per_fold=4,
        )

        self.assertEqual(status, "insufficient_target_variation")
        self.assertEqual(records[3], 4)
        self.assertEqual(unique_values[3], 1)

    def test_physiological_runner_contract_is_content_addressed(self):
        probe = load_script(
            "physiological_probe_runner_contract_test_module",
            "scripts/revision/44_physiological_interval_probe.py",
        )

        contract = probe.file_contract(probe.RUNNER_SOURCE_PATH)

        self.assertEqual(contract["path"], "scripts/revision/44_physiological_interval_probe.py")
        self.assertEqual(contract["sha256"], hashlib.sha256(
            probe.RUNNER_SOURCE_PATH.read_bytes()
        ).hexdigest())
        self.assertGreater(contract["size_bytes"], 0)

    def test_physiological_paired_view_delta_is_oriented_toward_view_a(self):
        probe = load_script(
            "physiological_probe_contrast_test_module",
            "scripts/revision/44_physiological_interval_probe.py",
        )
        y_true = np.linspace(50.0, 100.0, 200)
        prediction_a = y_true + np.sin(np.arange(200)) * 0.1
        prediction_b = y_true[::-1]

        result = probe.paired_view_bootstrap(
            y_true,
            prediction_a,
            prediction_b,
            n_boot=100,
            seed=42,
        )

        for metric in ["mae", "r2", "spearman"]:
            self.assertGreater(result[metric]["improvement_view_a_over_view_b"], 0)
            self.assertGreater(result[metric]["ci_low"], 0)
            self.assertEqual(
                result[metric]["interpretation"], "view_a_significantly_better"
            )

    def test_physiological_probe_tex_is_scale_free_and_claim_bounded(self):
        probe = load_script(
            "physiological_probe_tex_test_module",
            "scripts/revision/44_physiological_interval_probe.py",
        )
        rows = []
        for index, view in enumerate(probe.VIEWS):
            rows.append(
                {
                    "row_type": "aggregate",
                    "target": "qrs_ms",
                    "view": view,
                    "n_test": 1000,
                    "spearman": 0.1 + index * 0.01,
                    "spearman_ci_low": 0.05 + index * 0.01,
                    "spearman_ci_high": 0.15 + index * 0.01,
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "probe.tex"
            probe.write_tex_table(rows, output)
            rendered = output.read_text(encoding="utf-8")

        self.assertIn("Fold-held-out linear probes", rendered)
        self.assertIn("QRS duration (ms), 1000", rendered)
        self.assertIn("branch-associated linear information only", rendered)
        self.assertNotIn("disentanglement.", rendered)

    def test_physiological_probe_binds_embedding_to_manifest_and_folds(self):
        probe = load_script(
            "physiological_probe_embedding_contract_test_module",
            "scripts/revision/44_physiological_interval_probe.py",
        )
        record_id = np.arange(10, dtype=np.int64)
        fold_id = np.repeat(np.arange(1, 6, dtype=np.int16), 2)
        embeddings = {
            view: np.ones((10, 3), dtype=np.float32) * index
            for index, view in enumerate(probe.VIEWS, start=1)
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embedding_path = root / "embedding.npz"
            np.savez_compressed(
                embedding_path,
                record_id=record_id,
                fold_id=fold_id,
                oof_predictions_sha256=np.asarray("oof-sha"),
                freeze_manifest_sha256=np.asarray("freeze-sha"),
                **{f"{view}_embedding": values for view, values in embeddings.items()},
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "missing_records": 0,
                        "oof_predictions_sha256": "oof-sha",
                        "freeze_manifest_sha256": "freeze-sha",
                        "split_contract": {
                            "fold_assignment_sha256": probe.array_sha256(
                                fold_id, np.int16
                            )
                        },
                        "outputs": {
                            "embedding_npz_sha256": hashlib.sha256(
                                embedding_path.read_bytes()
                            ).hexdigest()
                        },
                    }
                ),
                encoding="utf-8",
            )

            contract = probe.validate_embedding_provenance(
                embedding_path=embedding_path,
                manifest_path=manifest_path,
                record_id=record_id,
                fold_id=fold_id,
                embeddings=embeddings,
                source_oof_sha256="oof-sha",
                source_freeze_sha256="freeze-sha",
            )
            self.assertEqual(contract["status"], "complete")

            with self.assertRaisesRegex(RuntimeError, "fold_assignment"):
                probe.validate_embedding_provenance(
                    embedding_path=embedding_path,
                    manifest_path=manifest_path,
                    record_id=record_id,
                    fold_id=np.roll(fold_id, 1),
                    embeddings=embeddings,
                    source_oof_sha256="oof-sha",
                    source_freeze_sha256="freeze-sha",
                )

    def test_physiological_probe_complete_path_is_end_to_end_runnable(self):
        probe = load_script(
            "physiological_probe_end_to_end_test_module",
            "scripts/revision/44_physiological_interval_probe.py",
        )
        rng = np.random.default_rng(42)
        n_records = 100
        record_id = np.arange(10_000, 10_000 + n_records, dtype=np.int64)
        fold_id = np.repeat(np.arange(1, 6, dtype=np.int16), n_records // 5)
        target = np.linspace(60.0, 120.0, n_records)
        embeddings = {
            view: np.column_stack(
                [target / 100.0 + rng.normal(0.0, 0.01, n_records), rng.normal(size=n_records)]
            ).astype(np.float32)
            for view in probe.VIEWS
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embedding_path = root / "embedding.npz"
            np.savez_compressed(
                embedding_path,
                record_id=record_id,
                fold_id=fold_id,
                oof_predictions_sha256=np.asarray("oof-sha"),
                freeze_manifest_sha256=np.asarray("freeze-sha"),
                **{f"{view}_embedding": values for view, values in embeddings.items()},
            )
            embedding_manifest = root / "embedding_manifest.json"
            embedding_manifest.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "missing_records": 0,
                        "oof_predictions_sha256": "oof-sha",
                        "freeze_manifest_sha256": "freeze-sha",
                        "split_contract": {
                            "fold_assignment_sha256": probe.array_sha256(fold_id, np.int16)
                        },
                        "outputs": {
                            "embedding_npz_sha256": hashlib.sha256(
                                embedding_path.read_bytes()
                            ).hexdigest()
                        },
                    }
                ),
                encoding="utf-8",
            )
            metadata_path = root / "physiology.csv"
            metadata_path.write_text(
                "record_id,qrs_ms\n"
                + "\n".join(
                    f"{record},{value:.8f}" for record, value in zip(record_id, target)
                )
                + "\n",
                encoding="utf-8",
            )
            provenance_path = root / "physiology.provenance.json"
            provenance_path.write_text(
                json.dumps(
                    {
                        "status": "reviewed",
                        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                        "record_id_column": "record_id",
                        "record_alignment": "one_row_per_record_id",
                        "independent_of_model_outputs": True,
                        "independent_of_ecg_ramba_feature_cache": True,
                        "reviewed_by": "unit-test reviewer",
                        "reviewed_utc": "2026-07-18T00:00:00+00:00",
                        "source_description": "Synthetic measured-target test fixture",
                        "targets": {
                            "qrs_ms": {
                                "source_column": "qrs_ms",
                                "unit": "ms",
                                "measurement_kind": "device_measured",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            outputs = {
                "summary": root / "summary.json",
                "table": root / "table.csv",
                "audit": root / "audit.csv",
                "contrasts": root / "contrasts.csv",
                "tex": root / "table.tex",
                "manifest": root / "manifest.json",
            }
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/revision/44_physiological_interval_probe.py"),
                "--embedding-npz",
                str(embedding_path),
                "--embedding-manifest",
                str(embedding_manifest),
                "--metadata-csv",
                str(metadata_path),
                "--metadata-provenance-json",
                str(provenance_path),
                "--min-records",
                "50",
                "--min-records-per-fold",
                "5",
                "--n-boot",
                "10",
                "--out-summary",
                str(outputs["summary"]),
                "--out-table",
                str(outputs["table"]),
                "--out-audit",
                str(outputs["audit"]),
                "--out-contrast-table",
                str(outputs["contrasts"]),
                "--out-tex-table",
                str(outputs["tex"]),
                "--out-manifest",
                str(outputs["manifest"]),
            ]

            result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)

            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "complete_measured_target_probe")
            self.assertTrue(summary["completeness_contract"]["aggregate_bootstrap_complete"])
            self.assertTrue(summary["completeness_contract"]["contrast_bootstrap_complete"])
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in outputs.values()))


if __name__ == "__main__":
    unittest.main()
