import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


paired = importlib.import_module("scripts.revision.32_paired_external_comparators")
fewshot = importlib.import_module("scripts.revision.35_true_fewshot_head_adaptation")
external_runner = importlib.import_module(
    "scripts.revision.31_generate_external_comparator_predictions"
)


class ExternalComparatorContractTests(unittest.TestCase):
    def _write_prediction(self, path: Path, probabilities: np.ndarray) -> None:
        y_true = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        np.savez_compressed(
            path,
            y_true=y_true,
            y_prob=np.asarray(probabilities, dtype=np.float32),
            record_id=np.asarray(["r1", "r2", "r3"]),
            group_id=np.asarray(["g1", "g2", "g3"]),
            split_id=np.asarray(["test", "test", "test"]),
            class_names=np.asarray(["a", "b"]),
            dataset=np.asarray("ptbxl"),
            protocol=np.asarray("test"),
            task_scope=np.asarray("record_level_mapped_external_task"),
            group_unit=np.asarray("patient"),
            adaptation_labels_used=np.asarray(0, dtype=np.int16),
        )

    def test_paired_external_loader_rejects_out_of_range_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.npz"
            self._write_prediction(
                path,
                np.asarray([[1.1, 0.2], [0.3, 0.8], [0.7, 0.9]], dtype=np.float32),
            )
            with self.assertRaises(ValueError):
                paired.load_predictions(path, "ptbxl")

    def test_external_group_assignment_and_cache_contract_are_content_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.npz"
            self._write_prediction(
                path,
                np.asarray([[0.8, 0.2], [0.3, 0.8], [0.7, 0.9]], dtype=np.float32),
            )
            loaded = paired.load_predictions(path, "ptbxl")
            args = SimpleNamespace(threshold=0.5, n_bins=15, n_boot=1000)
            canonical = {
                "oof_sha256": "a" * 64,
                "freeze_sha256": "b" * 64,
                "group_contract_sha256": "c" * 64,
                "group_sidecar_sha256": "d" * 64,
            }
            contract = paired.metric_cache_contract(
                "ptbxl",
                "resnet",
                "pr_auc_macro",
                loaded["sha256"],
                "e" * 64,
                args,
                group_assignment_sha256=loaded["group_assignment_sha256"],
                canonical=canonical,
                full_gate_sha256="f" * 64,
                comparator_manifest_sha256="1" * 64,
                bootstrap_seed=42,
            )
            changed_boot = {**contract, "n_boot": 999}
            changed_group = {**contract, "group_assignment_sha256": "2" * 64}
            self.assertNotEqual(paired.metric_cache_key(contract), paired.metric_cache_key(changed_boot))
            self.assertNotEqual(paired.metric_cache_key(contract), paired.metric_cache_key(changed_group))
            self.assertEqual(contract["canonical_group_sidecar_sha256"], "d" * 64)

    def test_external_bootstrap_cache_requires_exact_finite_count(self):
        valid = {
            "n_boot_valid": 10,
            "improvement_ci_low": -0.1,
            "improvement_ci_high": 0.2,
        }
        paired.validate_bootstrap_payload(valid, np.arange(10, dtype=float), n_boot=10)
        with self.assertRaisesRegex(RuntimeError, "exact finite bootstrap"):
            paired.validate_bootstrap_payload(valid, np.arange(9, dtype=float), n_boot=10)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            paired.validate_bootstrap_payload(
                {**valid, "improvement_ci_high": float("nan")},
                np.arange(10, dtype=float),
                n_boot=10,
            )

    def test_external_fold_cache_binds_dataset_sidecar_and_slice_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fold.npz"
            record_index = np.asarray([0, 0, 1], dtype=np.int64)
            starts = np.asarray([0, 1250, 0], dtype=np.int32)
            classes = np.asarray(["a", "b"])
            np.savez_compressed(
                path,
                slice_prob=np.asarray([[0.2, 0.8], [0.4, 0.6], [0.7, 0.3]], dtype=np.float32),
                slice_record_index=record_index,
                slice_start=starts,
                class_names=classes,
                dataset=np.asarray("ptbxl"),
                comparator=np.asarray("resnet"),
                fold=np.asarray(1),
                checkpoint_sha256=np.asarray("a" * 64),
                input_fingerprint=np.asarray("b" * 64),
                dataset_contract_sha256=np.asarray("c" * 64),
                protocol_version=np.asarray(external_runner.PROTOCOL_VERSION),
            )
            kwargs = dict(
                dataset="ptbxl",
                comparator="resnet",
                fold=1,
                checkpoint_sha="a" * 64,
                input_fingerprint="b" * 64,
                class_names=classes,
                dataset_contract_sha256="c" * 64,
                expected_record_index=record_index,
                expected_starts=starts,
            )
            self.assertTrue(external_runner.cache_matches(path, **kwargs))
            self.assertFalse(
                external_runner.cache_matches(
                    path,
                    **{**kwargs, "dataset_contract_sha256": "d" * 64},
                )
            )
            self.assertFalse(
                external_runner.cache_matches(
                    path,
                    **{**kwargs, "expected_starts": np.asarray([0, 0, 1250])},
                )
            )

    def test_cache_only_preflight_precedes_signal_loading(self):
        source = (
            Path(external_runner.__file__).read_text(encoding="utf-8")
        )
        run_dataset = source[source.index("def run_dataset("):source.index("\ndef main()")]
        self.assertLess(
            run_dataset.index("load_dataset_contract(args, dataset, sources, contract)"),
            run_dataset.index("external_helpers.load_records("),
        )
        self.assertIn("skipping archive extraction and signal loading", run_dataset)
        self.assertIn("all requested fold caches passed the v2 dataset-sidecar contract", run_dataset)

    def test_true_fewshot_metric_cache_binds_group_and_runner_contract(self):
        args = SimpleNamespace(
            dataset="ptbxl",
            threshold=0.5,
            n_bins=15,
            n_boot=1000,
        )
        canonical = {
            "oof_sha256": "a" * 64,
            "freeze_sha256": "b" * 64,
            "group_contract_sha256": "c" * 64,
            "group_sidecar_sha256": "d" * 64,
        }
        contract = fewshot.metric_cache_contract(
            args,
            comparison="full_vs_resnet",
            metric="f1_macro",
            seed=42,
            fraction=0.10,
            prediction_keys={"full": "e" * 64, "resnet": "f" * 64},
            train_groups=np.asarray(["g1", "g2"]),
            test_groups=np.asarray(["g3", "g4"]),
            canonical=canonical,
            analysis_lock_sha256="1" * 64,
        )
        self.assertEqual(contract["canonical_group_contract_sha256"], "c" * 64)
        self.assertEqual(contract["canonical_group_sidecar_sha256"], "d" * 64)
        self.assertEqual(contract["analysis_lock_sha256"], "1" * 64)
        self.assertNotEqual(
            fewshot.metric_cache_key(contract),
            fewshot.metric_cache_key({**contract, "n_boot": 999}),
        )
        valid = {"n_boot_valid": 10, "lo": -0.1, "hi": 0.2}
        fewshot.validate_interval_payload(valid, n_boot=10, low_field="lo", high_field="hi")
        with self.assertRaisesRegex(RuntimeError, "exactly 10"):
            fewshot.validate_interval_payload(
                {**valid, "n_boot_valid": 9},
                n_boot=10,
                low_field="lo",
                high_field="hi",
            )

    def test_embedding_requires_current_manifest_and_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction_path = root / "prediction.npz"
            self._write_prediction(
                prediction_path,
                np.asarray([[0.8, 0.2], [0.3, 0.8], [0.7, 0.9]], dtype=np.float32),
            )
            prediction = fewshot.load_prediction(prediction_path, "ptbxl")
            embedding_path = root / "embedding.npz"
            embeddings = np.arange(5 * 3 * 4, dtype=np.float32).reshape(5, 3, 4)
            np.savez_compressed(
                embedding_path,
                fold_embeddings=embeddings,
                y_true=prediction["y_true"],
                record_id=prediction["record_id"],
                group_id=prediction["group_id"],
                split_id=prediction["split_id"],
                class_names=prediction["class_names"],
                model=np.asarray("full"),
                source_prediction_sha256=np.asarray(prediction["sha256"]),
                input_fingerprint=np.asarray("f" * 64),
                protocol_version=np.asarray(2, dtype=np.int16),
                representation=np.asarray("mean_of_preclassifier_slice_embeddings_per_fold"),
            )
            canonical = {"oof_sha256": "a" * 64, "freeze_sha256": "b" * 64}
            manifest_path = root / "embedding_manifest.json"
            extractor = (
                fewshot.PROJECT_ROOT
                / "scripts"
                / "revision"
                / "34_extract_external_representations.py"
            )
            manifest = {
                "status": "complete",
                "protocol": "frozen_encoder_external_record_representation_v2_source_bound",
                "runner_sha256": fewshot.sha256_file(extractor),
                "canonical_contract": canonical,
                "input_fingerprint": "f" * 64,
                "representation": "mean_of_preclassifier_slice_embeddings_per_fold",
                "source_prediction": {"sha256": prediction["sha256"]},
                "output": {"sha256": fewshot.sha256_file(embedding_path)},
                "checkpoints": [
                    {"fold": fold, "sha256": str(fold) * 64}
                    for fold in range(1, 6)
                ],
                "checkpoint_source_contract": {"status": "authenticated"},
                "source_provenance": {"archive": {"sha256": "c" * 64}},
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = fewshot.load_embeddings(
                embedding_path,
                manifest_path,
                prediction,
                "full",
                canonical,
            )
            np.testing.assert_array_equal(loaded["embedding"], embeddings)

            manifest["canonical_contract"] = {"oof_sha256": "stale"}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                fewshot.load_embeddings(
                    embedding_path,
                    manifest_path,
                    prediction,
                    "full",
                    canonical,
                )

            manifest["canonical_contract"] = canonical
            manifest["source_provenance"] = {}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                fewshot.load_embeddings(
                    embedding_path,
                    manifest_path,
                    prediction,
                    "full",
                    canonical,
                )


if __name__ == "__main__":
    unittest.main()
