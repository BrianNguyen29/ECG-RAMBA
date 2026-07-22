import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "04_baselines_and_component_checks.ipynb"


class Notebook04DirectRunContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.code_cells = [
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        cls.source = "\n".join(cls.code_cells)

    def test_every_code_cell_compiles(self):
        for index, source in enumerate(self.code_cells):
            compile(source, f"{NOTEBOOK_PATH}:code_cell_{index}", "exec")

    def test_fair_baseline_matrix_validates_prediction_payloads(self):
        for token in (
            "CURRENT_OOF_REQUIRED_KEYS",
            "OOF_PREDICTION_REQUIRED_KEYS_04",
            "prediction_payload_matches_current_oof_04",
            "Frozen OOF payload contract failed",
            "_prediction_payload_matches_current_oof",
            "np.array_equal(np.asarray(candidate['y_true']",
            "np.array_equal(np.asarray(candidate['record_id']",
            "np.array_equal(np.asarray(candidate['fold_id']",
            "np.array_equal(np.asarray(candidate['class_names']",
            "prediction_contract_ok",
        ):
            self.assertIn(token, self.source)

    def test_runner_reuse_checks_payload_before_skipping(self):
        self.assertIn("Baseline record prediction payload rejected", self.source)
        self.assertIn("Baseline artifact SHA binding rejected", self.source)
        self.assertIn("Baseline checkpoint SHA binding rejected", self.source)
        self.assertIn("MiniRocket-only prediction payload rejected", self.source)
        self.assertIn("MiniRocket-only manifest artifact SHA binding rejected", self.source)
        self.assertIn("require_cuda_runtime_for_baseline('MiniRocket-only baseline')", self.source)
        self.assertIn("artifact_bindings={", self.source)
        self.assertGreaterEqual(self.source.count("artifact_bindings={"), 4)
        self.assertLess(
            self.source.index("def prediction_payload_matches_current_oof_04"),
            self.source.index("def _minirocket_artifacts_current"),
        )

    def test_auto_restore_uses_resolved_runner_state(self):
        for token in (
            "globals().get('minirocket_should_run', False)",
            "globals().get('resnet_should_run', False)",
            "globals().get('raw_mamba_should_run', False)",
            "globals().get('transformer_should_run', False)",
            "globals().get('hybrid_should_run', False)",
        ):
            self.assertIn(token, self.source)
        self.assertNotIn("globals().get('RUN_MINIROCKET_ONLY_BASELINE', False)", self.source)

    def test_paired_reuse_requires_complete_hash_bound_output_package(self):
        for token in (
            "def paired_output_artifacts_current_04",
            "'bootstrap_samples': Path(samples_path)",
            "manifest.get('artifact_sha256')",
            "paired output SHA mismatch for",
        ):
            self.assertIn(token, self.source)
        self.assertGreaterEqual(self.source.count("paired_output_artifacts_current_04("), 7)

    def test_hybrid_control_is_in_baseline_summary(self):
        for token in (
            "hybrid_summary, hybrid_artifact_valid, hybrid_stale_reason",
            "complete_optional_mechanism_baseline_from_script26",
            "family='mechanism_baseline_optional'",
            "hybrid_row,",
        ):
            self.assertIn(token, self.source)

    def test_direct_drive_fold_updates_repair_manifest_and_checkpoint_rows(self):
        self.assertIn("canonical_fold_manifest_mismatches_04", self.source)
        contracts = {
            "resnet1d_cnn": "fold{int(fold)}_resnet1d_cnn_final.pt",
            "raw_mamba": "fold{int(fold)}_raw_mamba_final_ema.pt",
            "transformer_ecg": "fold{int(fold)}_transformer_ecg_final.pt",
            "hybrid_morphology": "fold{int(fold)}_hybrid_morphology_final.pt",
        }
        for baseline, checkpoint_token in contracts.items():
            self.assertIn(f"{baseline}_fold{{fold}}_predictions.npz", self.source)
            self.assertIn(checkpoint_token, self.source)
        self.assertGreaterEqual(self.source.count("manifest repair only"), 4)

    def test_fold_cache_preflight_requires_current_oof_and_checkpoint_sha(self):
        for token in (
            "fold_prediction_cache_current_04",
            "oof_predictions_sha256",
            "checkpoint_sha256",
            "expected_checkpoint_sha",
            "Fold cache needs provenance upgrade",
            "LEGACY_PATCH_TRANSFORMER_SOURCE_COMMIT",
            "legacy_metadata_upgrade",
        ):
            self.assertIn(token, self.source)
        self.assertGreaterEqual(self.source.count("fold_prediction_cache_current_04("), 5)

    def test_transformer_reuse_adopts_consistent_checkpoint_training_batch_size(self):
        for token in (
            "TRANSFORMER_ADOPT_CHECKPOINT_TRAINING_BATCH_SIZE = True",
            "def _transformer_checkpoint_training_batch_sizes",
            "Transformer checkpoints were trained with inconsistent batch sizes",
            "and not transformer_force_retrain_requested",
            "TRANSFORMER_BATCH_SIZE = checkpoint_training_batch_size",
            "Resolved Transformer batch size",
        ):
            self.assertIn(token, self.source)

    def test_raw_mamba_installer_uses_the_capability_schema_pair(self):
        raw_mamba_cell = next(
            cell
            for cell in self.code_cells
            if "def ensure_mamba_runtime_for_raw_mamba()" in cell
        )
        for token in (
            "MAMBA_INSTALLER_CAPABILITY = 'ecg_ramba_mamba_installer_v1'",
            "MAMBA_INSTALLER_SCHEMA_VERSION = 1",
            "installer_candidates = []",
            "Could not locate exactly one canonical Mamba installer cell in Notebook 02.",
        ):
            self.assertIn(token, raw_mamba_cell)
        self.assertNotIn("'Mamba wheel environment' in candidate_source", raw_mamba_cell)

    def test_controlled_morphology_learnability_is_fold_resumable_and_paired(self):
        for token in (
            "39_morphology_learnability_control.py",
            "40_paired_morphology_learnability.py",
            "MORPHOLOGY_LEARNABILITY_FOLD_CACHE_DIR",
            "MORPHOLOGY_LEARNABILITY_CHECKPOINT_DIR",
            "matched_initialization_sha256_by_fold",
            "morphology_learnability_{variant}_fold{fold}_predictions.npz",
            "fold{fold}_morphology_learnability_{variant}_final.pt",
            "paired_morphology_learnability_bootstrap_samples.csv",
            "--n-boot 1000",
        ):
            self.assertIn(token, self.source)

        setup_cell = next(
            cell for cell in self.code_cells if "DIRECT_RUN_SOURCE_REQUIREMENTS_04" in cell
        )
        self.assertIn("scripts/revision/39_morphology_learnability_control.py", setup_cell)
        self.assertIn("scripts/revision/40_paired_morphology_learnability.py", setup_cell)

        summary_cell = next(
            cell for cell in self.code_cells if "reviewer_required_baseline_outputs" in cell
        )
        for token in (
            "morphology_learnability_summary.json",
            "morphology_learnability_frozen_oof_predictions.npz",
            "morphology_learnability_partial_oof_predictions.npz",
            "paired_morphology_learnability_comparison.json",
            "MORPHOLOGY_LEARNABILITY_CHECKPOINT_DIR",
        ):
            self.assertIn(token, summary_cell)


if __name__ == "__main__":
    unittest.main()
