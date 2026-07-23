import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_predictions_and_external_eval.ipynb"


class Notebook02DirectRunContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.cells = notebook["cells"]
        cls.code_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        cls.source = "\n".join(cls.code_cells)

    def test_external_runner_is_split_into_authenticated_cpu_and_a100_phases(self):
        feature_cells = [
            "".join(cell.get("source", []))
            for cell in self.cells
            if "EXTERNAL_FEATURE_PHASE_CELL_V1" in "".join(cell.get("source", []))
        ]
        fold9_feature_cells = [
            "".join(cell.get("source", []))
            for cell in self.cells
            if "PTBXL_FOLD9_FEATURE_PHASE_CELL_V1"
            in "".join(cell.get("source", []))
        ]
        inference_cells = [
            "".join(cell.get("source", []))
            for cell in self.cells
            if "EXTERNAL_RUN_PROFILE = os.environ.get(" in "".join(cell.get("source", []))
        ]
        fold9_inference_cells = [
            "".join(cell.get("source", []))
            for cell in self.cells
            if "RUN_PTBXL_FOLD9_EXPORT = 'auto'" in "".join(cell.get("source", []))
        ]
        self.assertEqual(len(feature_cells), 1)
        self.assertEqual(len(fold9_feature_cells), 1)
        self.assertEqual(len(inference_cells), 1)
        self.assertEqual(len(fold9_inference_cells), 1)

        feature = feature_cells[0]
        self.assertIn("--features-only", feature)
        self.assertIn("--feature-device cpu", feature)
        self.assertIn("external_feature_inference_handoff_v1", feature)
        self.assertIn("external_features_only_mirror_publish.log", feature)
        self.assertNotIn("require_gpu_inference_runtime", feature)
        self.assertNotIn("--inference-only", feature)

        inference = inference_cells[0]
        self.assertIn(
            "03_generate_external_predictions.py --inference-only", inference
        )
        self.assertIn("require_gpu_inference_runtime", inference)
        self.assertIn("external_{dataset}_feature_cache_manifest.json", inference)
        self.assertIn(
            "reports/revision/predictions/oof_final_ema_predictions.npz",
            inference,
        )
        self.assertIn(
            "reports/revision/manifests/oof_final_ema_prediction_run_manifest.json",
            inference,
        )
        self.assertNotIn(
            '--refresh-existing-prefix "predictions/external_feature_cache"',
            inference[inference.index("def external_publish_refresh_args(dataset):") :],
        )

        fold9_feature = fold9_feature_cells[0]
        self.assertIn("--output-tag fold9 --features-only", fold9_feature)
        self.assertNotIn("require_gpu_inference_runtime", fold9_feature)
        fold9_inference = fold9_inference_cells[0]
        self.assertIn(
            "03_generate_external_predictions.py --inference-only", fold9_inference
        )
        self.assertIn("PTBXL_FOLD9_FEATURE_HANDOFF", fold9_inference)
        self.assertIn("PTBXL_FOLD9_CONTRACT_INPUTS", fold9_inference)
        self.assertIn(
            "reports/revision/manifests/oof_final_ema_prediction_run_manifest.json",
            fold9_inference,
        )

    def test_every_code_cell_compiles(self):
        for index, source in enumerate(self.code_cells):
            compile(source, f"{NOTEBOOK_PATH}:code_cell_{index}", "exec")

    def test_only_true_fewshot_removes_unpublished_regenerable_outputs(self):
        self.assertIn(
            "def _restore_report_artifact(path, source_roots, remove_unpublished_active=False, allow_unpublished_active=False)",
            self.source,
        )
        self.assertIn("Active artifact is not authenticated by the canonical mirror manifest", self.source)
        self.assertEqual(self.source.count("remove_unpublished_active=True"), 1)
        self.assertIn(
            "_restore_report_artifact(path, true_fewshot_restore_roots, remove_unpublished_active=True)",
            self.source,
        )
        self.assertIn("'status': 'removed_unpublished_active'", self.source)

    def test_external_reuse_and_gate_handoff_are_source_bound(self):
        self.assertIn("validate_external_prediction_reuse", self.source)
        self.assertIn("external source-bound reuse contract ready", self.source)
        self.assertIn("external_archive_hash_cache", self.source)
        self.assertIn("table_georgia_snomed_code_inventory.csv", self.source)
        self.assertIn("table_cpsc2021_annotation_audit.csv", self.source)
        self.assertIn("External gate source-bound preflight:", self.source)
        self.assertIn("stopped before bootstrap because restored prediction artifacts are stale", self.source)
        self.assertIn("reports/revision/predictions/oof_final_ema_predictions.npz", self.source)
        self.assertIn("reports/revision/manifests/oof_final_ema_freeze_manifest.json", self.source)
        self.assertIn("--source-conflict-policy source", self.source)
        self.assertIn("--include-path", self.source)

    def test_external_rocket_features_are_gpu_parity_checked_and_resumable(self):
        self.assertIn("EXTERNAL_RUN_PROFILE = os.environ.get(", self.source)
        self.assertIn("'cpsc_resume_a100'", self.source)
        self.assertIn("'full_reviewer_a100'", self.source)
        self.assertIn("'cpu_gate_cpsc'", self.source)
        self.assertIn("'exports': {'ptbxl': False, 'georgia': False, 'cpsc2021': 'auto'}", self.source)
        self.assertIn("'allow_export_failures': False", self.source)
        self.assertIn("EXTERNAL_FEATURE_DEVICE = EXTERNAL_RUN_CONFIG['feature_device']", self.source)
        self.assertIn("EXTERNAL_FEATURE_BATCH_SIZE = 64", self.source)
        self.assertIn("EXTERNAL_FEATURE_PARITY_RECORDS = 4", self.source)
        self.assertIn("f'--feature-device {EXTERNAL_FEATURE_DEVICE} '", self.source)
        self.assertIn("f'--feature-batch-size {EXTERNAL_FEATURE_BATCH_SIZE} '", self.source)
        self.assertIn("f'--feature-parity-records {EXTERNAL_FEATURE_PARITY_RECORDS} '", self.source)
        self.assertIn("EXTERNAL_FEATURE_CACHE_ROOT = stable_mirror / 'predictions' / 'external_feature_cache'", self.source)
        self.assertIn("os.environ['ECG_RAMBA_EXTERNAL_FEATURE_CACHE_DIR']", self.source)
        self.assertIn('--refresh-existing-prefix "predictions/external_feature_cache"', self.source)
        self.assertIn(
            "cpsc2021_preprocessed_windows_source_bound_v3.npy.contract.npz",
            self.source,
        )
        self.assertIn("def external_publish_refresh_args(dataset):", self.source)
        self.assertIn("def publish_external_dataset_outputs(dataset, mirror_root, log_path):", self.source)
        self.assertIn("def _is_external_export_recovery_artifact(path):", self.source)
        self.assertIn(
            "allow_unpublished_active=_is_external_export_recovery_artifact(path)",
            self.source,
        )
        self.assertIn(
            "Recovering source-bound external outputs after an interrupted mirror publish",
            self.source,
        )
        self.assertIn("PTBXL_FOLD9_FEATURE_DEVICE = 'cpu'", self.source)
        self.assertIn("PTBXL_FOLD9_FEATURE_BATCH_SIZE = 64", self.source)
        self.assertIn("f'--feature-device {PTBXL_FOLD9_FEATURE_DEVICE} '", self.source)
        self.assertIn("PTBXL_FOLD9_EXTERNAL_FEATURE_CACHE_ROOT", self.source)
        self.assertIn("external_fixed_rocket_gpu_parity_checked_v1", self.source)
        self.assertIn("external_rocket_backend_bound_cache_v1", self.source)

    def test_external_gate_and_handoff_follow_the_explicit_dataset_selection(self):
        self.assertIn("EXTERNAL_GATE_DATASET_LIST =", self.source)
        self.assertIn("for dataset in EXTERNAL_GATE_DATASET_LIST", self.source)
        self.assertIn("if 'georgia' in EXTERNAL_GATE_DATASET_LIST:", self.source)
        self.assertIn("if 'cpsc2021' in EXTERNAL_GATE_DATASET_LIST:", self.source)
        self.assertIn("external_handoff_datasets = [", self.source)
        self.assertIn("for dataset in external_handoff_datasets", self.source)
        self.assertIn("EXTERNAL_GATE_DATASETS_DEFAULT", self.source)

    def test_missing_oof_group_sidecar_is_repaired_without_automatic_gpu_inference(self):
        self.assertIn("scripts/revision/49_build_oof_group_sidecar.py", self.source)
        self.assertIn("def ensure_oof_group_sidecar():", self.source)
        self.assertIn("Refreshing the strict OOF freeze metadata on CPU", self.source)
        self.assertIn(
            "freeze_refresh_command = freeze_command + ' --metadata-refresh-from-existing-oof'",
            self.source,
        )
        self.assertIn("verified_metadata_only_refresh", self.source)
        self.assertIn(
            "oof_inference_required = bool(FORCE_RERUN_OOF or not oof_core_available)",
            self.source,
        )
        self.assertIn("GPU inference was intentionally not started", self.source)

    def test_ptbxl_adaptation_is_locked_and_audited_before_metric_runners(self):
        lock = self.source.index("PTBXL_ADAPTATION_ANALYSIS_LOCK_CELL_V1")
        refresh = self.source.index("python -u scripts/revision/50_refresh_in_domain_paired_contracts.py")
        external = self.source.index("python -u scripts/revision/31_generate_external_comparator_predictions.py")
        audit = self.source.index("PTBXL_FOLD_PROTOCOL_AUDIT_CELL_V1")
        paired = self.source.index("python -u scripts/revision/32_paired_external_comparators.py")
        score = self.source.index("python -u scripts/revision/33_group_safe_score_calibration.py")
        head = self.source.index("python -u scripts/revision/35_true_fewshot_head_adaptation.py")
        self.assertLess(lock, refresh)
        self.assertLess(refresh, external)
        self.assertLess(external, audit)
        self.assertLess(audit, paired)
        self.assertLess(audit, score)
        self.assertLess(audit, head)
        self.assertIn("--analysis-lock \"{PTBXL_ADAPTATION_LOCK}\"", self.source)
        self.assertIn("ptbxl_adaptation_analysis_lock_source_attestation.json", self.source)
        self.assertIn("Runner source drift:", self.source)
        self.assertIn("table_ptbxl_unsupported_only_sensitivity.csv", self.source)
        self.assertIn("paired_refresh_complete_models", self.source)
        self.assertIn("{'resnet', 'raw_mamba'}.issubset(paired_refresh_complete_models)", self.source)

    def test_external_comparator_cpu_cache_aggregation_precedes_gpu_requirement(self):
        self.assertIn("CPU cache-only aggregation is supported", self.source)
        self.assertIn("A CPU runtime may rebuild aggregate artifacts from complete caches", self.source)
        self.assertNotIn(
            "require_gpu_inference_runtime('External learned-comparator inference for '",
            self.source,
        )
        self.assertIn("for dataset in datasets:", self.source)
        self.assertIn(
            "Publishing external comparator outputs immediately for dataset=",
            self.source,
        )
        self.assertIn(
            "external_learned_comparators_{dataset}_mirror_publish.log",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
