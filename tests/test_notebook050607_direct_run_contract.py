import ast
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def notebook_source(name: str) -> tuple[list[str], str]:
    path = PROJECT_ROOT / "notebooks" / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = [
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    ]
    return cells, "\n".join(cells)


def notebook_payload(name: str) -> dict:
    path = PROJECT_ROOT / "notebooks" / name
    return json.loads(path.read_text(encoding="utf-8"))


class Notebook050607DirectRunContractTests(unittest.TestCase):
    def test_notebook_v4_structure_is_complete(self):
        for notebook in (
            "05_hrv_domain_and_robustness.ipynb",
            "06_pooling_and_representation.ipynb",
            "07_results_freeze.ipynb",
        ):
            payload = notebook_payload(notebook)
            self.assertEqual(payload.get("nbformat"), 4)
            self.assertIsInstance(payload.get("nbformat_minor"), int)
            self.assertIsInstance(payload.get("metadata"), dict)
            self.assertIsInstance(payload.get("cells"), list)
            self.assertTrue(payload["cells"])
            for cell in payload["cells"]:
                self.assertIn(cell.get("cell_type"), {"code", "markdown", "raw"})
                self.assertIsInstance(cell.get("metadata"), dict)
                self.assertIsInstance(cell.get("source"), list)
                if cell.get("cell_type") == "code":
                    self.assertIn("execution_count", cell)
                    self.assertIsInstance(cell.get("outputs"), list)

    def test_every_code_cell_compiles(self):
        for notebook in (
            "05_hrv_domain_and_robustness.ipynb",
            "06_pooling_and_representation.ipynb",
            "07_results_freeze.ipynb",
        ):
            cells, _ = notebook_source(notebook)
            for index, source in enumerate(cells):
                compile(source, f"{notebook}:cell_{index}", "exec")

    def test_notebook05_selects_authenticated_robustness_profile(self):
        cells, source = notebook_source("05_hrv_domain_and_robustness.ipynb")
        for token in (
            "robustness_profile_audit.py",
            "select_best_profile",
            "robustness_multicomparator_audit",
            "metric_specific_ci_ready",
            "ROBUSTNESS_MULTI_RUN_PROFILE",
            "reviewer_minimal",
            "core_final",
            "artifact_mirror.py publish --verify-existing size",
            "--refresh-existing-cache-dirs",
            "oof_label_fold_contract_sha256",
            "hrv_only_oof_reuse_attestation.json",
            "semantic_contract_match",
            "BOOTSTRAP_ENGINE",
            "weighted_resample_metric",
            "ROBUSTNESS_MULTI_DEFAULT_BOOTSTRAP_JOBS",
            "ROBUSTNESS_MULTI_INNER_THREADS",
            "total paired metric jobs",
        ):
            self.assertIn(token, source)
        self.assertIn("'canonical_resume'", source)
        self.assertIn("Canonical six-stress robustness ledger is incomplete", source)
        self.assertIn("ROBUSTNESS_MULTI_STRICT = ROBUSTNESS_MULTI_RUN_PROFILE", source)
        summary_cell = next(cell for cell in cells if "Claim Evidence Summary requires" in cell)
        self.assertIn("oof_prediction_path = revision_root", summary_cell)
        self.assertIn("sha256_file(oof_prediction_path)", summary_cell)
        self.assertNotIn("sha256_file(record_path)", summary_cell)

    def test_notebook05_validates_stress_provenance_before_gpu_reuse(self):
        _, source = notebook_source("05_hrv_domain_and_robustness.ipynb")
        for token in (
            "comparator_stress_manifest_preflight.log",
            "--finalize-manifest-only",
            "Authenticated comparator/stress artifacts:",
            "Missing/stale comparator stress pairs:",
            "Only affected stress groups will run",
            "raw_mamba_needs_inference",
            "--refresh-existing-prefix predictions/robustness_",
            "All requested comparator stress artifacts passed provenance validation",
        ):
            self.assertIn(token, source)
        self.assertNotIn(
            "All requested comparator stress prediction artifacts are present; skipping GPU inference.",
            source,
        )

    def test_notebook05_finds_current_notebook02_mamba_installer(self):
        _, source = notebook_source("05_hrv_domain_and_robustness.ipynb")
        for token in (
            "MODEL_DEPS_SHOULD_RUN",
            "candidate_count={len(installer_candidates)}",
            "Mamba wheel environment",
        ):
            self.assertIn(token, source)
        self.assertNotIn("'INSTALL_MODEL_DEPS = True' in source", source)

    def test_notebook06_can_refresh_semantically_equivalent_embedding_on_cpu(self):
        _, source = notebook_source("06_pooling_and_representation.ipynb")
        for token in (
            "inspect_final_embedding_reuse",
            "validate_checkpoint_fold_contract",
            "Representation embedding semantic refresh audit",
            "Refreshing representation provenance from the verified final embedding; GPU/Mamba is not needed.",
            "The extractor now uses the fold_id frozen in the canonical OOF artifact",
        ):
            self.assertIn(token, source)

    def test_notebook06_is_direct_run_and_contract_strict(self):
        _, source = notebook_source("06_pooling_and_representation.ipynb")
        for token in (
            "ECG_RAMBA_INSTALL_NOTEBOOK06_BASE_DEPS",
            "pooling_sensitivity_immediate_mirror_publish.log",
            "external_pooling_immediate_mirror_publish.log",
            "representation_embedding_immediate_mirror_publish.log",
            "representation_probe_immediate_mirror_publish.log",
            "_pooling_artifacts_match_active_freeze()",
            "_embedding_is_current()",
            "_probe_is_v3_ready()",
            "representation_probe_fold_safe_v3_projection_and_fold_audit",
            "--refresh-existing-cache-dirs",
            "Repairing stale direct-cache manifest rows before Notebook 06 restore",
            "representation_cache_manifest_repair.log",
            "--refresh-existing-prefix predictions/folds",
        ):
            self.assertIn(token, source)
        for token in (
            "ptbxl,georgia,cpsc2021",
            "external_pooling_manifest.get('datasets') != ['ptbxl', 'georgia', 'cpsc2021']",
            "external_pooling_manifest.get('strict_group_bootstrap') is not True",
            "pooling_q3_paired_bootstrap.json",
        ):
            self.assertIn(token, source)

    def test_notebook06_finds_current_notebook02_mamba_installer(self):
        payload = notebook_payload("02_predictions_and_external_eval.ipynb")
        required_markers = (
            "INSTALL_MODEL_DEPS",
            "MODEL_DEPS_SHOULD_RUN",
            "AUTO_PIN_TORCH_FOR_MAMBA",
            "Mamba wheel environment",
        )
        candidates = [
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell.get("cell_type") == "code"
            and all(marker in "".join(cell.get("source", [])) for marker in required_markers)
        ]
        self.assertEqual(len(candidates), 1)
        self.assertIn("INSTALL_MODEL_DEPS = 'auto'", candidates[0])

    def test_notebook07_uses_targeted_restore_and_exports_profiles(self):
        _, source = notebook_source("07_results_freeze.ipynb")
        for token in (
            "ECG_RAMBA_FULL_MIRROR_RESTORE_07",
            "Skipping full mirror restore in Notebook 07",
            "--include-prefix",
            "robustness_multicomparator*_summary.csv",
            "learned_comparator_robustness_audit",
            "canonical_gate_ready",
            "final_evidence_tables",
            "--refresh-existing-cache-dirs",
        ):
            self.assertIn(token, source)

    def test_notebook07_uses_stable_final_generator_capabilities(self):
        _, source = notebook_source("07_results_freeze.ipynb")
        generator_source = (
            PROJECT_ROOT / "scripts" / "revision" / "13_final_evidence_matrix.py"
        ).read_text(encoding="utf-8")
        for token in (
            "FINAL_EVIDENCE_SCHEMA_VERSION",
            "FINAL_EVIDENCE_CAPABILITIES",
            "group_safe_score_calibration_v2",
            "true_fewshot_frozen_encoder_head_v2",
            "representation_probe_v3",
            "reviewer_presentation_assets",
            "reviewer_gap_closure_v1",
            "morphology_kernel_learnability_control",
            "external_zero_target_group_paired_ci",
            "pooling_q3_cross_dataset_sensitivity",
        ):
            self.assertIn(token, source)
            self.assertIn(token, generator_source)
        self.assertNotIn("def summarize_fewshot", source)
        self.assertNotIn("def combine_fewshot_summaries", source)

    def test_final_generator_closure_helper_does_not_truncate_main(self):
        generator_path = PROJECT_ROOT / "scripts" / "revision" / "13_final_evidence_matrix.py"
        tree = ast.parse(generator_path.read_text(encoding="utf-8"))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("reviewer_gap_closure_contract_issues", functions)
        main_node = functions["main"]
        self.assertFalse(any(isinstance(node, ast.FunctionDef) for node in main_node.body))
        assigned_names = {
            target.id
            for node in ast.walk(main_node)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
        self.assertIn("missing", assigned_names)
        self.assertIn("matrix_rows", assigned_names)

    def test_notebook07_refreshes_calibration_and_builds_reviewer_assets(self):
        _, source = notebook_source("07_results_freeze.ipynb")
        for token in (
            "bootstrap.unit=",
            "one_chapman_record_per_subject",
            "04_calibration_ci.py",
            "29_reviewer_presentation_assets.py --strict",
            "final_evidence_calibration_contract_refresh.log",
            "final_evidence_calibration_refresh_mirror_publish.log",
        ):
            self.assertIn(token, source)

    def test_notebook07_closes_four_partial_reviewer_items_before_final_freeze(self):
        _, source = notebook_source("07_results_freeze.ipynb")
        for token in (
            "41_reviewer_gap_closure.py --strict",
            "reviewer_gap_closure_mirror_publish.log",
            "R1-C2",
            "R1-C5",
            "R1-C6",
            "R2-C3",
            "table_external_zero_target_ci_compact.csv",
            "table_pooling_cross_dataset_compact.csv",
            "table_morphology_learnability_compact.csv",
            "table_robustness_six_stress_compact.csv",
        ):
            self.assertIn(token, source)

    def test_notebook03_rejects_legacy_calibration_bootstrap_metadata(self):
        _, source = notebook_source("03_calibration_and_ci.ipynb")
        self.assertIn("bootstrap_unit_missing_or_mismatched", source)
        self.assertIn("bootstrap_independence_contract_missing_or_mismatched", source)


if __name__ == "__main__":
    unittest.main()
