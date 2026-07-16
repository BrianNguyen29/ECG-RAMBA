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
        ):
            self.assertIn(token, source)
        summary_cell = next(cell for cell in cells if "Claim Evidence Summary requires" in cell)
        self.assertIn("oof_prediction_path = revision_root", summary_cell)
        self.assertIn("sha256_file(oof_prediction_path)", summary_cell)
        self.assertNotIn("sha256_file(record_path)", summary_cell)

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
        ):
            self.assertIn(token, source)

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


if __name__ == "__main__":
    unittest.main()
