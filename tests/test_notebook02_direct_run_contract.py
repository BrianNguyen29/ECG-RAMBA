import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_predictions_and_external_eval.ipynb"


class Notebook02DirectRunContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.code_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        cls.source = "\n".join(cls.code_cells)

    def test_every_code_cell_compiles(self):
        for index, source in enumerate(self.code_cells):
            compile(source, f"{NOTEBOOK_PATH}:code_cell_{index}", "exec")

    def test_only_true_fewshot_removes_unpublished_regenerable_outputs(self):
        self.assertIn(
            "def _restore_report_artifact(path, source_roots, remove_unpublished_active=False)",
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

    def test_missing_oof_group_sidecar_is_repaired_without_automatic_gpu_inference(self):
        self.assertIn("scripts/revision/49_build_oof_group_sidecar.py", self.source)
        self.assertIn("def ensure_oof_group_sidecar():", self.source)
        self.assertIn("Refreshing the strict OOF freeze metadata on CPU", self.source)
        self.assertIn(
            "oof_inference_required = bool(FORCE_RERUN_OOF or not oof_core_available)",
            self.source,
        )
        self.assertIn("GPU inference was intentionally not started", self.source)


if __name__ == "__main__":
    unittest.main()
