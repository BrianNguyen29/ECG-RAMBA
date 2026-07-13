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
            "_prediction_payload_matches_current_oof",
            "np.array_equal(np.asarray(candidate['y_true']",
            "np.array_equal(np.asarray(candidate['record_id']",
            "np.array_equal(np.asarray(candidate['fold_id']",
            "np.array_equal(np.asarray(candidate['class_names']",
            "prediction_contract_ok",
        ):
            self.assertIn(token, self.source)

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


if __name__ == "__main__":
    unittest.main()
