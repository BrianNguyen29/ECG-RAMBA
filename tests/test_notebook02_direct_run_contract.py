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


if __name__ == "__main__":
    unittest.main()
