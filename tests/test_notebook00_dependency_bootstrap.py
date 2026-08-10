import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "00_colab_bootstrap.ipynb"

class Notebook00DependencyBootstrapTests(unittest.TestCase):
    def test_cpu_bootstrap_skips_optional_dependencies_by_default(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cell = next(
            "".join(candidate.get("source", []))
            for candidate in notebook["cells"]
            if "FORENSIC_NOTEBOOK00_DEPENDENCY_BOOTSTRAP_V2"
            in "".join(candidate.get("source", []))
        )
        for token in (
            "ECG_RAMBA_INSTALL_NOTEBOOK00_OPTIONAL_DEPS",
            "NOTEBOOK00_OPTIONAL_DEPENDENCIES",
            "--prefer-binary",
            "--timeout', '60'",
            "timeout=900",
            "Notebook 02 owns version-pinned execution dependencies.",
        ):
            self.assertIn(token, cell)
        self.assertNotIn("!pip install -q", cell)
        compile(cell, f"{NOTEBOOK_PATH}:dependency_bootstrap", "exec")

if __name__ == "__main__":
    unittest.main()
