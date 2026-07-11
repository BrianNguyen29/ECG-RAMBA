import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "revision" / "37_artifact_source_audit.py"
SPEC = importlib.util.spec_from_file_location("artifact_source_audit", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ArtifactSourceAuditTest(unittest.TestCase):
    def test_notebooks_use_legacy_reports_only_for_bootstrap_audit(self):
        notebook_dir = Path(__file__).resolve().parents[1] / "notebooks"
        legacy_fragment = "ECG-RAMBA' / 'reports' / 'revision'"
        violations = []
        for path in sorted(notebook_dir.glob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook.get("cells", [])):
                source = "".join(cell.get("source", []))
                if legacy_fragment not in source:
                    continue
                allowed = (
                    path.name == "00_colab_bootstrap.ipynb"
                    and "LEGACY_DRIVE_REVISION" in source
                    and "37_artifact_source_audit.py" in source
                )
                if not allowed:
                    violations.append(f"{path.name}:cell{index}")
        self.assertEqual(violations, [])

    def test_classifies_identical_unique_and_conflicting_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "canonical"
            legacy = root / "legacy"
            canonical.mkdir()
            legacy.mkdir()

            (canonical / "same.txt").write_text("same", encoding="utf-8")
            (legacy / "same.txt").write_text("same", encoding="utf-8")
            (canonical / "canonical.txt").write_text("new", encoding="utf-8")
            (legacy / "legacy.txt").write_text("old", encoding="utf-8")
            (canonical / "conflict.txt").write_text("new", encoding="utf-8")
            (legacy / "conflict.txt").write_text("old", encoding="utf-8")
            (canonical / "logs").mkdir()
            (legacy / "logs").mkdir()
            (canonical / "logs" / "run.log").write_text("new", encoding="utf-8")
            (legacy / "logs" / "run.log").write_text("old", encoding="utf-8")

            rows = MODULE.audit_sources(canonical, legacy)
            statuses = {row.relative_path: row.status for row in rows}

            self.assertEqual(statuses["same.txt"], "identical")
            self.assertEqual(statuses["canonical.txt"], "canonical_only")
            self.assertEqual(statuses["legacy.txt"], "legacy_only")
            self.assertEqual(statuses["conflict.txt"], "conflict")
            self.assertNotIn("logs/run.log", statuses)

    def test_migration_refuses_conflicts_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "canonical"
            legacy = root / "legacy"
            canonical.mkdir()
            legacy.mkdir()
            (canonical / "value.txt").write_text("new", encoding="utf-8")
            (legacy / "value.txt").write_text("old", encoding="utf-8")

            rows = MODULE.audit_sources(canonical, legacy)
            with self.assertRaisesRegex(RuntimeError, "conflicts exist"):
                MODULE.migrate_legacy_only(rows, legacy, canonical, False)


if __name__ == "__main__":
    unittest.main()
