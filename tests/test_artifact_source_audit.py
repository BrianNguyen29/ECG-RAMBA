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

    def test_heavy_notebooks_use_canonical_drive_for_resumable_state(self):
        notebook_dir = Path(__file__).resolve().parents[1] / "notebooks"
        notebook04 = json.loads(
            (notebook_dir / "04_baselines_and_component_checks.ipynb").read_text(
                encoding="utf-8"
            )
        )
        source04 = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook04.get("cells", [])
        )
        for token in (
            "CANONICAL_FOLD_CACHE_DIR = MIRROR_REVISION_ROOT / 'predictions' / 'folds'",
            "CANONICAL_CHECKPOINT_ROOT = MIRROR_REVISION_ROOT / 'experimental'",
            "--fold-cache-dir",
            "--reuse-checkpoints",
            "Durable command log:",
        ):
            self.assertIn(token, source04)

        notebook05 = json.loads(
            (notebook_dir / "05_hrv_domain_and_robustness.ipynb").read_text(
                encoding="utf-8"
            )
        )
        source05 = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook05.get("cells", [])
        )
        self.assertIn(
            "CANONICAL_CHECKPOINT_ROOT = MIRROR_REVISION_ROOT / 'experimental'",
            source05,
        )
        self.assertIn("--raw-mamba-checkpoint-dir", source05)
        self.assertNotIn("def restore_learned_comparator_checkpoints_for_05", source05)
        self.assertIn("Durable command log:", source05)

    def test_all_active_notebooks_fail_fast_on_stale_drive_mount(self):
        notebook_dir = Path(__file__).resolve().parents[1] / "notebooks"
        active = [
            "00_colab_bootstrap.ipynb",
            "01_a0_protocol_audit.ipynb",
            "02_predictions_and_external_eval.ipynb",
            "02a_retrain_best_ema.ipynb",
            "03_calibration_and_ci.ipynb",
            "04_baselines_and_component_checks.ipynb",
            "05_hrv_domain_and_robustness.ipynb",
            "06_pooling_and_representation.ipynb",
            "07_results_freeze.ipynb",
        ]
        for name in active:
            notebook = json.loads((notebook_dir / name).read_text(encoding="utf-8"))
            source = "\n".join(
                "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
            )
            with self.subTest(notebook=name):
                self.assertIn("def _drive_root_ready", source)
                self.assertIn("force_remount=True", source)
                self.assertTrue(
                    "Google Drive root is not readable" in source
                    or "Google Drive root is not visible" in source
                )
                self.assertIn(
                    "MIRROR_REVISION_ROOT = DRIVE_ROOT / 'revision_artifacts' / 'reports' / 'revision'",
                    source,
                )

    def test_notebook05_uses_matching_minirocket_robustness_reference(self):
        notebook = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "notebooks"
                / "05_hrv_domain_and_robustness.ipynb"
            ).read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        self.assertIn(
            "'minirocket': 'robustness_minirocket_clean_ref_predictions.npz'",
            source,
        )
        self.assertIn("source_pairwise_sha256", source)
        self.assertIn("checkpoint_contract", source)

    def test_notebook07_treats_final_evidence_tables_as_output_only_snapshot(self):
        notebook = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "notebooks"
                / "07_results_freeze.ipynb"
            ).read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        self.assertNotIn("flat_source", source)
        self.assertNotIn("final_evidence_tables:{flat_source}", source)
        self.assertIn("pipeline_input_allowed", source)
        self.assertIn("final_evidence_export_manifest.json", source)
        self.assertIn("Final export source is not current in canonical mirror", source)

    def test_notebook00_reads_storage_audit_from_canonical_drive(self):
        notebook = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "notebooks"
                / "00_colab_bootstrap.ipynb"
            ).read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        self.assertIn(
            "storage_audit_json = CANONICAL_REVISION_MIRROR / 'metrics' / 'pipeline_storage_audit.json'",
            source,
        )
        self.assertNotIn(
            "REPO_DIR / 'reports/revision/metrics/pipeline_storage_audit.json'",
            source,
        )

    def test_notebook_restore_paths_require_canonical_manifest_authentication(self):
        notebook_dir = Path(__file__).resolve().parents[1] / "notebooks"
        sources = {}
        for name in (
            "02_predictions_and_external_eval.ipynb",
            "03_calibration_and_ci.ipynb",
            "04_baselines_and_component_checks.ipynb",
            "05_hrv_domain_and_robustness.ipynb",
            "06_pooling_and_representation.ipynb",
            "07_results_freeze.ipynb",
        ):
            notebook = json.loads((notebook_dir / name).read_text(encoding="utf-8"))
            sources[name] = "\n".join(
                "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
            )
            with self.subTest(notebook=name):
                self.assertNotIn("without a manifest row", sources[name])
                self.assertNotIn("Trying direct path fallback", sources[name])

        self.assertIn("Canonical mirror checksum mismatch", sources["02_predictions_and_external_eval.ipynb"])
        self.assertIn("_restore_verified_revision_artifact_04", sources["04_baselines_and_component_checks.ipynb"])
        self.assertIn("_restore_verified_revision_artifact_05", sources["05_hrv_domain_and_robustness.ipynb"])
        self.assertIn("verify_active_final_inputs_against_mirror", sources["07_results_freeze.ipynb"])

    def test_notebook02_direct_run_locks_current_oof_and_external_contracts(self):
        notebook_path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "02_predictions_and_external_eval.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )

        self.assertIn("--check-existing-freeze", source)
        self.assertNotIn("--check-only", source)
        self.assertIn("metric_implementation_sha256", source)
        self.assertIn("positive_label_multilabel_reduction", source)
        self.assertIn("Verified final external comparator artifacts were not reusable", source)
        self.assertIn("EXTERNAL_GATE_DATASETS = 'ptbxl,georgia,cpsc2021'", source)
        self.assertIn("EXTERNAL_GATE_STRICT = True", source)
        self.assertIn("EXTERNAL_GATE_INPUT_PATHS", source)
        self.assertIn("external_gate_input_restore.log", source)
        self.assertIn("External gate input contract: all 15 artifacts are present", source)
        self.assertIn("RUN_LEGACY_ROW_SPLIT_SCORE_CALIBRATION = False", source)
        self.assertIn("revision_artifacts' / 'reports' / 'revision", source)
        self.assertIn("def require_gpu_inference_runtime", source)
        self.assertIn("modules.extend(['mamba_ssm', 'causal_conv1d'])", source)
        self.assertIn("require_gpu_inference_runtime('Canonical final_ema OOF export')", source)
        self.assertIn("require_gpu_inference_runtime('External Full-model export')", source)
        self.assertIn("require_gpu_inference_runtime('PTB-XL fold 9 Full-model export')", source)
        self.assertIn("BATCH_SIZE = 256", source)
        self.assertIn("NUM_WORKERS = 2", source)

        comparator_runner = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "revision"
            / "31_generate_external_comparator_predictions.py"
        ).read_text(encoding="utf-8")
        self.assertIn("Verified final external comparator artifacts were not reusable", comparator_runner)
        self.assertIn('device.type != "cuda" and str(args.device).lower() == "auto"', comparator_runner)

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
