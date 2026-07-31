import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "colab_cli_pipeline.json"
STAGE_MODULE_PATH = ROOT / "scripts" / "colab_cli" / "stage_notebook.py"
PIPELINE_MODULE_PATH = ROOT / "scripts" / "colab_cli" / "pipeline.py"


def load_stage_module():
    spec = importlib.util.spec_from_file_location("stage_notebook", STAGE_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_pipeline_module(stage_module):
    import sys

    original = sys.modules.get("stage_notebook")
    sys.modules["stage_notebook"] = stage_module
    try:
        spec = importlib.util.spec_from_file_location(
            "colab_cli_pipeline", PIPELINE_MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if original is None:
            sys.modules.pop("stage_notebook", None)
        else:
            sys.modules["stage_notebook"] = original


class ColabCliPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_stage_module()
        cls.pipeline = load_pipeline_module(cls.module)
        cls.manifest = cls.module.load_manifest(MANIFEST_PATH)

    def test_manifest_sources_and_dependency_order_are_valid(self):
        self.assertEqual(
            self.module.validate_manifest_sources(ROOT, self.manifest),
            [],
        )

    def test_notebook_sources_match_immutable_authority_tag(self):
        self.assertEqual(
            self.module.validate_authority_sources(ROOT, self.manifest),
            [],
        )

    def test_retrain_stage_is_disabled(self):
        stage = self.module.stage_by_id(
            self.manifest, "nb02a_retrain_a100"
        )
        self.assertFalse(stage["enabled"])

    def test_oauth2_is_the_default_authentication_mode(self):
        self.assertEqual(self.manifest["default_auth"], "oauth2")

    def test_notebook07_uses_supported_snapshot_submission_mode(self):
        stage = self.module.stage_by_id(self.manifest, "nb07_cpu")
        self.assertEqual(
            stage["environment"]["ECG_RAMBA_SUBMISSION_MODE"],
            "snapshot",
        )

    def test_windows_drive_mount_bridge_uses_official_cli(self):
        source = (
            ROOT / "scripts" / "colab_cli" / "mount_drive_interactive.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("drivemount", source)
        self.assertIn("script -qefc", source)
        self.assertIn("/dev/tty", source)
        self.assertIn("$ProcessInfo.ArgumentList.Add", source)
        self.assertIn("RedirectStandardInput = $true", source)
        self.assertIn("^https://accounts\\.google\\.com/", source)
        self.assertIn("AutoConfirmAfterSeconds", source)
        self.assertIn("/content/drive/MyDrive", source)
        self.assertIn("$ObservedMountError", source)
        self.assertIn("$VerifyExitCode", source)

    def test_cpu_feature_stage_does_not_include_gpu_inference(self):
        stage = self.module.stage_by_id(
            self.manifest, "nb02_features_cpu"
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage.ipynb"
            notebook = self.module.build_stage_notebook(
                ROOT, self.manifest, stage, output
            )
        headings = [
            self.module.markdown_heading(cell)
            for cell in notebook["cells"]
            if self.module.markdown_heading(cell)
        ]
        self.assertIn("## CPU External Feature Preparation", headings)
        self.assertIn("## PTB-XL Fold 9 CPU Feature Preparation", headings)
        self.assertNotIn("## GPU External Prediction Inference", headings)
        self.assertNotIn(
            "## External Learned-Comparator Zero-Target-Label Inference",
            headings,
        )

    def test_a100_stage_disables_cpu_feature_cells_at_runtime(self):
        stage = self.module.stage_by_id(self.manifest, "nb02_a100")
        self.assertEqual(
            stage["environment"]["ECG_RAMBA_EXTERNAL_FEATURE_PROFILE"], "off"
        )
        self.assertEqual(
            stage["environment"]["ECG_RAMBA_RUN_PTBXL_FOLD9_FEATURES"], "0"
        )
        self.assertEqual(stage["hardware"], "a100")

    def test_adaptation_gpu_fallback_is_explicit_and_dependency_complete(self):
        stage = self.module.stage_by_id(
            self.manifest, "nb02_adaptation_gpu"
        )
        self.assertFalse(stage["enabled"])
        self.assertEqual(stage["hardware"], "gpu")
        self.assertEqual(stage["environment"]["ECG_RAMBA_GPU_FALLBACK"], "1")
        required_sections = {
            "## External Learned-Comparator Zero-Target-Label Inference",
            "## Paired External Comparator Audit",
            "## External Frozen-Encoder Representation Extraction",
            "## True Few-Shot Frozen-Encoder Head Adaptation",
        }
        self.assertTrue(required_sections.issubset(set(stage["sections"])))
        self.assertIn("## GPU External Prediction Inference", stage["sections"])
        self.assertIn(
            "## PTB-XL Fold 9 Adaptation-Pool Inference",
            stage["sections"],
        )
        self.assertEqual(
            stage["environment"]["ECG_RAMBA_EXTERNAL_RUN_PROFILE"],
            "cpu_gate_all",
        )
        with tempfile.TemporaryDirectory() as directory:
            notebook = self.module.build_stage_notebook(
                ROOT,
                self.manifest,
                stage,
                Path(directory) / "stage.ipynb",
            )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("def _comparator_artifacts", source)
        self.assertIn("external_archives = {}", source)
        self.assertIn("import pandas as pd", source)
        self.assertIn("def _restore_report_artifact", source)

    def test_adaptation_cpu_reuse_omits_model_installer_and_keeps_gates(self):
        stage = self.module.stage_by_id(
            self.manifest, "nb02_adaptation_cpu_reuse"
        )
        self.assertFalse(stage["enabled"])
        self.assertEqual(stage["hardware"], "cpu")
        self.assertEqual(stage["environment"]["ECG_RAMBA_CPU_REUSE_ONLY"], "1")
        self.assertNotIn("## Install Model Dependencies", stage["sections"])
        self.assertIn(
            "## External Learned-Comparator Zero-Target-Label Inference",
            stage["sections"],
        )
        self.assertIn(
            "## External Frozen-Encoder Representation Extraction",
            stage["sections"],
        )
        self.assertIn("## GPU External Prediction Inference", stage["sections"])
        self.assertIn(
            "## PTB-XL Fold 9 Adaptation-Pool Inference",
            stage["sections"],
        )
        self.assertEqual(
            stage["environment"]["ECG_RAMBA_EXTERNAL_RUN_PROFILE"],
            "cpu_gate_all",
        )
        with tempfile.TemporaryDirectory() as directory:
            notebook = self.module.build_stage_notebook(
                ROOT,
                self.manifest,
                stage,
                Path(directory) / "stage.ipynb",
            )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("external_archives = {}", source)
        self.assertIn("import pandas as pd", source)
        self.assertIn("def _restore_report_artifact", source)
        self.assertIn("PTBXL_FOLD9_REQUIRED = [", source)

    def test_adaptation_a100_includes_cache_restore_dependency_section(self):
        stage = self.module.stage_by_id(
            self.manifest, "nb02_adaptation_a100"
        )
        self.assertIn("## GPU External Prediction Inference", stage["sections"])
        self.assertIn(
            "## PTB-XL Fold 9 Adaptation-Pool Inference",
            stage["sections"],
        )
        self.assertEqual(
            stage["environment"]["ECG_RAMBA_EXTERNAL_RUN_PROFILE"],
            "cpu_gate_all",
        )

    def test_generated_notebook_is_clean_and_source_bound(self):
        stage = self.module.stage_by_id(self.manifest, "nb03_cpu")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage.ipynb"
            notebook = self.module.build_stage_notebook(
                ROOT, self.manifest, stage, output
            )
            reloaded = json.loads(output.read_text(encoding="utf-8"))
        contract = notebook["metadata"]["ecg_ramba_colab_cli"]
        self.assertEqual(contract["stage_id"], "nb03_cpu")
        self.assertEqual(len(contract["source_notebook_sha256"]), 64)
        self.assertEqual(len(contract["pipeline_manifest_sha256"]), 64)
        self.assertEqual(len(contract["stage_builder_sha256"]), 64)
        self.assertEqual(len(contract["pipeline_launcher_sha256"]), 64)
        self.assertEqual(reloaded["metadata"]["ecg_ramba_colab_cli"], contract)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_all_enabled_notebooks_finish_with_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            for stage in self.manifest["stages"]:
                if not stage["enabled"]:
                    continue
                notebook = self.module.build_stage_notebook(
                    ROOT,
                    self.manifest,
                    stage,
                    Path(directory) / f"{stage['id']}.ipynb",
                )
                final_source = "".join(notebook["cells"][-1]["source"])
                self.assertIn(
                    f"ECG_RAMBA_COLAB_CLI_STAGE_COMPLETE={stage['id']}",
                    final_source,
                )

    def test_completion_log_requires_exact_stage_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "run.log"
            log_path.write_text(
                "ECG_RAMBA_COLAB_CLI_STAGE_COMPLETE=other-stage\n",
                encoding="utf-8",
            )
            self.assertFalse(
                self.pipeline.completed_stage_log(log_path, "nb00_cpu")
            )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("ECG_RAMBA_COLAB_CLI_STAGE_COMPLETE=nb00_cpu\n")
            self.assertTrue(
                self.pipeline.completed_stage_log(log_path, "nb00_cpu")
            )

    def test_session_exists_rejects_cli_not_found_with_zero_exit_code(self):
        missing = SimpleNamespace(
            returncode=0,
            stdout="[ecgr-missing] Session not found.\n",
            stderr="",
        )
        active = SimpleNamespace(
            returncode=0,
            stdout=(
                "[ecgr-active] gpu-a100 | Hardware: A100 | "
                "Variant: GPU | Status: IDLE\n"
            ),
            stderr="",
        )
        with mock.patch.object(self.pipeline, "run_capture", return_value=missing):
            self.assertFalse(
                self.pipeline.session_exists(["colab"], "ecgr-missing")
            )
        with mock.patch.object(self.pipeline, "run_capture", return_value=active):
            self.assertTrue(
                self.pipeline.session_exists(["colab"], "ecgr-active")
            )

    def test_executed_notebook_is_preserved_with_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "stage.ipynb"
            executed = root / "stage_output.ipynb"
            source.write_text("{}", encoding="utf-8")
            executed.write_text('{"executed": true}', encoding="utf-8")
            with mock.patch.object(
                self.pipeline, "LOCAL_LOG_ROOT", root / "logs"
            ):
                preserved = self.pipeline.preserve_executed_notebook(
                    source,
                    "nb00_cpu",
                    "run-1",
                )
                destination = (
                    root / "logs" / "nb00_cpu" / "run-1_output.ipynb"
                )
                self.assertEqual(preserved, destination)
                self.assertEqual(
                    destination.read_text(encoding="utf-8"),
                    '{"executed": true}',
                )

    def test_executed_notebook_error_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            notebook_path = Path(directory) / "failed.ipynb"
            notebook_path.write_text(
                json.dumps(
                    {
                        "cells": [
                            {
                                "cell_type": "code",
                                "outputs": [
                                    {
                                        "output_type": "error",
                                        "ename": "RuntimeError",
                                        "evalue": "contract mismatch",
                                        "traceback": [],
                                    }
                                ],
                            },
                            {
                                "cell_type": "code",
                                "outputs": [
                                    {
                                        "output_type": "stream",
                                        "name": "stdout",
                                        "text": "completion marker",
                                    }
                                ],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                self.pipeline.executed_notebook_errors(notebook_path),
                ["cell 0: RuntimeError: contract mismatch"],
            )

    def test_clean_executed_notebook_has_no_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            notebook_path = Path(directory) / "clean.ipynb"
            notebook_path.write_text(
                json.dumps(
                    {
                        "cells": [
                            {
                                "cell_type": "code",
                                "outputs": [
                                    {
                                        "output_type": "stream",
                                        "name": "stdout",
                                        "text": "ok",
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                self.pipeline.executed_notebook_errors(notebook_path),
                [],
            )

    def test_execute_stage_rejects_cell_errors_even_with_zero_exit(self):
        stage = {
            "id": "nb03_cpu",
            "enabled": True,
            "hardware": "cpu",
            "timeout_seconds": 60,
        }
        args = SimpleNamespace(
            include_disabled=False,
            build_root=Path("build"),
            auth="oauth2",
            session="existing-session",
            dry_run=False,
            no_mount=True,
            remount=False,
            keep=True,
        )
        with (
            mock.patch.object(
                self.pipeline,
                "build_stage",
                return_value=Path("nb03_cpu.ipynb"),
            ),
            mock.patch.object(
                self.pipeline,
                "colab_base",
                return_value=["colab", "--auth=oauth2"],
            ),
            mock.patch.object(
                self.pipeline,
                "session_exists",
                return_value=True,
            ),
            mock.patch.object(self.pipeline, "run_stream", return_value=0),
            mock.patch.object(
                self.pipeline,
                "preserve_executed_notebook",
                return_value=Path("nb03_cpu_output.ipynb"),
            ),
            mock.patch.object(self.pipeline, "export_session_log"),
            mock.patch.object(
                self.pipeline,
                "executed_notebook_errors",
                return_value=["cell 3: RuntimeError: stale contract"],
            ),
        ):
            self.assertEqual(
                self.pipeline.execute_stage(self.manifest, stage, args),
                4,
            )

    def test_adc_scope_preflight_detects_missing_colaboratory_scope(self):
        complete = "\n".join(sorted(self.pipeline.REQUIRED_COLAB_SCOPES))
        missing = complete.replace(
            "https://www.googleapis.com/auth/colaboratory", ""
        )
        with mock.patch.object(
            self.pipeline,
            "run_capture",
            return_value=type(
                "Result", (), {"returncode": 0, "stdout": missing}
            )(),
        ):
            self.assertEqual(
                self.pipeline.validate_auth(["colab", "--auth=adc"], "adc"),
                2,
            )
        with mock.patch.object(
            self.pipeline,
            "run_capture",
            return_value=type(
                "Result", (), {"returncode": 0, "stdout": complete}
            )(),
        ):
            self.assertEqual(
                self.pipeline.validate_auth(["colab", "--auth=adc"], "adc"),
                0,
            )

    def test_oauth2_scope_preflight_detects_missing_colaboratory_scope(self):
        complete = "\n".join(sorted(self.pipeline.REQUIRED_COLAB_SCOPES))
        missing = complete.replace(
            "https://www.googleapis.com/auth/colaboratory", ""
        )
        with mock.patch.object(
            self.pipeline,
            "run_capture",
            return_value=type(
                "Result", (), {"returncode": 0, "stdout": missing}
            )(),
        ):
            self.assertEqual(
                self.pipeline.validate_auth(
                    ["colab", "--auth=oauth2"], "oauth2"
                ),
                2,
            )
        with mock.patch.object(
            self.pipeline,
            "run_capture",
            return_value=type(
                "Result", (), {"returncode": 0, "stdout": complete}
            )(),
        ):
            self.assertEqual(
                self.pipeline.validate_auth(
                    ["colab", "--auth=oauth2"], "oauth2"
                ),
                0,
            )

    def test_run_all_namespace_does_not_require_session_argument(self):
        stage = self.module.stage_by_id(self.manifest, "nb00_cpu")
        namespace = type(
            "Args",
            (),
            {
                "include_disabled": False,
                "build_root": Path(tempfile.gettempdir()) / "ecg-cli-test",
                "auth": "oauth2",
                "dry_run": True,
                "keep": False,
                "no_mount": False,
                "remount": False,
            },
        )()
        with mock.patch.object(
            self.pipeline, "colab_base", return_value=["colab", "--auth=oauth2"]
        ):
            self.assertEqual(
                self.pipeline.execute_stage(self.manifest, stage, namespace),
                0,
            )


if __name__ == "__main__":
    unittest.main()
