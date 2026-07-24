import importlib.util
import json
from pathlib import Path
import tempfile
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

    def test_windows_drive_mount_bridge_uses_official_cli(self):
        source = (
            ROOT / "scripts" / "colab_cli" / "mount_drive_interactive.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("drivemount", source)
        self.assertIn("RedirectStandardInput = $true", source)
        self.assertIn("^https://accounts\\.google\\.com/", source)

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
                self.pipeline.preserve_executed_notebook(
                    source,
                    "nb00_cpu",
                    "run-1",
                )
                destination = (
                    root / "logs" / "nb00_cpu" / "run-1_output.ipynb"
                )
                self.assertEqual(
                    destination.read_text(encoding="utf-8"),
                    '{"executed": true}',
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
