import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORENSIC_INTEGRATOR = PROJECT_ROOT / "scripts" / "revision" / "48_integrate_forensic_audit_notebooks.py"
PIPELINE_NOTEBOOKS = (
    "00_colab_bootstrap.ipynb",
    "01_a0_protocol_audit.ipynb",
    "02_predictions_and_external_eval.ipynb",
    "02a_retrain_best_ema.ipynb",
    "03_calibration_and_ci.ipynb",
    "04_baselines_and_component_checks.ipynb",
    "05_hrv_domain_and_robustness.ipynb",
    "06_pooling_and_representation.ipynb",
    "07_results_freeze.ipynb",
)


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


def literal_integrator_assignment(assignment_name: str):
    tree = ast.parse(FORENSIC_INTEGRATOR.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == assignment_name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Integrator assignment not found: {assignment_name}")


def literal_notebook_assignment(name: str, assignment_name: str):
    cells, _ = notebook_source(name)
    for source in cells:
        if assignment_name not in source:
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name) and target.id == assignment_name
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise AssertionError(f"Notebook assignment not found: {name}:{assignment_name}")


def compilable_notebook_source(source: str) -> str:
    """Replace IPython line magics with Python no-ops before syntax validation."""
    lines = []
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            indentation = line[: len(line) - len(stripped)]
            newline = "\n" if line.endswith("\n") else ""
            lines.append(indentation + "pass  # IPython line magic" + newline)
        else:
            lines.append(line)
    return "".join(lines)


def authority_block_source(name: str, *, occurrence: int = 0) -> str:
    cells, _ = notebook_source(name)
    blocks = []
    start_marker = "# BEGIN FORENSIC CODE AUTHORITY PIN"
    end_marker = "# END FORENSIC CODE AUTHORITY PIN"
    for cell in cells:
        if start_marker not in cell:
            continue
        start = cell.index(start_marker)
        end = cell.index(end_marker, start) + len(end_marker)
        blocks.append(cell[start:end] + "\n")
    return blocks[occurrence]


class Notebook050607DirectRunContractTests(unittest.TestCase):
    def test_notebook_source_token_preflights_match_the_pinned_repository(self):
        checks = (
            ("00_colab_bootstrap.ipynb", "required_source_tokens"),
            ("02_predictions_and_external_eval.ipynb", "REVISION_TOKEN_REQUIREMENTS"),
        )
        for notebook, assignment in checks:
            requirements = literal_notebook_assignment(notebook, assignment)
            for relative, tokens in requirements.items():
                path = PROJECT_ROOT / relative
                self.assertTrue(path.is_file(), f"{notebook}: missing {relative}")
                source = path.read_text(encoding="utf-8", errors="replace")
                missing = [token for token in tokens if token not in source]
                self.assertEqual(missing, [], f"{notebook}: stale requirements for {relative}")
        for notebook in ("00_colab_bootstrap.ipynb", "02_predictions_and_external_eval.ipynb"):
            _, source = notebook_source(notebook)
            self.assertNotIn("pre_specified_before_test_metric_evaluation", source)
            self.assertNotIn("pre_specified_0.10_no_test_set_budget_selection", source)

    def test_notebook_v4_structure_is_complete(self):
        for notebook in (
            "03_calibration_and_ci.ipynb",
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
        for notebook in PIPELINE_NOTEBOOKS:
            cells, _ = notebook_source(notebook)
            for index, source in enumerate(cells):
                compile(compilable_notebook_source(source), f"{notebook}:cell_{index}", "exec")

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
            "paired_record_resample_presorted_rank_sparse_ece_weighted_counts_v2",
            "weighted_resample_metric",
            "ece_context",
            "bootstrap_record_counts",
            "nominal_95_percentile_paired_record_bootstrap_unadjusted",
            "full_nominal_95ci_less_degraded",
            "nominal_95ci_inconclusive_stressed_difference",
            "fixed_trained_folds_and_checkpoints_not_retrained_within_bootstrap",
            "METRIC_CACHE_SCHEMA_VERSION = ROBUSTNESS_METRIC_CACHE_SCHEMA_VERSION",
            "rank_calibration_omit_single_resampled_class_f1_keeps_all_labels_zero_division_zero",
            "load_bootstrap_independence_contract",
            "expected_stress_spec",
            "load_validated_clean_artifact",
            "load_validated_stress_artifact",
            "artifact_status",
            "prediction artifact provenance grid is incomplete or invalid",
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
            "MAMBA_INSTALLER_CAPABILITY = 'ecg_ramba_mamba_installer_v1'",
            "candidate_count={len(installer_candidates)}",
            "len(installer_candidates) == 1",
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
            "Ignoring unauthenticated/stale representation cache rows",
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

    def test_notebook06_physological_probe_is_provenance_bound_and_reusable(self):
        _, source = notebook_source("06_pooling_and_representation.ipynb")
        for token in (
            "scripts/revision/44_physiological_interval_probe.py",
            "fold_held_out_measured_physiological_interval_probe_v3",
            "RUNNER_SOURCE_PATH",
            "independent_of_ecg_ramba_feature_cache",
            "metadata_sha256",
            "--embedding-manifest reports/revision/manifests/representation_embedding_manifest.json",
            "RUN_PHYSIOLOGICAL_INTERVAL_PROBE = 'auto'",
            "Physiological probe reusable:",
            "existing_manifest.get('runner')",
            "existing_inputs.get('embedding_manifest')",
            "physiology_common_output_relatives",
            "physiology_expected_output_relatives",
            "authenticated_output_paths",
            "table_physiological_interval_probe_contrasts.csv",
            "table_physiological_interval_probe.tex",
        ):
            self.assertIn(token, source)

    def test_notebook04_structured_ablation_uses_canonical_caches_and_source_guard(self):
        _, source = notebook_source("04_baselines_and_component_checks.ipynb")
        for token in (
            "scripts/revision/43_structured_ablation_5fold.py",
            "scripts/revision/01_generate_predictions.py",
            "scripts/train.py",
            "src/model.py",
            "src/features.py",
            "matched_retrained_structured_ablation_5fold_v3",
            "fold_seeded_full_reference_overlap_v1",
            "structured_ablation_metric_cache",
            "Structured-ablation OOF package reusable:",
            "expected_checkpoint_hashes",
            "prediction_contract.get('sha256')",
        ):
            self.assertIn(token, source)

    def test_notebook03_authenticates_every_matched_calibration_input(self):
        _, source = notebook_source("03_calibration_and_ci.ipynb")
        for token in (
            "PROTOCOL = MATCHED_CALIBRATION_PROTOCOL",
            "cannot reverse within-fold score ordering",
            "fully nested deploy-time calibration estimate",
            "require_canonical_matched_input",
            "Canonical matched calibration inputs authenticated:",
            "Deferred matched calibration audit until Notebook 04 publishes",
            "matched_required_prediction_paths",
            "matched_optional_prediction_paths",
            "Matched calibration input is not authenticated by canonical Drive",
            "Active matched input differs from canonical Drive",
        ):
            self.assertIn(token, source)

    def test_notebook06_finds_current_notebook02_mamba_installer(self):
        payload = notebook_payload("02_predictions_and_external_eval.ipynb")
        capability = "MAMBA_INSTALLER_CAPABILITY = 'ecg_ramba_mamba_installer_v1'"
        schema = "MAMBA_INSTALLER_SCHEMA_VERSION = 1"
        candidates = [
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell.get("cell_type") == "code"
            and capability in "".join(cell.get("source", []))
            and schema in "".join(cell.get("source", []))
        ]
        self.assertEqual(len(candidates), 1)
        self.assertIn("INSTALL_MODEL_DEPS = 'auto'", candidates[0])

    def test_notebook02_uses_authority_bound_capability_preflight(self):
        _, source = notebook_source("02_predictions_and_external_eval.ipynb")
        self.assertIn("# BEGIN FORENSIC NOTEBOOK 02 CAPABILITY PREFLIGHT", source)
        self.assertIn("NOTEBOOK_02_EXTERNAL_EXPORT_CAPABILITY", source)
        self.assertIn("external_export_full10s_grouped_v1", source)
        self.assertIn("NOTEBOOK_02_EXTERNAL_GATE_CAPABILITY", source)
        self.assertIn("external_gate_full10s_grouped_v1", source)
        self.assertIn("_compat_ast.parse", source)
        self.assertNotIn("raw.githubusercontent.com/BrianNguyen29/ECG-RAMBA", source)
        self.assertNotIn("annotation_aligned_nonoverlapping_10s_windows_majority_af_or_normal", source)
        self.assertNotIn("GATE_SCHEMA_VERSION = 4", source)

        export_source = (PROJECT_ROOT / "scripts/revision/03_generate_external_predictions.py").read_text(encoding="utf-8")
        gate_source = (PROJECT_ROOT / "scripts/revision/18_external_protocol_gate.py").read_text(encoding="utf-8")
        self.assertIn(
            'NOTEBOOK_02_EXTERNAL_EXPORT_CAPABILITY = "external_export_full10s_grouped_v1"',
            export_source,
        )
        self.assertIn("NOTEBOOK_02_EXTERNAL_EXPORT_SCHEMA_VERSION = 1", export_source)
        self.assertIn(
            'NOTEBOOK_02_EXTERNAL_GATE_CAPABILITY = "external_gate_full10s_grouped_v1"',
            gate_source,
        )
        self.assertIn("NOTEBOOK_02_EXTERNAL_GATE_SCHEMA_VERSION = 1", gate_source)

    def test_notebook02_shares_the_exact_sized_cpsc_window_cache(self):
        _, source = notebook_source("02_predictions_and_external_eval.ipynb")
        cache_name = "cpsc2021_preprocessed_windows_source_bound_v3.npy"
        self.assertGreaterEqual(source.count(cache_name), 2)
        self.assertIn("EXTERNAL_CPSC_SIGNAL_CACHE", source)
        self.assertIn(
            "--cpsc-signal-memmap \"{EXTERNAL_CPSC_SIGNAL_CACHE}\"",
            source,
        )
        self.assertIn("CPSC_EXACT_ELIGIBLE_WINDOW_CAPACITY_CAPABILITY", source)
        self.assertNotIn("cpsc2021_preprocessed_windows_source_bound_v2.npy", source)

    def test_installer_discovery_uses_exact_capability_schema_pair(self):
        _, notebook02_source = notebook_source("02_predictions_and_external_eval.ipynb")
        markers = (
            "MAMBA_INSTALLER_CAPABILITY = 'ecg_ramba_mamba_installer_v1'",
            "MAMBA_INSTALLER_SCHEMA_VERSION = 1",
            "BASE_INSTALLER_CAPABILITY = 'ecg_ramba_base_installer_v1'",
            "BASE_INSTALLER_SCHEMA_VERSION = 1",
        )
        for marker in markers:
            self.assertEqual(notebook02_source.count(marker), 1, marker)

        for notebook in (
            "00_colab_bootstrap.ipynb",
            "02a_retrain_best_ema.ipynb",
            "05_hrv_domain_and_robustness.ipynb",
            "06_pooling_and_representation.ipynb",
        ):
            _, source = notebook_source(notebook)
            self.assertIn(markers[0], source)
            self.assertIn(markers[1], source)
            self.assertTrue(
                "expected exactly one" in source
                or "len(installer_candidates) == 1" in source,
                notebook,
            )
        _, retrain_source = notebook_source("02a_retrain_best_ema.ipynb")
        self.assertIn("def canonical_installer_source(capability_marker, schema_marker):", retrain_source)
        self.assertIn("if len(candidates) != 1:", retrain_source)
        self.assertNotIn("def canonical_installer_source(*markers):", retrain_source)

    def test_code_authority_is_pinned_across_direct_runs(self):
        capability = "FORENSIC_CODE_AUTHORITY_CAPABILITY = 'canonical_versioned_git_release_v2'"
        schema = "FORENSIC_CODE_AUTHORITY_SCHEMA_VERSION = 2"
        authority_ref = literal_integrator_assignment("AUTHORITY_RELEASE_REF")
        self.assertRegex(authority_ref, r"^refs/tags/ecg-ramba-revision-\d{8}-v\d+$")
        for notebook in PIPELINE_NOTEBOOKS:
            cells, source = notebook_source(notebook)
            expected_count = 2 if notebook == "07_results_freeze.ipynb" else 1
            self.assertEqual(source.count(capability), expected_count, notebook)
            self.assertEqual(source.count(schema), expected_count, notebook)
            self.assertIn("notebook_code_authority.json", source)
            self.assertIn("git('checkout', '--detach', expected_commit)", source)
            self.assertIn("git('cat-file', '-e', expected_commit + '^{commit}')", source)
            self.assertIn("Tracked files differ from git before authority checkout", source)
            self.assertIn("verified_annotated_versioned_release_tag", source)
            self.assertIn(authority_ref, source)
            self.assertIn("Publish a new versioned tag instead of retagging", source)
            setup_cells = [cell for cell in cells if capability in cell]
            for setup in setup_cells:
                self.assertLess(setup.rfind("git pull --ff-only"), setup.index(capability))

        _, notebook00_source = notebook_source("00_colab_bootstrap.ipynb")
        self.assertIn("_AUTHORITY_BOOTSTRAP_ALLOWED = True", notebook00_source)
        self.assertIn("ECG_RAMBA_RESET_CODE_AUTHORITY", notebook00_source)
        self.assertIn("ECG_RAMBA_ROTATE_CODE_AUTHORITY_TO_BRANCH_HEAD", notebook00_source)
        self.assertIn("Implicit authority rotation to a moving branch head is disabled", notebook00_source)
        self.assertIn("Authority reset requires ECG_RAMBA_AUTHORITY_COMMIT", notebook00_source)
        self.assertNotIn("fetched_branch_head_at_initial_bootstrap", notebook00_source)
        for notebook in PIPELINE_NOTEBOOKS[1:]:
            _, source = notebook_source(notebook)
            self.assertNotIn("_AUTHORITY_BOOTSTRAP_ALLOWED = True", source)
            self.assertIn("_AUTHORITY_BOOTSTRAP_ALLOWED = False", source)

    def test_notebook00_scoped_audit_publish_repairs_direct_cpsc_cache_attestation(self):
        _, source = notebook_source("00_colab_bootstrap.ipynb")
        self.assertIn("artifact_source_audit_mirror_publish.log", source)
        self.assertIn(
            "--refresh-existing-prefix predictions/external_feature_cache",
            source,
        )
        self.assertIn(
            "--refresh-existing-prefix predictions/cpsc_window_cache/"
            "cpsc2021_preprocessed_windows_source_bound_v3.npy.contract.npz",
            source,
        )
        self.assertIn("--include-path manifests/artifact_source_audit.json", source)
        self.assertIn("--include-path tables/table_artifact_source_audit.csv", source)

    def test_code_authority_manifest_fails_closed_and_pins_moving_branch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            canonical = root / "canonical"
            repo.mkdir()
            subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, stdout=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "audit@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Audit Test"], cwd=repo, check=True)
            (repo / "authority.txt").write_text("first\n", encoding="utf-8")
            subprocess.run(["git", "add", "authority.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "first"], cwd=repo, check=True, stdout=subprocess.PIPE)
            first_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
            subprocess.run(
                ["git", "tag", "-a", "authority-v1", "-m", "authority v1", first_commit],
                cwd=repo,
                check=True,
            )

            namespace = {
                "MIRROR_REVISION_ROOT": canonical,
                "REPO_DIR": repo,
                "REPO_URL": "https://github.com/BrianNguyen29/ECG-RAMBA.git",
                "BRANCH": "main",
            }
            clean_environment = {
                "ECG_RAMBA_AUTHORITY_COMMIT": "",
                "ECG_RAMBA_RESET_CODE_AUTHORITY": "0",
                "ECG_RAMBA_AUTHORITY_REF": "refs/tags/authority-v1",
            }
            with mock.patch.dict(os.environ, clean_environment, clear=False):
                exec(authority_block_source("00_colab_bootstrap.ipynb"), namespace, namespace)
                manifest_path = canonical / "manifests" / "notebook_code_authority.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["git_commit"], first_commit)
                self.assertEqual(manifest["authority_ref"], "refs/tags/authority-v1")
                self.assertEqual(manifest["schema_version"], 2)

                subprocess.run(["git", "checkout", "-B", "main"], cwd=repo, check=True, stdout=subprocess.PIPE)
                (repo / "authority.txt").write_text("second\n", encoding="utf-8")
                subprocess.run(["git", "add", "authority.txt"], cwd=repo, check=True)
                subprocess.run(["git", "commit", "-m", "second"], cwd=repo, check=True, stdout=subprocess.PIPE)
                second_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo, text=True
                ).strip()
                self.assertNotEqual(first_commit, second_commit)

                downstream_namespace = dict(namespace)
                exec(
                    authority_block_source("01_a0_protocol_audit.ipynb"),
                    downstream_namespace,
                    downstream_namespace,
                )
                observed = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo, text=True
                ).strip()
                self.assertEqual(observed, first_commit)

            missing_namespace = {
                "MIRROR_REVISION_ROOT": root / "missing-canonical",
                "REPO_DIR": repo,
                "REPO_URL": namespace["REPO_URL"],
                "BRANCH": "main",
            }
            with mock.patch.dict(os.environ, clean_environment, clear=False):
                with self.assertRaises(FileNotFoundError):
                    exec(
                        authority_block_source("01_a0_protocol_audit.ipynb"),
                        missing_namespace,
                        missing_namespace,
                    )

    def test_notebook00_upgrades_to_a_new_reviewed_release_tag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            canonical = root / "canonical"
            repo.mkdir()
            subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, stdout=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "audit@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Audit Test"], cwd=repo, check=True)

            namespace = {
                "MIRROR_REVISION_ROOT": canonical,
                "REPO_DIR": repo,
                "REPO_URL": "https://github.com/BrianNguyen29/ECG-RAMBA.git",
                "BRANCH": "main",
            }
            clean_environment = {
                "ECG_RAMBA_AUTHORITY_COMMIT": "",
                "ECG_RAMBA_RESET_CODE_AUTHORITY": "0",
                "ECG_RAMBA_ROTATE_CODE_AUTHORITY_TO_BRANCH_HEAD": "0",
                "ECG_RAMBA_AUTHORITY_REF": "refs/tags/authority-v1",
            }

            (repo / "authority.txt").write_text("first\n", encoding="utf-8")
            subprocess.run(["git", "add", "authority.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "first"], cwd=repo, check=True, stdout=subprocess.PIPE)
            first_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
            subprocess.run(
                ["git", "tag", "-a", "authority-v1", "-m", "authority v1", first_commit],
                cwd=repo,
                check=True,
            )
            with mock.patch.dict(os.environ, clean_environment, clear=False):
                exec(authority_block_source("00_colab_bootstrap.ipynb"), namespace, namespace)

            subprocess.run(["git", "checkout", "-B", "main"], cwd=repo, check=True, stdout=subprocess.PIPE)
            (repo / "authority.txt").write_text("second\n", encoding="utf-8")
            subprocess.run(["git", "add", "authority.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "second"], cwd=repo, check=True, stdout=subprocess.PIPE)
            second_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
            self.assertNotEqual(first_commit, second_commit)
            subprocess.run(
                ["git", "tag", "-a", "authority-v2", "-m", "authority v2", second_commit],
                cwd=repo,
                check=True,
            )

            fresh_namespace = dict(namespace)
            implicit_rotation_environment = dict(clean_environment)
            implicit_rotation_environment["ECG_RAMBA_ROTATE_CODE_AUTHORITY_TO_BRANCH_HEAD"] = "1"
            with mock.patch.dict(os.environ, implicit_rotation_environment, clear=False):
                with self.assertRaisesRegex(RuntimeError, "Implicit authority rotation"):
                    exec(
                        authority_block_source("00_colab_bootstrap.ipynb"),
                        fresh_namespace,
                        fresh_namespace,
                    )

            release_upgrade_environment = dict(clean_environment)
            release_upgrade_environment["ECG_RAMBA_AUTHORITY_REF"] = "refs/tags/authority-v2"
            with mock.patch.dict(os.environ, release_upgrade_environment, clear=False):
                exec(
                    authority_block_source("00_colab_bootstrap.ipynb"),
                    fresh_namespace,
                    fresh_namespace,
                )

            manifest = json.loads(
                (canonical / "manifests" / "notebook_code_authority.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["git_commit"], second_commit)
            self.assertEqual(manifest["authority_ref"], "refs/tags/authority-v2")
            self.assertEqual(manifest["selection"], "verified_annotated_versioned_release_tag")
            self.assertEqual(manifest["update_reason"], "versioned_release_upgrade")
            self.assertEqual(
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                second_commit,
            )

    def test_notebook00_migrates_a_legacy_drive_authority_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            canonical = root / "canonical"
            repo.mkdir()
            subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, stdout=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "audit@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Audit Test"], cwd=repo, check=True)
            (repo / "authority.txt").write_text("legacy\n", encoding="utf-8")
            subprocess.run(["git", "add", "authority.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "legacy"], cwd=repo, check=True, stdout=subprocess.PIPE)
            legacy_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
            (repo / "authority.txt").write_text("reviewed release\n", encoding="utf-8")
            subprocess.run(["git", "add", "authority.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "reviewed"], cwd=repo, check=True, stdout=subprocess.PIPE)
            release_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
            subprocess.run(
                ["git", "tag", "-a", "authority-v2", "-m", "authority v2", release_commit],
                cwd=repo,
                check=True,
            )

            manifest_path = canonical / "manifests" / "notebook_code_authority.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "capability": "canonical_git_commit_pin_v1",
                        "schema_version": 1,
                        "git_commit": legacy_commit,
                        "repository_url": "https://github.com/BrianNguyen29/ECG-RAMBA.git",
                        "branch": "main",
                    }
                ),
                encoding="utf-8",
            )
            namespace = {
                "MIRROR_REVISION_ROOT": canonical,
                "REPO_DIR": repo,
                "REPO_URL": "https://github.com/BrianNguyen29/ECG-RAMBA.git",
                "BRANCH": "main",
            }
            environment = {
                "ECG_RAMBA_AUTHORITY_COMMIT": "",
                "ECG_RAMBA_RESET_CODE_AUTHORITY": "0",
                "ECG_RAMBA_AUTHORITY_REF": "refs/tags/authority-v2",
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                exec(authority_block_source("00_colab_bootstrap.ipynb"), namespace, namespace)

            migrated = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(migrated["schema_version"], 2)
            self.assertEqual(migrated["git_commit"], release_commit)
            self.assertEqual(migrated["previous_git_commit"], legacy_commit)
            self.assertEqual(migrated["update_reason"], "legacy_manifest_migration")

    def test_notebook00_rejects_a_moved_release_tag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            canonical = root / "canonical"
            repo.mkdir()
            subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, stdout=subprocess.PIPE)
            subprocess.run(["git", "config", "user.email", "audit@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Audit Test"], cwd=repo, check=True)
            (repo / "authority.txt").write_text("first\n", encoding="utf-8")
            subprocess.run(["git", "add", "authority.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "first"], cwd=repo, check=True, stdout=subprocess.PIPE)
            first_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
            subprocess.run(
                ["git", "tag", "-a", "authority-v1", "-m", "authority v1", first_commit],
                cwd=repo,
                check=True,
            )
            namespace = {
                "MIRROR_REVISION_ROOT": canonical,
                "REPO_DIR": repo,
                "REPO_URL": "https://github.com/BrianNguyen29/ECG-RAMBA.git",
                "BRANCH": "main",
            }
            environment = {
                "ECG_RAMBA_AUTHORITY_COMMIT": "",
                "ECG_RAMBA_RESET_CODE_AUTHORITY": "0",
                "ECG_RAMBA_AUTHORITY_REF": "refs/tags/authority-v1",
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                exec(authority_block_source("00_colab_bootstrap.ipynb"), namespace, namespace)

            subprocess.run(["git", "checkout", "-B", "main"], cwd=repo, check=True, stdout=subprocess.PIPE)
            (repo / "authority.txt").write_text("second\n", encoding="utf-8")
            subprocess.run(["git", "add", "authority.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "second"], cwd=repo, check=True, stdout=subprocess.PIPE)
            second_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
            subprocess.run(["git", "tag", "-d", "authority-v1"], cwd=repo, check=True, stdout=subprocess.PIPE)
            subprocess.run(
                ["git", "tag", "-a", "authority-v1", "-m", "moved", second_commit],
                cwd=repo,
                check=True,
            )
            fresh_namespace = dict(namespace)
            with mock.patch.dict(os.environ, environment, clear=False):
                with self.assertRaisesRegex(RuntimeError, "release tag moved or changed"):
                    exec(
                        authority_block_source("00_colab_bootstrap.ipynb"),
                        fresh_namespace,
                        fresh_namespace,
                    )

    def test_notebook02a_training_streams_stage_run_id_logs_to_drive(self):
        cells, source = notebook_source("02a_retrain_best_ema.ipynb")
        training = next(
            cell for cell in cells
            if "FORENSIC_RETRAIN_STREAMING_LOG_CAPABILITY" in cell
        )
        for token in (
            "stage_run_id_durable_stream_v1",
            "run(\n        training_command",
            "MIRROR_REVISION_ROOT / 'logs' / 'history' / 'retrain_best_ema_train'",
            "durable_model_log_path",
        ):
            self.assertIn(token, training)
        self.assertNotIn("subprocess.Popen", training)
        self.assertIn("FORENSIC_RUN_HISTORY_CAPABILITY = 'stage_run_id_v1'", source)

    def test_notebook07_updates_current_authority_then_rechecks_full_sha(self):
        cells, _ = notebook_source("07_results_freeze.ipynb")
        gate = next(cell for cell in cells if "FORENSIC_NOTEBOOK07_FINAL_GATE" in cell)
        for token in (
            "strict_full_sha_authority_update_v3",
            "--source-conflict-policy source",
            "final_pipeline_storage_audit_strict_full_sha.log",
            "final_forensic_audit_authority_publish.log",
            "final_pipeline_storage_audit_post_publish_strict_full_sha.log",
            "CODE_AUTHORITY.get('git_commit')",
        ):
            self.assertIn(token, gate)
        self.assertNotIn("--source-conflict-policy fail", gate)
        first_storage = gate.index("final_pipeline_storage_audit_strict_full_sha.log")
        forensic = gate.index("final_notebook_forensic_audit.log")
        authority_publish = gate.index("final_forensic_audit_authority_publish.log")
        final_storage = gate.index("final_pipeline_storage_audit_post_publish_strict_full_sha.log")
        self.assertLess(first_storage, forensic)
        self.assertLess(forensic, authority_publish)
        self.assertLess(authority_publish, final_storage)

    def test_forensic_integrator_is_idempotent(self):
        integrator = PROJECT_ROOT / "scripts" / "revision" / "48_integrate_forensic_audit_notebooks.py"

        def notebook_hashes():
            return {
                name: hashlib.sha256((PROJECT_ROOT / "notebooks" / name).read_bytes()).hexdigest()
                for name in PIPELINE_NOTEBOOKS
            }

        subprocess.run([sys.executable, str(integrator)], cwd=PROJECT_ROOT, check=True)
        first = notebook_hashes()
        subprocess.run([sys.executable, str(integrator)], cwd=PROJECT_ROOT, check=True)
        self.assertEqual(first, notebook_hashes())

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
            "ptbxl_adaptation_analysis_lock.json",
            "ptbxl_fold_protocol_audit.json",
            "table_ptbxl_unsupported_only_sensitivity.csv",
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
            "adaptation_learning_curve",
            "group_safe_score_calibration_v2",
            "matched_cross_fitted_calibration",
            "authenticated_matched_calibration_v5",
            "matched_structured_ablation_5fold",
            "matched_structured_ablation_fresh_full",
            "post_initial_review_adaptation_analysis_lock",
            "physiological_interval_probe_gate",
            "physiological_interval_probe_v3",
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
        self.assertIn("required_generator_schema = 12", source)
        self.assertIn("FINAL_EVIDENCE_SCHEMA_VERSION = 12", generator_source)
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
            "authenticated_source_patient_record",
            "physionet_ecg_arrhythmia_one_patient_per_record_v1",
            "04_calibration_ci.py",
            "29_reviewer_presentation_assets.py --strict",
            "final_evidence_calibration_contract_refresh.log",
            "final_evidence_calibration_refresh_mirror_publish.log",
        ):
            self.assertIn(token, source)

    def test_notebook07_closes_four_partial_reviewer_items_before_final_freeze(self):
        _, source = notebook_source("07_results_freeze.ipynb")
        for token in (
            "30_pooling_sensitivity_external.py",
            "--dataset ptbxl --dataset georgia --dataset cpsc2021",
            "--strict-group-bootstrap",
            "reviewer_gap_pooling_refresh.log",
            "reviewer_gap_pooling_refresh_mirror_publish.log",
            "experimental/external/cpsc2021/cpsc2021_full_slice_predictions.npz",
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
        self.assertLess(
            source.index("30_pooling_sensitivity_external.py"),
            source.index("41_reviewer_gap_closure.py --strict"),
        )

    def test_external_pooling_summary_uses_one_canonical_filename(self):
        producer = (
            PROJECT_ROOT / "scripts" / "revision" / "30_pooling_sensitivity_external.py"
        ).read_text(encoding="utf-8")
        final_generator = (
            PROJECT_ROOT / "scripts" / "revision" / "13_final_evidence_matrix.py"
        ).read_text(encoding="utf-8")
        readiness_gate = (
            PROJECT_ROOT / "scripts" / "revision" / "28_claim_readiness_gates.py"
        ).read_text(encoding="utf-8")
        for source in (producer, final_generator, readiness_gate):
            self.assertIn("pooling_sensitivity_external.csv", source)
        stale_metric = 'METRIC_DIR / "pooling_sensitivity_across_datasets.csv"'
        self.assertNotIn(stale_metric, final_generator)
        self.assertNotIn(stale_metric, readiness_gate)

    def test_notebook03_rejects_legacy_calibration_bootstrap_metadata(self):
        _, source = notebook_source("03_calibration_and_ci.ipynb")
        self.assertIn("bootstrap_unit_missing_or_mismatched", source)
        self.assertIn("bootstrap_independence_contract_missing_or_mismatched", source)


if __name__ == "__main__":
    unittest.main()
