from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = ROOT / "scripts" / "revision" / "47_forensic_notebook_audit.py"
MAMBA_MARKER = "MAMBA_INSTALLER_CAPABILITY = 'ecg_ramba_mamba_installer_v1'"
RUN_HISTORY_MARKER = "FORENSIC_RUN_HISTORY_CAPABILITY = 'stage_run_id_v1'"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("forensic_notebook_audit_test_module", AUDIT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def notebook_text(name: str) -> str:
    payload = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))


class PairedInferenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.audit = load_audit_module()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "metrics").mkdir(parents=True)
        (self.root / "tables").mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def write_json(self, name: str, payload: dict) -> None:
        (self.root / "metrics" / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_pointwise_ci_without_p_value_is_allowed(self):
        self.write_json(
            "paired_pointwise_comparison.json",
            {
                "inference_scope": "pointwise_percentile_ci_effect_size_only",
                "null_test": "not_run",
                "metrics": {
                    "f1": {
                        "interpretation": "full_nominal_95ci_better",
                        "p_value_two_sided": None,
                    }
                },
            },
        )
        rows, failures = self.audit.paired_inference_audit_rows(self.root)
        self.assertFalse(failures)
        self.assertEqual(rows[0]["status"], "pass")
        self.assertEqual(rows[0]["allowed_inference"], "pointwise_effect_size_and_percentile_ci_only")

    def test_bootstrap_tail_p_value_and_significance_label_are_rejected(self):
        self.write_json(
            "paired_unsafe_comparison.json",
            {
                "metrics": {
                    "f1": {
                        "interpretation": "full_significantly_better",
                        "p_value_two_sided": 0.001,
                        "holm_p_value_two_sided": 0.005,
                    }
                }
            },
        )
        rows, failures = self.audit.paired_inference_audit_rows(self.root)
        self.assertEqual(rows[0]["status"], "fail")
        self.assertTrue(any("significance_language" in failure for failure in failures))
        self.assertTrue(any("finite_p_values" in failure for failure in failures))

    def test_predeclared_paired_permutation_with_holm_is_accepted(self):
        self.write_json(
            "paired_valid_null_comparison.json",
            {
                "inference_contracts": {
                    "f1_primary": {
                        "inference_contract_id": "f1_primary",
                        "test_method": "paired_group_sign_flip_permutation",
                        "null_centered": True,
                        "n_perm": 10_000,
                        "multiplicity_adjustment": "Holm",
                        "multiplicity_family": "five pre-specified metrics",
                        "permutation_unit": "authenticated patient group",
                        "group_sidecar_sha256": "a" * 64,
                    }
                },
                "metrics": {
                    "f1": {
                        "inference_contract_id": "f1_primary",
                        "interpretation": "full_significantly_better",
                        "p_value_two_sided": 0.001,
                        "holm_p_value_two_sided": 0.005,
                    }
                },
            },
        )
        rows, failures = self.audit.paired_inference_audit_rows(self.root)
        self.assertFalse(failures)
        self.assertTrue(rows[0]["valid_paired_permutation_contract"])

    def test_unrelated_valid_contract_cannot_authorize_another_endpoint(self):
        self.write_json(
            "paired_cross_endpoint_comparison.json",
            {
                "inference_contracts": {
                    "valid_endpoint": {
                        "inference_contract_id": "valid_endpoint",
                        "test_method": "paired_group_sign_flip_permutation",
                        "null_centered": True,
                        "n_perm": 10_000,
                        "multiplicity_adjustment": "Holm",
                        "multiplicity_family": "five pre-specified metrics",
                        "permutation_unit": "authenticated patient group",
                        "group_sidecar_sha256": "b" * 64,
                    }
                },
                "metrics": {
                    "valid": {
                        "inference_contract_id": "valid_endpoint",
                        "interpretation": "full_significantly_better",
                        "p_value_two_sided": 0.001,
                        "holm_p_value_two_sided": 0.005,
                    },
                    "unsafe": {
                        "interpretation": "comparator_significantly_better",
                        "p_value_two_sided": 0.001,
                        "holm_p_value_two_sided": 0.005,
                    }
                },
            },
        )
        rows, failures = self.audit.paired_inference_audit_rows(self.root)
        self.assertTrue(failures)
        self.assertEqual(rows[0]["status"], "fail")
        self.assertEqual(rows[0]["unsafe_endpoint_count"], 1)

    def test_final_safe_wording_is_scanned_for_significance_language(self):
        (self.root / "tables" / "table_final_safe_wording.csv").write_text(
            "claim,safe_wording\nrobustness,full significantly better\n",
            encoding="utf-8",
        )
        rows, failures = self.audit.paired_inference_audit_rows(self.root)
        self.assertTrue(failures)
        target = next(row for row in rows if row["artifact"].endswith("table_final_safe_wording.csv"))
        self.assertEqual(target["status"], "fail")

    def test_audit_count_columns_are_not_misread_as_p_values(self):
        (self.root / "tables" / "table_paired_inference_audit.csv").write_text(
            "artifact,finite_p_value_count,status\nold.json,7,fail\n",
            encoding="utf-8",
        )
        rows, failures = self.audit.paired_inference_audit_rows(self.root)
        self.assertFalse(failures)
        self.assertEqual(rows, [])


class NotebookForensicContractTests(unittest.TestCase):
    def test_mamba_installer_has_one_authoritative_capability_marker(self):
        self.assertEqual(notebook_text("02_predictions_and_external_eval.ipynb").count(MAMBA_MARKER), 1)

    def test_notebooks_have_durable_run_history_wrapper(self):
        for name in (
            "00_colab_bootstrap.ipynb",
            "01_a0_protocol_audit.ipynb",
            "02_predictions_and_external_eval.ipynb",
            "02a_retrain_best_ema.ipynb",
            "03_calibration_and_ci.ipynb",
            "04_baselines_and_component_checks.ipynb",
            "05_hrv_domain_and_robustness.ipynb",
            "06_pooling_and_representation.ipynb",
            "07_results_freeze.ipynb",
        ):
            with self.subTest(notebook=name):
                self.assertIn(RUN_HISTORY_MARKER, notebook_text(name))

    def test_notebook00_cpu_audit_does_not_unconditionally_require_cuda(self):
        text = notebook_text("00_colab_bootstrap.ipynb")
        self.assertIn("INSTALL_MAMBA_IN_NOTEBOOK00", text)
        self.assertIn("Skipping Mamba installation in Notebook 00", text)
        self.assertIn("requires a CUDA runtime", text)

    def test_notebook02_freeze_is_group_bound_and_strict(self):
        text = notebook_text("02_predictions_and_external_eval.ipynb")
        self.assertIn("--manuscript-ready-strict", text)
        self.assertIn("oof_final_ema_group_sidecar.npz", text)

    def test_notebook07_has_strict_full_sha_forensic_gate(self):
        text = notebook_text("07_results_freeze.ipynb")
        for token in (
            "38_pipeline_storage_audit.py",
            "--strict --full-sha",
            "47_forensic_notebook_audit.py",
            "table_paired_inference_audit.csv",
        ):
            self.assertIn(token, text)

    def test_notebook07_exports_only_after_strict_forensic_gate(self):
        payload = json.loads((ROOT / "notebooks" / "07_results_freeze.ipynb").read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in payload.get("cells", [])]
        gate_index = next(index for index, text in enumerate(sources) if "FORENSIC_NOTEBOOK07_FINAL_GATE" in text)
        export_index = next(index for index, text in enumerate(sources) if "FINAL_TABLE_EXPORT_DIR" in text)
        self.assertGreater(export_index, gate_index)
        self.assertFalse(
            any("final_evidence_tables" in text and "shutil.copy2" in text for text in sources[:gate_index])
        )

    def test_source_bundle_contract_is_deterministic(self):
        audit = load_audit_module()
        first = audit.source_bundle_contract()
        second = audit.source_bundle_contract()
        self.assertEqual(first, second)
        self.assertRegex(first[0], r"^[0-9a-f]{64}$")
        self.assertGreater(first[1], 50)

    def test_all_notebook_code_cells_compile(self):
        audit = load_audit_module()
        _, failures = audit.notebook_cell_rows()
        self.assertEqual(failures, [])

    def test_reviewer_traceability_references_existing_runners(self):
        audit = load_audit_module()
        rows = audit.reviewer_traceability_rows()
        self.assertEqual(len(rows), 12)
        self.assertEqual(audit.traceability_contract_failures(rows), [])

    def test_training_loader_is_fold_local_by_source_contract(self):
        audit = load_audit_module()
        self.assertEqual(audit.training_loader_source_failures(), [])


class AuthenticatedBootstrapAuditTests(unittest.TestCase):
    def setUp(self):
        self.audit = load_audit_module()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "manifests").mkdir(parents=True)
        (self.root / "metrics").mkdir(parents=True)
        self.sidecar = self.root / "manifests" / "groups.npz"
        self.sidecar.write_bytes(b"authenticated-group-sidecar")
        self.oof_sha = "a" * 64

    def tearDown(self):
        self.temp.cleanup()

    def write_contracts(self, *, semantics: str | None = None) -> str:
        freeze_path = self.root / "manifests" / "oof_final_ema_freeze_manifest.json"
        freeze_payload = {
            "status": "frozen",
            "manuscript_ready": True,
            "strict_manuscript_contract": True,
            "validated_records": 4,
            "membership_contract": {"status": "verified"},
            "group_contract": {
                "status": "verified",
                "group_semantics": semantics or self.audit.GROUP_SEMANTICS,
                "group_semantics_reference": self.audit.GROUP_REFERENCE,
                "source_patient_record_counts": self.audit.GROUP_REFERENCE_COUNTS,
                "bootstrap_unit": self.audit.AUTHENTICATED_BOOTSTRAP_UNIT,
                "one_record_per_group": True,
                "n_records": 4,
                "n_groups": 4,
                "sidecar": {
                    "path": str(self.sidecar),
                    "sha256": self.audit.sha256_file(self.sidecar),
                },
            },
        }
        freeze_path.write_text(json.dumps(freeze_payload), encoding="utf-8")
        freeze_sha = self.audit.sha256_file(freeze_path)
        calibration = {
            "predictions_sha256": self.oof_sha,
            "freeze_manifest_sha256": freeze_sha,
            "bootstrap": {
                "unit": self.audit.AUTHENTICATED_BOOTSTRAP_UNIT,
                "independence_contract": self.audit.GROUP_SEMANTICS,
                "group_semantics_reference": self.audit.GROUP_REFERENCE,
                "group_sidecar": str(self.sidecar),
                "group_sidecar_sha256": self.audit.sha256_file(self.sidecar),
            },
        }
        (self.root / "metrics" / "calibration_ci_oof_final_ema_predictions.json").write_text(
            json.dumps(calibration),
            encoding="utf-8",
        )
        return freeze_sha

    def test_authenticated_bootstrap_contract_passes(self):
        freeze_sha = self.write_contracts()
        failures = self.audit.bootstrap_contract_failures(
            self.root,
            oof_sha=self.oof_sha,
            freeze_sha=freeze_sha,
        )
        self.assertEqual(failures, [])

    def test_group_semantic_mutation_is_blocked(self):
        freeze_sha = self.write_contracts(semantics="unreviewed_record_identifier_assumption")
        failures = self.audit.bootstrap_contract_failures(
            self.root,
            oof_sha=self.oof_sha,
            freeze_sha=freeze_sha,
        )
        self.assertTrue(any("group_contract.group_semantics" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
