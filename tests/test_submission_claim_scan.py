from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "revision" / "46_submission_claim_scan.py"


def load_module():
    spec = importlib.util.spec_from_file_location("submission_claim_scan_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SubmissionClaimScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scan = load_module()

    def statuses(self, text: str) -> list[str]:
        return [row["status"] for row in self.scan.scan_text(Path("test.txt"), text)]

    def test_positive_zero_shot_superiority_is_blocked(self):
        self.assertEqual(
            self.statuses("ECG-RAMBA demonstrates zero-shot superiority."),
            ["unsafe_positive_claim"],
        )

    def test_explicit_claim_boundary_is_allowed(self):
        self.assertEqual(
            self.statuses("We do not claim zero-shot superiority."),
            ["safe_boundary"],
        )

    def test_negation_does_not_leak_into_next_sentence(self):
        statuses = self.statuses(
            "We do not claim SOTA. ECG-RAMBA establishes global superiority."
        )
        self.assertEqual(statuses, ["safe_boundary", "unsafe_positive_claim"])

    def test_negation_does_not_leak_across_semicolon(self):
        statuses = self.statuses(
            "We do not claim zero-shot superiority; however, the experiments establish global superiority."
        )
        self.assertEqual(statuses, ["safe_boundary", "unsafe_positive_claim"])

    def test_all_baseline_outperformance_paraphrase_is_blocked(self):
        self.assertEqual(
            self.statuses("ECG-RAMBA outperforms all fair baselines across datasets."),
            ["unsafe_positive_claim"],
        )

    def test_full_hrv_implementation_is_blocked(self):
        statuses = self.statuses("The deployed model implements RMSSD and SDNN features.")
        self.assertEqual(statuses, ["unsafe_positive_claim"])

    def test_clinical_readiness_boundary_is_allowed(self):
        self.assertEqual(
            self.statuses("Clinical readiness is not established by this retrospective study."),
            ["safe_boundary"],
        )

    def test_conjoined_negation_and_blocked_status_are_allowed(self):
        self.assertEqual(
            self.statuses(
                "This is adaptation, but not end-to-end encoder fine-tuning or general superiority."
            ),
            ["safe_boundary"],
        )
        self.assertEqual(
            self.statuses("Broad robustness superiority remains blocked."),
            ["safe_boundary"],
        )

    def test_scan_output_binds_every_input_file_by_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            document = root / "submission.txt"
            out_json = root / "scan.json"
            out_table = root / "scan.csv"
            document.write_text("No unsupported broad superiority claim is made.", encoding="utf-8")
            argv = [
                "scan",
                "--path",
                str(document),
                "--out-json",
                str(out_json),
                "--out-table",
                str(out_table),
            ]
            from unittest import mock

            with mock.patch.object(sys, "argv", argv):
                self.scan.main()
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["files_scanned"], 1)
            self.assertEqual(payload["file_contracts"][0]["path"], str(document.resolve()))
            self.assertEqual(len(payload["file_contracts"][0]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
