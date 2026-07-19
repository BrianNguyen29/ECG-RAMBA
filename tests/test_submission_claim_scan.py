from __future__ import annotations

import importlib.util
import sys
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


if __name__ == "__main__":
    unittest.main()
