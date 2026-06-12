import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from configs.config import CLASS_TO_IDX, CLASSES


external = importlib.import_module("scripts.revision.03_generate_external_predictions")


class ExternalPredictionContractTests(unittest.TestCase):
    def test_checkpoint_provenance_requires_matching_explicit_ema_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for fold in (1, 2):
                path = Path(tmp) / f"fold{fold}_final_ema.pt"
                torch.save(
                    {
                        "model": {},
                        "weights_kind": "ema",
                        "epoch": 20,
                        "selection_rule": "fixed_final_epoch",
                        "config_hash": "source123",
                        "dataset_record_order_fingerprint": "records123",
                    },
                    path,
                )
                paths.append(path)
            rows, source_hash = external.checkpoint_provenance(paths, "final_ema")
            self.assertEqual(source_hash, "source123")
            self.assertEqual([row["weights_kind"] for row in rows], ["ema", "ema"])

    def test_ptb_mapping_uses_declared_chapman_classes(self):
        probs = np.zeros((2, len(CLASSES)), dtype=np.float32)
        probs[:, CLASS_TO_IDX["SNR"]] = [0.8, 0.1]
        probs[:, CLASS_TO_IDX["QAb"]] = [0.2, 0.7]
        mapped, names = external.map_model_probabilities("ptbxl", probs)
        self.assertEqual(list(names), ["NORM", "MI", "STTC", "CD"])
        np.testing.assert_allclose(mapped[:, 0], [0.8, 0.1])
        np.testing.assert_allclose(mapped[:, 1], [0.2, 0.7])

    def test_cpsc_af_output_combines_af_and_afl(self):
        probs = np.zeros((2, len(CLASSES)), dtype=np.float32)
        probs[:, CLASS_TO_IDX["AF"]] = [0.6, 0.2]
        probs[:, CLASS_TO_IDX["AFL"]] = [0.3, 0.9]
        mapped, names = external.map_model_probabilities("cpsc2021", probs)
        self.assertEqual(list(names), ["AF_or_AFL"])
        np.testing.assert_allclose(mapped[:, 0], [0.6, 0.9])

    def test_checkpoint_compatible_hrv_keeps_amplitude_slots_zero(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(size=(12, 5000)).astype(np.float32)
        features = external.checkpoint_compatible_hrv36(signal)
        self.assertEqual(features.shape, (36,))
        np.testing.assert_array_equal(features[25:30], np.zeros(5, dtype=np.float32))

    def test_cpsc_annotation_failure_is_not_converted_to_negative(self):
        with patch.object(external.wfdb, "rdann", side_effect=FileNotFoundError("missing atr")):
            with self.assertRaises(FileNotFoundError):
                external.cpsc_af_intervals(Path("missing"), signal_length=1000)

    def test_cpsc_intervals_follow_annotation_boundaries(self):
        annotation = SimpleNamespace(
            sample=np.asarray([0, 100, 300, 500]),
            aux_note=["(N", "(AFIB", "(N", "(AFL"],
        )
        with patch.object(external.wfdb, "rdann", return_value=annotation):
            intervals = external.cpsc_af_intervals(Path("record"), signal_length=700)
        self.assertEqual(intervals, [(100, 300), (500, 700)])
        self.assertEqual(external.interval_overlap(intervals, 250, 550), 100)

    def test_georgia_skips_records_without_mapped_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mapped.hea").write_text("#Dx: 164889003\n", encoding="utf-8")
            (root / "unmapped.hea").write_text("#Dx: 999999999\n", encoding="utf-8")
            rows, summary = external.georgia_metadata(root, limit=0)
        self.assertEqual([row["record_id"] for row in rows], ["mapped"])
        self.assertEqual(summary["skipped_records_without_mapped_label"], 1)
        self.assertEqual(summary["unmapped_snomed_codes"], {"999999999": 1})

    def test_ptb_uses_positive_likelihood_not_only_100(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(
                {
                    "ecg_id": [1],
                    "strat_fold": [10],
                    "scp_codes": ["{'NORM': 50.0, 'HYP': 25.0}"],
                    "filename_hr": ["records/00001_hr"],
                }
            ).set_index("ecg_id").to_csv(root / "ptbxl_database.csv")
            pd.DataFrame(
                {
                    "diagnostic": [1, 1],
                    "diagnostic_class": ["NORM", "HYP"],
                },
                index=["NORM", "HYP"],
            ).to_csv(root / "scp_statements.csv")
            rows, summary = external.ptb_metadata(root, limit=0)
        np.testing.assert_array_equal(rows[0]["y_true"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(summary["unsupported_superclasses"], {"HYP": 1})


if __name__ == "__main__":
    unittest.main()
