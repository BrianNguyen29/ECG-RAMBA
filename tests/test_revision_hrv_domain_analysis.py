import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

hrv_analysis = importlib.import_module("scripts.revision.09_hrv_domain_analysis")


class HRVDomainAnalysisTests(unittest.TestCase):
    def test_hrv_oof_baseline_covers_every_validation_record(self):
        rng = np.random.default_rng(42)
        n_records = 36
        X_hrv = rng.normal(size=(n_records, 6)).astype(np.float32)
        y = np.zeros((n_records, 3), dtype=np.float32)
        y[:, 0] = (X_hrv[:, 0] > 0).astype(np.float32)
        y[:, 1] = (X_hrv[:, 1] + X_hrv[:, 2] > 0).astype(np.float32)
        y[:, 2] = 0.0
        folds = [
            {"tr_idx": np.r_[12:36], "va_idx": np.r_[0:12]},
            {"tr_idx": np.r_[0:12, 24:36], "va_idx": np.r_[12:24]},
            {"tr_idx": np.r_[0:24], "va_idx": np.r_[24:36]},
        ]

        y_prob, fold_id, fold_rows = hrv_analysis.fit_predict_hrv_oof(
            X_hrv,
            y,
            folds,
            seed=123,
            max_iter=200,
        )

        self.assertEqual(y_prob.shape, y.shape)
        self.assertTrue(np.all(np.isfinite(y_prob)))
        self.assertTrue(np.all((y_prob >= 0.0) & (y_prob <= 1.0)))
        self.assertTrue(np.all(fold_id > 0))
        self.assertEqual(len(fold_rows), 3)
        np.testing.assert_allclose(y_prob[:, 2], 0.0)

    def test_domain_classifier_uses_balanced_cross_validated_domain_rows(self):
        rng = np.random.default_rng(7)
        features_by_domain = {
            "chapman": rng.normal(loc=0.0, scale=0.4, size=(30, 5)).astype(np.float32),
            "cpsc2021": rng.normal(loc=2.5, scale=0.4, size=(25, 5)).astype(np.float32),
            "ptbxl": rng.normal(loc=-2.5, scale=0.4, size=(20, 5)).astype(np.float32),
        }

        result = hrv_analysis.run_domain_classifier_cv(
            features_by_domain,
            max_per_domain=18,
            n_splits=3,
            seed=11,
        )

        self.assertEqual(result["X_shape"], [54, 5])
        self.assertEqual(sorted(result["domains"]), ["chapman", "cpsc2021", "ptbxl"])
        self.assertTrue(np.all(result["fold_id"] > 0))
        self.assertTrue(np.isfinite(result["metrics"]["domain_roc_auc_ovr_macro"]))
        self.assertTrue(0.0 <= result["metrics"]["balanced_accuracy"] <= 1.0)
        self.assertEqual(len(result["confusion_rows"]), 9)
        self.assertEqual(
            {row["sampled_records"] for row in result["sample_rows"]},
            {18},
        )


if __name__ == "__main__":
    unittest.main()
