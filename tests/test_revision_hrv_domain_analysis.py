import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import json

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

    def test_lightweight_oof_and_cached_hrv_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oof_path = root / "oof_full_predictions.npz"
            hrv_path = root / "hrv36_N12_C12_L5000.npz"
            y = np.zeros((12, 27), dtype=np.float32)
            y[::2, 0] = 1.0
            y[1::3, 1] = 1.0
            fold_id = np.asarray([1, 2, 3] * 4, dtype=np.int16)
            np.savez_compressed(
                oof_path,
                y_true=y,
                fold_id=fold_id,
                record_id=np.arange(12, dtype=np.int64),
                class_names=np.asarray(hrv_analysis.CLASSES),
            )
            X_hrv = np.arange(12 * 36, dtype=np.float32).reshape(12, 36)
            np.savez_compressed(hrv_path, X=X_hrv)

            y_loaded, folds, oof_info = hrv_analysis.load_oof_labels_and_folds(oof_path, limit_records=0)
            X_loaded, hrv_info = hrv_analysis.load_cached_chapman_hrv(
                n_records=12,
                explicit_cache=hrv_path,
                limit_records=0,
                allow_raw_fallback=False,
            )

            np.testing.assert_array_equal(y_loaded, y)
            self.assertEqual(len(folds), 3)
            self.assertEqual(oof_info["oof_records_total"], 12)
            np.testing.assert_array_equal(X_loaded, X_hrv)
            self.assertFalse(hrv_info["raw_chapman_loaded"])

    def test_cached_hrv_loader_prefers_record_fingerprinted_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_path = root / "hrv36_N6_C12_L5000.npz"
            fingerprinted_path = root / "hrv36_N6_C12_L5000_Rabc123.npz"
            legacy = np.zeros((6, 36), dtype=np.float32)
            fingerprinted = np.ones((6, 36), dtype=np.float32)
            np.savez_compressed(legacy_path, X=legacy)
            np.savez_compressed(fingerprinted_path, X=fingerprinted)

            with patch.dict(hrv_analysis.PATHS, {"cache_dir": str(root)}, clear=False):
                X_loaded, hrv_info = hrv_analysis.load_cached_chapman_hrv(
                    n_records=6,
                    explicit_cache=None,
                    limit_records=0,
                    allow_raw_fallback=False,
                )

            np.testing.assert_array_equal(X_loaded, fingerprinted)
            self.assertEqual(hrv_info["chapman_hrv_cache_kind"], "record_fingerprinted")

    def test_write_outputs_records_blocked_domain_classifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_dir = root / "predictions"
            metric_dir = root / "metrics"
            table_dir = root / "tables"
            manifest_dir = root / "manifests"
            args = SimpleNamespace(
                threshold=0.5,
                n_bins=15,
                n_boot=3,
                seed=42,
                limit_records=0,
                domain_max_per_domain=10,
                domain_n_splits=2,
                skip_domain_classifier=False,
                oof_predictions=root / "oof.npz",
                chapman_hrv_cache=root / "hrv.npz",
                ptbxl_hrv=root / "missing_ptbxl.npz",
                cpsc2021_hrv=root / "missing_cpsc.npz",
                allow_raw_chapman_fallback=False,
            )
            y = np.zeros((4, 27), dtype=np.float32)
            y[0, 0] = 1.0
            y[1, 1] = 1.0
            y_prob = np.full_like(y, 0.25)
            baseline = {
                "y_true": y,
                "y_prob": y_prob,
                "fold_id": np.asarray([1, 2, 1, 2], dtype=np.int16),
                "fold_rows": [{"fold": 1}, {"fold": 2}],
                "metrics": {"roc_auc_macro": 0.5, "pr_auc_macro": 0.2, "f1_macro": 0.1},
                "calibration": {},
                "bootstrap_ci": {},
                "per_class_rows": [
                    {
                        "class_index": idx,
                        "class_name": name,
                        "n_records": 4,
                        "n_positive": int(y[:, idx].sum()),
                        "prevalence": float(y[:, idx].mean()),
                        "predicted_positive": 0,
                        "predicted_positive_rate": 0.0,
                        "roc_auc": float("nan"),
                        "pr_auc": float("nan"),
                        "f1": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                    }
                    for idx, name in enumerate(hrv_analysis.CLASSES)
                ],
            }
            with patch.object(hrv_analysis, "PREDICTION_DIR", pred_dir), patch.object(
                hrv_analysis,
                "METRIC_DIR",
                metric_dir,
            ), patch.object(hrv_analysis, "TABLE_DIR", table_dir), patch.object(
                hrv_analysis,
                "MANIFEST_DIR",
                manifest_dir,
            ):
                summary = hrv_analysis.write_outputs(
                    args,
                    {"chapman_records": 4},
                    baseline,
                    domain_result=None,
                    domain_blocker="missing external HRV",
                )

            domain_summary_path = metric_dir / "hrv_domain_classifier_summary.json"
            self.assertTrue(domain_summary_path.exists())
            domain_summary = json.loads(domain_summary_path.read_text(encoding="utf-8"))
            self.assertEqual(domain_summary["status"], "blocked_external_hrv_missing")
            self.assertIn("missing external HRV", domain_summary["blocker"])
            self.assertEqual(summary["domain_status"], "blocked_external_hrv_missing")
            self.assertTrue((metric_dir / "hrv_domain_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
