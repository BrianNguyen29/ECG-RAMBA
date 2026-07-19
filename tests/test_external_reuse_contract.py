import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.revision.common import CACHE_SCHEMA_VERSION, sha256_file
from scripts.revision.external_reuse_contract import validate_external_prediction_reuse
from src.aggregation import aggregate_record_probabilities


class ExternalReuseContractTests(unittest.TestCase):
    def _fixture(self, root: Path):
        revision_root = root / "reports" / "revision"
        output = revision_root / "experimental" / "external" / "ptbxl"
        output.mkdir(parents=True)
        exporter = root / "03_generate_external_predictions.py"
        oof = root / "oof.npz"
        freeze = root / "freeze.json"
        archive = root / "PTB-XL.zip"
        exporter.write_text("# exporter v1\n", encoding="utf-8")
        np.savez_compressed(oof, y_true=np.asarray([[1]], dtype=np.float32))
        freeze.write_text('{"status":"frozen"}\n', encoding="utf-8")
        archive.write_bytes(b"archive-content")

        slice_prob = np.asarray(
            [[0.0, 0.8], [0.2, 0.6], [0.9, 0.1], [0.7, 0.3]], dtype=np.float32
        )
        record_index = np.asarray([0, 0, 1, 1], dtype=np.int64)
        y_prob, valid, _ = aggregate_record_probabilities(slice_prob, record_index, 2, q=3.0)
        self.assertTrue(valid.all())
        y_true = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
        record_id = np.asarray(["r1", "r2"])
        group_id = np.asarray(["p1", "p2"])
        split_id = np.asarray(["ptbxl_fold10", "ptbxl_fold10"])

        prediction = output / "ptbxl_full_predictions.npz"
        slice_prediction = output / "ptbxl_full_slice_predictions.npz"
        summary_path = output / "ptbxl_full_prediction_summary.json"
        class_summary = output / "ptbxl_full_class_summary.csv"
        manifest_path = output / "ptbxl_full_prediction_run_manifest.json"
        np.savez_compressed(
            prediction,
            y_true=y_true,
            y_prob=y_prob,
            record_id=record_id,
            group_id=group_id,
            group_unit=np.asarray("patient_id"),
            split_id=split_id,
            class_names=np.asarray(["NORM", "MI"]),
            dataset=np.asarray("ptbxl"),
            cache_schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
            evidence_status=np.asarray("experimental"),
            manuscript_ready=np.asarray(False),
            aggregation_method=np.asarray("power_mean"),
            aggregation_q=np.asarray(3.0, dtype=np.float32),
            threshold=np.asarray(0.5, dtype=np.float32),
        )
        np.savez_compressed(
            slice_prediction,
            slice_prob=slice_prob,
            record_index=record_index,
            record_id=record_id[record_index],
            group_id=group_id[record_index],
            split_id=split_id[record_index],
        )
        summary = {
            "dataset": "ptbxl",
            "evidence_status": "experimental",
            "manuscript_ready": False,
            "threshold": 0.5,
            "aggregation": {"method": "power_mean", "q": 3.0},
            "load_summary": {
                "label_protocol": "official_ptbxl_diagnostic_superclass_any_positive_likelihood"
            },
        }
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        class_summary.write_text("dataset,class_name\nptbxl,NORM\n", encoding="utf-8")
        outputs = {
            path.name: {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (prediction, slice_prediction, summary_path, class_summary)
        }
        manifest = {
            **summary,
            "runner_sha256": sha256_file(exporter),
            "canonical_contract": {
                "oof_sha256": sha256_file(oof),
                "freeze_sha256": sha256_file(freeze),
            },
            "archive": {
                "size_bytes": archive.stat().st_size,
                "sha256": sha256_file(archive),
            },
            "outputs": outputs,
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return revision_root, exporter, oof, freeze, archive, manifest_path

    def test_source_bound_contract_reuses_valid_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision_root, exporter, oof, freeze, archive, _ = self._fixture(root)
            result = validate_external_prediction_reuse(
                "ptbxl",
                revision_root=revision_root,
                archive_path=archive,
                exporter_path=exporter,
                oof_path=oof,
                freeze_path=freeze,
                archive_hash_cache_dir=root / "archive_hash_cache",
            )
            self.assertTrue(result["ready"], result["reasons"])
            self.assertLessEqual(result["diagnostics"]["q3_reconstruction_max_abs"], 1e-6)

    def test_source_or_protocol_mutation_forces_reexport(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision_root, exporter, oof, freeze, archive, manifest_path = self._fixture(root)
            exporter.write_text("# exporter v2\n", encoding="utf-8")
            result = validate_external_prediction_reuse(
                "ptbxl",
                revision_root=revision_root,
                archive_path=archive,
                exporter_path=exporter,
                oof_path=oof,
                freeze_path=freeze,
                archive_hash_cache_dir=root / "archive_hash_cache",
            )
            self.assertFalse(result["ready"])
            self.assertIn("external_exporter_sha_mismatch", result["reasons"])

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["runner_sha256"] = sha256_file(exporter)
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            summary_path = (
                revision_root
                / "experimental"
                / "external"
                / "ptbxl"
                / "ptbxl_full_prediction_summary.json"
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["load_summary"]["label_protocol"] = "stale_protocol"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            result = validate_external_prediction_reuse(
                "ptbxl",
                revision_root=revision_root,
                archive_path=archive,
                exporter_path=exporter,
                oof_path=oof,
                freeze_path=freeze,
                archive_hash_cache_dir=root / "archive_hash_cache",
            )
            self.assertFalse(result["ready"])
            self.assertIn("label_protocol_mismatch", result["reasons"])


if __name__ == "__main__":
    unittest.main()
