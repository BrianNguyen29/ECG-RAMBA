import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.revision import common


builder = importlib.import_module("scripts.revision.49_build_oof_group_sidecar")
freeze = importlib.import_module("scripts.revision.06_freeze_oof")


class OofGroupSidecarTests(unittest.TestCase):
    def test_builder_emits_freeze_compatible_sha_bound_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oof = root / "oof.npz"
            archive = root / "chapman.zip"
            output = root / "sidecar.npz"
            archive.write_bytes(b"reviewed-source-archive")
            np.savez_compressed(
                oof,
                record_id=np.asarray([2, 0, 1], dtype=np.int64),
                dataset=np.asarray("chapman_oof"),
                dataset_record_order_fingerprint=np.asarray("order-sha"),
            )

            result = builder.build_sidecar(oof, archive, output, expected_records=3)
            loaded = freeze.load_group_sidecar(output)

            self.assertEqual(result["status"], "complete")
            np.testing.assert_array_equal(loaded["record_id"], [2, 0, 1])
            np.testing.assert_array_equal(loaded["group_id"], [2, 0, 1])
            self.assertTrue(loaded["one_record_per_group"])
            self.assertEqual(loaded["group_semantics"], common.CHAPMAN_GROUP_SEMANTICS)
            self.assertEqual(
                loaded["group_semantics_reference"], common.CHAPMAN_GROUP_REFERENCE
            )
            self.assertEqual(loaded["record_file_sha256"], common.sha256_file(oof))
            self.assertEqual(loaded["source_archive_sha256"], common.sha256_file(archive))
            original_archive = freeze.PATHS["zip_path"]
            freeze.PATHS["zip_path"] = str(archive)
            try:
                contract = freeze.validate_group_contract(
                    sidecar_path=output,
                    expected_sidecar_sha256=result["output_sha256"],
                    record_file=oof,
                    record_id=np.asarray([2, 0, 1], dtype=np.int64),
                    dataset_record_order_fingerprint="order-sha",
                )
            finally:
                freeze.PATHS["zip_path"] = original_archive
            self.assertEqual(contract["status"], "verified")
            self.assertEqual(contract["n_groups"], 3)

    def test_builder_rejects_duplicate_record_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oof = root / "oof.npz"
            archive = root / "chapman.zip"
            archive.write_bytes(b"archive")
            np.savez_compressed(
                oof,
                record_id=np.asarray([0, 0, 1], dtype=np.int64),
                dataset=np.asarray("chapman_oof"),
                dataset_record_order_fingerprint=np.asarray("order-sha"),
            )
            with self.assertRaisesRegex(ValueError, "one unique source record"):
                builder.build_sidecar(oof, archive, root / "sidecar.npz", expected_records=3)


if __name__ == "__main__":
    unittest.main()
