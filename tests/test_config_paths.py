import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from configs.config import setup_paths


class ConfigPathTests(unittest.TestCase):
    def test_explicit_model_dir_is_used_even_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            drive_root = root / "drive"
            explicit_model_dir = drive_root / "model_runs" / "best_ema_revision_v1"
            legacy_model_dir = drive_root / "model"
            explicit_model_dir.mkdir(parents=True)
            legacy_model_dir.mkdir(parents=True)
            (legacy_model_dir / "fold1_best.pt").write_bytes(b"legacy")

            with patch.dict(
                os.environ,
                {
                    "ECG_RAMBA_DRIVE_ROOT": str(drive_root),
                    "ECG_RAMBA_MODEL_DIR": str(explicit_model_dir),
                },
                clear=False,
            ):
                paths = setup_paths(27, 3072, "testhash", drive_mounted=False)

            self.assertEqual(Path(paths["model_dir"]), explicit_model_dir)


if __name__ == "__main__":
    unittest.main()
