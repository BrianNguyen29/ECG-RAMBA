import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from configs.config import setup_paths


class ConfigPathTests(unittest.TestCase):
    def test_epoch_override_is_applied_before_config_hash(self):
        env = os.environ.copy()
        env["ECG_RAMBA_EPOCHS"] = "30"
        output = subprocess.check_output(
            [
                sys.executable,
                "-c",
                (
                    "from configs.config import CONFIG, CONFIG_HASH, EVALUATION_CONFIG_HASH; "
                    "print(CONFIG['epochs']); print(CONFIG_HASH); print(EVALUATION_CONFIG_HASH)"
                ),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            text=True,
        ).splitlines()
        self.assertEqual(output[0], "30")
        self.assertRegex(output[1], r"^[0-9a-f]{8}$")
        self.assertRegex(output[2], r"^[0-9a-f]{8}$")

        baseline = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "from configs.config import EVALUATION_CONFIG_HASH; print(EVALUATION_CONFIG_HASH)",
            ],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
        self.assertEqual(output[2], baseline)

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
