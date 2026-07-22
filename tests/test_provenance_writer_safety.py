import json
import os
import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.provenance import (
    exclusive_cache_writer,
    save_npz_atomic,
    source_bundle_sha256,
)


class ProvenanceWriterSafetyTests(unittest.TestCase):
    def test_live_same_host_writer_is_never_stolen_only_because_lock_is_old(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "cache.npz"
            lock = destination.with_name(f".{destination.name}.write.lock")
            lock.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "run_id": "live",
                        "pid": os.getpid(),
                        "hostname": socket.gethostname(),
                        "created_epoch": time.time() - 24 * 60 * 60,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "live process"):
                with exclusive_cache_writer(destination, stale_seconds=1):
                    pass

            self.assertTrue(lock.is_file())

    def test_dead_same_host_writer_is_recovered_without_waiting_for_stale_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "cache.npz"
            lock = destination.with_name(f".{destination.name}.write.lock")
            lock.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "run_id": "dead",
                        "pid": 999_999_999,
                        "hostname": socket.gethostname(),
                        "created_epoch": time.time(),
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch("src.provenance._same_host_pid_is_alive", return_value=False):
                with exclusive_cache_writer(destination, stale_seconds=24 * 60 * 60):
                    self.assertTrue(lock.is_file())

            self.assertFalse(lock.exists())
            self.assertEqual(len(list(destination.parent.glob(f"{lock.name}.stale.*"))), 1)

    def test_invalid_npz_is_rejected_before_final_name_is_exposed(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "cache.npz"

            def write_invalid(path, **arrays):
                del arrays
                Path(path).write_bytes(b"not-an-npz")

            with mock.patch("src.provenance.np.savez_compressed", side_effect=write_invalid):
                with self.assertRaises(Exception):
                    save_npz_atomic(destination, values=np.arange(4))

            self.assertFalse(destination.exists())

    def test_source_bundle_hash_is_clone_path_independent(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for root in (Path(first), Path(second)):
                (root / "src").mkdir()
                (root / "configs").mkdir()
                (root / "src" / "a.py").write_text("a = 1\n", encoding="utf-8")
                (root / "configs" / "b.py").write_text("b = 2\n", encoding="utf-8")
            first_hash = source_bundle_sha256(
                [Path(first) / "src" / "a.py", Path(first) / "configs" / "b.py"]
            )
            second_hash = source_bundle_sha256(
                [Path(second) / "src" / "a.py", Path(second) / "configs" / "b.py"]
            )
            self.assertEqual(first_hash, second_hash)


if __name__ == "__main__":
    unittest.main()
