import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.revision import artifact_mirror


class ArtifactMirrorStorageSafetyTests(unittest.TestCase):
    def _publish(self, revision: Path, mirror: Path, **kwargs) -> Path:
        with patch.object(artifact_mirror, "REVISION_DIR", revision), patch.object(
            artifact_mirror,
            "ensure_revision_dirs",
            return_value=None,
        ):
            return artifact_mirror.publish(mirror, **kwargs)

    def _restore(self, revision: Path, mirror: Path, **kwargs) -> Path:
        with patch.object(artifact_mirror, "REVISION_DIR", revision):
            return artifact_mirror.restore(mirror, **kwargs)

    def _seed_published_artifact(self, root: Path) -> tuple[Path, Path, Path, Path]:
        revision = root / "revision"
        mirror = root / "mirror"
        source = revision / "metrics" / "summary.json"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"old-complete-artifact")
        manifest = self._publish(revision, mirror)
        destination = mirror / "metrics" / "summary.json"
        return revision, mirror, source, destination

    def test_manifest_declares_byte_integrity_not_producer_attestation(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, mirror, _, _ = self._seed_published_artifact(Path(tmp))
            payload = json.loads(
                (mirror / "manifests" / "mirror_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("byte_integrity_only", payload["trust_scope"])
            self.assertTrue(
                all(
                    "byte_integrity_only" in row["attestation_scope"]
                    for row in payload["artifacts"]
                )
            )

    def test_source_conflict_override_is_limited_to_selected_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = root / "revision"
            mirror = root / "mirror"
            selected = revision / "metrics" / "selected.json"
            unselected = revision / "tables" / "unselected.csv"
            selected.parent.mkdir(parents=True)
            unselected.parent.mkdir(parents=True)
            selected.write_bytes(b"selected-old")
            unselected.write_bytes(b"unselected-old")
            self._publish(revision, mirror)

            selected.write_bytes(b"selected-new")
            unselected.write_bytes(b"unselected-new")
            manifest_path = self._publish(
                revision,
                mirror,
                source_conflict_policy="source",
                include_paths=["metrics/selected.json"],
            )

            self.assertEqual((mirror / "metrics" / "selected.json").read_bytes(), b"selected-new")
            self.assertEqual((mirror / "tables" / "unselected.csv").read_bytes(), b"unselected-old")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["source_selection"]["include_paths"],
                ["metrics/selected.json"],
            )

    def test_scoped_publish_re_attests_direct_authority_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = root / "revision"
            mirror = root / "mirror"
            local_authority = revision / "manifests" / "notebook_code_authority.json"
            local_authority.parent.mkdir(parents=True)
            local_authority.write_bytes(b'{"git_commit":"old"}')
            self._publish(revision, mirror)

            canonical_authority = mirror / "manifests" / "notebook_code_authority.json"
            canonical_authority.write_bytes(b'{"git_commit":"new-authority-with-new-size"}')
            audit_json = revision / "manifests" / "artifact_source_audit.json"
            audit_csv = revision / "tables" / "table_artifact_source_audit.csv"
            audit_json.write_bytes(b'{"status":true}')
            audit_csv.parent.mkdir(parents=True)
            audit_csv.write_bytes(b"status\ncomplete\n")

            manifest_path = self._publish(
                revision,
                mirror,
                verify_existing="size",
                refresh_existing_prefixes=["manifests/notebook_code_authority.json"],
                include_paths=[
                    "manifests/artifact_source_audit.json",
                    "tables/table_artifact_source_audit.csv",
                ],
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            rows = {row["relative_path"]: row for row in payload["artifacts"]}
            self.assertEqual(
                rows["manifests/notebook_code_authority.json"]["sha256"],
                artifact_mirror.sha256_file(canonical_authority),
            )
            self.assertEqual(
                canonical_authority.read_bytes(),
                b'{"git_commit":"new-authority-with-new-size"}',
            )
            self.assertEqual(
                payload["source_selection"]["include_paths"],
                [
                    "manifests/artifact_source_audit.json",
                    "tables/table_artifact_source_audit.csv",
                ],
            )

    def test_exact_cache_sidecar_refresh_does_not_refresh_large_sibling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = root / "revision"
            mirror = root / "mirror"
            source = revision / "metrics" / "summary.json"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"initial")
            self._publish(revision, mirror)

            cache_dir = mirror / "predictions" / "cpsc_window_cache"
            cache_dir.mkdir(parents=True)
            signal = cache_dir / "cpsc2021_preprocessed_windows_source_bound_v3.npy"
            contract = cache_dir / f"{signal.name}.contract.npz"
            signal.write_bytes(b"large-signal-placeholder")
            contract.write_bytes(b"old-contract")

            # Re-attest the direct-canonical cache files once so they enter the
            # manifest, then mutate only the small contract sidecar as happens
            # when a resumed CPSC cache is finalized.
            self._publish(
                revision,
                mirror,
                verify_existing="size",
                refresh_existing_prefixes=["predictions/cpsc_window_cache"],
            )
            old_signal_sha = artifact_mirror.sha256_file(signal)
            contract.write_bytes(b"new-source-bound-contract-with-a-different-size")

            with patch.object(
                artifact_mirror,
                "sha256_file",
                wraps=artifact_mirror.sha256_file,
            ) as sha_spy:
                manifest_path = self._publish(
                    revision,
                    mirror,
                    verify_existing="size",
                    refresh_existing_prefixes=[
                        "predictions/cpsc_window_cache/"
                        "cpsc2021_preprocessed_windows_source_bound_v3.npy.contract.npz"
                    ],
                )

            hashed_paths = {Path(call.args[0]) for call in sha_spy.call_args_list}
            self.assertIn(contract, hashed_paths)
            self.assertNotIn(signal, hashed_paths)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            rows = {row["relative_path"]: row for row in payload["artifacts"]}
            self.assertEqual(
                rows[
                    "predictions/cpsc_window_cache/"
                    "cpsc2021_preprocessed_windows_source_bound_v3.npy.contract.npz"
                ]["sha256"],
                artifact_mirror.sha256_file(contract),
            )
            self.assertEqual(
                rows[
                    "predictions/cpsc_window_cache/"
                    "cpsc2021_preprocessed_windows_source_bound_v3.npy"
                ]["sha256"],
                old_signal_sha,
            )

    def test_truncated_staging_never_replaces_complete_artifact_or_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            revision, mirror, source, destination = self._seed_published_artifact(
                Path(tmp)
            )
            manifest = mirror / "manifests" / "mirror_manifest.json"
            old_destination = destination.read_bytes()
            old_manifest = manifest.read_bytes()
            source.write_bytes(b"new-artifact-that-must-not-be-partially-published")

            def truncate_copy(source_handle, destination_handle):
                destination_handle.write(source_handle.read(7))

            with patch.object(artifact_mirror, "_copy_stream", truncate_copy):
                with self.assertRaisesRegex(RuntimeError, "Checksum mismatch in publish staging"):
                    self._publish(
                        revision,
                        mirror,
                        source_conflict_policy="source",
                    )

            self.assertEqual(destination.read_bytes(), old_destination)
            self.assertEqual(manifest.read_bytes(), old_manifest)
            self.assertFalse((mirror / artifact_mirror.PUBLISH_LOCK_NAME).exists())
            self.assertEqual(list(destination.parent.glob("*.partial.*")), [])

    def test_interrupted_staging_never_exposes_a_final_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            revision, mirror, source, destination = self._seed_published_artifact(
                Path(tmp)
            )
            manifest = mirror / "manifests" / "mirror_manifest.json"
            old_destination = destination.read_bytes()
            old_manifest = manifest.read_bytes()
            source.write_bytes(b"replacement-interrupted-before-commit")

            def interrupt_copy(source_handle, destination_handle):
                destination_handle.write(source_handle.read(5))
                destination_handle.flush()
                raise OSError("simulated disconnect")

            with patch.object(artifact_mirror, "_copy_stream", interrupt_copy):
                with self.assertRaisesRegex(OSError, "simulated disconnect"):
                    self._publish(
                        revision,
                        mirror,
                        source_conflict_policy="source",
                    )

            self.assertEqual(destination.read_bytes(), old_destination)
            self.assertEqual(manifest.read_bytes(), old_manifest)
            self.assertFalse((mirror / artifact_mirror.PUBLISH_LOCK_NAME).exists())
            self.assertEqual(list(destination.parent.glob("*.partial.*")), [])

    def test_active_writer_lock_rejects_a_concurrent_publish_even_with_recovery_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = root / "revision"
            mirror = root / "mirror"
            source = revision / "metrics" / "summary.json"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"payload")

            with artifact_mirror.PublishLock(mirror, run_id="active-writer"):
                with self.assertRaisesRegex(
                    artifact_mirror.MirrorPublishLockedError,
                    "Refusing to steal",
                ):
                    self._publish(
                        revision,
                        mirror,
                        recover_stale_lock=True,
                        stale_lock_seconds=0,
                    )
                payload = json.loads(
                    (mirror / artifact_mirror.PUBLISH_LOCK_NAME).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(payload["run_id"], "active-writer")

            self.assertFalse((mirror / artifact_mirror.PUBLISH_LOCK_NAME).exists())

    def test_separate_process_writer_blocks_publish_until_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = root / "revision"
            mirror = root / "mirror"
            ready = root / "writer.ready"
            release = root / "writer.release"
            source = revision / "metrics" / "summary.json"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"payload")
            child_code = (
                "import time\n"
                "from pathlib import Path\n"
                "from scripts.revision.artifact_mirror import PublishLock\n"
                f"mirror = Path({str(mirror)!r})\n"
                f"ready = Path({str(ready)!r})\n"
                f"release = Path({str(release)!r})\n"
                "with PublishLock(mirror, run_id='child-writer'):\n"
                "    ready.write_text('ready', encoding='utf-8')\n"
                "    deadline = time.time() + 20\n"
                "    while not release.exists() and time.time() < deadline:\n"
                "        time.sleep(0.02)\n"
            )
            process = subprocess.Popen(
                [sys.executable, "-c", child_code],
                cwd=Path(__file__).resolve().parents[1],
            )
            try:
                deadline = time.time() + 10
                while not ready.exists() and process.poll() is None and time.time() < deadline:
                    time.sleep(0.02)
                self.assertTrue(ready.exists(), "child writer did not acquire the lock")
                with self.assertRaises(artifact_mirror.MirrorPublishLockedError):
                    self._publish(revision, mirror)
            finally:
                release.write_text("release", encoding="utf-8")
                process.wait(timeout=10)
            self.assertEqual(process.returncode, 0)
            self.assertFalse((mirror / artifact_mirror.PUBLISH_LOCK_NAME).exists())

    def test_verifiably_dead_same_host_lock_is_quarantined_before_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = root / "revision"
            mirror = root / "mirror"
            source = revision / "metrics" / "summary.json"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"payload")
            mirror.mkdir(parents=True)
            lock_path = mirror / artifact_mirror.PUBLISH_LOCK_NAME
            lock_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "state": "active",
                        "run_id": "dead-writer",
                        "pid": 99999999,
                        "hostname": socket.gethostname(),
                        "created_utc": "2000-01-01T00:00:00+00:00",
                        "created_epoch": 0,
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(artifact_mirror, "_pid_is_alive", return_value=False):
                manifest_path = self._publish(revision, mirror)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            recovered = manifest["storage_safety"]["recovered_stale_lock"]
            self.assertIsNotNone(recovered)
            self.assertTrue(Path(recovered).exists())
            self.assertFalse(lock_path.exists())

    def test_stale_orphan_partial_is_quarantined_but_active_partials_are_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = root / "revision"
            mirror = root / "mirror"
            valid = mirror / "predictions" / "folds" / "fold1.npz"
            partial = valid.with_name(".fold2.npz.partial.abandoned")
            recent_partial = valid.with_name(".fold3.npz.partial.active")
            locked_partial = valid.with_name(".fold4.partial.old-writer.npz")
            locked_partial_lock = valid.with_name(".fold4.npz.write.lock")
            stale_lock = mirror / (
                artifact_mirror.PUBLISH_LOCK_NAME + ".stale.1.old-run"
            )
            valid.parent.mkdir(parents=True)
            valid.write_bytes(b"valid")
            partial.write_bytes(b"truncated")
            recent_partial.write_bytes(b"active")
            locked_partial.write_bytes(b"owned")
            locked_partial_lock.write_text("{}", encoding="utf-8")
            old_epoch = time.time() - artifact_mirror.DEFAULT_STALE_PARTIAL_SECONDS - 10
            os.utime(partial, (old_epoch, old_epoch))
            os.utime(locked_partial, (old_epoch, old_epoch))
            stale_lock.write_text("{}", encoding="utf-8")

            manifest_path = self._publish(revision, mirror, verify_existing="size")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            paths = {row["relative_path"] for row in manifest["artifacts"]}

            self.assertIn("predictions/folds/fold1.npz", paths)
            self.assertNotIn("predictions/folds/.fold2.npz.partial.abandoned", paths)
            self.assertFalse(any("artifact_mirror.publish.lock" in path for path in paths))
            self.assertFalse(partial.exists())
            quarantined = list(
                partial.parent.glob(".artifact_mirror.quarantined_partial.*")
            )
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(quarantined[0].read_bytes(), b"truncated")
            os.utime(quarantined[0], (old_epoch, old_epoch))
            self._publish(revision, mirror, verify_existing="size")
            quarantined_after_second_publish = list(
                partial.parent.glob(".artifact_mirror.quarantined_partial.*")
            )
            self.assertEqual(quarantined_after_second_publish, quarantined)
            self.assertTrue(recent_partial.exists())
            self.assertTrue(locked_partial.exists())
            self.assertTrue(locked_partial_lock.exists())
            self.assertTrue(stale_lock.exists())

    def test_manifest_rejects_parent_and_absolute_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            mirror = Path(tmp) / "mirror"
            mirror.mkdir()
            for unsafe in (
                "../outside.bin",
                "..\\outside.bin",
                "/outside.bin",
                str((Path(tmp) / "outside.bin").resolve()),
            ):
                payload = {
                    "mirror_root": str(mirror),
                    "artifacts": [
                        {
                            "relative_path": unsafe,
                            "size_bytes": 1,
                            "sha256": "0" * 64,
                        }
                    ],
                }
                with self.subTest(unsafe=unsafe):
                    with self.assertRaisesRegex(ValueError, "must stay below"):
                        artifact_mirror.normalize_manifest_rows(payload, mirror)

    def test_manifest_rejects_symlink_escape_when_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mirror = root / "mirror"
            outside = root / "outside"
            mirror.mkdir()
            outside.mkdir()
            link = mirror / "escaped"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")
            payload = {
                "mirror_root": str(mirror),
                "artifacts": [
                    {
                        "relative_path": "escaped/evidence.bin",
                        "size_bytes": 1,
                        "sha256": "0" * 64,
                    }
                ],
            }
            with self.assertRaisesRegex(ValueError, "outside its root"):
                artifact_mirror.normalize_manifest_rows(payload, mirror)

    def test_interrupted_restore_keeps_existing_destination_and_removes_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            published_revision, mirror, _, _ = self._seed_published_artifact(root)
            restore_revision = root / "restored-revision"
            destination = restore_revision / "metrics" / "summary.json"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"existing-local-artifact")

            def interrupt_copy(source_handle, destination_handle):
                destination_handle.write(source_handle.read(4))
                destination_handle.flush()
                raise OSError("simulated restore disconnect")

            with patch.object(artifact_mirror, "_copy_stream", interrupt_copy):
                with self.assertRaisesRegex(OSError, "simulated restore disconnect"):
                    self._restore(
                        restore_revision,
                        mirror,
                        replace_mismatched=True,
                    )

            self.assertEqual(destination.read_bytes(), b"existing-local-artifact")
            self.assertEqual(list(destination.parent.glob("*.partial.*")), [])
            self.assertFalse((mirror / artifact_mirror.PUBLISH_LOCK_NAME).exists())
            self.assertTrue(published_revision.exists())

    def test_restore_holds_same_lock_as_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, mirror, _, _ = self._seed_published_artifact(root)
            restore_revision = root / "restored-revision"
            with artifact_mirror.PublishLock(mirror, run_id="active-publisher"):
                with self.assertRaises(artifact_mirror.MirrorPublishLockedError):
                    self._restore(
                        restore_revision,
                        mirror,
                        replace_mismatched=True,
                    )

    def test_crash_after_artifact_replace_is_rolled_into_manifest_on_next_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            revision, mirror, source, destination = self._seed_published_artifact(
                Path(tmp)
            )
            source.write_bytes(b"replacement-committed-before-manifest")
            real_writer = artifact_mirror._atomic_write_json_verified

            def fail_manifest_only(destination_path, payload, **kwargs):
                if destination_path.name == "mirror_manifest.json":
                    raise OSError("simulated crash before manifest commit")
                return real_writer(destination_path, payload, **kwargs)

            with patch.object(
                artifact_mirror,
                "_atomic_write_json_verified",
                side_effect=fail_manifest_only,
            ):
                with self.assertRaisesRegex(OSError, "simulated crash"):
                    self._publish(
                        revision,
                        mirror,
                        source_conflict_policy="source",
                    )

            self.assertEqual(destination.read_bytes(), source.read_bytes())
            transaction = mirror / artifact_mirror.PUBLISH_TRANSACTION_NAME
            self.assertTrue(transaction.is_file())

            manifest_path = self._publish(
                revision,
                mirror,
                source_conflict_policy="source",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            row = next(
                row
                for row in manifest["artifacts"]
                if row["relative_path"] == "metrics/summary.json"
            )
            self.assertEqual(row["sha256"], artifact_mirror.sha256_file(destination))
            self.assertFalse(transaction.exists())
            self.assertEqual(
                manifest["transaction_recovery"]["rolled_forward_paths"],
                ["metrics/summary.json"],
            )


if __name__ == "__main__":
    unittest.main()
