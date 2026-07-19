"""Publish or restore revision artifacts with SHA256 verification."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import socket
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    REVISION_DIR,
    ensure_revision_dirs,
    save_json,
    sha256_file,
)


PUBLISH_LOCK_NAME = ".artifact_mirror.publish.lock"
PUBLISH_TRANSACTION_NAME = ".artifact_mirror.publish.transaction.json"
DEFAULT_STALE_LOCK_SECONDS = 6 * 60 * 60
DEFAULT_STALE_PARTIAL_SECONDS = DEFAULT_STALE_LOCK_SECONDS
ARTIFACT_MIRROR_CAPABILITIES = {
    "single_writer_lock",
    "atomic_file_replace",
    "atomic_restore_file_replace",
    "manifest_path_confinement",
    "stale_orphan_partial_quarantine",
    "recoverable_publish_transaction_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("publish", "restore"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--mirror-root", type=Path, required=True)
    subparsers.choices["publish"].add_argument(
        "--verify-existing",
        choices=("full", "size"),
        default="full",
        help=(
            "Verification for manifest rows preserved from an earlier publish. "
            "New/overwritten files are always SHA256-verified. Use size for frequent "
            "checkpoint publishes; restore still performs full SHA256 verification."
        ),
    )
    subparsers.choices["publish"].add_argument(
        "--source-conflict-policy",
        choices=("newer", "fail", "source"),
        default="newer",
        help=(
            "When a local reports/revision file differs from an existing canonical mirror file: "
            "newer publishes only if the local mtime is newer, fail rejects the conflict, and "
            "source explicitly overwrites. The default prevents stale Colab runtimes from rolling "
            "the canonical mirror backward."
        ),
    )
    subparsers.choices["publish"].add_argument(
        "--refresh-existing-prefix",
        action="append",
        default=[],
        help=(
            "Re-hash existing canonical files below this relative prefix before publishing. "
            "Use only for resumable cache/checkpoint directories that a successful runner "
            "writes directly into the canonical mirror. Repeatable."
        ),
    )
    subparsers.choices["publish"].add_argument(
        "--refresh-existing-cache-dirs",
        action="store_true",
        help=(
            "Re-hash existing canonical files below directories whose name contains 'cache' "
            "or a known resumable fold-cache root. Use after resumable runners write cache "
            "files directly to the mirror; non-cache evidence and checkpoints retain the "
            "normal conflict checks."
        ),
    )
    subparsers.choices["publish"].add_argument(
        "--recover-stale-lock",
        action="store_true",
        help=(
            "Explicitly recover a publish lock created by another host only after "
            "--stale-lock-seconds has elapsed. Same-host dead-PID locks are recovered "
            "automatically; live or unverifiable same-host locks are never stolen."
        ),
    )
    subparsers.choices["publish"].add_argument(
        "--stale-lock-seconds",
        type=float,
        default=DEFAULT_STALE_LOCK_SECONDS,
        help=(
            "Minimum age for explicit cross-host stale-lock recovery "
            f"(default: {DEFAULT_STALE_LOCK_SECONDS} seconds)."
        ),
    )
    subparsers.choices["restore"].add_argument(
        "--replace-mismatched",
        action="store_true",
        help="Replace an existing destination only after the mirror file passes its manifest checksum.",
    )
    subparsers.choices["restore"].add_argument(
        "--include-prefix",
        action="append",
        default=[],
        help="Restore only rows whose relative path begins with this prefix. Repeatable.",
    )
    subparsers.choices["restore"].add_argument(
        "--include-path",
        action="append",
        default=[],
        help="Restore this exact relative path. Repeatable.",
    )
    subparsers.choices["restore"].add_argument(
        "--exclude-cache-dirs",
        action="store_true",
        help="Exclude rows located below a directory whose name contains 'cache'.",
    )
    subparsers.choices["restore"].add_argument(
        "--recover-stale-lock",
        action="store_true",
        help=(
            "Explicitly recover a cross-host stale mirror lock before restore. "
            "Same-host dead-PID locks are recovered automatically."
        ),
    )
    subparsers.choices["restore"].add_argument(
        "--stale-lock-seconds",
        type=float,
        default=DEFAULT_STALE_LOCK_SECONDS,
        help="Minimum age for explicit cross-host stale-lock recovery.",
    )
    return parser.parse_args()


def skip_artifact(relative: Path) -> bool:
    normalized = relative.as_posix()
    name = relative.name.lower()
    return (
        normalized == "manifests/mirror_manifest.json"
        or name.startswith(".artifact_mirror.")
        # Atomic writers use same-directory hidden partial files. A Colab
        # disconnect can leave one behind before the finally block runs; it
        # must never be discovered and certified as reviewer evidence.
        or ".partial" in name
        or ".write.lock" in name
        or name.endswith((".tmp", ".part", ".lock"))
        # Logs are durable operational traces, not immutable evidence. They are
        # intentionally kept on Drive but excluded from the checksum manifest
        # so rerunning a command can safely truncate/append the same log path.
        or relative.parts[:1] == ("logs",)
        # Storage-audit outputs describe the mirror manifest and are written
        # directly to Drive. Including them would create a self-referential
        # manifest whose SHA changes immediately after every audit.
        or normalized == "metrics/pipeline_storage_audit.json"
        or normalized == "tables/table_pipeline_storage_audit.csv"
    )


def _safe_relative_path(value: str | os.PathLike[str], root: Path, *, label: str) -> Path:
    """Normalize an untrusted relative path and confine it below ``root``."""

    raw = os.fspath(value).strip()
    if not raw or "\x00" in raw:
        raise ValueError(f"{label} must be a non-empty relative path")
    normalized = raw.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(raw)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
    ):
        raise ValueError(f"{label} must stay below its root: {value!r}")
    parts = tuple(part for part in posix_path.parts if part not in {"", "."})
    if not parts:
        raise ValueError(f"{label} cannot name the root directory")

    relative = Path(*parts)
    root_resolved = Path(root).resolve()
    candidate = (root_resolved / relative).resolve(strict=False)
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside its root: {value!r}") from exc
    return relative


def _resolved_under_root(root: Path, relative: Path, *, label: str) -> Path:
    """Resolve a previously normalized path and reject symlink escapes."""

    safe_relative = _safe_relative_path(relative.as_posix(), root, label=label)
    root_resolved = Path(root).resolve()
    candidate = (root_resolved / safe_relative).resolve(strict=False)
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside its root: {relative}") from exc
    return candidate


def _relative_from_manifest_mirror(
    value: str, declared_root: str | os.PathLike[str]
) -> str:
    """Derive a relative path from legacy absolute ``mirror`` manifest fields."""

    raw = str(value).strip()
    declared_raw = os.fspath(declared_root).strip()
    use_windows = bool(PureWindowsPath(raw).drive or PureWindowsPath(declared_raw).drive)
    path_type = PureWindowsPath if use_windows else PurePosixPath
    mirror_path = path_type(raw)
    root_path = path_type(declared_raw)
    if not mirror_path.is_absolute():
        return raw
    if not root_path.is_absolute():
        raise ValueError("Mirror manifest declares an absolute artifact without an absolute root")
    try:
        return mirror_path.relative_to(root_path).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Mirror manifest artifact is outside its declared root: {value!r}"
        ) from exc


def normalize_refresh_prefixes(prefixes: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in prefixes:
        path = Path(str(value).strip().replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Refresh prefix must be a safe relative path: {value!r}")
        prefix = path.as_posix().strip("./")
        if not prefix:
            raise ValueError("Refresh prefix cannot be empty or the mirror root")
        normalized.append(prefix)
    return tuple(dict.fromkeys(normalized))


def matches_refresh_prefix(relative: Path, prefixes: tuple[str, ...]) -> bool:
    normalized = relative.as_posix()
    return any(
        normalized == prefix or normalized.startswith(prefix + "/")
        for prefix in prefixes
    )


DIRECT_CANONICAL_CACHE_PREFIXES = (
    "predictions/folds",
    "predictions/external_comparator_folds",
)


class MirrorPublishLockedError(RuntimeError):
    """Raised when another writer owns the canonical mirror publish lock."""


def _fsync_directory(directory: Path) -> None:
    """Durably persist a directory entry update where the platform supports it."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(str(directory), flags)
    except OSError:
        # Windows does not permit opening directories through os.open. The
        # file itself is still flushed and os.replace remains atomic there.
        return
    try:
        try:
            os.fsync(fd)
        except OSError as exc:
            unsupported = {
                errno.EBADF,
                errno.EINVAL,
                getattr(errno, "ENOTSUP", errno.EINVAL),
            }
            if exc.errno not in unsupported:
                raise
    finally:
        os.close(fd)


def _pid_is_alive(pid: int) -> bool | None:
    """Return process liveness, or None when the host cannot decide safely."""

    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            process_query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
                process_query_limited_information,
                False,
                pid,
            )
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
                return True
            error = ctypes.windll.kernel32.GetLastError()  # type: ignore[attr-defined]
            if error == 87:  # ERROR_INVALID_PARAMETER: no such PID.
                return False
            return None
        except (AttributeError, OSError):
            return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        return None
    return True


def _read_lock_payload(path: Path) -> tuple[dict, bytes]:
    try:
        raw = path.read_bytes()
    except OSError:
        return {}, b""
    try:
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("lock payload is not an object")
        return payload, raw
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {}, raw


class PublishLock:
    """Cross-runtime single-writer lock backed by atomic exclusive creation.

    A same-host lock is reclaimed only when its PID is verifiably dead. Locks
    from another host (for example, a disconnected Colab VM) are never stolen
    implicitly; recovery requires an explicit opt-in and a minimum stale age.
    """

    def __init__(
        self,
        mirror_root: Path,
        *,
        run_id: str | None = None,
        recover_stale_lock: bool = False,
        stale_lock_seconds: float = DEFAULT_STALE_LOCK_SECONDS,
    ) -> None:
        if stale_lock_seconds < 0:
            raise ValueError("stale_lock_seconds must be non-negative")
        self.mirror_root = Path(mirror_root)
        self.path = self.mirror_root / PUBLISH_LOCK_NAME
        self.run_id = run_id or uuid.uuid4().hex
        self.hostname = socket.gethostname()
        self.recover_stale_lock = bool(recover_stale_lock)
        self.stale_lock_seconds = float(stale_lock_seconds)
        self.acquired = False
        self.recovered_lock_path: Path | None = None

    def _metadata(self) -> dict:
        return {
            "schema_version": 1,
            "state": "active",
            "run_id": self.run_id,
            "pid": os.getpid(),
            "hostname": self.hostname,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "created_epoch": time.time(),
        }

    def _write_new_lock(self) -> None:
        self.mirror_root.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            str(self.path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        try:
            payload = json.dumps(
                self._metadata(),
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
            with os.fdopen(fd, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(fd)
        _fsync_directory(self.mirror_root)
        self.acquired = True

    def _lock_age_seconds(self, payload: dict) -> float:
        created_epoch = payload.get("created_epoch")
        try:
            return max(0.0, time.time() - float(created_epoch))
        except (TypeError, ValueError):
            try:
                return max(0.0, time.time() - self.path.stat().st_mtime)
            except OSError:
                return 0.0

    def _recovery_reason(self, payload: dict) -> str | None:
        owner_host = str(payload.get("hostname", ""))
        try:
            owner_pid = int(payload.get("pid", -1))
        except (TypeError, ValueError):
            owner_pid = -1
        if owner_host == self.hostname:
            alive = _pid_is_alive(owner_pid)
            if alive is True:
                return None
            if alive is False:
                return f"same-host owner PID {owner_pid} is not alive"
            return None

        age = self._lock_age_seconds(payload)
        if self.recover_stale_lock and age >= self.stale_lock_seconds:
            return (
                f"explicit cross-host stale recovery after {age:.1f}s "
                f"(owner_host={owner_host or 'unknown'})"
            )
        return None

    def _quarantine_stale_lock(self, expected_raw: bytes, reason: str) -> None:
        current_payload, current_raw = _read_lock_payload(self.path)
        if current_raw != expected_raw or not self.path.exists():
            raise MirrorPublishLockedError(
                "Publish lock changed during stale-lock validation; refusing recovery"
            )
        token = str(current_payload.get("run_id", "unknown"))[:16]
        quarantine = self.mirror_root / (
            f"{PUBLISH_LOCK_NAME}.stale.{int(time.time())}.{token}.{self.run_id}"
        )
        os.replace(self.path, quarantine)
        _fsync_directory(self.mirror_root)
        self.recovered_lock_path = quarantine
        print(f"Quarantined stale publish lock: {quarantine} | reason={reason}")

    def acquire(self) -> "PublishLock":
        for _ in range(3):
            try:
                self._write_new_lock()
                return self
            except FileExistsError:
                payload, raw = _read_lock_payload(self.path)
                reason = self._recovery_reason(payload)
                if reason is not None:
                    self._quarantine_stale_lock(raw, reason)
                    continue
                owner = {
                    key: payload.get(key)
                    for key in ("run_id", "pid", "hostname", "created_utc")
                }
                raise MirrorPublishLockedError(
                    "Canonical mirror already has an active or unverifiable publish lock: "
                    f"{self.path} owner={owner}. Refusing to steal it."
                )
        raise MirrorPublishLockedError(
            f"Could not acquire canonical mirror publish lock after recovery: {self.path}"
        )

    def assert_owned(self) -> None:
        if not self.acquired:
            raise MirrorPublishLockedError("Publish lock is not acquired")
        payload, _ = _read_lock_payload(self.path)
        if payload.get("run_id") != self.run_id:
            raise MirrorPublishLockedError(
                "Publish lock ownership changed before atomic commit; aborting publish"
            )

    def release(self) -> None:
        if not self.acquired:
            return
        self.assert_owned()
        self.path.unlink()
        _fsync_directory(self.mirror_root)
        self.acquired = False

    def __enter__(self) -> "PublishLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


def _partial_path(destination: Path, run_id: str) -> Path:
    return destination.with_name(f".{destination.name}.partial.{run_id}")


def _partial_write_lock_candidates(partial: Path) -> tuple[Path, ...]:
    """Return cache-writer locks that can own a same-directory partial."""

    name = partial.name
    if ".partial." not in name:
        return ()
    prefix, suffix = name.split(".partial.", 1)
    candidates = [partial.with_name(f"{prefix}.write.lock")]
    suffix_tail = Path(suffix).suffix
    if suffix_tail:
        candidates.append(partial.with_name(f"{prefix}{suffix_tail}.write.lock"))
    return tuple(dict.fromkeys(candidates))


def _quarantine_stale_orphan_partials(
    mirror_root: Path,
    *,
    active_run_id: str,
    stale_seconds: float = DEFAULT_STALE_PARTIAL_SECONDS,
) -> list[str]:
    """Quarantine old partials that have no writer lock and are not this run's."""

    if stale_seconds < 0:
        raise ValueError("stale partial age must be non-negative")
    mirror_root = Path(mirror_root)
    if not mirror_root.exists():
        return []
    quarantined: list[str] = []
    now = time.time()
    for candidate in sorted(mirror_root.rglob("*")):
        if not candidate.is_file() or ".partial." not in candidate.name:
            continue
        if candidate.name.endswith(f".partial.{active_run_id}"):
            continue
        try:
            age = max(0.0, now - candidate.stat().st_mtime)
        except OSError:
            continue
        if age < stale_seconds:
            continue
        # A cache partial with a writer lock is not an orphan. Conservatively
        # retain it even when old because a cross-host writer cannot be proven
        # dead from this process.
        if any(lock.exists() for lock in _partial_write_lock_candidates(candidate)):
            continue
        relative = candidate.relative_to(mirror_root).as_posix()
        quarantine = candidate.with_name(
            ".artifact_mirror.quarantined_partial."
            f"{int(now)}.{uuid.uuid4().hex}.{candidate.name.lstrip('.')[:80]}"
        )
        os.replace(candidate, quarantine)
        _fsync_directory(candidate.parent)
        quarantined.append(relative)
        print(f"Quarantined stale orphan partial: {relative} -> {quarantine.name}")
    return quarantined


def _copy_stream(source_handle, destination_handle) -> None:
    shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)


def _atomic_copy_verified(
    source: Path,
    destination: Path,
    *,
    run_id: str,
    ownership_check: Callable[[], None],
) -> str:
    """Copy one artifact without exposing a truncated final-name file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_path(destination, run_id)
    if partial.exists():
        raise RuntimeError(f"Publish staging path already exists: {partial}")
    source_sha = sha256_file(source)
    source_size = source.stat().st_size
    try:
        with source.open("rb") as source_handle, partial.open("xb") as partial_handle:
            _copy_stream(source_handle, partial_handle)
            partial_handle.flush()
            os.fsync(partial_handle.fileno())
        partial_size = partial.stat().st_size
        partial_sha = sha256_file(partial)
        if partial_size != source_size or partial_sha != source_sha:
            raise RuntimeError(
                f"Checksum mismatch in publish staging for {destination.name}: "
                f"source_size={source_size} partial_size={partial_size}"
            )
        shutil.copystat(source, partial)
        ownership_check()
        os.replace(partial, destination)
        _fsync_directory(destination.parent)
        return partial_sha
    finally:
        if partial.exists():
            partial.unlink()


def _atomic_restore_verified(
    source: Path,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    run_id: str,
    ownership_check: Callable[[], None],
) -> None:
    """Restore one verified artifact without exposing an incomplete final file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_path(destination, run_id)
    if partial.exists():
        raise RuntimeError(f"Restore staging path already exists: {partial}")
    try:
        ownership_check()
        with source.open("rb") as source_handle, partial.open("xb") as partial_handle:
            _copy_stream(source_handle, partial_handle)
            partial_handle.flush()
            os.fsync(partial_handle.fileno())

        partial_size = partial.stat().st_size
        partial_sha = sha256_file(partial)
        if partial_size != expected_size or partial_sha != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch in restore staging for {destination.name}: "
                f"expected_size={expected_size} partial_size={partial_size}"
            )
        shutil.copystat(source, partial)
        ownership_check()
        os.replace(partial, destination)
        _fsync_directory(destination.parent)

        # Read back the committed final name. The pre-commit readback prevents
        # truncation from replacing an existing destination; this check catches
        # storage corruption or a non-atomic filesystem implementation.
        if (
            destination.stat().st_size != expected_size
            or sha256_file(destination) != expected_sha256
        ):
            raise RuntimeError(f"Checksum mismatch after restoring {destination}")
    finally:
        if partial.exists():
            partial.unlink()
            _fsync_directory(partial.parent)


def _atomic_write_json_verified(
    destination: Path,
    payload: dict,
    *,
    run_id: str,
    ownership_check: Callable[[], None],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_path(destination, run_id)
    if partial.exists():
        raise RuntimeError(f"Manifest staging path already exists: {partial}")
    encoded = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    expected_sha = hashlib.sha256(encoded).hexdigest()
    try:
        with partial.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if partial.stat().st_size != len(encoded) or sha256_file(partial) != expected_sha:
            raise RuntimeError("Checksum mismatch in mirror manifest staging")
        ownership_check()
        os.replace(partial, destination)
        _fsync_directory(destination.parent)
    finally:
        if partial.exists():
            partial.unlink()


def _write_publish_transaction(
    mirror_root: Path,
    payload: dict,
    *,
    publish_lock: PublishLock,
) -> Path:
    path = mirror_root / PUBLISH_TRANSACTION_NAME
    _atomic_write_json_verified(
        path,
        payload,
        run_id=publish_lock.run_id,
        ownership_check=publish_lock.assert_owned,
    )
    return path


def _recover_publish_transaction(
    mirror_root: Path,
    *,
    publish_lock: PublishLock,
) -> dict | None:
    """Roll a prior interrupted file commit into the authenticated manifest."""

    transaction_path = mirror_root / PUBLISH_TRANSACTION_NAME
    if not transaction_path.exists():
        return None
    payload = json.loads(transaction_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise RuntimeError(f"Unsupported publish transaction: {transaction_path}")

    manifest_path = mirror_root / "manifests" / "mirror_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = {
            row["relative_path"]: row
            for row in normalize_manifest_rows(manifest, mirror_root)
        }
    else:
        manifest = {}
        rows = {}

    recovered: list[str] = []
    discarded_pending: list[str] = []
    prior_run_id = str(payload.get("run_id", ""))
    for update in payload.get("updates", []):
        relative = _safe_relative_path(
            str(update["relative_path"]),
            mirror_root,
            label="publish transaction relative_path",
        )
        destination = _resolved_under_root(
            mirror_root,
            relative,
            label="publish transaction destination",
        )
        expected_sha = str(update["sha256"])
        expected_size = int(update["size_bytes"])
        final_matches = (
            destination.is_file()
            and destination.stat().st_size == expected_size
            and sha256_file(destination) == expected_sha
        )
        if final_matches:
            rows[relative.as_posix()] = {
                "relative_path": relative.as_posix(),
                "size_bytes": expected_size,
                "sha256": expected_sha,
                "attestation_scope": str(
                    update.get("attestation_scope", "byte_integrity_only")
                ),
            }
            recovered.append(relative.as_posix())
        else:
            previous = rows.get(relative.as_posix())
            previous_matches = (
                previous is None
                and not destination.exists()
            ) or (
                previous is not None
                and destination.is_file()
                and destination.stat().st_size == int(previous["size_bytes"])
                and sha256_file(destination) == str(previous["sha256"])
            )
            if not previous_matches:
                raise RuntimeError(
                    "Interrupted publish cannot be rolled forward or safely ignored: "
                    f"{relative}"
                )
            discarded_pending.append(relative.as_posix())
        if prior_run_id:
            partial = _partial_path(destination, prior_run_id)
            partial.unlink(missing_ok=True)

    recovered_manifest = {
        **manifest,
        "schema_version": max(4, int(manifest.get("schema_version", 0))),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "publish_run_id": publish_lock.run_id,
        "artifact_count": len(rows),
        "artifacts": [rows[key] for key in sorted(rows)],
        "transaction_recovery": {
            "recovered_from_run_id": prior_run_id,
            "rolled_forward_paths": recovered,
            "discarded_uncommitted_paths": discarded_pending,
        },
    }
    _atomic_write_json_verified(
        manifest_path,
        recovered_manifest,
        run_id=publish_lock.run_id,
        ownership_check=publish_lock.assert_owned,
    )
    transaction_path.unlink()
    _fsync_directory(mirror_root)
    print(
        "Recovered interrupted mirror transaction: "
        f"rolled_forward={len(recovered)} discarded={len(discarded_pending)}"
    )
    return recovered_manifest


def is_cache_artifact(relative: Path) -> bool:
    named_cache_dir = any("cache" in part.lower() for part in relative.parts[:-1])
    return named_cache_dir or matches_refresh_prefix(
        relative, DIRECT_CANONICAL_CACHE_PREFIXES
    )


def can_refresh_existing(
    relative: Path,
    refresh_prefixes: tuple[str, ...],
    refresh_existing_cache_dirs: bool,
) -> bool:
    return matches_refresh_prefix(relative, refresh_prefixes) or (
        refresh_existing_cache_dirs and is_cache_artifact(relative)
    )


def _publish_locked(
    mirror_root: Path,
    publish_lock: PublishLock,
    verify_existing: str = "full",
    source_conflict_policy: str = "newer",
    refresh_existing_prefixes: list[str] | tuple[str, ...] = (),
    refresh_existing_cache_dirs: bool = False,
    transaction_recovery: dict | None = None,
    quarantined_orphan_partials: list[str] | None = None,
) -> Path:
    if verify_existing not in {"full", "size"}:
        raise ValueError(f"Unsupported existing verification mode: {verify_existing}")
    if source_conflict_policy not in {"newer", "fail", "source"}:
        raise ValueError(f"Unsupported source conflict policy: {source_conflict_policy}")
    refresh_prefixes = normalize_refresh_prefixes(refresh_existing_prefixes)
    ensure_revision_dirs()
    mirror_root.mkdir(parents=True, exist_ok=True)
    manifest_path = mirror_root / "manifests" / "mirror_manifest.json"
    existing_rows: dict[str, dict] = {}
    refreshed_existing_paths: list[str] = []
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in normalize_manifest_rows(payload, mirror_root):
            relative = Path(row["relative_path"])
            if skip_artifact(relative):
                continue
            source = mirror_root / relative
            if not source.exists():
                raise FileNotFoundError(
                    f"Existing mirror manifest references a missing file: {source}"
                )
            actual_size = source.stat().st_size
            refresh_allowed = can_refresh_existing(
                relative, refresh_prefixes, refresh_existing_cache_dirs
            )
            size_mismatch = row["size_bytes"] >= 0 and actual_size != row["size_bytes"]
            if size_mismatch and not refresh_allowed:
                raise RuntimeError(f"Existing mirror size mismatch before publish: {relative}")
            # Explicitly refreshed paths are always hashed so same-size direct
            # canonical updates cannot leave a stale SHA in the manifest.
            if refresh_allowed or verify_existing == "full" or row["size_bytes"] < 0:
                actual_sha = sha256_file(source)
                if (size_mismatch or actual_sha != row["sha256"]) and refresh_allowed:
                    refreshed_existing_paths.append(relative.as_posix())
                elif actual_sha != row["sha256"]:
                    raise RuntimeError(
                        f"Existing mirror checksum mismatch before publish: {relative}"
                    )
            else:
                actual_sha = row["sha256"]
            existing_rows[relative.as_posix()] = {
                "relative_path": relative.as_posix(),
                "size_bytes": actual_size,
                "sha256": actual_sha,
                "attestation_scope": str(
                    row.get("attestation_scope", "byte_integrity_only_legacy")
                ),
            }

    # Long-running runners may write resumable fold/stress caches directly to
    # the canonical Drive tree. Discover them on the next publish so they enter
    # the verified manifest instead of remaining invisible to future restores.
    discovered_unmanifested = 0
    for source in sorted(mirror_root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(mirror_root)
        key = relative.as_posix()
        if skip_artifact(relative) or key in existing_rows:
            continue
        actual_sha = sha256_file(source)
        existing_rows[key] = {
            "relative_path": key,
            "size_bytes": source.stat().st_size,
            "sha256": actual_sha,
            "attestation_scope": "byte_integrity_only_discovered_canonical",
        }
        discovered_unmanifested += 1

    merged_rows = dict(existing_rows)
    published_from_source = 0
    published_relative_paths: set[str] = set()
    skipped_stale_source_paths: list[str] = []
    preserved_direct_canonical_paths: list[str] = []
    transaction_path = mirror_root / PUBLISH_TRANSACTION_NAME
    transaction_payload = {
        "schema_version": 1,
        "run_id": publish_lock.run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "updates": [],
    }
    for source in sorted(REVISION_DIR.rglob("*")):
        if not source.is_file() or source.name == ".gitkeep":
            continue
        relative = source.relative_to(REVISION_DIR)
        if skip_artifact(relative):
            continue
        destination = mirror_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_sha = sha256_file(source)
        destination_sha = None
        if destination.exists():
            existing_row = merged_rows.get(relative.as_posix())
            if (
                existing_row
                and int(existing_row.get("size_bytes", -1)) == destination.stat().st_size
                and existing_row.get("sha256")
            ):
                destination_sha = str(existing_row["sha256"])
            else:
                destination_sha = sha256_file(destination)

        if destination_sha == source_sha:
            merged_rows[relative.as_posix()] = {
                "relative_path": relative.as_posix(),
                "size_bytes": destination.stat().st_size,
                "sha256": destination_sha,
                "attestation_scope": str(
                    merged_rows.get(relative.as_posix(), {}).get(
                        "attestation_scope", "byte_integrity_only"
                    )
                ),
            }
            continue

        if destination.exists() and destination_sha != source_sha:
            refresh_allowed = can_refresh_existing(
                relative, refresh_prefixes, refresh_existing_cache_dirs
            )
            if refresh_allowed:
                # A refresh opt-in means this cache was written directly to the
                # canonical mirror. Never roll it back from a stale runtime copy,
                # even when Drive/local timestamp resolution is ambiguous.
                preserved_direct_canonical_paths.append(relative.as_posix())
                continue
            if source_conflict_policy == "fail":
                raise RuntimeError(
                    f"Local/canonical publish conflict for {relative}; rerun with an explicit policy"
                )
            if (
                source_conflict_policy == "newer"
                and source.stat().st_mtime_ns <= destination.stat().st_mtime_ns
            ):
                skipped_stale_source_paths.append(relative.as_posix())
                continue

        transaction_payload["updates"].append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": source.stat().st_size,
                "sha256": source_sha,
                "attestation_scope": "byte_integrity_only_published_runtime",
            }
        )
        _write_publish_transaction(
            mirror_root,
            transaction_payload,
            publish_lock=publish_lock,
        )
        destination_sha = _atomic_copy_verified(
            source,
            destination,
            run_id=publish_lock.run_id,
            ownership_check=publish_lock.assert_owned,
        )
        merged_rows[relative.as_posix()] = {
            "relative_path": relative.as_posix(),
            "size_bytes": destination.stat().st_size,
            "sha256": destination_sha,
            "attestation_scope": "byte_integrity_only_published_runtime",
        }
        published_from_source += 1
        published_relative_paths.add(relative.as_posix())

    artifacts = [merged_rows[key] for key in sorted(merged_rows)]

    manifest = {
        "schema_version": 4,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "publish_run_id": publish_lock.run_id,
        "storage_safety": {
            "artifact_commit": "same_directory_partial_fsync_sha256_os_replace",
            "manifest_commit": "same_directory_partial_fsync_sha256_os_replace",
            "single_writer_lock": PUBLISH_LOCK_NAME,
            "recoverable_transaction_journal": PUBLISH_TRANSACTION_NAME,
            "quarantined_stale_orphan_partials": sorted(
                quarantined_orphan_partials or []
            ),
            "recovered_stale_lock": (
                str(publish_lock.recovered_lock_path)
                if publish_lock.recovered_lock_path is not None
                else None
            ),
        },
        "transaction_recovery": transaction_recovery,
        "source_root": str(REVISION_DIR),
        "mirror_root": str(mirror_root),
        "artifact_count": len(artifacts),
        "published_from_source_count": published_from_source,
        "preserved_existing_count": len(
            set(existing_rows) - published_relative_paths
        ),
        "publish_mode": "merge_verified_no_prune",
        "trust_scope": (
            "sha256_byte_integrity_only; producer and experiment provenance must be "
            "validated from artifact-internal contracts by the forensic audit"
        ),
        "existing_verification_mode": verify_existing,
        "source_conflict_policy": source_conflict_policy,
        "refresh_existing_prefixes": list(refresh_prefixes),
        "refresh_existing_cache_dirs": bool(refresh_existing_cache_dirs),
        "refreshed_existing_count": len(refreshed_existing_paths),
        "refreshed_existing_paths": sorted(refreshed_existing_paths),
        "skipped_stale_source_count": len(skipped_stale_source_paths),
        "skipped_stale_source_paths": skipped_stale_source_paths,
        "preserved_direct_canonical_count": len(preserved_direct_canonical_paths),
        "preserved_direct_canonical_paths": sorted(preserved_direct_canonical_paths),
        "discovered_unmanifested_count": discovered_unmanifested,
        "artifacts": artifacts,
    }
    _atomic_write_json_verified(
        manifest_path,
        manifest,
        run_id=publish_lock.run_id,
        ownership_check=publish_lock.assert_owned,
    )
    if transaction_path.exists():
        transaction_path.unlink()
        _fsync_directory(mirror_root)
    print(
        f"Published and byte-verified {published_from_source} source artifacts; "
        f"merged manifest contains {len(artifacts)} artifacts: {mirror_root}"
    )
    if refreshed_existing_paths:
        print(
            "Re-hashed direct canonical updates: "
            + ", ".join(sorted(refreshed_existing_paths))
        )
    print(f"Wrote: {manifest_path}")
    return manifest_path


def publish(
    mirror_root: Path,
    verify_existing: str = "full",
    source_conflict_policy: str = "newer",
    refresh_existing_prefixes: list[str] | tuple[str, ...] = (),
    refresh_existing_cache_dirs: bool = False,
    recover_stale_lock: bool = False,
    stale_lock_seconds: float = DEFAULT_STALE_LOCK_SECONDS,
) -> Path:
    """Publish revision artifacts under one crash-safe writer transaction."""

    mirror_root = Path(mirror_root)
    with PublishLock(
        mirror_root,
        recover_stale_lock=recover_stale_lock,
        stale_lock_seconds=stale_lock_seconds,
    ) as publish_lock:
        recovered_manifest = _recover_publish_transaction(
            mirror_root,
            publish_lock=publish_lock,
        )
        quarantined_orphan_partials = _quarantine_stale_orphan_partials(
            mirror_root,
            active_run_id=publish_lock.run_id,
        )
        return _publish_locked(
            mirror_root,
            publish_lock,
            verify_existing,
            source_conflict_policy,
            refresh_existing_prefixes,
            refresh_existing_cache_dirs,
            (
                recovered_manifest.get("transaction_recovery")
                if recovered_manifest is not None
                else None
            ),
            quarantined_orphan_partials,
        )


def normalize_manifest_rows(payload: dict, mirror_root: Path) -> list[dict]:
    normalized = []
    declared_root = payload.get("mirror_root", os.fspath(mirror_root))
    seen_paths: set[str] = set()
    for row in payload.get("artifacts", []):
        relative = row.get("relative_path")
        if not relative and row.get("mirror"):
            relative = _relative_from_manifest_mirror(str(row["mirror"]), declared_root)
        if not relative or not row.get("sha256"):
            raise ValueError("Mirror manifest row lacks relative_path/mirror or sha256")
        safe_relative = _safe_relative_path(
            str(relative),
            mirror_root,
            label="mirror manifest relative_path",
        )
        normalized_path = safe_relative.as_posix()
        if normalized_path in seen_paths:
            raise ValueError(f"Mirror manifest contains duplicate path: {normalized_path}")
        seen_paths.add(normalized_path)
        normalized.append(
            {
                "relative_path": normalized_path,
                "size_bytes": int(row.get("size_bytes", -1)),
                "sha256": str(row["sha256"]),
                "attestation_scope": str(
                    row.get("attestation_scope", "byte_integrity_only_legacy")
                ),
            }
        )
    return normalized


def _restore_locked(
    mirror_root: Path,
    mirror_lock: PublishLock,
    replace_mismatched: bool,
    include_prefixes: list[str] | None = None,
    include_paths: list[str] | None = None,
    exclude_cache_dirs: bool = False,
    quarantined_orphan_partials: list[str] | None = None,
) -> Path:
    manifest_path = _resolved_under_root(
        mirror_root,
        Path("manifests/mirror_manifest.json"),
        label="mirror manifest path",
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Verified restore requires: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = normalize_manifest_rows(payload, mirror_root)
    if not rows:
        raise ValueError("Mirror manifest contains no artifacts")

    prefixes = tuple(
        _safe_relative_path(item, mirror_root, label="restore include-prefix")
        .as_posix()
        .rstrip("/")
        + "/"
        for item in (include_prefixes or [])
        if str(item).strip().rstrip("/\\")
    )
    exact_paths = {
        _safe_relative_path(item, mirror_root, label="restore include-path").as_posix()
        for item in (include_paths or [])
    }
    if prefixes or exact_paths:
        rows = [
            row
            for row in rows
            if row["relative_path"] in exact_paths
            or row["relative_path"].startswith(prefixes)
        ]
        if not rows:
            raise ValueError("Mirror selection matched no manifest artifacts")
    if exclude_cache_dirs:
        rows = [
            row
            for row in rows
            if not any(
                "cache" in part.lower()
                for part in Path(row["relative_path"]).parts[:-1]
            )
        ]
        if not rows:
            raise ValueError("Mirror selection contains only excluded cache-directory artifacts")

    restored, reused = [], []
    for row in rows:
        relative = Path(row["relative_path"])
        if skip_artifact(relative):
            continue
        source = _resolved_under_root(
            mirror_root,
            relative,
            label="mirror artifact source",
        )
        if not source.exists():
            raise FileNotFoundError(f"Mirror manifest references a missing file: {source}")
        actual_size = source.stat().st_size
        actual_sha = sha256_file(source)
        if row["size_bytes"] >= 0 and actual_size != row["size_bytes"]:
            raise RuntimeError(f"Mirror size mismatch: {relative}")
        if actual_sha != row["sha256"]:
            raise RuntimeError(f"Mirror checksum mismatch: {relative}")

        destination = _resolved_under_root(
            REVISION_DIR,
            relative,
            label="restore destination",
        )
        if destination.exists():
            destination_sha = sha256_file(destination)
            if destination_sha == actual_sha:
                reused.append(relative.as_posix())
                continue
            if not replace_mismatched:
                raise RuntimeError(
                    f"Destination differs from verified mirror: {destination}. "
                    "Use --replace-mismatched to restore the verified copy."
                )
        _atomic_restore_verified(
            source,
            destination,
            expected_size=actual_size,
            expected_sha256=actual_sha,
            run_id=mirror_lock.run_id,
            ownership_check=mirror_lock.assert_owned,
        )
        restored.append(relative.as_posix())

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mirror_manifest": str(manifest_path),
        "mirror_manifest_sha256": sha256_file(manifest_path),
        "mirror_operation_run_id": mirror_lock.run_id,
        "storage_safety": {
            "restore_commit": "same_directory_partial_fsync_sha256_readback_os_replace",
            "single_writer_lock": PUBLISH_LOCK_NAME,
            "quarantined_stale_orphan_partials": sorted(
                quarantined_orphan_partials or []
            ),
        },
        "verified_artifact_count": len(restored) + len(reused),
        "selection": {
            "include_prefixes": list(include_prefixes or []),
            "include_paths": sorted(exact_paths),
            "exclude_cache_dirs": bool(exclude_cache_dirs),
        },
        "restored": restored,
        "reused": reused,
    }
    report_path = REVISION_DIR / "manifests" / "mirror_restore_report.json"
    save_json(report_path, report)
    print(f"Byte-verified mirror artifacts: {len(rows)}")
    print(f"Restored: {len(restored)} | reused: {len(reused)}")
    print(f"Wrote: {report_path}")
    return report_path


def restore(
    mirror_root: Path,
    replace_mismatched: bool,
    include_prefixes: list[str] | None = None,
    include_paths: list[str] | None = None,
    exclude_cache_dirs: bool = False,
    recover_stale_lock: bool = False,
    stale_lock_seconds: float = DEFAULT_STALE_LOCK_SECONDS,
) -> Path:
    """Restore verified files while holding the canonical mirror writer lock."""

    mirror_root = Path(mirror_root)
    with PublishLock(
        mirror_root,
        recover_stale_lock=recover_stale_lock,
        stale_lock_seconds=stale_lock_seconds,
    ) as mirror_lock:
        _recover_publish_transaction(mirror_root, publish_lock=mirror_lock)
        quarantined_orphan_partials = _quarantine_stale_orphan_partials(
            mirror_root,
            active_run_id=mirror_lock.run_id,
        )
        return _restore_locked(
            mirror_root,
            mirror_lock,
            replace_mismatched,
            include_prefixes,
            include_paths,
            exclude_cache_dirs,
            quarantined_orphan_partials,
        )


def main() -> None:
    args = parse_args()
    if args.command == "publish":
        publish(
            args.mirror_root,
            args.verify_existing,
            args.source_conflict_policy,
            args.refresh_existing_prefix,
            args.refresh_existing_cache_dirs,
            args.recover_stale_lock,
            args.stale_lock_seconds,
        )
    else:
        restore(
            args.mirror_root,
            args.replace_mismatched,
            args.include_prefix,
            args.include_path,
            args.exclude_cache_dirs,
            args.recover_stale_lock,
            args.stale_lock_seconds,
        )


if __name__ == "__main__":
    main()
