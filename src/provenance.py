"""Stable provenance fingerprints shared by training and evaluation.

The reviewer pipeline treats cache identity as a content contract.  File names
and record identifiers alone are insufficient because the same identifiers can
be paired with a different archive, preprocessing implementation, or signal
array after a partial Drive restore.
"""

from __future__ import annotations

import hashlib
import json
import os
import errno
import socket
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np


def record_order_fingerprint(record_ids: np.ndarray) -> str:
    """Hash record identifiers in order with unambiguous length framing."""
    digest = hashlib.sha256()
    for value in np.asarray(record_ids).astype(str):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()[:16]


def file_sha256(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Return the full SHA256 of a file using bounded memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ndarray_sha256(array: np.ndarray, rows_per_chunk: int = 64) -> str:
    """Hash dtype, shape, and every value of an ndarray without a full copy."""

    values = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(values.shape), separators=(",", ":")).encode("ascii"))
    if values.ndim == 0:
        digest.update(np.ascontiguousarray(values).tobytes(order="C"))
        return digest.hexdigest()
    for start in range(0, len(values), rows_per_chunk):
        chunk = np.ascontiguousarray(values[start : start + rows_per_chunk])
        digest.update(memoryview(chunk).cast("B"))
    return digest.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    """Hash a JSON-compatible contract with stable ordering and separators."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_bundle_sha256(paths: Iterable[str | os.PathLike[str]]) -> str:
    """Hash source files with clone-independent path framing."""

    resolved = [Path(item).resolve() for item in paths]
    if not resolved:
        return hashlib.sha256(b"").hexdigest()
    common_root = Path(os.path.commonpath([str(path.parent) for path in resolved]))
    labelled = sorted(
        ((path.relative_to(common_root).as_posix(), path) for path in resolved),
        key=lambda item: item[0],
    )
    digest = hashlib.sha256()
    for label, path in labelled:
        encoded = label.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def _same_host_pid_is_alive(metadata: dict) -> bool | None:
    """Return liveness for a same-host owner, or None for a foreign host."""

    if str(metadata.get("hostname", "")) != socket.gethostname():
        return None
    try:
        pid = int(metadata.get("pid", -1))
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        return True
    return True


@contextmanager
def exclusive_cache_writer(
    destination: str | os.PathLike[str],
    *,
    stale_seconds: float = 6 * 60 * 60,
):
    """Serialize final-name cache commits and recover only clearly stale locks."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(f".{path.name}.write.lock")
    run_id = uuid.uuid4().hex
    metadata = {
        "schema_version": 2,
        "run_id": run_id,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "created_epoch": time.time(),
    }
    for _ in range(2):
        try:
            fd = os.open(str(lock), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            try:
                existing = json.loads(lock.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            age = max(
                0.0,
                time.time() - float(existing.get("created_epoch", lock.stat().st_mtime)),
            )
            owner_alive = _same_host_pid_is_alive(existing)
            if owner_alive is True:
                raise RuntimeError(
                    f"Cache writer lock belongs to a live process: {lock} owner={existing}"
                )
            if owner_alive is None and os.environ.get(
                "ECG_RAMBA_RECOVER_FOREIGN_STALE_LOCK", "0"
            ) != "1":
                raise RuntimeError(
                    f"Cache writer lock belongs to another host and cannot be recovered "
                    f"automatically: {lock} owner={existing}. Set "
                    "ECG_RAMBA_RECOVER_FOREIGN_STALE_LOCK=1 only after confirming that "
                    "the foreign runtime is no longer active."
                )
            if age < stale_seconds:
                raise RuntimeError(
                    f"Cache writer lock is active or unverifiable: {lock} owner={existing}"
                )
            quarantine = lock.with_name(f"{lock.name}.stale.{int(time.time())}.{run_id}")
            os.replace(lock, quarantine)
            continue
        else:
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                lock.unlink(missing_ok=True)
                raise
            break
    else:
        raise RuntimeError(f"Could not acquire cache writer lock: {lock}")
    try:
        yield
    finally:
        try:
            current = json.loads(lock.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
        if current.get("run_id") != run_id:
            raise RuntimeError(f"Cache writer lock ownership changed: {lock}")
        lock.unlink()


def save_npz_atomic(path: str | os.PathLike[str], **arrays: np.ndarray) -> None:
    """Write a compressed NPZ durably, then atomically expose its final name."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_id = f"{os.getpid()}-{os.urandom(6).hex()}"
    temporary = destination.with_name(f".{destination.stem}.partial.{run_id}.npz")
    with exclusive_cache_writer(destination):
        try:
            np.savez_compressed(temporary, **arrays)
            with temporary.open("r+b") as handle:
                os.fsync(handle.fileno())
            expected = {
                key: ndarray_sha256(np.asarray(value)) for key, value in arrays.items()
            }
            with np.load(temporary, allow_pickle=False) as payload:
                if set(payload.files) != set(arrays):
                    raise RuntimeError(
                        f"NPZ readback keys differ before commit: {temporary}"
                    )
                observed = {
                    key: ndarray_sha256(np.asarray(payload[key])) for key in payload.files
                }
            if observed != expected:
                raise RuntimeError(
                    f"NPZ readback checksum differs before commit: {temporary}"
                )
            os.replace(temporary, destination)
            try:
                directory_fd = os.open(destination.parent, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if temporary.exists():
                temporary.unlink()
