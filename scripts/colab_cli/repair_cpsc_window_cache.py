from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(
    "/content/drive/MyDrive/ECG-Ramba/revision_artifacts/reports/revision"
)
RELATIVE = Path(
    "predictions/cpsc_window_cache/"
    "cpsc2021_preprocessed_windows_source_bound_v3.npy"
)
CANDIDATE = RELATIVE.with_name("." + RELATIVE.name + ".partial.npy")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_row() -> dict[str, object]:
    manifest_path = ROOT / "manifests/mirror_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("artifacts", payload.get("files", []))
    expected_relative = RELATIVE.as_posix()
    for row in rows:
        relative = str(
            row.get("relative_path", row.get("path", ""))
        ).replace("\\", "/")
        if relative == expected_relative:
            return row
    raise RuntimeError(f"Mirror manifest lacks required row: {expected_relative}")


def main() -> None:
    expected = manifest_row()
    expected_size = int(expected["size_bytes"])
    expected_sha = str(expected["sha256"])
    destination = ROOT / RELATIVE
    candidate = ROOT / CANDIDATE

    if destination.is_file():
        source = destination
        action = "validated_existing"
    else:
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Neither final nor resumable CPSC window cache exists: {candidate}"
            )
        source = candidate
        action = "restored_from_partial"

    actual_size = source.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"CPSC window cache size mismatch: {actual_size} != {expected_size}"
        )
    actual_sha = sha256_file(source)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"CPSC window cache SHA256 mismatch: {actual_sha} != {expected_sha}"
        )

    array = np.load(source, mmap_mode="r", allow_pickle=False)
    if array.shape != (65445, 12, 5000):
        raise RuntimeError(f"Unexpected CPSC window shape: {array.shape}")
    if array.dtype != np.float32:
        raise RuntimeError(f"Unexpected CPSC window dtype: {array.dtype}")
    if source != destination:
        os.replace(source, destination)

    print(
        json.dumps(
            {
                "action": action,
                "path": str(destination),
                "sha256": actual_sha,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "size_bytes": actual_size,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
