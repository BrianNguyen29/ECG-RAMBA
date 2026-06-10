"""Create a checksum inventory for reviewer-revision artifacts.

Run from repo root:
    python scripts/revision/05_artifact_inventory.py

The manifest helps freeze which files support a manuscript/rebuttal version.
Large generated outputs remain ignored by Git; the manifest can be archived with
the submission package when the final numbers are frozen.
"""

from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    MANIFEST_DIR,
    REVISION_DIR,
    ensure_revision_dirs,
    save_csv,
    save_json,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ensure_revision_dirs()

    rows = []
    for path in sorted(REVISION_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.name in {".gitkeep", "artifacts_manifest.json", "artifacts_manifest.csv"}:
            continue
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        rows.append(
            {
                "path": rel,
                "size_bytes": path.stat().st_size,
                "modified_utc": datetime.fromtimestamp(
                    path.stat().st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
                "sha256": sha256_file(path),
            }
        )

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "revision_dir": str(REVISION_DIR),
        "artifact_count": len(rows),
        "artifacts": rows,
    }

    save_json(MANIFEST_DIR / "artifacts_manifest.json", payload)
    save_csv(MANIFEST_DIR / "artifacts_manifest.csv", rows)

    print("=" * 80)
    print("REVISION ARTIFACT INVENTORY")
    print("=" * 80)
    print(f"Revision dir  : {REVISION_DIR}")
    print(f"Artifact count: {len(rows)}")
    print(f"Wrote         : {MANIFEST_DIR / 'artifacts_manifest.json'}")
    print(f"Wrote         : {MANIFEST_DIR / 'artifacts_manifest.csv'}")
    for row in rows:
        print(f"- {row['path']} | {row['size_bytes']} bytes")


if __name__ == "__main__":
    main()
