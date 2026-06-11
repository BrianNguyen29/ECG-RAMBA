"""Publish or restore revision artifacts with SHA256 verification."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import (  # noqa: E402
    REVISION_DIR,
    ensure_revision_dirs,
    save_json,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("publish", "restore"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--mirror-root", type=Path, required=True)
    subparsers.choices["restore"].add_argument(
        "--replace-mismatched",
        action="store_true",
        help="Replace an existing destination only after the mirror file passes its manifest checksum.",
    )
    return parser.parse_args()


def skip_artifact(relative: Path) -> bool:
    normalized = relative.as_posix()
    return (
        normalized == "manifests/mirror_manifest.json"
        or relative.parts[:1] == ("logs",) and "mirror" in relative.name
    )


def publish(mirror_root: Path) -> Path:
    ensure_revision_dirs()
    mirror_root.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in sorted(REVISION_DIR.rglob("*")):
        if not source.is_file() or source.name == ".gitkeep":
            continue
        relative = source.relative_to(REVISION_DIR)
        if skip_artifact(relative):
            continue
        destination = mirror_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_sha = sha256_file(source)
        destination_sha = sha256_file(destination)
        if source_sha != destination_sha:
            raise RuntimeError(f"Checksum mismatch after publishing {relative}")
        copied.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": destination.stat().st_size,
                "sha256": destination_sha,
            }
        )

    manifest = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(REVISION_DIR),
        "mirror_root": str(mirror_root),
        "artifact_count": len(copied),
        "artifacts": copied,
    }
    manifest_path = mirror_root / "manifests" / "mirror_manifest.json"
    save_json(manifest_path, manifest)
    print(f"Published and verified {len(copied)} artifacts: {mirror_root}")
    print(f"Wrote: {manifest_path}")
    return manifest_path


def normalize_manifest_rows(payload: dict, mirror_root: Path) -> list[dict]:
    normalized = []
    declared_root = Path(payload.get("mirror_root", mirror_root))
    for row in payload.get("artifacts", []):
        relative = row.get("relative_path")
        if not relative and row.get("mirror"):
            mirror_path = Path(row["mirror"])
            try:
                relative = mirror_path.relative_to(declared_root).as_posix()
            except ValueError:
                relative = mirror_path.name
        if not relative or not row.get("sha256"):
            raise ValueError("Mirror manifest row lacks relative_path/mirror or sha256")
        normalized.append(
            {
                "relative_path": Path(relative).as_posix(),
                "size_bytes": int(row.get("size_bytes", -1)),
                "sha256": str(row["sha256"]),
            }
        )
    return normalized


def restore(mirror_root: Path, replace_mismatched: bool) -> Path:
    manifest_path = mirror_root / "manifests" / "mirror_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Verified restore requires: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = normalize_manifest_rows(payload, mirror_root)
    if not rows:
        raise ValueError("Mirror manifest contains no artifacts")

    restored, reused = [], []
    for row in rows:
        relative = Path(row["relative_path"])
        if skip_artifact(relative):
            continue
        source = mirror_root / relative
        if not source.exists():
            raise FileNotFoundError(f"Mirror manifest references a missing file: {source}")
        actual_size = source.stat().st_size
        actual_sha = sha256_file(source)
        if row["size_bytes"] >= 0 and actual_size != row["size_bytes"]:
            raise RuntimeError(f"Mirror size mismatch: {relative}")
        if actual_sha != row["sha256"]:
            raise RuntimeError(f"Mirror checksum mismatch: {relative}")

        destination = REVISION_DIR / relative
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
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256_file(destination) != actual_sha:
            raise RuntimeError(f"Checksum mismatch after restoring {relative}")
        restored.append(relative.as_posix())

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mirror_manifest": str(manifest_path),
        "mirror_manifest_sha256": sha256_file(manifest_path),
        "verified_artifact_count": len(restored) + len(reused),
        "restored": restored,
        "reused": reused,
    }
    report_path = REVISION_DIR / "manifests" / "mirror_restore_report.json"
    save_json(report_path, report)
    print(f"Verified mirror artifacts: {len(rows)}")
    print(f"Restored: {len(restored)} | reused: {len(reused)}")
    print(f"Wrote: {report_path}")
    return report_path


def main() -> None:
    args = parse_args()
    if args.command == "publish":
        publish(args.mirror_root)
    else:
        restore(args.mirror_root, args.replace_mismatched)


if __name__ == "__main__":
    main()
