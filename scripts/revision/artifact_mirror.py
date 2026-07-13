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
    return parser.parse_args()


def skip_artifact(relative: Path) -> bool:
    normalized = relative.as_posix()
    name = relative.name.lower()
    return (
        normalized == "manifests/mirror_manifest.json"
        # Atomic writers use same-directory hidden partial files. A Colab
        # disconnect can leave one behind before the finally block runs; it
        # must never be discovered and certified as reviewer evidence.
        or ".partial" in name
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


def publish(
    mirror_root: Path,
    verify_existing: str = "full",
    source_conflict_policy: str = "newer",
    refresh_existing_prefixes: list[str] | tuple[str, ...] = (),
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
            refresh_allowed = matches_refresh_prefix(relative, refresh_prefixes)
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
        }
        discovered_unmanifested += 1

    merged_rows = dict(existing_rows)
    published_from_source = 0
    published_relative_paths: set[str] = set()
    skipped_stale_source_paths: list[str] = []
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
            }
            continue

        if destination.exists() and destination_sha != source_sha:
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

        shutil.copy2(source, destination)
        destination_sha = sha256_file(destination)
        if source_sha != destination_sha:
            raise RuntimeError(f"Checksum mismatch after publishing {relative}")
        merged_rows[relative.as_posix()] = {
            "relative_path": relative.as_posix(),
            "size_bytes": destination.stat().st_size,
            "sha256": destination_sha,
        }
        published_from_source += 1
        published_relative_paths.add(relative.as_posix())

    artifacts = [merged_rows[key] for key in sorted(merged_rows)]

    manifest = {
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(REVISION_DIR),
        "mirror_root": str(mirror_root),
        "artifact_count": len(artifacts),
        "published_from_source_count": published_from_source,
        "preserved_existing_count": len(
            set(existing_rows) - published_relative_paths
        ),
        "publish_mode": "merge_verified_no_prune",
        "existing_verification_mode": verify_existing,
        "source_conflict_policy": source_conflict_policy,
        "refresh_existing_prefixes": list(refresh_prefixes),
        "refreshed_existing_count": len(refreshed_existing_paths),
        "refreshed_existing_paths": sorted(refreshed_existing_paths),
        "skipped_stale_source_count": len(skipped_stale_source_paths),
        "skipped_stale_source_paths": skipped_stale_source_paths,
        "discovered_unmanifested_count": discovered_unmanifested,
        "artifacts": artifacts,
    }
    save_json(manifest_path, manifest)
    print(
        f"Published and verified {published_from_source} source artifacts; "
        f"merged manifest contains {len(artifacts)} artifacts: {mirror_root}"
    )
    if refreshed_existing_paths:
        print(
            "Re-hashed direct canonical updates: "
            + ", ".join(sorted(refreshed_existing_paths))
        )
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


def restore(
    mirror_root: Path,
    replace_mismatched: bool,
    include_prefixes: list[str] | None = None,
    include_paths: list[str] | None = None,
    exclude_cache_dirs: bool = False,
) -> Path:
    manifest_path = mirror_root / "manifests" / "mirror_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Verified restore requires: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = normalize_manifest_rows(payload, mirror_root)
    if not rows:
        raise ValueError("Mirror manifest contains no artifacts")

    prefixes = tuple(
        Path(item).as_posix().rstrip("/") + "/"
        for item in (include_prefixes or [])
        if Path(item).as_posix().rstrip("/")
    )
    exact_paths = {Path(item).as_posix() for item in (include_paths or [])}
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
    print(f"Verified mirror artifacts: {len(rows)}")
    print(f"Restored: {len(restored)} | reused: {len(reused)}")
    print(f"Wrote: {report_path}")
    return report_path


def main() -> None:
    args = parse_args()
    if args.command == "publish":
        publish(
            args.mirror_root,
            args.verify_existing,
            args.source_conflict_policy,
            args.refresh_existing_prefix,
        )
    else:
        restore(
            args.mirror_root,
            args.replace_mismatched,
            args.include_prefix,
            args.include_path,
            args.exclude_cache_dirs,
        )


if __name__ == "__main__":
    main()
