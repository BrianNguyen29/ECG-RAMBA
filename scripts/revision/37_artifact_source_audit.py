"""Audit and safely consolidate legacy Drive artifacts into the canonical mirror.

The canonical mirror is authoritative. Migration copies only files that are absent
from the canonical mirror and refuses to overwrite conflicting canonical content.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.artifact_mirror import publish  # noqa: E402
from scripts.revision.common import REVISION_DIR, save_json, sha256_file  # noqa: E402


IGNORED_PREFIXES = ("logs/",)
IGNORED_PATHS = {
    "manifests/mirror_manifest.json",
    "manifests/mirror_restore_report.json",
    "manifests/artifact_source_audit.json",
    "tables/table_artifact_source_audit.csv",
    "metrics/pipeline_storage_audit.json",
    "tables/table_pipeline_storage_audit.csv",
}


@dataclass(frozen=True)
class AuditRow:
    relative_path: str
    status: str
    canonical_size_bytes: int | None
    canonical_sha256: str | None
    legacy_size_bytes: int | None
    legacy_sha256: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument(
        "--apply-legacy-only",
        action="store_true",
        help=(
            "Copy legacy-only files into the active reports/revision tree and publish "
            "them to the canonical mirror. Conflicts are never overwritten."
        ),
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse a prior canonical audit when both source-tree metadata fingerprints match.",
    )
    parser.add_argument(
        "--keep-canonical-on-conflict",
        action="store_true",
        help="Allow legacy-only migration while retaining conflicting canonical files.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REVISION_DIR / "manifests" / "artifact_source_audit.json",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=REVISION_DIR / "tables" / "table_artifact_source_audit.csv",
    )
    return parser.parse_args()


def ignored(relative: Path) -> bool:
    normalized = relative.as_posix()
    return normalized in IGNORED_PATHS or normalized.startswith(IGNORED_PREFIXES)


def inventory(root: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if ignored(relative):
            continue
        rows[relative.as_posix()] = {
            "path": path,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return rows


def quick_tree_fingerprint(root: Path) -> str:
    """Hash relative path, size, and mtime without reading multi-GB file contents."""

    digest = hashlib.sha256()
    if not root.exists():
        digest.update(b"missing")
        return digest.hexdigest()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if ignored(relative):
            continue
        stat = path.stat()
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def audit_sources(canonical_root: Path, legacy_root: Path) -> list[AuditRow]:
    canonical = inventory(canonical_root)
    legacy = inventory(legacy_root)
    rows: list[AuditRow] = []
    for relative in sorted(set(canonical) | set(legacy)):
        current = canonical.get(relative)
        old = legacy.get(relative)
        if current and old:
            status = "identical" if current["sha256"] == old["sha256"] else "conflict"
        elif current:
            status = "canonical_only"
        else:
            status = "legacy_only"
        rows.append(
            AuditRow(
                relative_path=relative,
                status=status,
                canonical_size_bytes=int(current["size_bytes"]) if current else None,
                canonical_sha256=str(current["sha256"]) if current else None,
                legacy_size_bytes=int(old["size_bytes"]) if old else None,
                legacy_sha256=str(old["sha256"]) if old else None,
            )
        )
    return rows


def write_csv(path: Path, rows: list[AuditRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(AuditRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def migrate_legacy_only(
    rows: list[AuditRow],
    legacy_root: Path,
    canonical_root: Path,
    keep_canonical_on_conflict: bool,
) -> list[str]:
    conflicts = [row for row in rows if row.status == "conflict"]
    if conflicts and not keep_canonical_on_conflict:
        raise RuntimeError(
            f"Refusing migration because {len(conflicts)} canonical/legacy conflicts exist. "
            "Review table_artifact_source_audit.csv or explicitly use "
            "--keep-canonical-on-conflict to retain canonical versions."
        )

    copied: list[str] = []
    for row in rows:
        if row.status != "legacy_only":
            continue
        source = legacy_root / row.relative_path
        destination = REVISION_DIR / row.relative_path
        if destination.exists():
            if sha256_file(destination) != row.legacy_sha256:
                raise RuntimeError(
                    f"Active artifact conflicts with legacy-only migration source: {destination}"
                )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256_file(destination) != row.legacy_sha256:
            raise RuntimeError(f"Checksum mismatch after staging {row.relative_path}")
        copied.append(row.relative_path)

    if copied:
        publish(canonical_root)
    return copied


def main() -> None:
    args = parse_args()
    canonical_root = args.canonical_root.expanduser().resolve()
    legacy_root = args.legacy_root.expanduser().resolve()
    quick_fingerprints = {
        "canonical": quick_tree_fingerprint(canonical_root),
        "legacy": quick_tree_fingerprint(legacy_root),
    }
    cached_json = canonical_root / "manifests" / "artifact_source_audit.json"
    cached_csv = canonical_root / "tables" / "table_artifact_source_audit.csv"
    if args.reuse_existing and not args.apply_legacy_only and cached_json.exists():
        cached = json.loads(cached_json.read_text(encoding="utf-8"))
        if cached.get("quick_tree_fingerprints") == quick_fingerprints:
            save_json(args.out_json, cached)
            if cached_csv.exists() and cached_csv.resolve() != args.out_csv.resolve():
                args.out_csv.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached_csv, args.out_csv)
            print("Artifact source audit: reused cached SHA audit")
            print("counts   :", cached.get("counts", {}))
            print(f"Wrote: {args.out_json}")
            print(f"Wrote: {args.out_csv}")
            return
    rows = audit_sources(canonical_root, legacy_root)

    def status_counts(items: list[AuditRow]) -> dict[str, int]:
        result: dict[str, int] = {}
        for item in items:
            result[item.status] = result.get(item.status, 0) + 1
        return result

    pre_migration_counts = status_counts(rows)

    copied: list[str] = []
    migration_error = None
    if args.apply_legacy_only:
        try:
            copied = migrate_legacy_only(
                rows,
                legacy_root,
                canonical_root,
                args.keep_canonical_on_conflict,
            )
        except Exception as exc:
            migration_error = str(exc)

    # Report the actual post-migration state so Notebook 00 does not require a
    # second audit pass to confirm that legacy-only files became identical.
    if copied:
        rows = audit_sources(canonical_root, legacy_root)
        quick_fingerprints = {
            "canonical": quick_tree_fingerprint(canonical_root),
            "legacy": quick_tree_fingerprint(legacy_root),
        }
    counts = status_counts(rows)

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_root": str(canonical_root),
        "legacy_root": str(legacy_root),
        "canonical_is_authoritative": True,
        "ignored_prefixes": list(IGNORED_PREFIXES),
        "quick_tree_fingerprints": quick_fingerprints,
        "pre_migration_counts": pre_migration_counts,
        "counts": counts,
        "migration_requested": bool(args.apply_legacy_only),
        "migrated_legacy_only_count": len(copied),
        "migrated_relative_paths": copied,
        "migration_error": migration_error,
        "rows": [asdict(row) for row in rows],
    }
    save_json(args.out_json, payload)
    write_csv(args.out_csv, rows)

    print("Artifact source audit")
    print(f"canonical: {canonical_root}")
    print(f"legacy   : {legacy_root} exists={legacy_root.exists()}")
    print("counts   :", counts)
    print(f"migrated : {len(copied)}")
    print(f"Wrote: {args.out_json}")
    print(f"Wrote: {args.out_csv}")
    if migration_error:
        raise RuntimeError(migration_error)


if __name__ == "__main__":
    main()
