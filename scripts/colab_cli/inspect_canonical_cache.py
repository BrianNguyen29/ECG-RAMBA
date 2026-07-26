from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(
    "/content/drive/MyDrive/ECG-Ramba/revision_artifacts/reports/revision"
)


def describe(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if path.exists():
        stat = path.stat()
        result.update(
            {
                "is_file": path.is_file(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return result


def main() -> None:
    patterns = [
        "predictions/cpsc_window_cache/*",
        "predictions/external_feature_cache/*cpsc2021*.npz",
        "predictions/external_feature_cache/*ptbxl*.npz",
        "manifests/external_*feature_cache_manifest.json",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(ROOT.glob(pattern))

    mirror_manifest = ROOT / "manifests/mirror_manifest.json"
    payload = json.loads(mirror_manifest.read_text(encoding="utf-8"))
    entries = payload.get("artifacts", payload.get("files", []))
    relevant_entries: list[dict[str, object]] = []
    for entry in entries:
        relative = str(
            entry.get("relative_path", entry.get("path", ""))
        ).replace("\\", "/")
        if (
            "cpsc_window_cache" in relative
            or "external_cpsc2021_feature_cache_manifest" in relative
            or "external_ptbxl_fold9_feature_cache_manifest" in relative
        ):
            relevant_entries.append(entry)

    print(
        json.dumps(
            {
                "root": str(ROOT),
                "paths": [describe(path) for path in sorted(set(paths))],
                "mirror_entries": relevant_entries,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
