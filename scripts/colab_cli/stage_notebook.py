#!/usr/bin/env python3
"""Build deterministic Colab CLI stage notebooks from reviewed source notebooks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


SCHEMA_VERSION = 1
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Colab CLI manifest schema: {payload.get('schema_version')}"
        )
    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Colab CLI manifest must contain a non-empty stages list")
    stage_ids = [str(stage.get("id", "")) for stage in stages]
    if any(not item for item in stage_ids) or len(stage_ids) != len(set(stage_ids)):
        raise ValueError("Colab CLI stage ids must be non-empty and unique")
    known = set(stage_ids)
    for stage in stages:
        unknown = set(stage.get("depends_on", [])) - known
        if unknown:
            raise ValueError(f"Stage {stage['id']} has unknown dependencies: {unknown}")
        if stage.get("hardware") not in {"cpu", "a100"}:
            raise ValueError(f"Stage {stage['id']} has unsupported hardware")
        if stage.get("mode") not in {"full", "sections"}:
            raise ValueError(f"Stage {stage['id']} has unsupported mode")
        if stage.get("mode") == "sections" and not stage.get("sections"):
            raise ValueError(f"Stage {stage['id']} requires sections")
    return payload


def stage_by_id(manifest: dict[str, Any], stage_id: str) -> dict[str, Any]:
    for stage in manifest["stages"]:
        if stage["id"] == stage_id:
            return stage
    raise KeyError(f"Unknown Colab CLI stage: {stage_id}")


def authority_blob_bytes(
    repo_root: Path,
    manifest: dict[str, Any],
    relative: str,
) -> bytes:
    commit = manifest["authority"]["git_commit"]
    blob = subprocess.run(
        ["git", "show", f"{commit}:{Path(relative).as_posix()}"],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if blob.returncode:
        raise RuntimeError(
            f"{relative} is unavailable from authority commit {commit}: "
            + blob.stderr.decode("utf-8", errors="replace").strip()
        )
    return blob.stdout


def validate_authority_sources(
    repo_root: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    authority = manifest.get("authority") or {}
    commit = str(authority.get("git_commit") or "").strip().lower()
    git_ref = str(authority.get("git_ref") or "").strip()
    if not GIT_SHA_PATTERN.fullmatch(commit):
        return ["authority.git_commit must be a full lowercase Git SHA"]
    if not git_ref.startswith("refs/tags/"):
        return ["authority.git_ref must be an immutable refs/tags/... reference"]

    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", f"{git_ref}^{{commit}}"],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if resolved.returncode:
        return [
            f"authority ref cannot be resolved locally: {git_ref}: "
            + resolved.stdout.strip()
        ]
    if resolved.stdout.strip().lower() != commit:
        return [
            f"authority ref {git_ref} resolves to {resolved.stdout.strip()}, "
            f"not {commit}"
        ]

    for relative in sorted({stage["notebook"] for stage in manifest["stages"]}):
        source_path = repo_root / relative
        if not source_path.is_file():
            continue
        try:
            authority_notebook = json.loads(
                authority_blob_bytes(repo_root, manifest, relative).decode("utf-8")
            )
            working_notebook = json.loads(
                source_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            errors.append(
                f"{relative}: authority comparison failed: {exc}"
            )
            continue
        if working_notebook != authority_notebook:
            errors.append(
                f"{relative}: notebook JSON differs from authority commit {commit}"
            )
    return errors


def markdown_heading(cell: dict[str, Any]) -> str | None:
    if cell.get("cell_type") != "markdown":
        return None
    lines = "".join(cell.get("source", [])).strip().splitlines()
    if not lines:
        return None
    first = lines[0].strip()
    return first if first.startswith("#") else None


def section_blocks(cells: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    starts: list[tuple[str, int]] = []
    for index, cell in enumerate(cells):
        heading = markdown_heading(cell)
        if heading and heading.startswith("## "):
            starts.append((heading, index))
    blocks: dict[str, tuple[int, int]] = {}
    for offset, (heading, start) in enumerate(starts):
        end = starts[offset + 1][1] if offset + 1 < len(starts) else len(cells)
        if heading in blocks:
            raise ValueError(f"Duplicate level-2 notebook heading: {heading}")
        blocks[heading] = (start, end)
    return blocks


def clean_cell(cell: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(cell)
    result["id"] = result.get("id") or hashlib.sha1(
        "".join(result.get("source", [])).encode("utf-8")
    ).hexdigest()[:8]
    if result.get("cell_type") == "code":
        result["execution_count"] = None
        result["outputs"] = []
    return result


def configuration_cell(
    manifest: dict[str, Any],
    stage: dict[str, Any],
    source_sha256: str,
    manifest_sha256: str,
    builder_sha256: str,
    launcher_sha256: str,
) -> dict[str, Any]:
    environment = {
        "ECG_RAMBA_AUTHORITY_COMMIT": manifest["authority"]["git_commit"],
        "ECG_RAMBA_AUTHORITY_REF": manifest["authority"]["git_ref"],
        "ECG_RAMBA_COLAB_CLI_STAGE": stage["id"],
        "ECG_RAMBA_COLAB_CLI_MANIFEST_SHA256": manifest_sha256,
        "ECG_RAMBA_COLAB_CLI_BUILDER_SHA256": builder_sha256,
        "ECG_RAMBA_COLAB_CLI_LAUNCHER_SHA256": launcher_sha256,
        **{str(key): str(value) for key, value in stage.get("environment", {}).items()},
    }
    expected_hardware = stage["hardware"]
    code = f"""# @title Colab CLI Stage Configuration
import json as _stage_json
import os as _stage_os
import platform as _stage_platform
import subprocess as _stage_subprocess

_STAGE_ID = {stage["id"]!r}
_STAGE_SOURCE_SHA256 = {source_sha256!r}
_STAGE_EXPECTED_HARDWARE = {expected_hardware!r}
_STAGE_ENVIRONMENT = {_json_literal(environment)}
_stage_os.environ.update(_STAGE_ENVIRONMENT)

_gpu_name = ""
try:
    _gpu_name = _stage_subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
except FileNotFoundError:
    pass

if _STAGE_EXPECTED_HARDWARE == "a100" and "A100" not in _gpu_name:
    raise RuntimeError(
        f"{{_STAGE_ID}} requires an A100 runtime; observed GPU={{_gpu_name or 'none'}}"
    )

print("COLAB CLI STAGE START")
print("stage       :", _STAGE_ID)
print("source sha  :", _STAGE_SOURCE_SHA256)
print("hardware    :", _STAGE_EXPECTED_HARDWARE)
print("gpu         :", _gpu_name or "none")
print("python      :", _stage_platform.python_version())
print("environment :", _stage_json.dumps(_STAGE_ENVIRONMENT, sort_keys=True))
"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"cli-{stage['id'][:20]}",
        "metadata": {"tags": ["colab-cli-stage-config"]},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


def _json_literal(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def completion_cell(stage: dict[str, Any]) -> dict[str, Any]:
    code = f"""# @title Colab CLI Stage Completion
print("ECG_RAMBA_COLAB_CLI_STAGE_COMPLETE={stage['id']}")
"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"done-{stage['id'][:19]}",
        "metadata": {"tags": ["colab-cli-stage-complete"]},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


def build_stage_notebook(
    repo_root: Path,
    manifest: dict[str, Any],
    stage: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    source_path = (repo_root / stage["notebook"]).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Source notebook is missing: {source_path}")
    source_bytes = authority_blob_bytes(
        repo_root,
        manifest,
        stage["notebook"],
    )
    notebook = json.loads(source_bytes.decode("utf-8"))
    source_cells = notebook.get("cells", [])
    if not source_cells:
        raise ValueError(f"Source notebook is empty: {source_path}")

    if stage["mode"] == "full":
        selected = [clean_cell(cell) for cell in source_cells]
    else:
        blocks = section_blocks(source_cells)
        missing = [name for name in stage["sections"] if name not in blocks]
        if missing:
            raise ValueError(f"Stage {stage['id']} is missing source sections: {missing}")
        selected_indices = {0}
        for name in stage["sections"]:
            start, end = blocks[name]
            selected_indices.update(range(start, end))
        selected = [
            clean_cell(cell)
            for index, cell in enumerate(source_cells)
            if index in selected_indices
        ]

    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    builder_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    launcher_sha256 = hashlib.sha256(
        Path(__file__).with_name("pipeline.py").read_bytes()
    ).hexdigest()
    config = configuration_cell(
        manifest,
        stage,
        source_sha256,
        manifest_sha256,
        builder_sha256,
        launcher_sha256,
    )
    insert_at = 1 if selected and selected[0].get("cell_type") == "markdown" else 0
    selected.insert(insert_at, config)
    selected.append(completion_cell(stage))

    result = copy.deepcopy(notebook)
    result["cells"] = selected
    result.setdefault("metadata", {})["ecg_ramba_colab_cli"] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_id": manifest["pipeline_id"],
        "stage_id": stage["id"],
        "source_notebook": stage["notebook"],
        "source_notebook_sha256": source_sha256,
        "pipeline_manifest_sha256": manifest_sha256,
        "stage_builder_sha256": builder_sha256,
        "pipeline_launcher_sha256": launcher_sha256,
        "mode": stage["mode"],
        "sections": stage.get("sections", []),
        "hardware": stage["hardware"],
        "authority": manifest["authority"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def validate_manifest_sources(repo_root: Path, manifest: dict[str, Any]) -> list[str]:
    errors = validate_authority_sources(repo_root, manifest)
    stage_ids = {stage["id"] for stage in manifest["stages"]}
    seen: set[str] = set()
    for stage in manifest["stages"]:
        for dependency in stage.get("depends_on", []):
            if dependency not in seen and dependency in stage_ids:
                errors.append(
                    f"{stage['id']} appears before dependency {dependency} in stage order"
                )
        source_path = repo_root / stage["notebook"]
        if not source_path.is_file():
            errors.append(f"{stage['id']}: missing notebook {stage['notebook']}")
            seen.add(stage["id"])
            continue
        if stage["mode"] == "sections":
            notebook = json.loads(source_path.read_text(encoding="utf-8"))
            blocks = section_blocks(notebook.get("cells", []))
            for section in stage["sections"]:
                if section not in blocks:
                    errors.append(f"{stage['id']}: missing section {section}")
        seen.add(stage["id"])
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest = load_manifest(args.manifest.resolve())
    errors = validate_manifest_sources(repo_root, manifest)
    if errors:
        raise RuntimeError("Invalid Colab CLI pipeline:\n- " + "\n- ".join(errors))
    stage = stage_by_id(manifest, args.stage)
    build_stage_notebook(repo_root, manifest, stage, args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
