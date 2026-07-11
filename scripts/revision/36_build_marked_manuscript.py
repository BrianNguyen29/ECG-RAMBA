"""Build a reviewer-marked manuscript with latexdiff and verify the PDF.

The decision letter explicitly requests a highlighted/marked manuscript. This
runner treats that deliverable as complete only after ``latexdiff`` generates a
non-empty TeX file and ``latexmk`` compiles a non-empty PDF. Missing tools are
recorded as an explicit blocker rather than silently accepting an unmarked PDF.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import MANIFEST_DIR, git_commit, save_json_atomic, sha256_file  # noqa: E402


DEFAULT_MANUSCRIPT_DIR = PROJECT_ROOT.parent / "docs" / "IEEE_JBHI___ECG_RAMBA___XT_Reviewed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, default=DEFAULT_MANUSCRIPT_DIR / "BACKUP.tex")
    parser.add_argument("--revised", type=Path, default=DEFAULT_MANUSCRIPT_DIR / "main.tex")
    parser.add_argument("--out-tex", type=Path, default=DEFAULT_MANUSCRIPT_DIR / "main_marked.tex")
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_MANUSCRIPT_DIR / "main_marked.pdf")
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=MANIFEST_DIR / "marked_manuscript_manifest.json",
    )
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def executable(name: str) -> str | None:
    return shutil.which(name)


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout[-8000:], flush=True)
    if result.stderr:
        print(result.stderr[-8000:], flush=True)
    return result


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def blocker_payload(
    *,
    status: str,
    issue: str,
    original: Path,
    revised: Path,
    out_tex: Path,
    out_pdf: Path,
    tools: dict[str, str | None],
) -> dict[str, Any]:
    return {
        "status": status,
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "issue": issue,
        "tools": tools,
        "inputs": {"original": str(original), "revised": str(revised)},
        "outputs": {"marked_tex": str(out_tex), "marked_pdf": str(out_pdf)},
        "editorial_ready": False,
    }


def main() -> None:
    args = parse_args()
    original = resolve(args.original)
    revised = resolve(args.revised)
    out_tex = resolve(args.out_tex)
    out_pdf = resolve(args.out_pdf)
    manifest_path = resolve(args.out_manifest)
    tools = {name: executable(name) for name in ("latexdiff", "latexmk", "pdftotext")}
    print("=" * 80, flush=True)
    print("MARKED/HIGHLIGHTED MANUSCRIPT BUILD", flush=True)
    print("=" * 80, flush=True)

    missing_inputs = [str(path) for path in (original, revised) if not path.exists() or path.stat().st_size == 0]
    if missing_inputs:
        payload = blocker_payload(
            status="blocked_missing_input",
            issue="Missing TeX input(s): " + "; ".join(missing_inputs),
            original=original,
            revised=revised,
            out_tex=out_tex,
            out_pdf=out_pdf,
            tools=tools,
        )
        save_json_atomic(manifest_path, payload)
        if args.strict:
            raise FileNotFoundError(payload["issue"])
        return
    if tools["latexdiff"] is None or (args.compile and tools["latexmk"] is None):
        missing_tools = [name for name in ("latexdiff", "latexmk" if args.compile else "") if name and tools[name] is None]
        payload = blocker_payload(
            status="blocked_missing_tool",
            issue="Missing required executable(s): " + ", ".join(missing_tools),
            original=original,
            revised=revised,
            out_tex=out_tex,
            out_pdf=out_pdf,
            tools=tools,
        )
        save_json_atomic(manifest_path, payload)
        print(json.dumps(payload, indent=2), flush=True)
        if args.strict:
            raise RuntimeError(payload["issue"])
        return

    original_text = original.read_text(encoding="utf-8", errors="replace")
    revised_text = revised.read_text(encoding="utf-8", errors="replace")
    line_diff = list(difflib.unified_diff(original_text.splitlines(), revised_text.splitlines()))
    if not line_diff:
        payload = blocker_payload(
            status="blocked_no_detectable_revision",
            issue="Original and revised TeX inputs are identical; a marked revision cannot be demonstrated.",
            original=original,
            revised=revised,
            out_tex=out_tex,
            out_pdf=out_pdf,
            tools=tools,
        )
        save_json_atomic(manifest_path, payload)
        if args.strict:
            raise RuntimeError(payload["issue"])
        return

    diff_result = run_command(
        [str(tools["latexdiff"]), "--flatten", str(original), str(revised)],
        cwd=revised.parent,
    )
    if diff_result.returncode != 0 or not diff_result.stdout.strip():
        raise RuntimeError(f"latexdiff failed with exit code {diff_result.returncode}")
    write_text_atomic(out_tex, diff_result.stdout)

    compile_log = None
    if args.compile:
        job_name = out_pdf.stem
        compile_result = run_command(
            [
                str(tools["latexmk"]),
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-jobname={job_name}",
                out_tex.name,
            ],
            cwd=out_tex.parent,
        )
        compile_log = out_tex.parent / f"{job_name}.log"
        generated_pdf = out_tex.parent / f"{job_name}.pdf"
        if compile_result.returncode != 0 or not generated_pdf.exists() or generated_pdf.stat().st_size == 0:
            raise RuntimeError(f"Marked manuscript LaTeX compilation failed; inspect {compile_log}")
        if generated_pdf.resolve() != out_pdf.resolve():
            shutil.copy2(generated_pdf, out_pdf)

    outputs = {"marked_tex": {"path": str(out_tex), "sha256": sha256_file(out_tex)}}
    if args.compile:
        outputs["marked_pdf"] = {
            "path": str(out_pdf),
            "sha256": sha256_file(out_pdf),
            "size_bytes": out_pdf.stat().st_size,
        }
    payload = {
        "status": "complete_marked_manuscript" if args.compile else "complete_marked_tex_only",
        "created_utc": now_utc(),
        "git_commit": git_commit(),
        "editorial_ready": bool(args.compile),
        "tools": tools,
        "inputs": {
            "original": {"path": str(original), "sha256": sha256_file(original)},
            "revised": {"path": str(revised), "sha256": sha256_file(revised)},
        },
        "diff": {
            "unified_diff_lines": len(line_diff),
            "original_lines": len(original_text.splitlines()),
            "revised_lines": len(revised_text.splitlines()),
        },
        "outputs": outputs,
        "compile_log": str(compile_log) if compile_log else None,
    }
    save_json_atomic(manifest_path, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
