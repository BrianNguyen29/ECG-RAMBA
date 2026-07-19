"""Scan submission artifacts for unsupported high-risk claims.

The scanner is intentionally conservative but negation-aware. A phrase such
as "we do not claim zero-shot superiority" is retained as a safe boundary;
the positive form is a strict failure.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import METRIC_DIR, TABLE_DIR, save_csv, save_json_atomic  # noqa: E402


TEXT_SUFFIXES = {".tex", ".md", ".txt", ".csv", ".json", ".tsv"}
CLAIM_PATTERNS = {
    "sota": re.compile(r"\bSOTA\b|state[- ]of[- ]the[- ]art", re.I),
    "best_in_domain": re.compile(r"\bbest\s+in[- ]domain\b", re.I),
    "zero_shot_superiority": re.compile(r"\bzero[- ]shot\s+superiority\b", re.I),
    "broad_or_global_superiority": re.compile(
        r"\b(?:broad|global|general|universal)\s+(?:external\s+|in[- ]domain\s+|robustness\s+)?superiority\b",
        re.I,
    ),
    "proven_disentanglement": re.compile(
        r"\b(?:proven|proved|established|validated|confirmed)\s+(?:mechanistic\s+)?disentanglement\b|"
        r"\bdisentanglement\s+(?:is|was|has been)\s+(?:proven|established|validated|confirmed)\b",
        re.I,
    ),
    "clinical_readiness": re.compile(
        r"\bclinical(?:ly)?\s+ready\b|\bclinical\s+readiness\b|"
        r"\bready\s+for\s+(?:real[- ]time\s+)?clinical\s+deployment\b|"
        r"\bfeasibility\s+for\s+real[- ]time\s+clinical\s+deployment\b",
        re.I,
    ),
    "fewshot_finetuning": re.compile(
        r"\bfew[- ]shot\s+(?:weight\s+)?fine[- ]?tuning\b|\bend[- ]to[- ]end\s+few[- ]shot\b",
        re.I,
    ),
    "full_hrv_implementation": re.compile(
        r"\b(?:implement(?:s|ed)?|uses?|includes?|extracts?)\b[^\n.!?]{0,80}"
        r"\b(?:RMSSD|SDNN|LF\s*/\s*HF|LF-HF)\b",
        re.I,
    ),
}

SAFE_PREFIX = re.compile(
    r"(?:\bdo\s+not\b|\bdoes\s+not\b|\bdid\s+not\b|\bmust\s+not\b|"
    r"\bshould\s+not\b|\bcannot\b|\bcan\s+not\b|\bno\s+longer\b|"
    r"\bnot\s+supported\b|\bunsupported\b|\bwe\s+avoid\b|\bwe\s+refrain\b|"
    r"\bforbidden\b|\bprohibited\b|\bclaim\s+boundary\b)[^.!?\n]{0,100}$|"
    r"\bnot\b[^.!?\n]{0,80}\bor\s+$",
    re.I,
)
SAFE_SUFFIX = re.compile(
    r"^[^.!?\n]{0,40}\b(?:is|are|was|were|remains?|has\s+been)\s+not\s+"
    r"(?:supported|established|demonstrated|claimed)\b|"
    r"^[^.!?\n]{0,40}\b(?:is\s+|remains?\s+)?(?:blocked|unsupported|prohibited)\b",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        action="append",
        type=Path,
        required=True,
        help="File or directory to scan. Repeat for manuscript, response, PDF text, and evidence root.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=METRIC_DIR / "submission_forbidden_claim_scan.json",
    )
    parser.add_argument(
        "--out-table",
        type=Path,
        default=TABLE_DIR / "table_submission_forbidden_claim_scan.csv",
    )
    return parser.parse_args()


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    except Exception:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pdf.txt"
            completed = subprocess.run(
                ["pdftotext", str(path), str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0 or not output.is_file():
                raise RuntimeError(
                    f"Could not extract PDF text from {path}: {completed.stderr.strip()}"
                )
            return output.read_text(encoding="utf-8", errors="replace")


def collect_files(paths: list[Path]) -> tuple[list[Path], list[str]]:
    files: set[Path] = set()
    missing: list[str] = []
    for raw in paths:
        path = raw if raw.is_absolute() else (PROJECT_ROOT / raw)
        path = path.resolve()
        if not path.exists():
            missing.append(str(path))
        elif path.is_file():
            files.add(path)
        else:
            for candidate in path.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in TEXT_SUFFIXES | {".pdf"}:
                    files.add(candidate.resolve())
    return sorted(files), missing


def local_context(text: str, start: int, end: int) -> tuple[str, str, str]:
    left_boundary = max(text.rfind(marker, 0, start) for marker in ("\n", ".", "!", "?"))
    right_candidates = [position for marker in ("\n", ".", "!", "?") if (position := text.find(marker, end)) >= 0]
    right_boundary = min(right_candidates) if right_candidates else len(text)
    prefix = text[left_boundary + 1 : start]
    suffix = text[end:right_boundary]
    context = text[left_boundary + 1 : right_boundary].strip()
    return prefix, suffix, re.sub(r"\s+", " ", context)


def negated_or_boundary(prefix: str, suffix: str) -> bool:
    return bool(SAFE_PREFIX.search(prefix[-160:]) or SAFE_SUFFIX.search(suffix[:100]))


def scan_text(path: Path, text: str) -> list[dict]:
    rows: list[dict] = []
    line_starts = [0]
    line_starts.extend(match.end() for match in re.finditer(r"\n", text))
    for claim, pattern in CLAIM_PATTERNS.items():
        for match in pattern.finditer(text):
            prefix, suffix, context = local_context(text, match.start(), match.end())
            safe = negated_or_boundary(prefix, suffix)
            line_number = 1
            lo, hi = 0, len(line_starts)
            while lo < hi:
                mid = (lo + hi) // 2
                if line_starts[mid] <= match.start():
                    line_number = mid + 1
                    lo = mid + 1
                else:
                    hi = mid
            rows.append(
                {
                    "path": str(path),
                    "line": line_number,
                    "claim": claim,
                    "matched_text": match.group(0),
                    "status": "safe_boundary" if safe else "unsafe_positive_claim",
                    "context": context[:500],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    files, missing = collect_files(args.path)
    if missing and not args.allow_missing:
        raise FileNotFoundError("Missing scan path(s): " + "; ".join(missing))
    rows: list[dict] = []
    read_failures: list[str] = []
    for path in files:
        try:
            text = read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(
                encoding="utf-8", errors="replace"
            )
            rows.extend(scan_text(path, text))
        except Exception as exc:
            read_failures.append(f"{path}: {exc}")
    unsafe = [row for row in rows if row["status"] == "unsafe_positive_claim"]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": not unsafe and not read_failures and (not missing or args.allow_missing),
        "files_scanned": len(files),
        "hits": len(rows),
        "safe_boundary_hits": len(rows) - len(unsafe),
        "unsafe_positive_claim_hits": len(unsafe),
        "missing_paths": missing,
        "read_failures": read_failures,
        "patterns": sorted(CLAIM_PATTERNS),
        "rows": rows,
    }
    save_json_atomic(args.out_json, payload)
    save_csv(args.out_table, rows or [{"status": "clean", "path": "", "line": "", "claim": "", "matched_text": "", "context": ""}])
    print(json.dumps({key: payload[key] for key in ("status", "files_scanned", "hits", "safe_boundary_hits", "unsafe_positive_claim_hits")}, indent=2))
    print(f"Wrote: {args.out_json}")
    print(f"Wrote: {args.out_table}")
    if args.strict and not payload["status"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
