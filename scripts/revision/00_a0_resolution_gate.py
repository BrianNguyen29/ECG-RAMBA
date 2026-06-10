"""Validate A0 blocker decisions and optionally update the revision task board."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.common import REVISION_DIR, save_json  # noqa: E402


CHECKLIST_PATH = PROJECT_ROOT / "docs" / "revision_plan" / "a0_resolution_checklist.csv"
TASK_BOARD_PATH = PROJECT_ROOT / "docs" / "revision_plan" / "task_board.csv"
ALLOWED_STATUSES = {"resolved", "deferred", "manuscript-corrected"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-task-board", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evidence_status(value: str) -> tuple[list[str], list[str]]:
    paths = [item.strip() for item in value.split(";") if item.strip()]
    missing = [path for path in paths if not (PROJECT_ROOT / path).exists()]
    return paths, missing


def main() -> None:
    args = parse_args()
    rows = read_rows(CHECKLIST_PATH)
    checked = []
    invalid = []
    for row in rows:
        status = row["resolution_status"].strip()
        evidence, missing = evidence_status(row["evidence_paths"])
        valid = status in ALLOWED_STATUSES and bool(row["decision"].strip()) and not missing
        item = {
            **row,
            "evidence": evidence,
            "missing_evidence": missing,
            "valid": valid,
        }
        checked.append(item)
        if not valid:
            invalid.append(item)

    complete = bool(rows) and not invalid
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checklist_path": str(CHECKLIST_PATH),
        "allowed_resolution_statuses": sorted(ALLOWED_STATUSES),
        "blocker_count": len(rows),
        "valid_blocker_count": len(rows) - len(invalid),
        "a0_audit_complete": complete,
        "meaning": (
            "A0 completion means every mismatch has a recorded resolution, deferral, "
            "or manuscript correction. Deferred experiment claims remain blocked."
        ),
        "blockers": checked,
    }
    output = REVISION_DIR / "a0_resolution_status.json"
    save_json(output, payload)

    if args.update_task_board:
        tasks = read_rows(TASK_BOARD_PATH)
        found = False
        for task in tasks:
            if task["id"] == "A0":
                task["status"] = "completed" if complete else "in_progress"
                task["notes"] = (
                    "A0 decision gate passed; deferred claims remain blocked by their owner tasks."
                    if complete
                    else f"A0 decision gate failed for {len(invalid)} blocker(s)."
                )
                found = True
        if not found:
            raise ValueError("A0 row not found in task_board.csv")
        write_rows(TASK_BOARD_PATH, tasks)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote: {output}")
    if invalid:
        raise SystemExit(f"A0 remains in progress: {len(invalid)} invalid blocker decision(s)")


if __name__ == "__main__":
    main()
