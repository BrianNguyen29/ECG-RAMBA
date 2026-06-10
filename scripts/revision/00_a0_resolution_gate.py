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


def update_a0_task_board(path: Path, status: str, notes: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    updated = []
    found = False
    escaped_notes = notes.replace('"', '""')
    for line in lines:
        if line.startswith("A0,"):
            prefix, _, _ = line.rsplit(",", 2)
            line = f'{prefix},{status},"{escaped_notes}"'
            found = True
        updated.append(line)
    if not found:
        raise ValueError("A0 row not found in task_board.csv")
    path.write_text("\n".join(updated) + "\n", encoding="utf-8")


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

    audit_complete = bool(rows) and not invalid
    deferred = [item for item in checked if item["resolution_status"] == "deferred"]
    protocol_ready = audit_complete and not deferred
    if protocol_ready:
        gate_status = "protocol_ready"
    elif audit_complete:
        gate_status = "audit_complete_with_deferred_blockers"
    else:
        gate_status = "in_progress"
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checklist_path": str(CHECKLIST_PATH),
        "allowed_resolution_statuses": sorted(ALLOWED_STATUSES),
        "blocker_count": len(rows),
        "valid_blocker_count": len(rows) - len(invalid),
        "status": gate_status,
        "audit_complete": audit_complete,
        "protocol_ready": protocol_ready,
        "deferred_blocker_count": len(deferred),
        "deferred_blocker_ids": [item["blocker_id"] for item in deferred],
        "meaning": (
            "audit_complete means every mismatch has a recorded decision and evidence path. "
            "protocol_ready is true only when no blocker remains deferred."
        ),
        "blockers": checked,
    }
    output = REVISION_DIR / "a0_resolution_status.json"
    save_json(output, payload)

    if args.update_task_board:
        tasks = read_rows(TASK_BOARD_PATH)
        notes = ""
        for task in tasks:
            if task["id"] == "A0":
                if protocol_ready:
                    notes = "A0 audit and protocol gates passed."
                elif audit_complete:
                    notes = (
                        "A0 decisions are complete, but deferred blockers still "
                        "prevent protocol-ready status."
                    )
                else:
                    notes = f"A0 decision gate failed for {len(invalid)} blocker(s)."
                break
        if not notes:
            raise ValueError("A0 row not found in task_board.csv")
        update_a0_task_board(TASK_BOARD_PATH, gate_status, notes)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote: {output}")
    if invalid:
        raise SystemExit(f"A0 remains in progress: {len(invalid)} invalid blocker decision(s)")


if __name__ == "__main__":
    main()
