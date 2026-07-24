#!/usr/bin/env python3
"""Provision Colab sessions and execute the ECG-RAMBA notebook stage plan."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable

from stage_notebook import (
    build_stage_notebook,
    load_manifest,
    stage_by_id,
    validate_manifest_sources,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "colab_cli_pipeline.json"
DEFAULT_BUILD_ROOT = Path.home() / ".cache" / "ecg-ramba-colab-cli" / "stages"
LOCAL_LOG_ROOT = REPO_ROOT / "reports" / "revision" / "logs" / "colab_cli"
REQUIRED_COLAB_SCOPES = {
    "openid",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/colaboratory",
}


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def colab_base(auth: str) -> list[str]:
    executable = shutil.which("colab")
    if not executable:
        raise FileNotFoundError(
            "colab is not on PATH. Run scripts/colab_cli/install_wsl.sh first."
        )
    return [executable, f"--auth={auth}"]


def render_command(command: Iterable[str]) -> str:
    return " ".join(json.dumps(str(part)) for part in command)


def run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def run_stream(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$", render_command(command), flush=True)
    with log_path.open("w", encoding="utf-8", newline="") as log_handle:
        log_handle.write("$ " + render_command(command) + "\n")
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return process.wait()


def stage_completion_marker(stage_id: str) -> str:
    return f"ECG_RAMBA_COLAB_CLI_STAGE_COMPLETE={stage_id}"


def completed_stage_log(log_path: Path, stage_id: str) -> bool:
    if not log_path.is_file():
        return False
    return stage_completion_marker(stage_id) in log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def validate_auth(base: list[str], auth: str) -> int:
    result = run_capture(base + ["whoami"])
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode:
        return result.returncode
    missing = sorted(
        scope for scope in REQUIRED_COLAB_SCOPES if scope not in result.stdout
    )
    if missing:
        remediation = (
            "Run scripts/colab_cli/setup_oauth2.sh inside WSL, then retry "
            "auth-check."
            if auth == "oauth2"
            else "Run scripts/colab_cli/setup_adc.sh inside WSL, then retry "
            "auth-check."
        )
        print(
            f"{auth.upper()} is missing Colab-required scopes:\n- "
            + "\n- ".join(missing)
            + "\n"
            + remediation,
            file=sys.stderr,
        )
        return 2
    print(f"{auth.upper()} scope preflight: OK")
    return 0


def session_name(stage_id: str) -> str:
    compact = "".join(
        character if character.isalnum() else "-" for character in stage_id.lower()
    ).strip("-")
    return ("ecgr-" + compact)[:48]


def stage_notebook_path(stage_id: str, build_root: Path) -> Path:
    return build_root / f"{stage_id}.ipynb"


def stage_log_dir(stage_id: str) -> Path:
    return LOCAL_LOG_ROOT / stage_id


def session_exists(base: list[str], name: str) -> bool:
    result = run_capture(base + ["status", "-s", name])
    return result.returncode == 0


def create_session(base: list[str], stage: dict[str, Any], name: str) -> None:
    command = base + ["new", "-s", name]
    if stage["hardware"] == "a100":
        command += ["--gpu", "A100"]
    print("$", render_command(command), flush=True)
    subprocess.run(command, check=True)


def mount_drive(base: list[str], name: str) -> None:
    command = base + ["drivemount", "-s", name, "/content/drive"]
    print("$", render_command(command), flush=True)
    print(
        "Complete the Google Drive consent in the browser, then return to this terminal.",
        flush=True,
    )
    subprocess.run(command, check=True)


def export_session_log(
    base: list[str],
    name: str,
    stage_id: str,
    run_id: str,
) -> None:
    output = stage_log_dir(stage_id) / f"{run_id}.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = base + ["log", "-s", name, "-o", str(output)]
    result = run_capture(command)
    if result.returncode:
        print("WARNING: failed to export Colab session log")
        print(result.stdout)
    else:
        print("Exported Colab session log:", output)


def preserve_executed_notebook(
    notebook_path: Path,
    stage_id: str,
    run_id: str,
) -> None:
    executed = notebook_path.with_name(notebook_path.stem + "_output.ipynb")
    if not executed.is_file():
        print(
            "WARNING: Colab CLI did not produce the expected executed notebook:",
            executed,
        )
        return
    destination = stage_log_dir(stage_id) / f"{run_id}_output.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(executed, destination)
    print("Preserved executed stage notebook:", destination)


def stop_session(base: list[str], name: str) -> None:
    command = base + ["stop", "-s", name]
    result = run_capture(command)
    if result.returncode:
        print("WARNING: failed to stop Colab session")
        print(result.stdout)
    else:
        print(result.stdout.strip())


def build_stage(
    manifest: dict[str, Any],
    stage: dict[str, Any],
    build_root: Path,
) -> Path:
    destination = stage_notebook_path(stage["id"], build_root)
    build_stage_notebook(REPO_ROOT, manifest, stage, destination)
    return destination


def dry_run_commands(
    base: list[str],
    stage: dict[str, Any],
    notebook_path: Path,
    name: str,
) -> list[list[str]]:
    create = base + ["new", "-s", name]
    if stage["hardware"] == "a100":
        create += ["--gpu", "A100"]
    return [
        create,
        base + ["drivemount", "-s", name, "/content/drive"],
        base
        + [
            "exec",
            "-s",
            name,
            "-f",
            str(notebook_path),
            "--timeout",
            str(stage["timeout_seconds"]),
        ],
        base + ["log", "-s", name, "-o", "<local-stage-log.ipynb>"],
        base + ["stop", "-s", name],
    ]


def execute_stage(
    manifest: dict[str, Any],
    stage: dict[str, Any],
    args: argparse.Namespace,
) -> int:
    if not stage.get("enabled", True) and not args.include_disabled:
        print(
            f"SKIP {stage['id']}: disabled by protocol. "
            "Use --include-disabled only after documented review."
        )
        return 0

    notebook_path = build_stage(manifest, stage, args.build_root)
    base = colab_base(args.auth)
    name = getattr(args, "session", None) or session_name(stage["id"])
    if args.dry_run:
        print(f"DRY RUN stage={stage['id']} hardware={stage['hardware']}")
        for command in dry_run_commands(base, stage, notebook_path, name):
            print("$", render_command(command))
        return 0

    run_id = utc_run_id()
    command_log = stage_log_dir(stage["id"]) / f"{run_id}.log"
    created = False
    try:
        if not session_exists(base, name):
            create_session(base, stage, name)
            created = True
        else:
            print("Reusing active Colab session:", name)

        if not args.no_mount and (created or args.remount):
            mount_drive(base, name)

        exec_command = base + [
            "exec",
            "-s",
            name,
            "-f",
            str(notebook_path),
            "--timeout",
            str(stage["timeout_seconds"]),
        ]
        return_code = run_stream(exec_command, command_log)
        preserve_executed_notebook(
            notebook_path,
            stage["id"],
            run_id,
        )
        export_session_log(base, name, stage["id"], run_id)
        if return_code:
            print(
                f"Stage {stage['id']} failed with exit code {return_code}. "
                f"Session {name} remains active for inspection."
            )
            return return_code
        if not completed_stage_log(command_log, stage["id"]):
            print(
                f"Stage {stage['id']} returned zero but its completion marker is "
                f"missing from {command_log}. Session {name} remains active for "
                "inspection.",
                file=sys.stderr,
            )
            return 3

        print(f"Stage {stage['id']} completed successfully.")
        if not args.keep:
            stop_session(base, name)
        else:
            print("Keeping Colab session active:", name)
        return 0
    except KeyboardInterrupt:
        print(
            f"Interrupted. Session {name} remains active; rerun the same stage to resume.",
            file=sys.stderr,
        )
        return 130
    except subprocess.CalledProcessError as exc:
        print(
            f"Colab command failed with exit code {exc.returncode}. "
            f"Session {name} remains active for inspection.",
            file=sys.stderr,
        )
        return exc.returncode


def print_plan(manifest: dict[str, Any]) -> None:
    print(
        "order  stage                         hardware  enabled  mode      dependencies"
    )
    for index, stage in enumerate(manifest["stages"], start=1):
        dependencies = ",".join(stage.get("depends_on", [])) or "-"
        print(
            f"{index:>5}  {stage['id']:<28}  {stage['hardware']:<8}  "
            f"{str(stage.get('enabled', True)):<7}  {stage['mode']:<8}  "
            f"{dependencies}"
        )


def selected_stages(
    manifest: dict[str, Any],
    from_stage: str | None,
    to_stage: str | None,
) -> list[dict[str, Any]]:
    stages = manifest["stages"]
    ids = [stage["id"] for stage in stages]
    start = ids.index(from_stage) if from_stage else 0
    end = ids.index(to_stage) + 1 if to_stage else len(stages)
    if end < start:
        raise ValueError("--to-stage occurs before --from-stage")
    return stages[start:end]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ECG-RAMBA notebook stages through Google Colab CLI"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--auth", choices=["oauth2", "adc"])
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    subparsers = parser.add_subparsers(dest="action", required=True)

    subparsers.add_parser("plan")
    subparsers.add_parser("validate")
    subparsers.add_parser("auth-check")
    subparsers.add_parser("sessions")

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--stage")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--stage", required=True)
    _add_run_options(run_parser)

    all_parser = subparsers.add_parser("run-all")
    all_parser.add_argument("--from-stage")
    all_parser.add_argument("--to-stage")
    _add_run_options(all_parser, include_session=False)
    return parser.parse_args()


def _add_run_options(
    parser: argparse.ArgumentParser,
    *,
    include_session: bool = True,
) -> None:
    if include_session:
        parser.add_argument("--session")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--no-mount", action="store_true")
    parser.add_argument("--remount", action="store_true")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest.resolve())
    args.auth = args.auth or manifest.get("default_auth", "oauth2")
    args.build_root = args.build_root.expanduser().resolve()
    errors = validate_manifest_sources(REPO_ROOT, manifest)
    if errors:
        print("Invalid Colab CLI pipeline:\n- " + "\n- ".join(errors), file=sys.stderr)
        return 2

    if args.action == "plan":
        print_plan(manifest)
        return 0
    if args.action == "validate":
        print("Colab CLI pipeline manifest and source notebook sections are valid.")
        return 0
    if args.action == "auth-check":
        return validate_auth(colab_base(args.auth), args.auth)
    if args.action == "sessions":
        return subprocess.run(colab_base(args.auth) + ["sessions"], check=False).returncode
    if args.action == "build":
        stages = (
            [stage_by_id(manifest, args.stage)]
            if args.stage
            else manifest["stages"]
        )
        for stage in stages:
            print(build_stage(manifest, stage, args.build_root))
        return 0
    if args.action == "run":
        return execute_stage(manifest, stage_by_id(manifest, args.stage), args)
    if args.action == "run-all":
        for stage in selected_stages(
            manifest, args.from_stage, args.to_stage
        ):
            return_code = execute_stage(manifest, stage, args)
            if return_code:
                return return_code
        return 0
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
