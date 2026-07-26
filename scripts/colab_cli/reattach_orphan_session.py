#!/usr/bin/env python3
"""Reattach a named Colab CLI session after an intentional kernel restart."""

from __future__ import annotations

import argparse

from colab_cli.commands.session import spawn_keep_alive
from colab_cli.common import state
from colab_cli.state import SessionState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--accelerator", default="A100")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    existing = state.store.get(args.session)
    if existing is not None:
        if args.endpoint and existing.endpoint != args.endpoint:
            raise RuntimeError(
                f"Session {args.session} already names another endpoint."
            )
        print(
            f"Session already attached: {args.session} -> {existing.endpoint}"
        )
        return

    assignments = list(state.client.list_assignments())
    candidates = [
        assignment
        for assignment in assignments
        if (not args.endpoint or assignment.endpoint == args.endpoint)
        and assignment.accelerator.value.upper() == args.accelerator.upper()
    ]
    if len(candidates) != 1:
        safe_endpoints = [assignment.endpoint for assignment in candidates]
        raise RuntimeError(
            "Expected exactly one matching orphaned assignment; "
            f"found={safe_endpoints}"
        )

    assignment = candidates[0]
    proxy = assignment.runtime_proxy_info
    session = SessionState(
        name=args.session,
        token=proxy.token,
        url=proxy.url,
        endpoint=assignment.endpoint,
        variant=assignment.variant.name,
        accelerator=assignment.accelerator.value,
        kernel_id=None,
        session_id=None,
    )
    state.store.add(session)
    session.keep_alive_pid = spawn_keep_alive(
        session.endpoint,
        session.name,
        auth_provider=state.auth_provider,
        config_path=state.config_path,
    )
    state.store.add(session)
    state.history.log_event(
        session.name,
        "session_reattached",
        {
            "endpoint": session.endpoint,
            "variant": session.variant,
            "accelerator": session.accelerator,
            "kernel_reset": True,
        },
    )
    print(
        f"Reattached session: {session.name} -> {session.endpoint} "
        f"({session.accelerator})"
    )


if __name__ == "__main__":
    main()
