"""System-1 Decider plugin for Hermes subagent delegation.

Registers:
- ``DeciderDelegationProvider`` (``delegation.backend: decider``)
- ``pre_tool_call`` hook that batch-evaluates unassigned ``delegate_task`` spawns in one
  ``/v1/systemone`` forward pass (both under ``delegation.backend: decider`` and alongside
  ``delegation.backend: model_map`` when auto-routing is enabled)
- ``/decider`` slash command (``slash.py``)
- ``hermes decider`` CLI & interactive TUI configurator (``cli.py``)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _on_pre_tool_call(tool_name: str = "", args: dict[str, Any] | None = None, **_: Any):
    """Batch-route unassigned ``delegate_task`` items via ``/v1/systemone`` before spawn."""
    if tool_name != "delegate_task" or not isinstance(args, dict):
        return None
    action = str(args.get("action") or "spawn").strip().lower()
    if action != "spawn":
        return None
    tasks = args.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return None
    try:
        from . import decider_router, store

        cfg = store.full_config()
        section = store.delegation_section(cfg)
        mode = store.current_mode(section)
        if mode not in ("decider", "model_map"):
            return None
        if not store.decider_enabled(cfg):
            return None
        cloned_tasks = [dict(t) if isinstance(t, dict) else t for t in tasks]
        assigned = decider_router.route_unassigned_tasks(cloned_tasks, config=cfg)
        if assigned:
            return {"action": "modify", "args": {"tasks": cloned_tasks}}
    except Exception as exc:
        logger.debug("decider pre_tool_call hook skipped: %s", exc)
    return None


def register(ctx) -> None:
    try:
        from .provider import DeciderDelegationProvider
    except ModuleNotFoundError as exc:
        if exc.name != "agent.delegation_provider":
            raise
        logger.warning(
            "decider needs a Hermes core with pluggable delegation providers "
            "(agent.delegation_provider); update Hermes. The plugin is inactive until then."
        )
        return

    ctx.register_delegation_provider(DeciderDelegationProvider())
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)

    from . import cli, slash

    ctx.register_command(
        "decider",
        slash.make_handler(ctx),
        description="Configure, toggle, or test the System-1 Decider subagent auto-router",
        args_hint="[status | on | off | test <goal> | history | mode [decider|model_map|default] [--now]]",
    )
    ctx.register_cli_command(
        name="decider",
        help="Configure or test the System-1 Decider subagent auto-router (/v1/systemone)",
        setup_fn=cli.setup_parser,
        handler_fn=cli.handle,
        description=(
            "Interactive configurator, CLI flags, live probe, and routing log for the "
            "System-1 Decider (`hermes decider`, `hermes decider --show`, `hermes decider --test ...`)."
        ),
    )
