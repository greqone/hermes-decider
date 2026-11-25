"""``delegation.backend: decider`` — System-1 Decider delegation provider.

Offers the configured model routes as an optional per-task ``option`` enum on ``delegate_task``
(reading ``delegation.backends.decider.options`` or falling back to
``delegation.backends.model_map.options``). Tasks with an explicit ``option`` run on that route;
tasks that omit ``option`` are routed automatically by the System-1 Decider (batched ahead of time
by the ``pre_tool_call`` hook when multiple tasks or image attachments are present, or evaluated
directly here for single-task/continue calls), degrading cleanly to the ``default`` route when
the decision endpoint is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from agent.delegation_provider import DelegationProvider

from . import decider_router, store

logger = logging.getLogger(__name__)

PARAMETER = "option"


class DeciderDelegationProvider(DelegationProvider):
    """Pluggable ``delegation.backend: decider`` provider."""

    @property
    def name(self) -> str:
        return "decider"

    @property
    def summary(self) -> str:
        return "System-1 Decider auto-selects a model route when `option` is omitted"

    def is_available(self) -> bool:
        # Must not perform network I/O: an unreachable /v1/systemone server degrades gracefully
        # to the configured default route at spawn time rather than bricking delegate_task.
        return True

    def task_parameters(self, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        options = store.configured_options(config)
        if not options:
            return {}
        enabled = store.decider_enabled()
        header = (
            "Which configured model runs this subagent. Omit to let the System-1 Decider auto-select "
            "between the default route and a specialist route."
            if enabled
            else "Which configured model runs this subagent. Omit for the default subagent route."
        )
        lines = [
            header,
            *(
                f"{name}: {entry['description']}" if entry["description"] else name
                for name, entry in options.items()
            ),
        ]
        return {
            PARAMETER: {
                "type": "string",
                "enum": list(options),
                "description": "\n".join(lines),
            }
        }

    def description(self, config: Mapping[str, Any]) -> str:
        if not store.configured_options(config):
            return ""
        if store.decider_enabled():
            return (
                "Each task may set `option` to run that subagent on one of the listed models "
                "(if omitted, System-1 Decider routes automatically)."
            )
        return "Each task may set `option` to run that subagent on one of the listed models."

    def route_task(self, task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        options = store.configured_options(config)
        if not options:
            return {}

        chosen_raw = task.get(PARAMETER)
        chosen = chosen_raw.strip() if isinstance(chosen_raw, str) else ""

        if not chosen and store.decider_enabled():
            decided = decider_router.route_single_task(task)
            if decided and decided in options:
                chosen = decided

        if not chosen:
            chosen = store.route_default(config)

        if not chosen:
            return {}

        if chosen not in options:
            logger.warning("decider: route %r is no longer in configured options; using default route.", chosen)
            fallback_default = store.route_default(config)
            if not fallback_default or fallback_default not in options:
                return {}
            chosen = fallback_default

        route = options[chosen]["route"]
        return {**route, "label": chosen}
