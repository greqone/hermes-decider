"""Decider: register one delegation backend and nothing else.

``plugin.yaml`` declares ``requires_hermes: ">=0.22.5"``, which current Hermes enforces before it
imports a plugin. Older cores ignore that field, so on a core without pluggable delegation
providers this logs one warning and registers nothing rather than crashing the plugin load.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    try:
        from .decider import DeciderDelegationProvider
    except ModuleNotFoundError as exc:
        if exc.name != "agent.delegation_provider":
            raise
        logger.warning(
            "decider needs a Hermes core with pluggable delegation providers "
            "(agent.delegation_provider); update Hermes. The plugin is inactive until then."
        )
        return
    ctx.register_delegation_provider(DeciderDelegationProvider())
