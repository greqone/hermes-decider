"""Test bootstrap for the decider plugin.

The plugin imports ``agent.delegation_provider`` and ``agent.secret_scope`` from Hermes core. Those
are real modules in a full checkout, but the behaviour under test does not need the rest of core,
so this installs minimal stand-ins before the plugin is imported. A real core on ``PYTHONPATH``
wins: the stand-ins are only inserted when the import would otherwise fail.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _install_agent_stubs() -> None:
    try:
        import agent.delegation_provider  # noqa: F401
        import agent.secret_scope  # noqa: F401
    except ImportError:
        _install_stub_modules()
        return
    # A real core is importable, but its get_secret pulls in the whole scope
    # runtime. Tests never want a real key, so the fixture below replaces it.


def _install_stub_modules() -> None:


    agent = sys.modules.get("agent") or types.ModuleType("agent")
    sys.modules["agent"] = agent

    provider = types.ModuleType("agent.delegation_provider")
    provider.ROUTE_KEYS = frozenset({
        "model", "provider", "base_url", "api_key", "api_mode", "request_overrides",
        "reasoning_effort", "service_tier", "fallback_providers", "profile", "label",
    })

    class DelegationRouteError(ValueError):
        pass

    class DelegationProvider:
        name = ""

        def settings(self, config):
            backends = config.get("backends") if isinstance(config, dict) else None
            own = backends.get(self.name) if isinstance(backends, dict) else None
            return own if isinstance(own, dict) else {}

    provider.DelegationRouteError = DelegationRouteError
    provider.DelegationProvider = DelegationProvider
    sys.modules["agent.delegation_provider"] = provider

    scope = types.ModuleType("agent.secret_scope")
    scope.get_secret = lambda name, default=None: default
    sys.modules["agent.secret_scope"] = scope


_install_agent_stubs()


def _load_plugin():
    name = "decider_plugin"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, _ROOT / "__init__.py", submodule_search_locations=[str(_ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load_plugin()


@pytest.fixture(autouse=True)
def _no_real_secrets(monkeypatch):
    """No test reads a real key. ``get_secret`` returns empty unless a test says otherwise.

    Patched on the plugin module, not on ``agent.secret_scope``: the plugin bound the name at
    import, so patching the source module would not change what ``decide`` calls.
    """
    import decider_plugin.decider as decider

    monkeypatch.setattr(decider, "get_secret", lambda name, default=None: default)
