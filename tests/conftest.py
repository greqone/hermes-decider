"""Hermetic test bootstrap for ``hermes-decider``.

Installs lightweight stand-ins for Hermes core modules only when running outside a Hermes checkout,
and isolates every test to an in-memory/tmp profile config, temporary ``HERMES_HOME`` cache dir, and
empty ``get_secret`` stub so no test ever reads or mutates the operator's real Hermes installation.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import types
from typing import Any
import urllib.request

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]


def _ensure_hermes_stubs() -> None:
    try:
        import agent.delegation_provider  # noqa: F401
    except ImportError:
        agent_mod = sys.modules.setdefault("agent", types.ModuleType("agent"))
        dp = types.ModuleType("agent.delegation_provider")
        dp.ROUTE_KEYS = frozenset({
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

        dp.DelegationRouteError = DelegationRouteError
        dp.DelegationProvider = DelegationProvider
        agent_mod.delegation_provider = dp
        sys.modules["agent.delegation_provider"] = dp

    try:
        import agent.secret_scope  # noqa: F401
    except ImportError:
        agent_mod = sys.modules.setdefault("agent", types.ModuleType("agent"))
        ss = types.ModuleType("agent.secret_scope")
        ss.get_secret = lambda name, default=None: default
        agent_mod.secret_scope = ss
        sys.modules["agent.secret_scope"] = ss

    try:
        import hermes_constants  # noqa: F401
    except ImportError:
        hc = types.ModuleType("hermes_constants")
        hc.get_hermes_home = lambda: Path.cwd()
        sys.modules["hermes_constants"] = hc

    try:
        import hermes_cli.config  # noqa: F401
    except ImportError:
        cli_pkg = sys.modules.setdefault("hermes_cli", types.ModuleType("hermes_cli"))
        cfg_mod = types.ModuleType("hermes_cli.config")
        cfg_mod._state = {}
        cfg_mod.load_config_readonly = lambda: copy.deepcopy(cfg_mod._state)

        def _update(mutate):
            mutate(cfg_mod._state)

        cfg_mod.update_user_config = _update
        cfg_mod.is_managed = lambda: False
        cfg_mod.format_managed_message = lambda action="": f"Config is managed ({action})"
        cfg_mod.get_config_path = lambda: Path.cwd() / "config.yaml"
        cfg_mod.require_readable_config_before_write = lambda p: copy.deepcopy(cfg_mod._state)
        cfg_mod.atomic_config_write = lambda p, d: p.write_text(yaml.safe_dump(d), encoding="utf-8")
        cli_pkg.config = cfg_mod
        sys.modules["hermes_cli.config"] = cfg_mod

    try:
        import hermes_cli.plugin_events  # noqa: F401
    except ImportError:
        cli_pkg = sys.modules.setdefault("hermes_cli", types.ModuleType("hermes_cli"))
        ev_mod = types.ModuleType("hermes_cli.plugin_events")
        ev_mod.broadcast_plugin_event = lambda *a, **kw: None
        cli_pkg.plugin_events = ev_mod
        sys.modules["hermes_cli.plugin_events"] = ev_mod

    try:
        import hermes_cli.urllib_security  # noqa: F401
    except ImportError:
        cli_pkg = sys.modules.setdefault("hermes_cli", types.ModuleType("hermes_cli"))
        sec_mod = types.ModuleType("hermes_cli.urllib_security")
        sec_mod.open_credentialed_url = lambda req, timeout=18.0: urllib.request.urlopen(req, timeout=timeout)
        cli_pkg.urllib_security = sec_mod
        sys.modules["hermes_cli.urllib_security"] = sec_mod

    try:
        import hermes_cli.version_info  # noqa: F401
    except ImportError:
        cli_pkg = sys.modules.setdefault("hermes_cli", types.ModuleType("hermes_cli"))
        ver_mod = types.ModuleType("hermes_cli.version_info")
        ver_mod.hermes_user_agent = lambda component="": f"hermes-test/{component}"
        cli_pkg.version_info = ver_mod
        sys.modules["hermes_cli.version_info"] = ver_mod


_ensure_hermes_stubs()


def load_plugin_package():
    name = "decider_plugin"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, _ROOT / "__init__.py", submodule_search_locations=[str(_ROOT)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load_plugin_package()


def sample_routes() -> list[dict[str, Any]]:
    return [
        {
            "name": "FastSub",
            "description": "Fast general-purpose coding, research, and debugging worker.",
            "provider": "openai-codex",
            "model": "gpt-5.4-mini",
        },
        {
            "name": "VisualProwess",
            "description": "Visual & multimodal worker for screenshots, UI layout, OCR, and charts.",
            "provider": "solstice",
            "model": "gemini-4-hydrogen",
        },
        {
            "name": "FrontierOpenAI",
            "description": "Rigorous adversarial verification, analytical reasoning, and hard bugs.",
            "provider": "openai-codex",
            "model": "gpt-5.4",
        },
        {
            "name": "UltraFastPaid",
            "description": "PAID per token high-throughput flash worker.",
            "provider": "openrouter",
            "model": "deepseek-v4.1-flash",
        },
    ]


@pytest.fixture(autouse=True)
def isolated_hermes(monkeypatch, tmp_path):
    """Give every test an isolated in-memory config, cache home, and empty secret store."""
    import hermes_constants
    from hermes_cli import config as hermes_cfg
    import decider_plugin.decider_router as router_mod

    home = tmp_path / "hermes_home"
    home.mkdir(parents=True, exist_ok=True)
    cfg_state: dict[str, Any] = {
        "delegation": {
            "backend": "model_map",
            "backends": {
                "model_map": {
                    "default": "FastSub",
                    "options": sample_routes(),
                }
            },
        },
        "plugins": {
            "entries": {
                "decider": {
                    "enabled": True,
                    "settings": {
                        "enabled": True,
                        "url": "http://127.0.0.1:11434/v1/systemone",
                        "model": "pplx-decider:27b-nvfp4",
                    },
                }
            }
        },
    }
    secrets: dict[str, str] = {}

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    monkeypatch.setattr(hermes_cfg, "load_config_readonly", lambda: copy.deepcopy(cfg_state))

    def _update_user_config(mutate):
        mutate(cfg_state)

    monkeypatch.setattr(hermes_cfg, "update_user_config", _update_user_config, raising=False)
    monkeypatch.setattr(hermes_cfg, "is_managed", lambda: False, raising=False)
    monkeypatch.setattr(
        router_mod,
        "get_secret",
        lambda name, default="": secrets.get(name, default),
    )
    router_mod.reset_runtime_caches()

    return {
        "home": home,
        "config": cfg_state,
        "secrets": secrets,
    }
