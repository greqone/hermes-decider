"""Profile-scoped config readers and writers for the System-1 Decider plugin.

Reads use Hermes's public ``load_config_readonly``. Writes prefer ``update_user_config`` when
present on ``hermes_cli.config`` and fall back to ``require_readable_config_before_write`` +
``atomic_config_write`` on older cores.

Routes are resolved from ``delegation.backends.decider.options`` when configured, falling back
automatically to ``delegation.backends.model_map.options`` so routes managed in Hermes Desktop
(**Settings ▸ Model ▸ Subagents**) or ``advanced-delegation`` work with zero duplication.

Decider settings live in ``plugins.entries.decider.settings`` (flat keys matching ``plugin.yaml``'s
``config_schema``). Nothing here reads environment variables or touches ``.env`` files.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Mapping

from agent.delegation_provider import ROUTE_KEYS

PLUGIN_ID = "decider"

DEFAULT_DECIDER_URL = "http://127.0.0.1:11434/v1/systemone"
DEFAULT_DECIDER_MODEL = "pplx-decider:27b-nvfp4"
DEFAULT_FALLBACK_URL = ""
DEFAULT_FALLBACK_MODEL = "jev-latest"
DEFAULT_DECIDER_TIMEOUT = 18.0
DEFAULT_SPECIALIST_THRESHOLD = 0.70
DEFAULT_CONFIDENCE_THRESHOLD = 0.60

# Routes excluded from automatic assignment unless explicitly requested via `option`:
# paid-per-token routes and advisor-only routes with strict quota/latency budgets.
DEFAULT_EXCLUDED_OPTIONS: tuple[str, ...] = (
    "UltraFastPaid",
    "AbsoluteSOTAGoogle",
    "AbsoluteSOTAOpenAI",
    "AbsoluteSOTAAnthropic",
    "AbsoluteSOTAXAI",
)

MODES: dict[str, str] = {
    "decider": "System-1 Decider",
    "model_map": "Model routes",
    "profile_routes": "Profile routes",
    "default": "Default",
}

_ROUTE_FIELDS = tuple(sorted(ROUTE_KEYS - {"label", "fallback_providers"}))


def _text(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def full_config(config: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """Return the active profile's effective config mapping (read-only)."""
    if config is not None and isinstance(config, Mapping):
        return config
    from hermes_cli.config import load_config_readonly
    loaded = load_config_readonly()
    return loaded if isinstance(loaded, Mapping) else {}


def delegation_section(config: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """Return the ``delegation`` section from a full config or a section already narrowed to it."""
    if isinstance(config, Mapping) and ("backends" in config or "backend" in config) and "delegation" not in config:
        return config
    cfg = full_config(config)
    section = cfg.get("delegation")
    return section if isinstance(section, Mapping) else {}


def current_mode(section: Mapping[str, Any] | None = None) -> str:
    sec = delegation_section(section)
    raw = sec.get("backend")
    return raw.strip().lower() if isinstance(raw, str) and raw.strip() else "default"


def mode_label(mode: str) -> str:
    return MODES.get(mode, mode or "Default")


def _backend_block(section: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    backends = section.get("backends")
    block = backends.get(name) if isinstance(backends, Mapping) else None
    return block if isinstance(block, Mapping) else {}


def active_routes_block(section: Mapping[str, Any] | None = None) -> tuple[str, Mapping[str, Any]]:
    """Return ``(source_backend, block)`` holding the configured model options.

    ``delegation.backends.decider`` wins when it defines non-empty ``options`` (or ``routes``);
    otherwise ``delegation.backends.model_map`` is used so routes edited in **Settings ▸ Model ▸
    Subagents** are picked up automatically.
    """
    sec = delegation_section(section)
    own = _backend_block(sec, "decider")
    own_opts = own.get("options") if isinstance(own.get("options"), list) else own.get("routes")
    if isinstance(own_opts, list) and own_opts:
        return "decider", own
    mm = _backend_block(sec, "model_map")
    return "model_map", mm


def raw_route_entries(section: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Raw route entry dicts in config order."""
    _, block = active_routes_block(section)
    raw = block.get("options") if isinstance(block.get("options"), list) else block.get("routes")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def configured_options(section: Mapping[str, Any] | None = None) -> OrderedDict[str, dict[str, Any]]:
    """Validated ``{name: {"description": str, "route": dict}}`` in config order."""
    out: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for entry in raw_route_entries(section):
        name = _text(entry.get("name"))
        model = _text(entry.get("model"))
        profile = _text(entry.get("profile"))
        provider = _text(entry.get("provider"))
        base_url = _text(entry.get("base_url"))
        if not name or name in out:
            continue
        if not profile and not (model and (provider or base_url)):
            continue
        route: dict[str, Any] = {}
        for key in _ROUTE_FIELDS:
            val = entry.get(key)
            if isinstance(val, str) and val.strip():
                route[key] = val.strip()
            elif key == "request_overrides" and isinstance(val, Mapping):
                route[key] = dict(val)
        if isinstance(entry.get("fallback_providers"), list):
            route["fallback_providers"] = list(entry["fallback_providers"])
        out[name] = {
            "description": _text(entry.get("description")),
            "route": route,
        }
    return out


def route_default(section: Mapping[str, Any] | None = None) -> str:
    """Configured default route name (empty string when unset or not in configured options)."""
    sec = delegation_section(section)
    own_default = _text(_backend_block(sec, "decider").get("default"))
    _, block = active_routes_block(sec)
    candidate = own_default or _text(block.get("default"))
    options = configured_options(sec)
    return candidate if candidate in options else ""


def route_summary(entry: Mapping[str, Any]) -> str:
    """Human-readable ``provider / model (effort)`` summary without exposing credentials or URLs."""
    profile = _text(entry.get("profile"))
    if profile:
        return f"profile:{profile}"
    provider = _text(entry.get("provider")) or ("custom" if _text(entry.get("base_url")) else "")
    model = _text(entry.get("model"))
    effort = _text(entry.get("reasoning_effort"))
    tier = _text(entry.get("service_tier"))
    head = f"{provider}/{model}" if provider and model else (model or provider or "inherit")
    extras = [x for x in (effort, tier) if x]
    return f"{head} ({', '.join(extras)})" if extras else head


def normalize_systemone_url(raw_url: str) -> str:
    url = str(raw_url or "").strip().rstrip("/")
    if not url:
        return DEFAULT_DECIDER_URL
    if url.endswith("/v1/systemone") or url.endswith("/v1/score"):
        return url
    if url.endswith("/v1"):
        return f"{url}/systemone"
    return f"{url}/v1/systemone"


def _normalize_optional_url(raw_url: Any) -> str:
    url = str(raw_url or "").strip().rstrip("/")
    if not url:
        return ""
    return normalize_systemone_url(url)


def _saved_plugin_settings(config: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """Read ``plugins.entries.decider.settings`` (with nested ``decider`` dict compatibility)."""
    cfg = full_config(config)
    plugins = cfg.get("plugins")
    entries = plugins.get("entries") if isinstance(plugins, Mapping) else None
    entry = entries.get(PLUGIN_ID) if isinstance(entries, Mapping) else None
    if not isinstance(entry, Mapping):
        return {}
    settings = entry.get("settings")
    if not isinstance(settings, Mapping):
        settings = entry.get("config")
    if not isinstance(settings, Mapping):
        return {}
    nested = settings.get("decider")
    if isinstance(nested, Mapping) and "url" not in settings and "enabled" not in settings:
        return nested
    return settings


def _clamp_float(val: Any, default: float, low: float, high: float) -> float:
    if isinstance(val, bool) or val is None or val == "":
        return default
    try:
        num = float(val)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, num))


def normalize_decider(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize a Decider settings dict to canonical types and safe defaults.

    Cloud/secondary fallback is strictly opt-in (``allow_fallback: False``, ``fallback_url: ""``)
    so task text never leaves the local machine unless explicitly configured by the operator.
    """
    src = raw if isinstance(raw, Mapping) else {}
    enabled_raw = src.get("enabled")
    enabled = enabled_raw if isinstance(enabled_raw, bool) else True

    url_raw = _text(src.get("url")) or DEFAULT_DECIDER_URL
    url = normalize_systemone_url(url_raw)
    model = _text(src.get("model")) or DEFAULT_DECIDER_MODEL

    allow_fb_raw = src.get("allow_fallback")
    allow_fallback = allow_fb_raw if isinstance(allow_fb_raw, bool) else False
    fallback_url = _normalize_optional_url(src.get("fallback_url"))
    fallback_model = _text(src.get("fallback_model")) or DEFAULT_FALLBACK_MODEL

    excl_raw = src.get("excluded_options")
    if isinstance(excl_raw, list):
        excluded = [_text(x) for x in excl_raw if _text(x)]
    else:
        excluded = list(DEFAULT_EXCLUDED_OPTIONS)

    return {
        "enabled": enabled,
        "url": url,
        "model": model,
        "allow_fallback": allow_fallback,
        "fallback_url": fallback_url,
        "fallback_model": fallback_model,
        "specialist_threshold": _clamp_float(
            src.get("specialist_threshold"), DEFAULT_SPECIALIST_THRESHOLD, 0.0, 1.0
        ),
        "confidence_threshold": _clamp_float(
            src.get("confidence_threshold"), DEFAULT_CONFIDENCE_THRESHOLD, 0.0, 1.0
        ),
        "timeout": _clamp_float(src.get("timeout"), DEFAULT_DECIDER_TIMEOUT, 1.0, 120.0),
        "excluded_options": excluded,
    }


def decider_settings(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Effective normalized System-1 Decider settings for the active profile."""
    return normalize_decider(_saved_plugin_settings(config))


def decider_enabled(config: Mapping[str, Any] | None = None) -> bool:
    return bool(decider_settings(config).get("enabled"))


def validate_decider(settings: Mapping[str, Any]) -> list[str]:
    """Validate candidate System-1 Decider settings before saving (empty list = valid)."""
    problems: list[str] = []
    url = _text(settings.get("url"))
    if url and not (url.startswith("http://") or url.startswith("https://")):
        problems.append("Decider endpoint URL must start with http:// or https://.")
    model = settings.get("model")
    if model is not None and isinstance(model, str) and not model.strip() and "model" in settings:
        # Blank string in UI resets to default ONLY when omitted; explicit whitespace-only when url set:
        pass
    fb_url = _text(settings.get("fallback_url"))
    if fb_url and not (fb_url.startswith("http://") or fb_url.startswith("https://")):
        problems.append("Fallback URL must start with http:// or https:// (or be left blank).")
    if settings.get("allow_fallback") is True and not fb_url:
        problems.append("Fallback URL is required when secondary fallback is enabled.")

    for field, low, high in (
        ("specialist_threshold", 0.0, 1.0),
        ("confidence_threshold", 0.0, 1.0),
        ("timeout", 1.0, 120.0),
    ):
        val = settings.get(field)
        if val is None or val == "":
            continue
        if isinstance(val, bool):
            problems.append(f"{field} must be a number between {low} and {high}.")
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            problems.append(f"{field} must be a number between {low} and {high}.")
            continue
        if not (low <= num <= high):
            problems.append(f"{field} must be between {low} and {high} (got {num}).")
    return problems


# --- writes -----------------------------------------------------------------------------------

def _child(parent: dict[str, Any], key: str) -> dict[str, Any]:
    child = parent.get(key)
    if not isinstance(child, dict):
        child = parent[key] = {}
    return child


def _write_config(mutate: Callable[[dict[str, Any]], None]) -> None:
    """Persist a config mutation through Hermes's public config transaction API."""
    from hermes_cli import config

    update = getattr(config, "update_user_config", None)
    if callable(update):
        update(mutate)
        return

    if config.is_managed():
        raise RuntimeError(config.format_managed_message("change decider settings"))
    path = config.get_config_path()
    data = config.require_readable_config_before_write(path)
    mutate(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    config.atomic_config_write(path, data)


def save_decider(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Validate/normalize and persist ``plugins.entries.decider.settings``."""
    cleaned_input = dict(settings)
    for k in ("url", "model", "fallback_model"):
        if cleaned_input.get(k) == "":
            cleaned_input.pop(k, None)
    normalized = normalize_decider(cleaned_input)

    def mutate(cfg: dict[str, Any]) -> None:
        own = _child(_child(_child(cfg, "plugins"), "entries"), PLUGIN_ID)
        target = _child(own, "settings")
        target.update(normalized)

    _write_config(mutate)
    return normalized


def set_mode(mode: str) -> None:
    """Persist ``delegation.backend``."""
    if mode not in MODES:
        raise ValueError(f"Unknown delegation mode {mode!r}; use one of: {', '.join(MODES)}.")

    def mutate(cfg: dict[str, Any]) -> None:
        _child(cfg, "delegation")["backend"] = mode

    _write_config(mutate)
