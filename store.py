"""Profile-scoped config and secret helpers for the System-1 Decider plugin.

Reads use Hermes's public ``load_config_readonly``. Writes prefer ``update_user_config`` when
present on ``hermes_cli.config`` and fall back to ``require_readable_config_before_write`` +
``atomic_config_write`` on older cores.

Secrets (``DECIDER_API_KEY`` and ``DECIDER_FALLBACK_API_KEY``) are read exclusively through
``agent.secret_scope.get_secret`` (propagating ``UnscopedSecretError``) and written/cleared
exclusively through Hermes's public ``hermes_cli.config.save_env_value_secure`` /
``remove_env_value``. Nothing in this plugin accesses ``os.environ`` or parses ``.env`` files.

Routes are resolved from ``delegation.backends.decider.options`` when configured, falling back
automatically to ``delegation.backends.model_map.options`` so routes managed in Hermes Desktop
(**Settings ▸ Model ▸ Subagents**) or ``advanced-delegation`` work with zero duplication.
"""

from __future__ import annotations

from collections import OrderedDict
import math
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit, urlunsplit

from agent.delegation_provider import ROUTE_KEYS
try:
    from agent.secret_scope import UnscopedSecretError, get_secret
except ImportError:  # pragma: no cover - older core fallback
    from agent.secret_scope import get_secret  # type: ignore[misc]

    class UnscopedSecretError(RuntimeError):  # type: ignore[no-redef]
        pass

PLUGIN_ID = "decider"

PRIMARY_KEY_ENV = "DECIDER_API_KEY"
FALLBACK_KEY_ENV = "DECIDER_FALLBACK_API_KEY"
ALLOWED_SECRET_ENVS = frozenset({PRIMARY_KEY_ENV, FALLBACK_KEY_ENV})

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
_TRUE_STRINGS = frozenset({"1", "true", "yes", "on"})
_FALSE_STRINGS = frozenset({"0", "false", "no", "off", ""})


def _text(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in _TRUE_STRINGS:
            return True
        if norm in _FALSE_STRINGS:
            return False
    return default


def is_loopback_host(hostname: str | None) -> bool:
    return (hostname or "").strip().lower() in ("127.0.0.1", "localhost", "::1")


def is_ollama_endpoint(url: str) -> bool:
    """Return True for local/LAN Ollama-style endpoints that never take Bearer auth."""
    raw = (url or "").strip()
    if not raw:
        return False
    try:
        parts = urlsplit(raw)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return False
    if port == 11434 or (is_loopback_host(host) and port in (None, 11434)):
        return True
    if host.startswith("192.168.") or host.startswith("10."):
        return port in (None, 11434)
    if host.startswith("172."):
        octets = host.split(".")
        if len(octets) == 4 and octets[1].isdigit() and 16 <= int(octets[1]) <= 31:
            return port in (None, 11434)
    return False


def display_url(raw_url: str) -> str:
    """Strip any embedded userinfo from ``raw_url`` so logs and errors never echo credentials."""
    text = (raw_url or "").strip()
    if not text:
        return ""
    try:
        parts = urlsplit(text)
    except ValueError:
        return "<invalid-url>"
    if not parts.username and not parts.password:
        return text
    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = f"{host}:{port}" if port else host
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def read_secret_value(env_name: str) -> str:
    """Read an allowed plugin secret strictly via ``agent.secret_scope.get_secret``.

    Propagates ``UnscopedSecretError`` untouched so unscoped calls fail closed.
    """
    if env_name not in ALLOWED_SECRET_ENVS:
        raise ValueError(f"Unsupported Decider secret slot: {env_name!r}")
    val = (get_secret(env_name, "") or "").strip()
    if val and (not val.isascii() or not val.isprintable()):
        return ""
    return val


def primary_api_key() -> str:
    return read_secret_value(PRIMARY_KEY_ENV)


def fallback_api_key() -> str:
    return read_secret_value(FALLBACK_KEY_ENV)


def effective_fallback_api_key(primary_url: str) -> str:
    """Resolve the Bearer key for ``fallback_url``.

    ``DECIDER_FALLBACK_API_KEY`` always takes precedence. For convenience when the primary endpoint
    is local/keyless Ollama (``http://127.0.0.1:11434`` or LAN ``:11434``), ``DECIDER_API_KEY`` is
    accepted as a fallback alias; when the primary endpoint is itself a non-Ollama remote server,
    the two keys stay strictly isolated so credentials are never shared across distinct remote hosts.
    """
    dedicated = fallback_api_key()
    if dedicated:
        return dedicated
    if is_ollama_endpoint(primary_url):
        return primary_api_key()
    return ""


def validate_secret_input(raw_value: str, *, label: str = "API key") -> str | None:
    """Validate a candidate secret string before persisting it."""
    val = str(raw_value or "").strip()
    if not val:
        return f"{label} cannot be empty (use clear to remove a saved key)."
    if set(val) == {"*"} or "«redacted" in val or val.endswith("..."):
        return f"{label} looks like a redacted placeholder; paste the full key."
    if not val.isascii() or not val.isprintable() or any(ch.isspace() for ch in val):
        return f"{label} must contain printable ASCII characters with no spaces."
    return None


def save_secret(env_name: str, raw_value: str) -> None:
    """Persist ``env_name`` via Hermes's unified credential lifecycle."""
    if env_name not in ALLOWED_SECRET_ENVS:
        raise ValueError(f"Unsupported Decider secret slot: {env_name!r}")
    problem = validate_secret_input(raw_value, label=env_name)
    if problem:
        raise ValueError(problem)
    cleaned = raw_value.strip()

    from hermes_cli import config

    if config.is_managed():
        raise RuntimeError(config.format_managed_message(f"set {env_name}"))
    secure_save = getattr(config, "save_env_value_secure", None)
    if callable(secure_save):
        secure_save(env_name, cleaned)
    else:
        config.save_env_value(env_name, cleaned)


def clear_secret(env_name: str) -> bool:
    """Remove ``env_name`` via Hermes's public ``remove_env_value`` helper."""
    if env_name not in ALLOWED_SECRET_ENVS:
        raise ValueError(f"Unsupported Decider secret slot: {env_name!r}")
    from hermes_cli import config

    if config.is_managed():
        raise RuntimeError(config.format_managed_message(f"remove {env_name}"))
    remover = getattr(config, "remove_env_value", None)
    if callable(remover):
        return bool(remover(env_name))
    return False


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
        if entry.get("daybreak") is True:
            overrides = dict(route.get("request_overrides") or {})
            overrides["daybreak"] = True
            route["request_overrides"] = overrides
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
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(num):
        return default
    return max(low, min(high, num))


def normalize_decider(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize a Decider settings dict to canonical types and safe defaults.

    Cloud/secondary fallback is strictly opt-in (``allow_fallback: False``, ``fallback_url: ""``)
    so task text never leaves the local machine unless explicitly configured by the operator.
    """
    src = raw if isinstance(raw, Mapping) else {}
    enabled = _parse_bool(src.get("enabled"), True)

    url_raw = _text(src.get("url")) or DEFAULT_DECIDER_URL
    url = normalize_systemone_url(url_raw)
    primary_model = (
        _text(src.get("primary_model"))
        or _text(src.get("model"))
        or DEFAULT_DECIDER_MODEL
    )

    allow_fallback = _parse_bool(src.get("allow_fallback"), False)
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
        "primary_model": primary_model,
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


def validate_endpoint_url(
    raw_url: str,
    *,
    label: str,
    require_https_if_keyed: bool = False,
) -> str | None:
    """Validate a candidate ``/v1/systemone`` endpoint URL."""
    text = (raw_url or "").strip()
    if not text:
        return None
    try:
        parts = urlsplit(text)
        _ = parts.port
    except ValueError:
        return f"{label} has an invalid URL or port."
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return f"{label} must start with http:// or https:// and include a hostname."
    if parts.username or parts.password:
        return (
            f"{label} must not embed credentials in the URL; store API keys in plugin secrets "
            f"({PRIMARY_KEY_ENV} / {FALLBACK_KEY_ENV})."
        )
    if require_https_if_keyed and parts.scheme == "http" and not is_loopback_host(parts.hostname):
        return (
            f"{label} must use https:// when an API key is configured "
            f"(plain http:// is only permitted on localhost/loopback)."
        )
    return None


def validate_decider(
    settings: Mapping[str, Any],
    *,
    has_primary_key: bool | None = None,
    has_fallback_key: bool | None = None,
) -> list[str]:
    """Validate candidate System-1 Decider settings before saving (empty list = valid)."""
    problems: list[str] = []
    url = _text(settings.get("url"))
    fb_url = _text(settings.get("fallback_url"))
    allow_fb = _parse_bool(settings.get("allow_fallback"), False)

    if has_primary_key is None:
        try:
            has_primary_key = bool(primary_api_key())
        except UnscopedSecretError:
            has_primary_key = False
    if has_fallback_key is None:
        try:
            eff_url = normalize_systemone_url(url or DEFAULT_DECIDER_URL)
            has_fallback_key = bool(effective_fallback_api_key(eff_url))
        except UnscopedSecretError:
            has_fallback_key = False

    url_will_send_key = bool(has_primary_key and url and not is_ollama_endpoint(url))
    url_err = validate_endpoint_url(
        url,
        label="Primary endpoint URL",
        require_https_if_keyed=url_will_send_key,
    )
    if url_err:
        problems.append(url_err)

    fb_will_send_key = bool(has_fallback_key and fb_url and not is_ollama_endpoint(fb_url))
    fb_err = validate_endpoint_url(
        fb_url,
        label="Fallback URL",
        require_https_if_keyed=fb_will_send_key,
    )
    if fb_err:
        problems.append(fb_err)

    if allow_fb and not fb_url:
        problems.append("Fallback URL is required when secondary fallback is enabled.")
    if (
        allow_fb
        and fb_url
        and url
        and normalize_systemone_url(fb_url) == normalize_systemone_url(url)
    ):
        problems.append("Fallback URL must differ from the primary endpoint URL.")

    for field, low, high in (
        ("specialist_threshold", 0.0, 1.0),
        ("confidence_threshold", 0.0, 1.0),
        ("timeout", 1.0, 120.0),
    ):
        val = settings.get(field)
        if val is None or val == "":
            continue
        if isinstance(val, bool):
            problems.append(f"{field} must be a finite number between {low} and {high}.")
            continue
        try:
            num = float(val)
        except (TypeError, ValueError, OverflowError):
            problems.append(f"{field} must be a finite number between {low} and {high}.")
            continue
        if not math.isfinite(num) or not (low <= num <= high):
            problems.append(f"{field} must be between {low} and {high} (got {val}).")
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
    if "primary_model" not in cleaned_input and "model" in cleaned_input:
        cleaned_input["primary_model"] = cleaned_input.pop("model")
    else:
        cleaned_input.pop("model", None)
    for k in ("url", "primary_model", "fallback_model"):
        if cleaned_input.get(k) == "":
            cleaned_input.pop(k, None)
    normalized = normalize_decider(cleaned_input)

    def mutate(cfg: dict[str, Any]) -> None:
        own = _child(_child(_child(cfg, "plugins"), "entries"), PLUGIN_ID)
        target = _child(own, "settings")
        target.pop("model", None)
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
