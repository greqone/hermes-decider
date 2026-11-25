"""System-1 Decider ``/v1/systemone`` client, batch router, probe, and decision history.

Evaluates unassigned ``delegate_task`` items in a single batched ``/v1/systemone`` forward pass:
1. ``t{i}_needs_specialist`` (``noul``): does this task genuinely require a specialist/frontier
   route (visual/multimodal analysis, complex architecture, adversarial verification, offensive
   security, or deep multi-source synthesis), or is the default fast worker sufficient?
2. ``t{i}_route`` (``choice``): which eligible route best matches the task?

Security & privacy invariants (matching catalog admission rules):
- Zero ``os.environ`` reads and zero manual ``.env`` file reads.
- Optional ``DECIDER_API_KEY`` and ``DECIDER_FALLBACK_API_KEY`` are read exclusively via
  ``agent.secret_scope.get_secret`` (propagating ``UnscopedSecretError``).
- Bearer credentials are strictly host-locked to their configured endpoint, never sent to local/LAN
  Ollama endpoints, never sent over cleartext ``http://`` off localhost/loopback, and stripped on
  cross-origin redirects via ``open_credentialed_url``.
- Secondary fallback is disabled by default (``allow_fallback: False``, ``fallback_url: ""``), so
  task text never leaves the machine unless explicitly opted into by the operator.
- Never attaches undeclared properties (such as ``_decider_evaluated``) to ``delegate_task`` items,
  preserving strict schema compatibility with Hermes core's ``_route_one`` guard.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping
from urllib.parse import urlsplit
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_MAX_LOG_ENTRIES = 100
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024  # 2 MiB safety ceiling on /v1/systemone responses
_PRIMARY_COOLDOWN_SECONDS: float = 45.0

_STATE_LOCK = threading.RLock()
_LOG_LOCK = threading.RLock()

# Cooldown keyed by (primary_url, primary_model) so editing the URL/model immediately clears any
# prior cooldown instead of routing to fallback for 45s on stale state.
_PRIMARY_COOLDOWN_UNTIL: dict[tuple[str, str], float] = {}

# Short-lived in-memory cache so a batch decision evaluated in `pre_tool_call` that intentionally
# leaves `option` unset (when `default` is empty -> inherit parent route) is not re-queried a
# second time in `DeciderDelegationProvider.route_task` milliseconds later.
_RECENT_DECISIONS: OrderedDict[str, tuple[float, str]] = OrderedDict()
_RECENT_TTL_SECONDS: float = 30.0
_RECENT_MAX_ENTRIES: int = 256


def _load_store():
    if __package__:
        from . import store
        return store
    path = Path(__file__).resolve().with_name("store.py")
    spec = importlib.util.spec_from_file_location(f"{__name__}_store", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load decider store: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reset_runtime_caches() -> None:
    """Clear in-memory cooldown and recent-decision caches (used by tests)."""
    with _STATE_LOCK:
        _PRIMARY_COOLDOWN_UNTIL.clear()
        _RECENT_DECISIONS.clear()


def _head_tail(text: str, head: int = 320, tail: int = 160) -> str:
    """Keep the beginning and end of ``text`` so batch prefill stays compact."""
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= head + tail + 24:
        return cleaned
    omitted = len(cleaned) - head - tail
    return f"{cleaned[:head]} ...[+{omitted} chars]... {cleaned[-tail:]}"


def _host_and_scheme(url: str) -> tuple[str, str]:
    try:
        parts = urlsplit(url)
        return (parts.hostname or "").lower(), (parts.scheme or "").lower()
    except ValueError:
        return "", ""


def _adapt_questions_for_endpoint(questions: Mapping[str, Any], url: str, *, is_ollama: bool) -> dict[str, Any]:
    """Ollama expects ``criteria`` on all question types; remote ``/v1/systemone`` expects ``criterion`` on ``noul``."""
    adapted: dict[str, Any] = {}
    for qname, qspec in questions.items():
        if not isinstance(qspec, Mapping):
            continue
        q = dict(qspec)
        qtype = q.get("type")
        if is_ollama:
            if "criterion" in q and "criteria" not in q:
                q["criteria"] = q.pop("criterion")
        elif qtype == "noul":
            if "criteria" in q and "criterion" not in q:
                q["criterion"] = q.pop("criteria")
        adapted[qname] = q
    return adapted


def attach_bearer_if_allowed(
    request: urllib.request.Request,
    *,
    configured_url: str,
    api_key: str,
    key_label: str = "DECIDER_API_KEY",
) -> None:
    """Attach ``Authorization: Bearer`` strictly when safe and host-matched.

    Refuses:
    - empty or non-ASCII/non-printable keys
    - local/LAN Ollama endpoints
    - mismatched request vs configured hostnames
    - cleartext ``http://`` on any non-loopback host
    """
    store = _load_store()
    cleaned = (api_key or "").strip()
    if not cleaned or not cleaned.isascii() or not cleaned.isprintable():
        return
    if store.is_ollama_endpoint(configured_url):
        return
    req_host, req_scheme = _host_and_scheme(request.full_url)
    cfg_host, _ = _host_and_scheme(configured_url)
    if not req_host or req_host != cfg_host:
        logger.warning(
            "decider: refusing to send %s to host %r (configured host is %r)",
            key_label,
            req_host,
            cfg_host,
        )
        return
    if req_scheme != "https" and not store.is_loopback_host(req_host):
        logger.warning(
            "decider: refusing to send %s over cleartext http:// to non-loopback host %r",
            key_label,
            req_host,
        )
        return
    request.add_header("Authorization", f"Bearer {cleaned}")


def _user_agent() -> str:
    try:
        from hermes_cli.version_info import hermes_user_agent
        return hermes_user_agent("hermes-decider")
    except Exception:
        return "hermes-decider/1.2"


def _open_url(request: urllib.request.Request, timeout: float):
    """Use Hermes's redirect-safe ``open_credentialed_url`` when available, falling back to ``urlopen``."""
    try:
        from hermes_cli.urllib_security import open_credentialed_url
        return open_credentialed_url(request, timeout=timeout)
    except ImportError:
        return urllib.request.urlopen(request, timeout=timeout)


def _short_route_desc(desc: str, max_chars: int = 85) -> str:
    cleaned = " ".join(str(desc or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


def eligible_routes(
    options: OrderedDict[str, dict[str, Any]],
    dec_cfg: Mapping[str, Any],
) -> OrderedDict[str, str]:
    """Return ``{route_name: concise_description}`` for routes the Decider may auto-select."""
    excluded_set = set(dec_cfg.get("excluded_options") or [])
    out: OrderedDict[str, str] = OrderedDict()
    for name, entry in options.items():
        if name in excluded_set:
            continue
        desc = _short_route_desc(str(entry.get("description") or "") or f"Route {name}", max_chars=85)
        out[name] = desc
    return out


def _extract_answers_map(resp: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(resp, Mapping):
        return {}
    ans = resp.get("answers")
    if isinstance(ans, Mapping):
        return ans
    dec = resp.get("decisions")
    if isinstance(dec, Mapping):
        return dec
    return {}


def _extract_noul_prob(entry: Mapping[str, Any] | None) -> float:
    if not isinstance(entry, Mapping):
        return 0.0
    for key in ("noul", "yes_probability", "probability"):
        val = entry.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
            return float(val)
    return 0.0


def _post_systemone_once(
    *,
    url: str,
    model: str,
    state: dict[str, Any],
    questions: Mapping[str, Any],
    timeout: float,
    api_key: str = "",
    key_label: str = "DECIDER_API_KEY",
) -> dict[str, Any]:
    store = _load_store()
    use_ollama = store.is_ollama_endpoint(url)
    serialized_state: Any = (
        state if use_ollama else json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    )
    body: dict[str, Any] = {
        "model": model,
        "state": serialized_state,
        "questions": _adapt_questions_for_endpoint(questions, url, is_ollama=use_ollama),
    }
    if use_ollama:
        body["keep_alive"] = -1

    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": _user_agent(),
        },
        method="POST",
    )
    attach_bearer_if_allowed(req, configured_url=url, api_key=api_key, key_label=key_label)

    try:
        with _open_url(req, timeout=timeout) as resp:
            chunk = resp.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        safe = store.display_url(url)
        raise RuntimeError(f"HTTP {exc.code} from {safe}") from None
    except urllib.error.URLError as exc:
        safe = store.display_url(url)
        raise RuntimeError(f"Connection to {safe} failed: {exc.reason}") from None

    if len(chunk) > _MAX_RESPONSE_BYTES:
        raise RuntimeError(f"Response from {store.display_url(url)} exceeded {_MAX_RESPONSE_BYTES} bytes")
    parsed = json.loads(chunk.decode("utf-8"))
    return parsed if isinstance(parsed, dict) else {}


def _is_primary_in_cooldown(key: tuple[str, str]) -> bool:
    with _STATE_LOCK:
        until = _PRIMARY_COOLDOWN_UNTIL.get(key, 0.0)
        if until <= 0.0:
            return False
        if time.monotonic() >= until:
            _PRIMARY_COOLDOWN_UNTIL.pop(key, None)
            return False
        return True


def _mark_primary_cooldown(key: tuple[str, str]) -> None:
    with _STATE_LOCK:
        _PRIMARY_COOLDOWN_UNTIL[key] = time.monotonic() + _PRIMARY_COOLDOWN_SECONDS


def _clear_primary_cooldown(key: tuple[str, str]) -> None:
    with _STATE_LOCK:
        _PRIMARY_COOLDOWN_UNTIL.pop(key, None)


def call_systemone(
    state: dict[str, Any],
    questions: Mapping[str, Any],
    dec_cfg: Mapping[str, Any],
    *,
    bypass_cooldown: bool = False,
) -> dict[str, Any]:
    """Call primary ``/v1/systemone`` endpoint; use secondary fallback ONLY when explicitly enabled."""
    store = _load_store()
    primary_url = str(dec_cfg["url"])
    primary_model = str(dec_cfg.get("primary_model") or dec_cfg.get("model") or store.DEFAULT_DECIDER_MODEL)
    fb_url = str(dec_cfg.get("fallback_url") or "")
    fb_model = str(dec_cfg.get("fallback_model") or store.DEFAULT_FALLBACK_MODEL)
    timeout = float(dec_cfg["timeout"])

    # Note: UnscopedSecretError propagates untouched if secret_scope is unbound.
    primary_key = store.primary_api_key()
    fb_key = store.effective_fallback_api_key(primary_url) if fb_url else ""

    fb_host, fb_scheme = _host_and_scheme(fb_url)
    fb_transport_ok = bool(
        fb_url
        and (
            store.is_ollama_endpoint(fb_url)
            or (
                bool(fb_key)
                and (fb_scheme == "https" or store.is_loopback_host(fb_host))
            )
        )
    )
    can_fallback = (
        bool(dec_cfg.get("allow_fallback"))
        and bool(fb_url)
        and fb_url != primary_url
        and fb_transport_ok
    )

    cooldown_key = (primary_url, primary_model)
    t0 = time.perf_counter()
    if can_fallback and not bypass_cooldown and _is_primary_in_cooldown(cooldown_key):
        data = _post_systemone_once(
            url=fb_url,
            model=fb_model,
            state=state,
            questions=questions,
            timeout=timeout,
            api_key=fb_key,
            key_label=store.FALLBACK_KEY_ENV,
        )
        data["fallback_used"] = True
        data["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        return data

    try:
        data = _post_systemone_once(
            url=primary_url,
            model=primary_model,
            state=state,
            questions=questions,
            timeout=timeout,
            api_key=primary_key,
            key_label=store.PRIMARY_KEY_ENV,
        )
        _clear_primary_cooldown(cooldown_key)
        data["fallback_used"] = False
        data["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        return data
    except Exception as primary_err:
        _mark_primary_cooldown(cooldown_key)
        if not can_fallback:
            if (
                bool(dec_cfg.get("allow_fallback"))
                and bool(fb_url)
                and not store.is_ollama_endpoint(fb_url)
                and not fb_key
            ):
                logger.warning(
                    "decider: secondary fallback %s is enabled without %s; skipping unauthenticated fallback",
                    store.display_url(fb_url),
                    store.FALLBACK_KEY_ENV,
                )
            raise
        logger.warning(
            "decider: primary endpoint %s failed (%s); using opted-in fallback %s",
            store.display_url(primary_url),
            primary_err,
            store.display_url(fb_url),
        )
        t_fb = time.perf_counter()
        data = _post_systemone_once(
            url=fb_url,
            model=fb_model,
            state=state,
            questions=questions,
            timeout=timeout,
            api_key=fb_key,
            key_label=store.FALLBACK_KEY_ENV,
        )
        data["fallback_used"] = True
        data["latency_ms"] = round((time.perf_counter() - t_fb) * 1000.0, 1)
        return data


def _decision_log_path() -> Path | None:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home()) / "cache" / "decider" / "decisions.jsonl"
    except Exception:
        return None


def _record_decision(entry: dict[str, Any]) -> None:
    """Append ``entry`` to the bounded JSONL log using a PID+TID temp file and atomic replace."""
    path = _decision_log_path()
    if path is None:
        return
    line = json.dumps(entry, ensure_ascii=False)
    with _LOG_LOCK:
        tmp: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            existing: list[str] = []
            if path.is_file():
                existing = [
                    ln for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()
                ]
            existing.append(line)
            if len(existing) > _MAX_LOG_ENTRIES:
                existing = existing[-_MAX_LOG_ENTRIES:]
            tmp = path.parent / f".decisions.{os.getpid()}.{threading.get_ident()}.tmp"
            tmp.write_text("\n".join(existing) + "\n", encoding="utf-8")
            for attempt in range(3):
                try:
                    os.replace(tmp, path)
                    break
                except PermissionError:
                    if attempt == 2:
                        break
                    time.sleep(0.01 * (attempt + 1))
        except Exception:
            pass
        finally:
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass


def read_decision_log(limit: int = 30) -> list[dict[str, Any]]:
    """Return the most recent Decider routing decisions (newest first)."""
    path = _decision_log_path()
    if path is None:
        return []
    with _LOG_LOCK:
        if not path.is_file():
            return []
        try:
            lines = [
                ln for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()
            ]
            items: list[dict[str, Any]] = []
            for ln in reversed(lines[-max(1, min(int(limit), _MAX_LOG_ENTRIES)):]):
                try:
                    row = json.loads(ln)
                    if isinstance(row, dict):
                        items.append(row)
                except Exception:
                    continue
            return items
        except Exception:
            return []


def clear_decision_log() -> int:
    """Clear the cached Decider routing log and return the number of removed entries."""
    path = _decision_log_path()
    if path is None:
        return 0
    with _LOG_LOCK:
        if not path.is_file():
            return 0
        try:
            lines = [
                ln for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()
            ]
            path.unlink(missing_ok=True)
            return len(lines)
        except Exception:
            return 0


def _cache_key(eligible: Mapping[str, str], default_opt: str, goal: str, context: str) -> str:
    raw = json.dumps(
        [list(eligible.items()), default_opt, _head_tail(goal), _head_tail(context, 200, 100)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _remember_decision(key: str, chosen_or_empty: str) -> None:
    with _STATE_LOCK:
        now = time.monotonic()
        expired = [k for k, (exp, _) in _RECENT_DECISIONS.items() if exp <= now]
        for k in expired:
            _RECENT_DECISIONS.pop(k, None)
        _RECENT_DECISIONS[key] = (now + _RECENT_TTL_SECONDS, chosen_or_empty)
        _RECENT_DECISIONS.move_to_end(key)
        while len(_RECENT_DECISIONS) > _RECENT_MAX_ENTRIES:
            _RECENT_DECISIONS.popitem(last=False)


def _lookup_recent_decision(key: str) -> tuple[bool, str]:
    with _STATE_LOCK:
        hit = _RECENT_DECISIONS.get(key)
        if hit is None:
            return False, ""
        exp, val = hit
        if exp <= time.monotonic():
            _RECENT_DECISIONS.pop(key, None)
            return False, ""
        return True, val


def route_unassigned_tasks(
    tasks: list[dict[str, Any]],
    *,
    config: Mapping[str, Any] | None = None,
    section: Mapping[str, Any] | None = None,
) -> dict[int, str]:
    """Batch-route unassigned ``delegate_task`` items in place (setting ONLY ``task['option']``).

    Never attaches undeclared keys onto ``task`` dicts so Hermes core's ``_route_one`` parameter
    validator accepts every task cleanly.
    """
    store = _load_store()
    unassigned: list[tuple[int, dict[str, Any]]] = []
    for idx, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        opt = task.get("option")
        if isinstance(opt, str) and opt.strip():
            continue
        goal = str(task.get("goal") or "").strip()
        if not goal:
            continue
        unassigned.append((idx, task))

    if not unassigned:
        return {}

    dec_cfg = store.decider_settings(config)
    if not dec_cfg.get("enabled"):
        return {}

    eff_section = store.delegation_section(section if section is not None else config)
    options = store.configured_options(eff_section)
    eligible = eligible_routes(options, dec_cfg)
    if not eligible:
        return {}

    default_opt = store.route_default(eff_section)
    primary_model = str(dec_cfg.get("primary_model") or dec_cfg.get("model") or store.DEFAULT_DECIDER_MODEL)

    # TTFT optimization on non-causal decision models (e.g. pplx-decider:27b-nvfp4 on Apple Silicon
    # MLX): keep full route descriptions once in `state.available_routes` when batching >1 tasks.
    compact_criteria = (
        {name: f"Route {name}" for name in eligible}
        if len(unassigned) > 1
        else dict(eligible)
    )

    state_tasks: dict[str, Any] = {}
    questions: dict[str, Any] = {}
    for idx, task in unassigned:
        key = f"t{idx}"
        images = task.get("images")
        has_images = bool(isinstance(images, list) and len(images) > 0)
        state_tasks[key] = {
            "goal": _head_tail(str(task.get("goal") or ""), head=320, tail=160),
            "context": _head_tail(str(task.get("context") or ""), head=200, tail=100),
            "has_images": has_images,
        }
        questions[f"{key}_needs_specialist"] = {
            "type": "noul",
            "instructions": (
                f"Does task `{key}` require a specialist route (visual/image analysis, complex "
                f"multi-file architecture, adversarial verification, offensive security, or deep "
                f"multi-source synthesis) instead of `{default_opt or 'default'}`?"
            ),
            "criteria": {
                "true": (
                    "Task has images (`has_images: true`), offensive security, complex architecture, "
                    "adversarial review, or deep synthesis needing a specialist route."
                ),
                "false": (
                    f"Standard coding, debugging, test writing, research, or tool-heavy work handled "
                    f"well by `{default_opt or 'default'}`."
                ),
            },
        }
        questions[f"{key}_route"] = {
            "type": "choice",
            "instructions": (
                f"Select the best route from `available_routes` for task `{key}`."
                if len(unassigned) > 1
                else f"Select the best route for task `{key}`."
            ),
            "criteria": compact_criteria,
        }

    state: dict[str, Any] = {
        "default_route": default_opt or "inherit",
        "tasks": state_tasks,
    }
    if len(unassigned) > 1:
        state["available_routes"] = dict(eligible)

    try:
        resp = call_systemone(state, questions, dec_cfg)
    except store.UnscopedSecretError:
        raise
    except Exception as exc:
        logger.warning(
            "decider: endpoint unreachable (%s); keeping default route %r",
            exc,
            default_opt or "inherit",
        )
        for _, task in unassigned:
            ckey = _cache_key(
                eligible, default_opt, str(task.get("goal") or ""), str(task.get("context") or "")
            )
            _remember_decision(ckey, default_opt)
        return {}

    decisions = _extract_answers_map(resp)
    if not decisions:
        return {}

    spec_thresh = float(dec_cfg["specialist_threshold"])
    conf_thresh = float(dec_cfg["confidence_threshold"])
    latency_ms = resp.get("latency_ms", 0.0)
    used_model = str(resp.get("model") or primary_model)
    fallback_used = bool(resp.get("fallback_used", False))

    assigned: dict[int, str] = {}
    for idx, task in unassigned:
        key = f"t{idx}"
        needs_d = decisions.get(f"{key}_needs_specialist") or {}
        route_d = decisions.get(f"{key}_route") or {}

        needs_p = _extract_noul_prob(needs_d if isinstance(needs_d, Mapping) else None)
        picked = str((route_d if isinstance(route_d, Mapping) else {}).get("choice") or "").strip()
        raw_conf = (route_d if isinstance(route_d, Mapping) else {}).get("confidence", 0.0)
        conf = (
            float(raw_conf)
            if isinstance(raw_conf, (int, float)) and not isinstance(raw_conf, bool) and math.isfinite(float(raw_conf))
            else 0.0
        )
        has_images = bool(state_tasks[key]["has_images"])

        if (
            picked in eligible
            and picked != default_opt
            and (
                (needs_p >= spec_thresh and conf >= conf_thresh)
                or (has_images and "visual" in picked.lower() and conf >= 0.40)
            )
        ):
            task["option"] = picked
            assigned[idx] = picked
            action = "upgraded"
            effective_route = picked
            chosen_for_cache = picked
            logger.info(
                "decider: upgraded task[%d] -> %s (needs_specialist=%.2f, confidence=%.2f, %.0fms)",
                idx,
                picked,
                needs_p,
                conf,
                latency_ms,
            )
        else:
            if default_opt and default_opt in options:
                task["option"] = default_opt
                assigned[idx] = default_opt
            action = "kept_default"
            effective_route = default_opt or "inherit"
            chosen_for_cache = default_opt
            logger.info(
                "decider: kept task[%d] on %s (picked=%s, needs_specialist=%.2f, confidence=%.2f, %.0fms)",
                idx,
                effective_route,
                picked or "-",
                needs_p,
                conf,
                latency_ms,
            )

        ckey = _cache_key(
            eligible, default_opt, str(task.get("goal") or ""), str(task.get("context") or "")
        )
        _remember_decision(ckey, chosen_for_cache)

        _record_decision(
            {
                "timestamp": round(time.time(), 3),
                "goal": state_tasks[key]["goal"][:140],
                "has_images": has_images,
                "action": action,
                "route": effective_route,
                "candidate": picked or default_opt or "inherit",
                "needs_specialist": round(needs_p, 4),
                "confidence": round(conf, 4),
                "latency_ms": latency_ms,
                "model": used_model,
                "fallback_used": fallback_used,
            }
        )

    return assigned


def route_single_task(
    task: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
    section: Mapping[str, Any] | None = None,
) -> str:
    """Resolve one unassigned task inside ``DeciderDelegationProvider.route_task`` (cache-aware)."""
    store = _load_store()
    goal = str(task.get("goal") or "").strip()
    if not goal:
        return ""
    dec_cfg = store.decider_settings(config)
    if not dec_cfg.get("enabled"):
        return ""
    eff_section = store.delegation_section(section if section is not None else config)
    options = store.configured_options(eff_section)
    eligible = eligible_routes(options, dec_cfg)
    if not eligible:
        return ""
    default_opt = store.route_default(eff_section)
    context = str(task.get("context") or "")

    ckey = _cache_key(eligible, default_opt, goal, context)
    hit, cached_choice = _lookup_recent_decision(ckey)
    if hit:
        return cached_choice

    probe = dict(task)
    route_unassigned_tasks([probe], config=config, section=eff_section)
    opt = probe.get("option")
    return opt.strip() if isinstance(opt, str) and opt.strip() else ""


def probe_decider(
    goal: str,
    *,
    context: str = "",
    has_images: bool = False,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a live diagnostic probe against the configured System-1 Decider without spawning a child."""
    store = _load_store()
    dec_cfg = store.decider_settings(config)
    section = store.delegation_section(config)
    options = store.configured_options(section)
    eligible = eligible_routes(options, dec_cfg)
    default_opt = store.route_default(section)
    primary_model = str(dec_cfg.get("primary_model") or dec_cfg.get("model") or store.DEFAULT_DECIDER_MODEL)

    if not eligible:
        return {
            "ok": False,
            "error": (
                "No eligible model routes configured in delegation.backends.decider.options "
                "or delegation.backends.model_map.options."
            ),
            "config": dec_cfg,
        }

    questions = {
        "t0_needs_specialist": {
            "type": "noul",
            "instructions": (
                f"Does task `t0` require a specialist route (visual/image analysis, complex "
                f"multi-file architecture, adversarial verification, offensive security, or deep "
                f"multi-source synthesis) instead of `{default_opt or 'default'}`?"
            ),
            "criteria": {
                "true": (
                    "Task has images (`has_images: true`), offensive security, complex architecture, "
                    "adversarial review, or deep synthesis needing a specialist route."
                ),
                "false": (
                    f"Standard coding, debugging, test writing, research, or tool-heavy work handled "
                    f"well by `{default_opt or 'default'}`."
                ),
            },
        },
        "t0_route": {
            "type": "choice",
            "instructions": "Select the best route for task `t0`.",
            "criteria": dict(eligible),
        },
    }
    state = {
        "default_route": default_opt or "inherit",
        "tasks": {
            "t0": {
                "goal": _head_tail(goal, head=320, tail=160),
                "context": _head_tail(context, head=200, tail=100),
                "has_images": bool(has_images),
            }
        },
    }

    try:
        resp = call_systemone(state, questions, dec_cfg, bypass_cooldown=True)
    except store.UnscopedSecretError:
        raise
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "config": dec_cfg,
        }

    decisions = _extract_answers_map(resp)
    needs_d = decisions.get("t0_needs_specialist") if isinstance(decisions, Mapping) else {}
    route_d = decisions.get("t0_route") if isinstance(decisions, Mapping) else {}
    needs_p = _extract_noul_prob(needs_d if isinstance(needs_d, Mapping) else None)
    picked = str((route_d or {}).get("choice") or "").strip()
    raw_conf = (route_d or {}).get("confidence", 0.0) if isinstance(route_d, Mapping) else 0.0
    conf = (
        float(raw_conf)
        if isinstance(raw_conf, (int, float)) and not isinstance(raw_conf, bool) and math.isfinite(float(raw_conf))
        else 0.0
    )
    scores = (
        (route_d or {}).get("probabilities") or (route_d or {}).get("scores")
        if isinstance(route_d, Mapping)
        else {}
    )

    upgraded = bool(
        picked in eligible
        and picked != default_opt
        and (
            (needs_p >= float(dec_cfg["specialist_threshold"]) and conf >= float(dec_cfg["confidence_threshold"]))
            or (has_images and "visual" in picked.lower() and conf >= 0.40)
        )
    )
    effective = picked if upgraded else (default_opt or "inherit")

    return {
        "ok": True,
        "goal": goal,
        "has_images": bool(has_images),
        "upgraded": upgraded,
        "effective_route": effective,
        "candidate_route": picked,
        "default_route": default_opt or "inherit",
        "needs_specialist": round(needs_p, 4),
        "confidence": round(conf, 4),
        "scores": scores if isinstance(scores, Mapping) else {},
        "latency_ms": resp.get("latency_ms", 0.0),
        "model": resp.get("model", primary_model),
        "fallback_used": bool(resp.get("fallback_used", False)),
        "config": dec_cfg,
    }
