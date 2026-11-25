"""``/decider`` slash command: status, toggle, probe, history, and backend mode switch."""

from __future__ import annotations

from typing import Any, Callable

from . import decider_router, store

USAGE = "/decider [status | on | off | test <goal> | history | clear | mode [decider|model_map|default] [--now]]"


def make_handler(ctx) -> Callable[..., str]:
    def decider_command(raw_args: str = "", session_key: str = "", **_: Any) -> str:
        return handle(
            raw_args,
            session_key=session_key,
            api=getattr(ctx, "delegation_backends", None),
        )

    return decider_command


def _switch_mode(raw_rest: str, session_key: str, api) -> str:
    words = raw_rest.split()
    now = "--now" in words
    tokens = [w for w in words if w != "--now"]
    mode = tokens[0].lower() if tokens else "decider"
    if mode not in store.MODES:
        return f"Unknown mode {mode!r}. Choose one of: {', '.join(store.MODES)}."
    try:
        if api is None:
            store.set_mode(mode)
            applied = False
        elif now:
            try:
                api.set_global_backend(mode, session_key=session_key, apply_now=True)
                applied = True
            except TypeError:
                api.set_global_backend(mode, session_key=session_key)
                applied = False
        else:
            api.set_global_backend(mode, session_key=session_key)
            applied = False
    except (ValueError, PermissionError, RuntimeError) as exc:
        return f"Could not switch delegation backend: {exc}"

    if applied:
        return (
            f"Delegation backend set to '{mode}' ({store.mode_label(mode)}) and applied to this chat "
            f"(tools rebuild on next turn)."
        )
    return (
        f"Delegation backend set to '{mode}' ({store.mode_label(mode)}) for new sessions (/new). "
        f"Add --now to rebuild this chat's tools immediately."
    )


def handle(raw_args: str = "", *, session_key: str = "", api=None) -> str:
    stripped = (raw_args or "").strip()
    tokens = stripped.split(None, 1)
    sub = tokens[0].lower() if tokens else "status"
    tail = tokens[1].strip() if len(tokens) > 1 else ""

    cfg = store.full_config()
    dec = dict(store.decider_settings(cfg))

    if sub in ("on", "enable", "enabled"):
        dec["enabled"] = True
        try:
            saved = store.save_decider(dec)
        except RuntimeError as exc:
            return f"Could not enable System-1 Decider: {exc}"
        return f"System-1 Decider auto-router: ON ({saved['primary_model']} @ {store.display_url(saved['url'])})."

    if sub in ("off", "disable", "disabled"):
        dec["enabled"] = False
        try:
            store.save_decider(dec)
        except RuntimeError as exc:
            return f"Could not disable System-1 Decider: {exc}"
        return "System-1 Decider auto-router: OFF."

    if sub in ("history", "log"):
        rows = decider_router.read_decision_log(limit=10)
        if not rows:
            return "No System-1 Decider routing decisions recorded yet."
        lines = [f"Recent System-1 Decider decisions ({len(rows)}):"]
        for r in rows:
            fb = " [fallback]" if r.get("fallback_used") else ""
            lines.append(
                f"  • {r.get('action', '-')} -> {r.get('route', '-')} "
                f"(spec={r.get('needs_specialist', 0):.2f}, conf={r.get('confidence', 0):.2f}, "
                f"{r.get('latency_ms', 0):.0f}ms{fb}): {r.get('goal', '')}"
            )
        return "\n".join(lines)

    if sub == "clear":
        removed = decider_router.clear_decision_log()
        return f"Cleared {removed} cached System-1 Decider routing decision(s)."

    if sub in ("test", "probe"):
        goal = tail or "Analyze screenshot.png and extract the error dialog text"
        has_img = any(k in goal.lower() for k in ("screenshot", "image", ".png", ".jpg"))
        res = decider_router.probe_decider(goal, has_images=has_img, config=cfg)
        if not res.get("ok"):
            return f"Decider probe failed: {res.get('error')}"
        action = "upgraded to specialist" if res.get("upgraded") else "kept on default"
        fb = " (fallback)" if res.get("fallback_used") else ""
        return (
            f"Decider probe ({res.get('latency_ms', 0):.0f}ms on {res.get('model')}{fb}):\n"
            f"  Goal: {res.get('goal')}\n"
            f"  Route: {res.get('effective_route')} ({action}; candidate={res.get('candidate_route')}, "
            f"needs_specialist={res.get('needs_specialist', 0):.2f}, confidence={res.get('confidence', 0):.2f})"
        )

    if sub == "mode":
        return _switch_mode(tail, session_key=session_key, api=api)

    if sub not in ("status", "show", "help"):
        return f"Unknown subcommand {sub!r}.\nUsage: {USAGE}"

    mode = store.current_mode(cfg)
    state_lbl = "ON" if dec["enabled"] else "OFF"
    if dec["allow_fallback"] and dec["fallback_url"]:
        try:
            has_fb_key = bool(store.effective_fallback_api_key(dec["url"]))
        except store.UnscopedSecretError:
            has_fb_key = False
        fb_tag = (
            ""
            if store.is_ollama_endpoint(dec["fallback_url"])
            else (" [key: set]" if has_fb_key else " [key: NOT SET]")
        )
        fb_lbl = f"{dec['fallback_model']} @ {store.display_url(dec['fallback_url'])}{fb_tag}"
    else:
        fb_lbl = "off (local-only)"
    return (
        f"System-1 Decider auto-router: {state_lbl} (delegation.backend: {mode})\n"
        f"  Primary: {dec['primary_model']} @ {store.display_url(dec['url'])}\n"
        f"  Fallback: {fb_lbl}\n"
        f"  Thresholds: specialist >= {dec['specialist_threshold']:.2f}, confidence >= {dec['confidence_threshold']:.2f}, timeout {dec['timeout']:.0f}s\n"
        f"  Excluded routes: {', '.join(dec['excluded_options']) if dec['excluded_options'] else '(none)'}\n"
        f"Usage: {USAGE}"
    )
