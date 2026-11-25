"""Host-authenticated backend API for the System-1 Decider Desktop UI.

Mounted by Hermes at ``/api/plugins/decider/`` when ``decider`` is enabled. Loads its adjacent
``store.py`` and ``decider_router.py`` files directly without mutating ``sys.modules`` or
``sys.path``.

Routes:
    GET    /state       Current delegation mode, configured routes, Decider settings, recent decisions
    PUT    /decider     Validate and save Decider settings (plugins.entries.decider.settings)
    POST   /probe       Run a live /v1/systemone routing probe for a sample subagent goal
    DELETE /decisions   Clear cached System-1 Decider routing history
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException

_PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def _load_sibling(filename: str):
    path = _PLUGIN_ROOT / filename
    spec = importlib.util.spec_from_file_location(f"{__name__}_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


store = _load_sibling("store.py")
decider_router = _load_sibling("decider_router.py")
router = APIRouter()


def _broadcast(event: str, payload: dict[str, Any]) -> None:
    try:
        from hermes_cli.plugin_events import broadcast_plugin_event
        broadcast_plugin_event("decider", event, payload)
    except Exception:
        pass


def _build_state(decision_limit: int = 25) -> dict[str, Any]:
    cfg = store.full_config()
    section = store.delegation_section(cfg)
    mode = store.current_mode(section)
    decider = dict(store.decider_settings(cfg))
    try:
        decider["has_api_key"] = bool(store.primary_api_key())
        decider["has_fallback_api_key"] = bool(store.effective_fallback_api_key(decider["url"]))
    except store.UnscopedSecretError:
        decider["has_api_key"] = False
        decider["has_fallback_api_key"] = False
    excluded = set(decider.get("excluded_options") or [])
    # Never expose raw credentials or base_urls over the transport: summary + eligibility only.
    routes = []
    for opt in store.raw_route_entries(section):
        name = str(opt.get("name") or "").strip()
        if not name:
            continue
        routes.append({
            "name": name,
            "description": str(opt.get("description") or ""),
            "summary": store.route_summary(opt),
            "decider_eligible": bool(name not in excluded),
        })
    return {
        "mode": mode,
        "mode_label": store.mode_label(mode),
        "routes": routes,
        "route_default": store.route_default(section),
        "decider": decider,
        "decisions": decider_router.read_decision_log(limit=decision_limit),
    }


@router.get("/state")
async def get_state(limit: int = 25):
    safe_limit = max(1, min(int(limit), 100))
    return await asyncio.to_thread(_build_state, safe_limit)


@router.put("/decider")
async def update_decider(payload: dict[str, Any] = Body(...)):
    def _run():
        cfg = store.full_config()
        current = dict(store.decider_settings(cfg))
        if "model" in payload and "primary_model" not in payload:
            current["primary_model"] = payload["model"]
        for key in (
            "enabled",
            "url",
            "primary_model",
            "fallback_url",
            "fallback_model",
            "allow_fallback",
            "timeout",
            "specialist_threshold",
            "confidence_threshold",
            "excluded_options",
        ):
            if key in payload:
                current[key] = payload[key]

        raw_api_key = str(payload.get("api_key") or "").strip()
        clear_api_key = payload.get("clear_api_key") is True
        raw_fb_key = str(payload.get("fallback_api_key") or "").strip()
        clear_fb_key = payload.get("clear_fallback_api_key") is True

        secret_problems: list[str] = []
        if raw_api_key:
            err = store.validate_secret_input(raw_api_key, label=store.PRIMARY_KEY_ENV)
            if err:
                secret_problems.append(err)
        if raw_fb_key:
            err = store.validate_secret_input(raw_fb_key, label=store.FALLBACK_KEY_ENV)
            if err:
                secret_problems.append(err)

        try:
            cur_pri = bool(store.primary_api_key())
            cur_fb = bool(store.fallback_api_key())
        except store.UnscopedSecretError:
            cur_pri = False
            cur_fb = False

        next_pri = True if raw_api_key else (False if clear_api_key else cur_pri)
        eff_url = store.normalize_systemone_url(str(current.get("url") or store.DEFAULT_DECIDER_URL))
        next_ded_fb = True if raw_fb_key else (False if clear_fb_key else cur_fb)
        next_fb = next_ded_fb or (next_pri and store.is_ollama_endpoint(eff_url))

        problems = secret_problems + store.validate_decider(
            current,
            has_primary_key=next_pri,
            has_fallback_key=next_fb,
        )
        if problems:
            raise HTTPException(status_code=400, detail=" ".join(problems))

        try:
            if clear_api_key and not raw_api_key:
                store.clear_secret(store.PRIMARY_KEY_ENV)
            elif raw_api_key:
                store.save_secret(store.PRIMARY_KEY_ENV, raw_api_key)

            if clear_fb_key and not raw_fb_key:
                store.clear_secret(store.FALLBACK_KEY_ENV)
            elif raw_fb_key:
                store.save_secret(store.FALLBACK_KEY_ENV, raw_fb_key)

            store.save_decider(current)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        state = _build_state()
        _broadcast("updated", {"decider": state["decider"]})
        return state

    return await asyncio.to_thread(_run)


@router.post("/probe")
async def run_probe(payload: dict[str, Any] = Body(...)):
    goal = str((payload or {}).get("goal") or "").strip()
    context = str((payload or {}).get("context") or "").strip()
    has_images = bool((payload or {}).get("has_images", False))
    if not goal:
        raise HTTPException(status_code=400, detail="A non-empty subagent goal is required to run a probe.")

    def _run():
        return decider_router.probe_decider(goal, context=context, has_images=has_images)

    return await asyncio.to_thread(_run)


@router.delete("/decisions")
async def delete_decisions():
    def _run():
        cleared = decider_router.clear_decision_log()
        _broadcast("updated", {"decisions_cleared": cleared})
        return {"ok": True, "cleared": cleared}

    return await asyncio.to_thread(_run)
