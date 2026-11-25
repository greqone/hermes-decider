"""FastAPI contract tests for ``dashboard/plugin_api.py``."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

import decider_plugin.decider_router as router_mod
from hermes_cli import config as hermes_cfg

_API_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "plugin_api.py"


def _load_api_module():
    before_path = list(sys.path)
    before_keys = set(sys.modules)
    spec = importlib.util.spec_from_file_location("test_decider_dashboard_api", _API_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert list(sys.path) == before_path, "dashboard/plugin_api.py must not mutate sys.path"
    leaked = set(sys.modules) - before_keys
    assert "store" not in leaked and "decider_router" not in leaked
    return mod


def _make_client() -> tuple[TestClient, Any]:
    mod = _load_api_module()
    app = FastAPI()
    app.include_router(mod.router, prefix="/api/plugins/decider")
    return TestClient(app), mod


def test_get_state_returns_routes_settings_and_redacts_secrets(isolated_hermes):
    isolated_hermes["config"]["delegation"]["backends"]["model_map"]["options"][0]["api_key"] = "sk-do-not-leak"
    isolated_hermes["config"]["delegation"]["backends"]["model_map"]["options"][0]["base_url"] = "https://priv.internal/v1"
    client, _ = _make_client()

    res = client.get("/api/plugins/decider/state")
    assert res.status_code == 200
    body = res.json()
    assert body["mode"] == "model_map"
    assert body["route_default"] == "FastSub"
    assert body["decider"]["allow_fallback"] is False
    assert len(body["routes"]) == 4
    raw = res.text
    assert "sk-do-not-leak" not in raw
    assert "priv.internal" not in raw


def test_put_decider_validates_and_persists(isolated_hermes):
    client, _ = _make_client()

    bad = client.put("/api/plugins/decider/decider", json={"url": "ftp://bad-scheme"})
    assert bad.status_code == 400

    bad_fb = client.put("/api/plugins/decider/decider", json={"allow_fallback": True, "fallback_url": ""})
    assert bad_fb.status_code == 400

    ok = client.put(
        "/api/plugins/decider/decider",
        json={
            "enabled": True,
            "url": "http://192.168.1.50:11434",
            "model": "pplx-decider:27b-nvfp4",
            "specialist_threshold": 0.75,
            "confidence_threshold": 0.65,
            "timeout": 22.0,
        },
    )
    assert ok.status_code == 200
    saved = ok.json()["decider"]
    assert saved["url"] == "http://192.168.1.50:11434/v1/systemone"
    assert saved["specialist_threshold"] == 0.75
    assert saved["confidence_threshold"] == 0.65
    assert saved["timeout"] == 22.0


def test_put_decider_returns_409_when_config_is_managed(monkeypatch):
    def _refuse(_mutate):
        raise RuntimeError("Configuration is managed by system policy")

    monkeypatch.setattr(hermes_cfg, "update_user_config", _refuse, raising=False)
    client, _ = _make_client()
    res = client.put("/api/plugins/decider/decider", json={"enabled": False})
    assert res.status_code == 409


def test_post_probe_and_delete_decisions(monkeypatch):
    client, api_mod = _make_client()

    monkeypatch.setattr(
        api_mod.decider_router,
        "probe_decider",
        lambda goal, context="", has_images=False: {
            "ok": True,
            "goal": goal,
            "has_images": has_images,
            "upgraded": True,
            "effective_route": "VisualProwess",
            "candidate_route": "VisualProwess",
            "default_route": "FastSub",
            "needs_specialist": 0.98,
            "confidence": 0.97,
            "latency_ms": 42.0,
            "model": "pplx-decider:27b-nvfp4",
            "fallback_used": False,
        },
    )
    empty = client.post("/api/plugins/decider/probe", json={"goal": "   "})
    assert empty.status_code == 400

    probe = client.post("/api/plugins/decider/probe", json={"goal": "Inspect UI screenshot", "has_images": True})
    assert probe.status_code == 200
    assert probe.json()["effective_route"] == "VisualProwess"

    router_mod._record_decision({"goal": "sample", "route": "FastSub", "action": "kept_default"})
    cleared = client.delete("/api/plugins/decider/decisions")
    assert cleared.status_code == 200
    assert cleared.json() == {"ok": True, "cleared": 1}
