"""Unit & security tests for ``decider_router``, ``provider``, ``store``, and ``__init__``."""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

import pytest

import decider_plugin as pkg
import decider_plugin.decider_router as router_mod
from decider_plugin.provider import DeciderDelegationProvider
import decider_plugin.store as store_mod

_ROOT = Path(__file__).resolve().parents[1]


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]):
        self._bytes = json.dumps(payload).encode("utf-8")

    def read(self, amt: int | None = None) -> bytes:
        return self._bytes if amt is None else self._bytes[:amt]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_security_no_os_environ_or_dotenv_reads_anywhere():
    """Verify none of the plugin source files read os.environ or parse .env files."""
    py_files = [
        p for p in _ROOT.rglob("*.py")
        if "tests" not in p.parts and ".git" not in p.parts
    ]
    assert py_files, "Expected plugin Python files"
    for path in py_files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "environ":
                raise AssertionError(f"Forbidden os.environ access in {path}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("getenv",):
                    raise AssertionError(f"Forbidden os.getenv call in {path}")
        assert "api.typesafe.ai" not in source, f"Hardcoded remote fallback host found in {path}"
        assert '".env"' not in source and "'.env'" not in source, f"Direct .env reference in {path}"


def test_default_settings_keep_fallback_disabled():
    dec = store_mod.normalize_decider({})
    assert dec["enabled"] is True
    assert dec["allow_fallback"] is False
    assert dec["fallback_url"] == ""
    assert "UltraFastPaid" in dec["excluded_options"]


def test_batch_routing_upgrades_specialist_and_keeps_default_without_extra_task_keys(
    monkeypatch, isolated_hermes
):
    captured_requests: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def fake_open(req: urllib.request.Request, timeout: float = 18.0):
        body = json.loads(req.data.decode("utf-8"))
        headers = {k.lower(): v for k, v in req.header_items()}
        captured_requests.append((req.full_url, body, headers))
        return _FakeHTTPResponse({
            "model": "pplx-decider:27b-nvfp4",
            "answers": {
                "t0_needs_specialist": {"noul": 0.12},
                "t0_route": {"choice": "FastSub", "confidence": 0.91},
                "t1_needs_specialist": {"noul": 0.96},
                "t1_route": {"choice": "FrontierOpenAI", "confidence": 0.88},
            },
        })

    monkeypatch.setattr(router_mod, "_open_url", fake_open)

    tasks = [
        {"goal": "Write unit tests for string formatting helpers"},
        {"goal": "Perform adversarial correctness verification of the consensus state machine"},
        {"goal": "Already pinned task", "option": "VisualProwess"},
    ]
    assigned = router_mod.route_unassigned_tasks(tasks)

    assert assigned == {0: "FastSub", 1: "FrontierOpenAI"}
    assert tasks[0] == {"goal": "Write unit tests for string formatting helpers", "option": "FastSub"}
    assert tasks[1] == {
        "goal": "Perform adversarial correctness verification of the consensus state machine",
        "option": "FrontierOpenAI",
    }
    assert tasks[2] == {"goal": "Already pinned task", "option": "VisualProwess"}
    # Crucial: no _decider_evaluated or any undeclared property may touch task dicts.
    for t in tasks:
        assert set(t).issubset({"goal", "context", "images", "option"})

    assert len(captured_requests) == 1
    _, body, headers = captured_requests[0]
    assert body["keep_alive"] == -1
    assert isinstance(body["state"], dict)
    assert "available_routes" in body["state"]
    assert "UltraFastPaid" not in body["state"]["available_routes"]
    assert "authorization" not in headers

    log_rows = router_mod.read_decision_log()
    assert len(log_rows) == 2
    assert {r["route"] for r in log_rows} == {"FastSub", "FrontierOpenAI"}


def test_visual_fast_path_upgrades_image_task_even_below_specialist_threshold(monkeypatch):
    def fake_open(req: urllib.request.Request, timeout: float = 18.0):
        return _FakeHTTPResponse({
            "model": "pplx-decider:27b-nvfp4",
            "answers": {
                "t0_needs_specialist": {"noul": 0.42},
                "t0_route": {"choice": "VisualProwess", "confidence": 0.55},
            },
        })

    monkeypatch.setattr(router_mod, "_open_url", fake_open)
    tasks = [{"goal": "Inspect screenshot", "images": ["/tmp/ui.png"]}]
    assigned = router_mod.route_unassigned_tasks(tasks)
    assert assigned == {0: "VisualProwess"}
    assert tasks[0]["option"] == "VisualProwess"


def test_primary_failure_does_not_call_any_fallback_when_disabled(monkeypatch, isolated_hermes):
    called_urls: list[str] = []

    def fake_open(req: urllib.request.Request, timeout: float = 18.0):
        called_urls.append(req.full_url)
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(router_mod, "_open_url", fake_open)
    isolated_hermes["secrets"]["DECIDER_API_KEY"] = "sk-secret"
    # Even if fallback_url is lingered in config, allow_fallback=False blocks it.
    isolated_hermes["config"]["plugins"]["entries"]["decider"]["settings"].update({
        "allow_fallback": False,
        "fallback_url": "https://decider.example.com/v1/systemone",
    })

    tasks = [{"goal": "Refactor parser module"}]
    assigned = router_mod.route_unassigned_tasks(tasks)
    assert assigned == {}
    assert "option" not in tasks[0]
    assert called_urls == ["http://127.0.0.1:11434/v1/systemone"]


def test_opt_in_fallback_attaches_key_only_to_configured_nonlocal_host(monkeypatch, isolated_hermes):
    calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def fake_open(req: urllib.request.Request, timeout: float = 18.0):
        headers = {k.lower(): v for k, v in req.header_items()}
        body = json.loads(req.data.decode("utf-8"))
        calls.append((req.full_url, headers, body))
        if "127.0.0.1" in req.full_url:
            raise urllib.error.URLError("local ollama asleep")
        return _FakeHTTPResponse({
            "model": "jev-latest",
            "decisions": {
                "t0_needs_specialist": {"yes_probability": 0.93},
                "t0_route": {"choice": "FrontierOpenAI", "confidence": 0.89},
            },
        })

    monkeypatch.setattr(router_mod, "_open_url", fake_open)
    isolated_hermes["secrets"]["DECIDER_API_KEY"] = "decider-token-xyz"
    isolated_hermes["config"]["plugins"]["entries"]["decider"]["settings"].update({
        "allow_fallback": True,
        "fallback_url": "https://decider.example.com/v1/systemone",
        "fallback_model": "jev-latest",
    })

    tasks = [{"goal": "Audit cryptographic state transitions across modules"}]
    assigned = router_mod.route_unassigned_tasks(tasks)
    assert assigned == {0: "FrontierOpenAI"}
    assert len(calls) == 2
    assert "authorization" not in calls[0][1]
    assert calls[1][1].get("authorization") == "Bearer decider-token-xyz"
    # Non-Ollama remote /v1/systemone receives JSON-encoded state string + criterion on noul
    assert isinstance(calls[1][2]["state"], str)
    assert "criterion" in calls[1][2]["questions"]["t0_needs_specialist"]


def test_bearer_key_never_attached_when_host_mismatches_or_plain_http():
    req = urllib.request.Request("https://evil.example.net/v1/systemone", method="POST")
    router_mod.attach_bearer_if_allowed(
        req,
        configured_url="https://decider.example.com/v1/systemone",
        api_key="super-secret",
    )
    assert req.get_header("Authorization") is None

    # Cleartext http:// off loopback must NEVER carry Authorization header
    cleartext_req = urllib.request.Request("http://decider.example.com/v1/systemone", method="POST")
    router_mod.attach_bearer_if_allowed(
        cleartext_req,
        configured_url="http://decider.example.com/v1/systemone",
        api_key="super-secret",
    )
    assert cleartext_req.get_header("Authorization") is None


def test_distinct_remote_primary_and_fallback_keys_never_cross_contaminate(monkeypatch, isolated_hermes):
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_open(req: urllib.request.Request, timeout: float = 18.0):
        headers = {k.lower(): v for k, v in req.header_items()}
        calls.append((req.full_url, headers))
        if "primary.example.com" in req.full_url:
            raise urllib.error.URLError("primary down")
        return _FakeHTTPResponse({
            "model": "jev-latest",
            "decisions": {
                "t0_needs_specialist": {"yes_probability": 0.91},
                "t0_route": {"choice": "FrontierOpenAI", "confidence": 0.87},
            },
        })

    monkeypatch.setattr(router_mod, "_open_url", fake_open)
    isolated_hermes["secrets"]["DECIDER_API_KEY"] = "pri-token-111"
    isolated_hermes["config"]["plugins"]["entries"]["decider"]["settings"].update({
        "url": "https://primary.example.com/v1/systemone",
        "allow_fallback": True,
        "fallback_url": "https://fallback.example.com/v1/systemone",
        "fallback_model": "jev-latest",
    })

    # Without DECIDER_FALLBACK_API_KEY, primary's remote key MUST NOT be sent to fallback.example.com!
    tasks = [{"goal": "Audit cryptographic state transitions"}]
    assigned = router_mod.route_unassigned_tasks(tasks)
    assert assigned == {}
    assert len(calls) == 1
    assert calls[0][1].get("authorization") == "Bearer pri-token-111"

    # Now set dedicated DECIDER_FALLBACK_API_KEY: each host gets ONLY its own key!
    calls.clear()
    router_mod.reset_runtime_caches()
    isolated_hermes["secrets"]["DECIDER_FALLBACK_API_KEY"] = "fb-token-222"
    tasks2 = [{"goal": "Audit cryptographic state transitions"}]
    assigned2 = router_mod.route_unassigned_tasks(tasks2)
    assert assigned2 == {0: "FrontierOpenAI"}
    assert len(calls) == 2
    assert calls[0][1].get("authorization") == "Bearer pri-token-111"
    assert calls[1][1].get("authorization") == "Bearer fb-token-222"


def test_unscoped_secret_error_propagates_fail_closed(monkeypatch):
    def _raise_unscoped(name: str, default: str = "") -> str:
        raise store_mod.UnscopedSecretError("secret_scope is not bound")

    monkeypatch.setattr(store_mod, "get_secret", _raise_unscoped)
    with pytest.raises(store_mod.UnscopedSecretError):
        router_mod.route_unassigned_tasks([{"goal": "Test unscoped"}])


def test_validation_rejects_nan_embedded_credentials_and_cleartext_remote_keys(isolated_hermes):
    assert store_mod.validate_decider({"specialist_threshold": float("nan")})
    assert store_mod.validate_decider({"timeout": float("inf")})
    assert store_mod.validate_decider({"url": "https://user:secret@decider.example.com/v1/systemone"})
    assert store_mod.display_url("https://user:secret@decider.example.com:8443/v1/systemone") == (
        "https://decider.example.com:8443/v1/systemone"
    )
    assert store_mod.validate_decider(
        {"allow_fallback": True, "fallback_url": "http://decider.example.com/v1/systemone"},
        has_fallback_key=True,
    )
    # Saving legacy `model` key migrates cleanly to `primary_model` with no reserved `model` key left
    store_mod.save_decider({"model": "custom-decider:9b"})
    raw_saved = isolated_hermes["config"]["plugins"]["entries"]["decider"]["settings"]
    assert "model" not in raw_saved
    assert raw_saved["primary_model"] == "custom-decider:9b"


def test_concurrent_decision_logging_is_thread_safe():
    import concurrent.futures

    def _writer(i: int) -> None:
        router_mod._record_decision({"goal": f"task-{i}", "route": "FastSub", "action": "kept_default"})

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_writer, range(40)))

    rows = router_mod.read_decision_log(limit=100)
    assert len(rows) == 40


def test_delegation_provider_routes_both_explicit_and_unassigned_tasks(monkeypatch, isolated_hermes):
    isolated_hermes["config"]["delegation"]["backend"] = "decider"

    def fake_open(req: urllib.request.Request, timeout: float = 18.0):
        return _FakeHTTPResponse({
            "model": "pplx-decider:27b-nvfp4",
            "answers": {
                "t0_needs_specialist": {"noul": 0.95},
                "t0_route": {"choice": "FrontierOpenAI", "confidence": 0.91},
            },
        })

    monkeypatch.setattr(router_mod, "_open_url", fake_open)

    prov = DeciderDelegationProvider()
    section = isolated_hermes["config"]["delegation"]
    params = prov.task_parameters(section)
    assert "option" in params
    assert params["option"]["enum"] == ["FastSub", "VisualProwess", "FrontierOpenAI", "UltraFastPaid"]
    assert "System-1 Decider" in prov.description(section)

    explicit = prov.route_task({"goal": "Quick run", "option": "UltraFastPaid"}, section)
    assert explicit["model"] == "deepseek-v4.1-flash"
    assert explicit["label"] == "UltraFastPaid"

    from agent.delegation_provider import DelegationRouteError
    with pytest.raises(DelegationRouteError):
        prov.route_task({"goal": "Removed route", "option": "NonExistentRoute"}, section)

    auto = prov.route_task({"goal": "Prove deadlock freedom of lock-free ring buffer"}, section)
    assert auto["model"] == "gpt-5.4"
    assert auto["label"] == "FrontierOpenAI"


def test_pre_tool_call_hook_modifies_spawn_and_avoids_double_query_in_provider(
    monkeypatch, isolated_hermes
):
    isolated_hermes["config"]["delegation"]["backend"] = "decider"
    # Unset default route to verify even "inherit" decisions are cached across hook -> provider!
    isolated_hermes["config"]["delegation"]["backends"]["model_map"]["default"] = ""
    http_calls = 0

    def fake_open(req: urllib.request.Request, timeout: float = 18.0):
        nonlocal http_calls
        http_calls += 1
        return _FakeHTTPResponse({
            "model": "pplx-decider:27b-nvfp4",
            "answers": {
                "t0_needs_specialist": {"noul": 0.08},
                "t0_route": {"choice": "FastSub", "confidence": 0.80},
            },
        })

    monkeypatch.setattr(router_mod, "_open_url", fake_open)

    hook_res = pkg._on_pre_tool_call(
        tool_name="delegate_task",
        args={"action": "spawn", "tasks": [{"goal": "List files in src/"}]},
    )
    # With no default route and needs_specialist=0.08, task stays on inherit (no modification).
    assert hook_res is None
    assert http_calls == 1

    # Immediately afterwards, Hermes calls DeciderDelegationProvider.route_task: cache hit -> 0 new HTTP calls!
    prov = DeciderDelegationProvider()
    route = prov.route_task({"goal": "List files in src/"}, isolated_hermes["config"]["delegation"])
    assert route == {}
    assert http_calls == 1
