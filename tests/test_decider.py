"""Decider backend behaviour: route selection, degradation, and where the key may go."""

from __future__ import annotations

import json
import urllib.request

import pytest

from decider_plugin import decider as decider_mod
from decider_plugin.decider import (
    API_KEY_ENV,
    DeciderDelegationProvider,
    decide,
)

_ROUTES = [
    {"name": "fast", "description": "cheap and quick", "provider": "openai", "model": "small"},
    {"name": "deep", "description": "slow and thorough", "provider": "anthropic", "model": "large"},
]


def _settings(**overrides):
    base = {"endpoint": "http://127.0.0.1:11434/v1/systemone", "routes": _ROUTES, "default": "fast"}
    base.update(overrides)
    return base


def _config(settings):
    return {"backends": {"decider": settings}}


def _answer(name):
    return json.dumps({"answers": {"route": {"choice": name}}}).encode("utf-8")


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(payload):
    def urlopen(request, timeout):
        urlopen.last_request = request
        urlopen.last_timeout = timeout
        return _Response(payload)

    return urlopen


def test_chosen_route_is_returned_whole():
    provider = DeciderDelegationProvider()
    captured = _fake_urlopen(_answer("deep"))
    urllib.request.urlopen = captured
    try:
        route = provider.route_task(
            {"goal": "rewrite the parser"}, _config(_settings())
        )
    finally:
        del urllib.request.urlopen
    assert route["provider"] == "anthropic"
    assert route["model"] == "large"
    assert route["label"] == "deep"


def test_goal_and_route_names_are_what_gets_sent():
    captured = _fake_urlopen(_answer("fast"))
    urllib.request.urlopen = captured
    try:
        decide("rewrite the parser", "it is slow", {"fast": {}, "deep": {}}, _settings())
    finally:
        del urllib.request.urlopen
    sent = json.loads(captured.last_request.data.decode("utf-8"))
    assert sent["state"]["goal"] == "rewrite the parser"
    assert sent["questions"]["route"]["choices"] == ["fast", "deep"]


def test_endpoint_down_uses_the_default_route():
    provider = DeciderDelegationProvider()

    def boom(request, timeout):
        raise urllib.error.URLError("connection refused")

    urllib.request.urlopen = boom
    try:
        route = provider.route_task({"goal": "rewrite the parser"}, _config(_settings()))
    finally:
        del urllib.request.urlopen
    assert route["label"] == "fast"
    assert route["model"] == "small"


def test_garbage_answer_uses_the_default_route():
    provider = DeciderDelegationProvider()
    urllib.request.urlopen = _fake_urlopen(_answer("nonexistent"))
    try:
        route = provider.route_task({"goal": "rewrite the parser"}, _config(_settings()))
    finally:
        del urllib.request.urlopen
    assert route["label"] == "fast"


def test_no_default_and_no_decision_inherits_parent_route():
    provider = DeciderDelegationProvider()
    urllib.request.urlopen = _fake_urlopen(b"not json at all")
    try:
        route = provider.route_task(
            {"goal": "rewrite the parser"}, _config(_settings(default=""))
        )
    finally:
        del urllib.request.urlopen
    assert route == {}


def test_no_routes_refuses_the_task():
    provider = DeciderDelegationProvider()
    with pytest.raises(Exception, match="No routes configured"):
        provider.route_task({"goal": "rewrite the parser"}, _config({"routes": []}))


def _request(url):
    return urllib.request.Request(url, data=b"{}", method="POST")


def test_key_is_sent_to_the_configured_host(monkeypatch):
    monkeypatch.setattr(decider_mod, "get_secret", lambda name, default=None: "sekret-123")
    request = _request("http://127.0.0.1:11434/v1/systemone")
    decider_mod._attach_key(request, "http://127.0.0.1:11434/v1/systemone")
    assert request.get_header("Authorization") == "Bearer sekret-123"


def test_key_is_not_sent_to_a_different_host(monkeypatch):
    monkeypatch.setattr(decider_mod, "get_secret", lambda name, default=None: "sekret-123")
    request = _request("http://evil.example/v1/systemone")
    decider_mod._attach_key(request, "http://127.0.0.1:11434/v1/systemone")
    assert request.get_header("Authorization") is None


def test_empty_key_sends_no_authorization():
    request = _request("http://127.0.0.1:11434/v1/systemone")
    decider_mod._attach_key(request, "http://127.0.0.1:11434/v1/systemone")
    assert request.get_header("Authorization") is None


def test_no_per_task_parameter_is_declared():
    provider = DeciderDelegationProvider()
    assert provider.task_parameters({}) == {}


def test_malformed_routes_are_skipped():
    from decider_plugin.decider import configured_routes

    routes = configured_routes({
        "routes": [
            "nope",
            {"description": "nameless"},
            {"name": "ok", "provider": "openai", "model": "m", "junk": "dropped"},
            {"name": "ok", "provider": "other"},
        ]
    })
    assert list(routes) == ["ok"]
    assert routes["ok"] == {"provider": "openai", "model": "m"}
    assert "junk" not in routes["ok"]
