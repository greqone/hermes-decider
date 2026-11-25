"""CLI (``hermes decider``) and slash (``/decider``) tests."""

from __future__ import annotations

import argparse

import decider_plugin.cli as cli_mod
import decider_plugin.decider_router as router_mod
import decider_plugin.slash as slash_mod
import decider_plugin.store as store_mod


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="hermes decider")
    cli_mod.setup_parser(parser)
    return parser.parse_args(argv)


def test_cli_show_and_flag_mutations(capsys, isolated_hermes):
    rc = cli_mod.handle(_parse(["--show"]))
    assert rc == 0
    out = capsys.readouterr().out
    assert "System-1 Decider auto-router: ON" in out
    assert "disabled (local-only)" in out

    rc2 = cli_mod.handle(
        _parse([
            "--url", "http://192.168.1.50:11434",
            "--threshold", "0.72",
            "--confidence", "0.64",
            "--mode", "decider",
        ])
    )
    assert rc2 == 0
    capsys.readouterr()
    saved = store_mod.decider_settings()
    assert saved["url"] == "http://192.168.1.50:11434/v1/systemone"
    assert saved["specialist_threshold"] == 0.72
    assert saved["confidence_threshold"] == 0.64
    assert store_mod.current_mode() == "decider"


def test_cli_rejects_enabling_fallback_without_url(capsys):
    rc = cli_mod.handle(_parse(["--allow-fallback"]))
    assert rc == 2
    err = capsys.readouterr().err
    assert "Fallback URL is required" in err


def test_slash_status_toggle_probe_history_and_mode(monkeypatch):
    status = slash_mod.handle("status")
    assert "System-1 Decider auto-router: ON" in status
    assert "off (local-only)" in status

    off_msg = slash_mod.handle("off")
    assert "OFF" in off_msg
    assert store_mod.decider_enabled() is False

    on_msg = slash_mod.handle("on")
    assert "ON" in on_msg
    assert store_mod.decider_enabled() is True

    monkeypatch.setattr(
        router_mod,
        "probe_decider",
        lambda goal, has_images=False, config=None: {
            "ok": True,
            "goal": goal,
            "upgraded": True,
            "effective_route": "VisualProwess",
            "candidate_route": "VisualProwess",
            "needs_specialist": 0.99,
            "confidence": 0.99,
            "latency_ms": 55.0,
            "model": "pplx-decider:27b-nvfp4",
            "fallback_used": False,
        },
    )
    probe_msg = slash_mod.handle("test Analyze screenshot.png")
    assert "VisualProwess" in probe_msg

    router_mod._record_decision({
        "action": "upgraded",
        "route": "VisualProwess",
        "needs_specialist": 0.99,
        "confidence": 0.99,
        "latency_ms": 55.0,
        "goal": "Analyze screenshot.png",
    })
    hist = slash_mod.handle("history")
    assert "VisualProwess" in hist

    clr = slash_mod.handle("clear")
    assert "Cleared 1" in clr

    mode_msg = slash_mod.handle("mode decider")
    assert "'decider'" in mode_msg
    assert store_mod.current_mode() == "decider"
