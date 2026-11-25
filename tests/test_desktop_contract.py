"""Runs the Node ESM contract suite against ``desktop/plugin.js``."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

_HARNESS = Path(__file__).resolve().with_name("desktop-contract.mjs")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize(
    "scenario",
    ["registration", "path-guard", "fallback-default-off", "happy-path", "field-width"],
)
def test_desktop_plugin_contract(scenario: str) -> None:
    proc = subprocess.run(
        ["node", str(_HARNESS), scenario],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
