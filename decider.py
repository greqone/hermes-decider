"""Decider: a delegation backend that picks a configured route per task.

``delegation.backend: decider``. For each ``delegate_task`` task the plugin asks one decision
endpoint which of the routes in ``delegation.backends.decider.routes`` should run it, then returns
that route. The decision model only ever answers with a route *name*; this module turns the name
into the route core actually runs.

It fails closed and it fails local. One endpoint, no second host. If the endpoint is unset, down,
slow, or answers with anything that is not one of the configured route names, the task takes the
route marked ``default`` (or the parent's own route, when no default is set). Task text is sent
only to the endpoint named in config.

The only credential is ``DECIDER_API_KEY``, declared as a ``secret`` in ``plugin.yaml`` and read
through :func:`agent.secret_scope.get_secret`. Nothing here touches ``os.environ`` or reads ``.env``
by hand. The key is attached only when the request host is exactly the host configured in
``endpoint``; a redirect to any other host is dropped rather than followed with the key.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any, Mapping
from urllib.parse import urlsplit

from agent.delegation_provider import (
    ROUTE_KEYS,
    DelegationProvider,
    DelegationRouteError,
)
from agent.secret_scope import get_secret

logger = logging.getLogger(__name__)

API_KEY_ENV = "DECIDER_API_KEY"
SETTINGS_PATH = "delegation.backends.decider"

_DEFAULT_ENDPOINT = "http://127.0.0.1:11434/v1/systemone"
_DEFAULT_MODEL = "pplx-decider:27b-nvfp4"
_DEFAULT_TIMEOUT_S = 18.0

# Route fields copied verbatim from a configured entry. Anything else the entry carries (a
# description, a typo) is not a route and is dropped. Kept inside ROUTE_KEYS so core honours them.
_ROUTE_FIELDS = tuple(sorted(ROUTE_KEYS - {"label"}))


def _host(url: str) -> str:
    """Lowercased host of ``url``, or ``""`` when it has none."""
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _text(entry: Mapping[str, Any], key: str) -> str:
    value = entry.get(key)
    return value.strip() if isinstance(value, str) else ""


class DeciderDelegationProvider(DelegationProvider):
    """``delegation.backend: decider``: one local decision model chooses a named route per task."""

    @property
    def name(self) -> str:
        return "decider"

    @property
    def summary(self) -> str:
        return "A local decision model picks one of your configured routes for each subagent."

    def is_available(self) -> bool:
        # No network here: a down endpoint degrades to the default route at decision time, it does
        # not make the backend uninstallable.
        return True

    def task_parameters(self, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        # The decision is the backend's, not the calling model's. No per-task parameter: adding one
        # would put the choice back in the model's hands, which is exactly what this backend is for.
        return {}

    def description(self, config: Mapping[str, Any]) -> str:
        names = ", ".join(configured_routes(self.settings(config))) or "none configured"
        return (
            "Each subagent is routed by a local decision model to one of the configured routes "
            f"({names}). You do not choose the route."
        )

    def route_task(self, task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        settings = self.settings(config)
        routes = configured_routes(settings)
        if not routes:
            raise DelegationRouteError(
                f"No routes configured under {SETTINGS_PATH}.routes. Add at least one route, "
                "or switch delegation.backend away from decider."
            )
        goal = _text(task, "goal")
        if not goal:
            raise DelegationRouteError("Task has no goal, so it cannot be routed.")

        chosen = decide(goal, _text(task, "context"), routes, settings)
        if chosen not in routes:
            chosen = default_route_name(settings, routes)
            if chosen is None:
                # Nothing was chosen and nothing is marked default: inherit the parent's route
                # rather than guess. An empty route means exactly that to core.
                return {}
        route = routes[chosen]
        return {**route, "label": chosen}


def configured_routes(settings: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """``{name: route}`` from ``settings['routes']``, in order, skipping malformed entries.

    A route keeps only the fields core understands (:data:`ROUTE_KEYS` minus ``label``). An entry
    that is not a mapping, has no name, or repeats a name is skipped with one warning.
    """
    raw = settings.get("routes")
    if not isinstance(raw, list):
        return {}
    routes: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            logger.warning("%s.routes[%d] is not a mapping; skipped.", SETTINGS_PATH, index)
            continue
        name = _text(entry, "name")
        if not name:
            logger.warning("%s.routes[%d] has no name; skipped.", SETTINGS_PATH, index)
            continue
        if name in routes:
            logger.warning("%s.routes[%d] repeats route %r; skipped.", SETTINGS_PATH, index, name)
            continue
        route = {
            key: entry[key].strip()
            for key in _ROUTE_FIELDS
            if isinstance(entry.get(key), str) and entry[key].strip()
        }
        fallbacks = entry.get("fallback_providers")
        if isinstance(fallbacks, list):
            route["fallback_providers"] = [str(item) for item in fallbacks]
        routes[name] = route
    return routes


def default_route_name(settings: Mapping[str, Any], routes: Mapping[str, Any]) -> str | None:
    """The route a task takes when no decision is available, or ``None`` to inherit the parent."""
    name = _text(settings, "default")
    if name and name in routes:
        return name
    if name:
        logger.warning("%s.default %r is not one of the configured routes; ignored.", SETTINGS_PATH, name)
    return None


def _endpoint(settings: Mapping[str, Any]) -> str:
    return _text(settings, "endpoint") or _DEFAULT_ENDPOINT


def _timeout(settings: Mapping[str, Any]) -> float:
    value = settings.get("timeout_s")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return _DEFAULT_TIMEOUT_S



def _attach_key(request: urllib.request.Request, endpoint: str) -> None:
    """Add the bearer key, but only when the request targets the configured host.

    ``request.full_url`` is the address actually dialed. If it is not exactly the host named in
    ``endpoint``, the key stays home and a warning is logged. An empty key adds nothing.
    """
    api_key = (get_secret(API_KEY_ENV, "") or "").strip()
    if not api_key:
        return
    if _host(request.full_url) != _host(endpoint):
        logger.warning("decider: refusing to send %s to a host other than the configured endpoint.", API_KEY_ENV)
        return
    request.add_header("Authorization", f"Bearer {api_key}")


def decide(
    goal: str,
    context: str,
    routes: Mapping[str, Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> str | None:
    """Ask the endpoint which route should run this task.

    Returns a route name, or ``None`` when the endpoint cannot be reached or does not answer with
    one of ``routes``. Never raises: a failed decision is the caller's cue to use the default.
    """
    endpoint = _endpoint(settings)
    names = list(routes)
    state = {"goal": goal}
    if context:
        state["context"] = context
    body = {
        "model": _text(settings, "model") or _DEFAULT_MODEL,
        "state": state,
        "questions": {
            "route": {
                "type": "choice",
                "prompt": "Which route should run this task? Answer with exactly one of the names.",
                "choices": names,
            }
        },
        "keep_alive": -1,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "hermes-decider/1.0",
        },
        method="POST",
    )
    _attach_key(request, endpoint)

    try:
        with urllib.request.urlopen(request, timeout=_timeout(settings)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # endpoint down, slow, or not JSON: degrade, don't fail the task
        logger.warning("decider: endpoint %s did not answer (%s); using the default route.", endpoint, exc)
        return None

    return _route_name(payload, names)


def _route_name(payload: Any, names: list[str]) -> str | None:
    """Pull the chosen route name out of a ``/v1/systemone`` response, if it is one of ``names``."""
    if not isinstance(payload, Mapping):
        return None
    answers = payload.get("answers")
    if not isinstance(answers, Mapping):
        answers = payload.get("decisions")
    if not isinstance(answers, Mapping):
        return None
    entry = answers.get("route")
    if isinstance(entry, str):
        chosen = entry.strip()
    elif isinstance(entry, Mapping):
        chosen = next(
            (str(entry[key]).strip() for key in ("choice", "value", "answer") if isinstance(entry.get(key), str)),
            "",
        )
    else:
        chosen = ""
    return chosen if chosen in names else None
