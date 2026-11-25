"""CLI & interactive terminal configurator for ``hermes decider``.

Supports non-interactive flags (``--show``, ``--enable``, ``--disable``, ``--url``, ``--model``,
``--allow-fallback`` / ``--no-fallback``, ``--fallback-url``, ``--fallback-model``,
``--threshold``, ``--confidence``, ``--timeout``, ``--exclude``, ``--mode``, ``--test``,
``--history``, ``--clear-history``) as well as an interactive menu using Hermes's public
``hermes_cli.setup`` helpers when invoked with no flags on a TTY.
"""

from __future__ import annotations

import sys
from typing import Any, Mapping

from . import decider_router, store


def setup_parser(parser) -> None:
    parser.add_argument("--show", action="store_true", help="Print current System-1 Decider settings and exit")
    parser.add_argument("--enable", action="store_true", help="Enable System-1 Decider auto-routing")
    parser.add_argument("--disable", action="store_true", help="Disable System-1 Decider auto-routing")
    parser.add_argument("--url", type=str, default=None, help="Primary /v1/systemone endpoint URL")
    parser.add_argument("--model", type=str, default=None, help="Primary decision model name")
    parser.add_argument("--fallback-url", type=str, default=None, help="Optional secondary /v1/systemone URL")
    parser.add_argument("--fallback-model", type=str, default=None, help="Secondary decision model name")
    parser.add_argument(
        "--allow-fallback",
        dest="allow_fallback",
        action="store_true",
        default=None,
        help="Opt in to secondary fallback when the primary endpoint fails",
    )
    parser.add_argument(
        "--no-fallback",
        dest="allow_fallback",
        action="store_false",
        help="Disable secondary fallback (local-only default)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Specialist probability threshold (0.0-1.0, default 0.70)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Route choice confidence threshold (0.0-1.0, default 0.60)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Request timeout in seconds (1.0-120.0, default 18.0)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Comma-separated route names excluded from automatic routing",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(store.MODES),
        default=None,
        help="Set delegation.backend mode (decider, model_map, profile_routes, default)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        metavar="GOAL",
        help="Run a live routing probe for GOAL against the configured Decider",
    )
    parser.add_argument(
        "--images",
        action="store_true",
        help="With --test: simulate image attachments on the task",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Print recent System-1 Decider routing decisions",
    )
    parser.add_argument(
        "--clear-history",
        action="store_true",
        help="Clear cached System-1 Decider routing decisions",
    )
    parser.set_defaults(func=handle)


def describe_status(cfg: Mapping[str, Any] | None = None) -> str:
    cfg_map = store.full_config(cfg)
    section = store.delegation_section(cfg_map)
    mode = store.current_mode(section)
    dec = store.decider_settings(cfg_map)
    options = store.configured_options(section)
    default_route = store.route_default(section)
    opt_names = list(options)
    excluded = set(dec["excluded_options"])
    eligible = [n for n in opt_names if n not in excluded]
    state_str = "ON" if dec["enabled"] else "OFF"
    fb_str = (
        f"{dec['fallback_model']} @ {dec['fallback_url']}"
        if dec["allow_fallback"] and dec["fallback_url"]
        else "disabled (local-only)"
    )
    lines = [
        f"System-1 Decider auto-router: {state_str}",
        f"  Delegation backend:   {mode} ({store.mode_label(mode)})",
        f"  Primary endpoint:     {dec['url']}",
        f"  Primary model:        {dec['model']}",
        f"  Secondary fallback:   {fb_str}",
        f"  Thresholds:           specialist >= {dec['specialist_threshold']:.2f}, confidence >= {dec['confidence_threshold']:.2f}",
        f"  Timeout:              {dec['timeout']:.1f}s",
        f"  Default route:        {default_route or '(inherit parent)'}",
        f"  Eligible routes ({len(eligible)}/{len(opt_names)}): {', '.join(eligible) if eligible else '(none)'}",
        f"  Excluded routes:      {', '.join(dec['excluded_options']) if dec['excluded_options'] else '(none)'}",
    ]
    return "\n".join(lines)


def _print_probe_result(res: Mapping[str, Any]) -> int:
    if not res.get("ok"):
        print(f"✗ Decider probe failed: {res.get('error')}", file=sys.stderr)
        return 1
    action = "UPGRADED to specialist" if res.get("upgraded") else "KEPT on default"
    fb = " (via secondary fallback)" if res.get("fallback_used") else ""
    print(f"✓ Decider probe succeeded in {res.get('latency_ms', 0):.0f} ms on {res.get('model')}{fb}:")
    print(f"  Goal:              {res.get('goal')}")
    print(f"  Effective route:   {res.get('effective_route')} ({action})")
    print(f"  Candidate route:   {res.get('candidate_route')} (confidence={res.get('confidence', 0):.2f})")
    print(f"  Needs specialist:  {res.get('needs_specialist', 0):.2f}")
    return 0


def handle(args) -> int:
    if getattr(args, "clear_history", False):
        removed = decider_router.clear_decision_log()
        print(f"Cleared {removed} cached Decider routing decision(s).")
        return 0

    if getattr(args, "history", False):
        rows = decider_router.read_decision_log(limit=25)
        if not rows:
            print("No System-1 Decider routing decisions recorded yet.")
            return 0
        print(f"Recent System-1 Decider routing decisions ({len(rows)}):")
        for r in rows:
            fb = " [fallback]" if r.get("fallback_used") else ""
            print(
                f"  • [{r.get('action', '-')}] -> {r.get('route', '-')} "
                f"(spec={r.get('needs_specialist', 0):.2f}, conf={r.get('confidence', 0):.2f}, "
                f"{r.get('latency_ms', 0):.0f}ms{fb}): {r.get('goal', '')}"
            )
        return 0

    cfg = store.full_config()
    current = dict(store.decider_settings(cfg))
    changed = False

    if getattr(args, "enable", False):
        current["enabled"] = True
        changed = True
    if getattr(args, "disable", False):
        current["enabled"] = False
        changed = True
    if getattr(args, "url", None) is not None:
        current["url"] = args.url.strip()
        changed = True
    if getattr(args, "model", None) is not None:
        current["model"] = args.model.strip()
        changed = True
    if getattr(args, "fallback_url", None) is not None:
        current["fallback_url"] = args.fallback_url.strip()
        changed = True
    if getattr(args, "fallback_model", None) is not None:
        current["fallback_model"] = args.fallback_model.strip()
        changed = True
    if getattr(args, "allow_fallback", None) is not None:
        current["allow_fallback"] = bool(args.allow_fallback)
        changed = True
    if getattr(args, "threshold", None) is not None:
        current["specialist_threshold"] = args.threshold
        changed = True
    if getattr(args, "confidence", None) is not None:
        current["confidence_threshold"] = args.confidence
        changed = True
    if getattr(args, "timeout", None) is not None:
        current["timeout"] = args.timeout
        changed = True
    if getattr(args, "exclude", None) is not None:
        current["excluded_options"] = [x.strip() for x in args.exclude.split(",") if x.strip()]
        changed = True

    if changed:
        problems = store.validate_decider(current)
        if problems:
            print("✗ Invalid Decider configuration:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 2
        try:
            store.save_decider(current)
        except RuntimeError as exc:
            print(f"✗ {exc}", file=sys.stderr)
            return 1

    target_mode = getattr(args, "mode", None)
    if target_mode:
        try:
            store.set_mode(target_mode)
        except (ValueError, RuntimeError) as exc:
            print(f"✗ {exc}", file=sys.stderr)
            return 1

    if getattr(args, "test", None) is not None:
        res = decider_router.probe_decider(
            args.test.strip(),
            has_images=bool(getattr(args, "images", False)),
        )
        return _print_probe_result(res)

    if getattr(args, "show", False) or changed or target_mode or not sys.stdin.isatty():
        print(describe_status())
        return 0

    return _interactive_menu()


def _interactive_menu() -> int:
    """Interactive curses/numbered configurator using Hermes's public setup helpers."""
    try:
        from hermes_cli.setup import prompt, prompt_choice
    except ImportError:
        print(describe_status())
        return 0

    while True:
        cfg = store.full_config()
        dec = dict(store.decider_settings(cfg))
        mode = store.current_mode(cfg)
        fb_status = (
            f"ON ({dec['fallback_model']} @ {dec['fallback_url']})"
            if dec["allow_fallback"] and dec["fallback_url"]
            else "OFF (local-only)"
        )
        choices = [
            f"Auto-routing:           {'ON' if dec['enabled'] else 'OFF'}",
            f"Delegation backend:     {mode} ({store.mode_label(mode)})",
            f"Primary endpoint & mdl: {dec['model']} @ {dec['url']}",
            f"Secondary fallback:     {fb_status}",
            f"Thresholds & timeout:   specialist>={dec['specialist_threshold']:.2f}, conf>={dec['confidence_threshold']:.2f}, {dec['timeout']:.0f}s",
            f"Excluded routes:        {', '.join(dec['excluded_options']) if dec['excluded_options'] else '(none)'}",
            "Run live /v1/systemone probe",
            "Done",
        ]
        idx = prompt_choice(
            "System-1 Decider (/v1/systemone) auto-router",
            choices,
            default=0,
            description="  Evaluates unassigned subagent tasks in one /v1/systemone forward pass.\n",
        )
        if idx < 0 or idx == 7:
            return 0
        if idx == 0:
            dec["enabled"] = not dec["enabled"]
            store.save_decider(dec)
        elif idx == 1:
            modes = list(store.MODES)
            cur = modes.index(mode) if mode in modes else 0
            pick = prompt_choice(
                "Select delegation.backend mode:",
                [f"{m} — {store.MODES[m]}" for m in modes],
                default=cur,
            )
            if 0 <= pick < len(modes):
                store.set_mode(modes[pick])
        elif idx == 2:
            dec["url"] = prompt("Primary /v1/systemone URL", default=dec["url"]).strip() or dec["url"]
            dec["model"] = prompt("Primary decision model", default=dec["model"]).strip() or dec["model"]
            problems = store.validate_decider(dec)
            if problems:
                print("✗ " + " ".join(problems))
            else:
                store.save_decider(dec)
        elif idx == 3:
            toggle = prompt_choice(
                "Secondary fallback endpoint (sends task text to fallback URL on primary failure):",
                ["Disabled (local-only, recommended)", "Enabled"],
                default=1 if dec["allow_fallback"] else 0,
            )
            if toggle == 1:
                dec["allow_fallback"] = True
                dec["fallback_url"] = prompt("Fallback /v1/systemone URL", default=dec["fallback_url"]).strip()
                dec["fallback_model"] = (
                    prompt("Fallback model", default=dec["fallback_model"]).strip() or dec["fallback_model"]
                )
            elif toggle == 0:
                dec["allow_fallback"] = False
            problems = store.validate_decider(dec)
            if problems:
                print("✗ " + " ".join(problems))
            else:
                store.save_decider(dec)
        elif idx == 4:
            raw_s = prompt("Specialist threshold (0.0-1.0)", default=str(dec["specialist_threshold"])).replace(",", ".")
            raw_c = prompt("Confidence threshold (0.0-1.0)", default=str(dec["confidence_threshold"])).replace(",", ".")
            raw_t = prompt("Timeout seconds (1-120)", default=str(dec["timeout"])).replace(",", ".")
            cand = {**dec, "specialist_threshold": raw_s, "confidence_threshold": raw_c, "timeout": raw_t}
            problems = store.validate_decider(cand)
            if problems:
                print("✗ " + " ".join(problems))
            else:
                store.save_decider(cand)
        elif idx == 5:
            raw_ex = prompt(
                "Comma-separated excluded route names",
                default=", ".join(dec["excluded_options"]),
            )
            dec["excluded_options"] = [x.strip() for x in raw_ex.split(",") if x.strip()]
            store.save_decider(dec)
        elif idx == 6:
            goal = prompt(
                "Sample subagent goal",
                default="Analyze screenshot.png and list UI layout bugs",
            ).strip()
            if goal:
                _print_probe_result(
                    decider_router.probe_decider(goal, has_images=("screenshot" in goal.lower()))
                )
