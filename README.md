# System-1 Decider (`hermes-decider`)

System-1 Decider (`/v1/systemone`) auto-router for [Hermes Agent](https://hermes-agent.nousresearch.com/docs) subagent delegation.

When the parent agent spawns `delegate_task` items without specifying `option`, `hermes-decider` evaluates the unassigned batch in **one `/v1/systemone` forward pass**:

1. **`t{i}_needs_specialist` (`noul`)** — estimates whether the task genuinely requires a specialist/frontier route (visual/multimodal input, multi-file architecture, adversarial verification, offensive security, or deep synthesis) versus staying on your fast `default` route.
2. **`t{i}_route` (`choice`)** — selects the best-matching eligible route from your configured routes (automatically excluding paid-per-token and advisor-only routes unless toggled on).

Works both as a standalone `delegation.backend: decider` provider **and** alongside `delegation.backend: model_map` (reusing routes configured in **Settings ▸ Model ▸ Subagents** with zero duplication).

---

## Privacy & Security Guarantees

Built specifically as a self-contained catalog plugin with strict local-first defaults:

- **Local-only by default (`allow_fallback: false`, `fallback_url: ""`).** Task text goes only to your configured `/v1/systemone` endpoint (`http://127.0.0.1:11434/v1/systemone` by default) and never leaves your machine unless you explicitly opt in to a secondary fallback URL.
- **Zero environment variables & zero `.env` parsing.** The plugin never touches `os.environ` or reads `.env` files. Optional `DECIDER_API_KEY` is declared as `type: secret` in `plugin.yaml` and read exclusively via Hermes's `agent.secret_scope.get_secret`.
- **Host-locked credentials.** When `DECIDER_API_KEY` is set for an authenticated non-local endpoint, the `Authorization: Bearer` header is attached **only** when the request host matches your configured endpoint host, and requests run through `hermes_cli.urllib_security.open_credentialed_url` so cross-origin redirects strip credentials.
- **Graceful fallback.** If the local decision server is asleep or unreachable, tasks degrade cleanly to your configured `default` route without failing `delegate_task`.

---

## Install

```bash
hermes plugins install decider
```

Or directly from GitHub:

```bash
hermes plugins install https://github.com/greqone/hermes-decider
```

Requires Hermes `>=0.22.5`.

---

## Configure

### CLI & Interactive TUI

```bash
hermes decider                          # interactive TUI configurator
hermes decider --show                   # print current settings & eligible routes
hermes decider --url http://127.0.0.1:11434/v1/systemone --model pplx-decider:27b-nvfp4
hermes decider --test "Analyze screenshot.png and list layout bugs" --images
hermes decider --history                # view recent routing decisions
```

### Slash Command (in chat)

```text
/decider status
/decider on | off
/decider test Analyze screenshot.png and extract the stack trace
/decider history | clear
/decider mode decider --now
```

### Hermes Desktop

Open **Settings ▸ Plugins ▸ System-1 Decider** (or press <kbd>Ctrl/Cmd+K</kbd> → **System-1 Decider settings**) to configure the endpoint, thresholds, per-route eligibility switches, run a live probe, or inspect recent routing decisions.

### `config.yaml` Example

Routes can live in `delegation.backends.model_map.options` (managed by Hermes Desktop's **Model ▸ Subagents** screen) or directly under `delegation.backends.decider.options`:

```yaml
delegation:
  backend: decider   # or "model_map" — pre_tool_call batch routing works in both modes
  backends:
    decider:
      default: FastSub
      options:
        - name: FastSub
          description: Fast general-purpose worker for coding, tests, debugging, and research.
          provider: openai-codex
          model: gpt-5.4-mini
        - name: VisualProwess
          description: Multimodal worker for screenshots, UI layout, diagrams, and OCR.
          provider: solstice
          model: gemini-4-hydrogen
        - name: FrontierOpenAI
          description: Rigorous verification, adversarial review, and complex architecture.
          provider: openai-codex
          model: gpt-5.4

plugins:
  entries:
    decider:
      enabled: true
      settings:
        enabled: true
        url: http://127.0.0.1:11434/v1/systemone
        model: pplx-decider:27b-nvfp4
        allow_fallback: false
        specialist_threshold: 0.70
        confidence_threshold: 0.60
        timeout: 18.0
```

---

## License

MIT
