# decider

A Hermes delegation backend. Set `delegation.backend: decider` and, for each `delegate_task`, a
single local decision model picks which of your configured routes the subagent runs on.

It is deliberately small, and the constraints are the point:

- **One endpoint.** Configured under `plugins.entries.decider.settings.endpoint`. There is no
  fallback host. If the endpoint is down, slow, or answers with something that is not one of your
  route names, the task takes the route you marked `default`. Task text goes to that one host and
  nowhere else.
- **No environment variables.** The only credential is `DECIDER_API_KEY`, declared as a `secret` in
  `plugin.yaml`. Hermes stores it and the plugin reads it through `agent.secret_scope.get_secret`.
  The plugin never reads `.env` itself and never calls `os.environ`.
- **The key travels only to the host you named.** The bearer token is attached only when the
  request host is exactly the host in `endpoint`. Anything else and the key stays put.

## Configure

Routes live in `delegation.backends.decider`, the same place every delegation backend keeps its own
settings:

```yaml
delegation:
  backend: decider
  backends:
    decider:
      endpoint: http://127.0.0.1:11434/v1/systemone
      model: pplx-decider:27b-nvfp4
      timeout_s: 18
      default: fast
      routes:
        - name: fast
          description: cheap and quick
          provider: openai
          model: small-model
        - name: deep
          description: slow and thorough
          provider: anthropic
          model: large-model
```

The decision model is shown the route names and returns exactly one of them. It never sees provider
names or model ids, and it never chooses a route that is not in that list.

Requires Hermes `>=0.22.5`.
