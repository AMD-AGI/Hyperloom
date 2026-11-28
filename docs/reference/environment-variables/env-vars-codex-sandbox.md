---
myst:
    html_meta:
        "description": "Codex agent sandbox variable for Hyperloom: HYPERLOOM_CODEX_SANDBOX_MODE selects workspace-write, read-only, or bypass isolation for Codex agent sessions and TraceLens analysis."
        "keywords": "Hyperloom, environment variables, Codex sandbox, HYPERLOOM_CODEX_SANDBOX_MODE, bubblewrap, agent isolation, workspace-write, bypass"
---
# Hyperloom Codex agent sandbox

Selects how a Codex agent session (TraceLens analysis and every future
Codex-based agent) is contained. The secure default is `workspace-write`.
Codex implements both contained presets with bubblewrap, so Hyperloom executes
a real namespace-and-mount capability probe before starting the SDK. Merely
finding a `bwrap` executable is insufficient: if the current kernel or
container prevents it from establishing the sandbox, `workspace-write` and
`read-only` fail closed before the app-server starts. Under `workspace-write`
the probe also writes to the session's working directory and each declared
writable root, because Codex creates its protected paths (`.agents`, `.codex`)
there; a directory whose owner is not mapped inside the sandbox's user
namespace, such as a checkout owned by another host user inside a root
container, fails the same way. There is no automatic
fallback to `bypass`.

`bypass` is an explicit operator opt-in. Set
`HYPERLOOM_CODEX_SANDBOX_MODE=bypass` when an external container or sandbox
already enforces the required isolation. Hyperloom does not create that
boundary. A confirmed bypass maps to Codex full access even when no writable
roots are declared, because the external sandbox is authoritative. Under the
contained modes, no writable roots remains `read-only`. Unknown modes fail
immediately.

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_`<br>`CODEX_SANDBOX_MODE` | `workspace-write` | `workspace-write` restricts writes to the session directory plus declared output roots; `read-only` forbids writes; `bypass` selects Codex full access when an external sandbox already enforces isolation. |
