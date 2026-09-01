---
myst:
  html_meta:
    "description": "KernelForge deployment modes: Claude Code in-session live-stream and the autonomous forge-loop."
    "keywords": "KernelForge, deployment modes, Claude Code, in-session, forge-loop, autonomous loop, billing"
---

# Deployment modes

KernelForge can run in two modes that differ in where the agent runs, how much
human oversight the run gets, and how the run is billed.

## Claude Code — native in-session live-stream (default)

Launch from inside a Claude Code session and watch every state transition, tool
call, and decision land in chat as it happens. Everything routes through the
`claude` CLI subprocess and bills to your Claude Code Max subscription.

That billing depends on the `claude` in the container being authenticated, and
the container has no `~/.claude/.credentials.json` of its own — its `claude` is a
fresh install and your host's home is not mounted in. Either mount that file or
pass `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`. An `ANTHROPIC_API_KEY`
left in the environment moves the run to API credits, since the CLI reads it
ahead of the subscription token.

## Autonomous loop — overnight optimization

```bash
kernelforge forge-loop --workspace <W> \
    --kernel <f> --driver <f> --snr-threshold 30 --max-hours 8
```

Runs unattended with the driver-owned complete correctness suite, three
independent benchmarks, and automatic git keep/revert. Stop it between
iterations with `touch <workspace>/.stop`. See
{doc}`Autonomous overnight loop </kernelforge/how-to/autonomous-loop>`.

## Comparison

| | Claude Code (default) | Autonomous loop |
|:--|:--:|:--:|
| Agent runs on | Your machine (docker) | Your machine |
| GPU tools run on | Container | Your machine |
| Human oversight | Live in chat | None (overnight) |
| Billing | Max subscription | API credits |
| Git integration | Manual | Auto commit/revert |
| Best for | Interactive debug | Overnight optimization |
