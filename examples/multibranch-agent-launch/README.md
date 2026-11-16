# The launch a component agent is served

A component agent (an `--autonomous-agent` whose checkout an installed adapter recognises) is served the root recipe's
launch added to the launch it holds only the additions of (`extra_server_args`, `extra_envs`). By default that is the
session's frozen base launch, the one multi-branch (MB) measures every change on, resolved from the recipe sources
(InferenceX, then the framework's cookbook) or `--reference-script`; no flag is needed:

```bash
hyperloom-multibranch run ... --autonomous-agent <agent> --agent-pack <pack>
```

`--agent-launch FILE` (below) serves a launch file instead.

## A base launch record: `--reference-script`

A `base_launch.json` (`"schema": "hyperloom-base-launch/1"`) frozen by another session, or on another host, is not an
`--agent-launch` file: give it as the session's base, as `hyperloom optimize` takes one, and leave `--agent-launch` out:

```bash
hyperloom-multibranch run ... --reference-script base_launch.json --autonomous-agent <agent> --agent-pack <pack>
```

The session freezes it as its base without reading any recipe source, so this is also how a host that cannot reach
InferenceX runs a session. `--reference-script` also takes a shell launch script, and a session without a component
agent takes it the same way. A resumed session keeps the base it froze.

Each component agent is served the session's frozen base, however it was resolved: its args and env, its golden
layer's among them, with the root recipe's added. MB gives the agent's adapter the record, and the adapter writes the
agent's config from it.

A file that is no launch of the session's framework, sets a flag from a shell variable, or is a record that fails its
own digest, is refused before PRELUDE, as is a record given as `--agent-launch` or an `--agent-launch` beside
`--reference-script`; one for another topology than the session's, when the session freezes it.

## A launch file: `--agent-launch`

A launch file replaces the frozen base as what a component agent is served. Copy
[`launch.template.json`](launch.template.json) and fill it at the workload's concurrency:

| Key | Required | What it holds |
| --- | --- | --- |
| `server_flags` | yes | The server arguments, with the recipe's `override_c<CONC>` applied. The root recipe's `extra_server_args` are added after them. |
| `server_env` | yes | The server's environment, as an object of strings: the recipe's variables, its golden `SGLANG_SIMULATE_ACC_*` among them. The root recipe's `extra_envs` and its start tables are set over them. |
| `model_name` | no | The name the server answers to, which the client requests. Without it, the client's `--model`, else the model's path. |

MB gives every agent the gate's client to measure with, so the file names none; a `client` key is refused.

The session's base is still resolved from the recipe sources and is what MB measures on; nothing checks the file
against it.

A kept change imports as what it changes from this launch, so the root's recipe gets only that change.

`hyperloom-multibranch run` refuses a file it cannot read as a launch before PRELUDE, with every reason. That includes
the template with its `<...>` placeholders unfilled.
