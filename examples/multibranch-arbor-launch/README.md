# The launch file of an Arbor agent on an AgentX recipe

An Arbor agent (`--autonomous-agent` on an Arbor checkout) is served the root recipe's launch. An AgentX recipe holds
only what is added to its launch (`extra_server_args`, `extra_envs`): the round resolves the rest from the InferenceX recipe.
Arbor would then serve defaults that multi-branch (MB) does not measure, so the adapter refuses an AgentX branch without
the resolved launch. Give it with `--agent-launch FILE`:

```bash
hyperloom-multibranch run ... --autonomous-agent <arbor> --agent-pack <pack> --agent-launch launch.json
```

`FILE` is one of two things.

## A base launch record

A `base_launch.json` (`"schema": "hyperloom-base-launch/1"`) frozen by another session, or on another host. The session
freezes it as its base without reading any recipe source, so this is also how a host that cannot reach InferenceX runs
a session. A resumed session keeps the base it froze.

Each Arbor is then served the session's frozen base: its args and env, its golden layer's among them, with the root
recipe's added. Arbor gets the record as `component.base_launch`, and its entrypoint as `component.entrypoint` when it
is not the framework's own. Before any GPU work, Arbor compares the launch it would run with the record and refuses to
start on any difference. It compares the whole launch, the root recipe's additions included, so a branch whose root
recipe changes the base launch is refused. An Arbor older than `ba2f5a8` runs the launch without that check, and its
branch notes so.

A record that fails its own digest, or launches no server, is refused before PRELUDE; one for another framework or
topology than the session's, when the session freezes it.

## A launch file

Copy [`launch.template.json`](launch.template.json) and fill it from the InferenceX recipe the round resolves, at the workload's
concurrency:

| Key | Required | What it holds |
| --- | --- | --- |
| `server_flags` | yes | The server arguments, with the recipe's `override_c<CONC>` applied. The root recipe's `extra_server_args` are added after them, and the adapter adds SGLang's TP and EP sizes when neither sets them. |
| `server_env` | yes | The server's environment, as an object of strings: the recipe's variables, its golden `SGLANG_SIMULATE_ACC_*` among them. The root recipe's `extra_envs` and its start tables are set over them. |
| `model_name` | no | The name the server answers to, which the client requests. Without it, the client's `--model`, else the model's path. |
| `client` | no | The gate's benchmark client, a command with `--port` (or `--base-url`) and `--result-dir`, which the adapter makes Arbor's `component.bench_command`. Empty, the pack's own `validator/bench.sh` measures. MB's own client, when the session has one, comes first. |

The session's base is still resolved from the recipe sources, and nothing checks the file against it.

A KEEP imports as what it changes from this launch, so the root's recipe gets only that change.

`hyperloom-multibranch run` refuses a file the adapter cannot use before PRELUDE, with every reason. That includes the
template with its `<...>` placeholders unfilled.
