# The launch file of an Arbor agent on an AgentX recipe

An Arbor agent (`--autonomous-agent` on an Arbor checkout) is served the root recipe's launch. An AgentX recipe holds
only what is added to its launch (`extra_server_args`, `extra_envs`): the round resolves the rest from the InferenceX recipe.
Arbor would then serve defaults that multi-branch (MB) does not measure, so the adapter refuses an AgentX branch without
the resolved launch. Give it with `--agent-launch FILE`, or give the session's base with `--reference-script` (below):

```bash
hyperloom-multibranch run ... --autonomous-agent <arbor> --agent-pack <pack> --agent-launch launch.json
```

## A base launch record: `--reference-script`

A `base_launch.json` (`"schema": "hyperloom-base-launch/1"`) frozen by another session, or on another host, is not an
`--agent-launch` file: give it as the session's base, as `hyperloom optimize` takes one, and leave `--agent-launch` out:

```bash
hyperloom-multibranch run ... --reference-script base_launch.json --autonomous-agent <arbor> --agent-pack <pack>
```

The session freezes it as its base without reading any recipe source, so this is also how a host that cannot reach
InferenceX runs a session. `--reference-script` also takes a shell launch script, and a session without an Arbor takes
it the same way. A resumed session keeps the base it froze.

Each Arbor is then served the session's frozen base: its args and env, its golden layer's among them, with the root
recipe's added. Arbor gets the record as `component.base_launch`, and its entrypoint as `component.entrypoint` when it
is not the framework's own. Before any GPU work, Arbor compares the launch it would run with the record and refuses to
start on any difference. It compares the whole launch, the root recipe's additions included, so a branch whose root
recipe changes the base launch is refused. An Arbor older than `ba2f5a8` runs the launch without that check, and its
branch notes so.

A file that is no launch of the session's framework, sets a flag from a shell variable, or is a record that fails its
own digest, is refused before PRELUDE, as is a record given as `--agent-launch` or an `--agent-launch` beside
`--reference-script`; one for another topology than the session's, when the session freezes it.

## How Arbor measures an AgentX branch

MB gives every Arbor the gate's own AgentX round as `component.bench_command`: `python -m inference_suite run` against
Arbor's server, on the session's frozen base, staged tools, trace corpus, seed, concurrency and trace window. Each run
writes `report.json` in Arbor's output directory, from which Arbor's harness reads each metric by path
(`component.bench_read`, Arbor `6a62def` or later; an older Arbor is refused at preflight). Neither file needs bench
variables (`MODEL_PREFIX`, `DURATION`, `INFMAX_CONTAINER_WORKSPACE` and so on), and the pack's `bench_agentx.sh` does
not run. Arbor's bench phase is capped at one test attempt: the trace window (`--agentx-duration-sec`, an hour by
default) plus an hour.

## A launch file: `--agent-launch`

Copy [`launch.template.json`](launch.template.json) and fill it from the InferenceX recipe the round resolves, at the workload's
concurrency:

| Key | Required | What it holds |
| --- | --- | --- |
| `server_flags` | yes | The server arguments, with the recipe's `override_c<CONC>` applied. The root recipe's `extra_server_args` are added after them, and the adapter adds SGLang's TP and EP sizes when neither sets them. |
| `server_env` | yes | The server's environment, as an object of strings: the recipe's variables, its golden `SGLANG_SIMULATE_ACC_*` among them. The root recipe's `extra_envs` and its start tables are set over them. |
| `model_name` | no | The name the server answers to, which the client requests. Without it, the client's `--model`, else the model's path. |
| `client` | no | The gate's benchmark client, a command with `--port` (or `--base-url`) and `--result-dir`, or an InferenceSuite round's `--run-dir`, which the adapter makes Arbor's `component.bench_command`. MB's own client comes first, and MB gives one to every session's Arbors. |

The session's base is still resolved from the recipe sources, and nothing checks the file against it.

A KEEP imports as what it changes from this launch, so the root's recipe gets only that change.

`hyperloom-multibranch run` refuses a file the adapter cannot use before PRELUDE, with every reason. That includes the
template with its `<...>` placeholders unfilled.
