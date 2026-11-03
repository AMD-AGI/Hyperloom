# The launch file of an Arbor agent on an AgentX recipe

A stand-in until multi-branch (MB) carries the launch Magpie resolves.

An Arbor agent (`--autonomous-agent` on an Arbor checkout) is served the root recipe's launch. A Magpie AgentX recipe
holds only what is added to its launch (`extra_server_args`, `extra_envs`): Magpie resolves the rest from the InferenceX
recipe. Arbor would then serve defaults that MB does not measure, so the adapter refuses an AgentX branch without the
resolved launch. Give it with `--agent-launch FILE`:

```bash
hyperloom-multibranch run ... --autonomous-agent <arbor> --agent-pack <pack> --agent-launch launch.json
```

Copy [`launch.template.json`](launch.template.json) and fill it from the InferenceX recipe Magpie resolves, at the
workload's concurrency:

| Key | Required | What it holds |
| --- | --- | --- |
| `server_flags` | yes | The server arguments, with the recipe's `override_c<CONC>` applied. The root recipe's `extra_server_args` are added after them, and the adapter adds SGLang's TP and EP sizes when neither sets them. |
| `server_env` | yes | The server's environment, as an object of strings: the recipe's variables, its golden `SGLANG_SIMULATE_ACC_*` among them. The root recipe's `extra_envs` and its start tables are set over them. |
| `model_name` | no | The name the server answers to, which the client requests. Without it, the client's `--model`, else the model's path. |
| `client` | no | The gate's benchmark client, a command with `--port` (or `--base-url`) and `--result-dir`, which the adapter makes Arbor's `component.bench_command`. Empty, the pack's own `validator/bench.sh` measures. MB's own client, when the session has one, comes first. |

A KEEP imports as what it changes from this launch, so the root's recipe gets only that change.

`hyperloom-multibranch run` refuses a file the adapter cannot use before PRELUDE, with every reason.
