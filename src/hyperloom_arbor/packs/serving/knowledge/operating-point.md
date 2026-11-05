# The operating point

What the bench sends, and where each number comes from. The config sets them; the agent never
changes them (pack.yaml §8).

| What | Where it is set | Template default |
| --- | --- | --- |
| Concurrency | the config's `concurrency` | 14 |
| Prompts per measured replay | the config's `num_prompts`; the short bench uses about a fifth | 640 |
| Input and output length | `ISL`, `OSL` in the config's `server_env` | 1024 / 1024 |
| GPUs the server uses (for tok/s per GPU) | `NUM_GPUS` in `server_env` | the TP, else 1 |
| Seed of the measured replay | `BENCH_SEED` in `server_env` | 1 |
| Correctness size and floor | `GSM8K_LIMIT`, `GSM8K_MIN` in `server_env` | 400 questions; a 0.30 sanity floor until `GSM8K_MIN` is set |

In component mode a host may give its own bench (`component.bench_command`): then the operating
point is the host's, and its command and output say what it is. Read them before you reason about
a number.

The template's bench is synthetic: random tokens, `--ignore-eos`, temperature 0, no prefix reuse.
Every request is exactly `ISL` in and `OSL` out, so the batch follows the concurrency. Work whose
value depends on real text, real stopping or cache reuse is invisible here; a workload that has
them (a trace replay such as InferenceX AgentX) needs its own bench and a pack derived from this one.
