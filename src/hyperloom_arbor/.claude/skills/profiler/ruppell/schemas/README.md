# schemas: workload schemas

A workload schema fixes the names, types and alignment of the per-step and
per-run fields an adapter records: `serving.json` (LLM serving, checked at
export on every engine step of every rank) and `training.json`. A run that
lacks a required field is refused.

## Why a schema

It is a contract rather than a layer of code: the adapter produces the values,
the schema fixes what they are called. It earns its place for three reasons:

- **Regimes.** A per-step cost is only meaningful with the step's load. The
  census, and any comparison between runs, groups steps by these fields.
- **One schema, several adapters and workloads.** The same vLLM adapter serves
  an AgentX replay, an offline batch and a microbenchmark; a SGLang adapter
  should fill the same serving fields. Without fixed names, every adapter
  invents its own, and runs cannot be compared.
- **Knowledge base conditions.** Entries record the operating point they were
  measured at (rows per step, context, concurrency). The schema is the
  vocabulary for that.

M1 did not need it (the demo's step fields are free-form). In M2 the census
reads `serving.json`'s fields: on DeepSeek V4.1 Flash the time-weighted
running count over aiperf's profiling phase agreed with aiperf's own in-flight
count within 0.3%.
