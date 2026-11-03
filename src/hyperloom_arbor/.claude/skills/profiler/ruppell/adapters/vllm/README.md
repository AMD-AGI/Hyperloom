# adapters/vllm

The vLLM adapter: where vLLM captures graphs and steps its engine, it calls
`ruppell.capture`, `ruppell.scope`, `ruppell.step_begin` and `ruppell.step_end`,
and hands the scheduler's per-step state to the serving schema.

## Hooks it needs

| Hook | Where in vLLM | Records |
| --- | --- | --- |
| Capture scope | around each graph capture in the model runner | graph key (batch descriptor); the scope stack at each launch |
| Op scopes during capture | module forward calls, or PyTorch `RecordFunction` callbacks | module path, layer index, op, input shapes |
| Step boundaries | around each engine step | marker kernels on the compute stream |
| Scheduler state | after scheduling each step | running requests, rows, context lengths, prefill or decode, graph key or eager |

## How it attaches

A vLLM plugin if the hook points are reachable without code changes;
otherwise one commit per hook on a branch of the pinned vLLM, behind one env
flag (`VLLM_RUPPELL`, default off), with the flag-off tree behaving as the base.
The v0 plan decides which after its feasibility item.

Known vLLM facts that matter (from the runner's plans):

- A graph replay runs no Python, so a Python hook inside a captured region
  fires at capture and never at replay. Everything a replay needs must be
  registered at capture.
- The served cudagraph mode decides which steps are replays and which are
  eager; record it per step.

## Why an adapter

For a plain PyTorch program, the script itself calls `instrument`, `capture`
and `step_begin`/`step_end`. vLLM owns every one of those places, so the
calls have to be made from inside it. That is the adapter (roadmap M2), and it
is necessary for five reasons, all confirmed against the pinned vLLM in M2:

1. **vLLM captures the graphs, not the user.** Its model runner captures one
   graph per batch size at startup. `ruppell.capture(key)` has to wrap each of
   those captures, with the batch descriptor as the key, or the replays cannot
   be joined to anything.
2. **vLLM defines the step.** An engine step (schedule, prepare inputs, run the
   model, sample) happens inside vLLM's engine loop. The step markers must be
   enqueued there, on the stream the model runs on, or steps mean nothing.
3. **Scopes may not come from module hooks.** Where vLLM compiles the model
   (`torch.compile` with piecewise graphs), forward hooks may not run inside
   compiled regions, so `instrument()` alone could leave captured launches
   unscoped. The adapter must get scopes another way: from the compiled
   graph's module metadata, or from PyTorch's `RecordFunction` callbacks at
   capture time.
4. **The workload's state lives in the scheduler.** Running requests, rows,
   context lengths, prefill or decode, which graph ran or eager: only vLLM's
   scheduler knows them per step, and a step without them cannot be placed in
   a regime (C8, the two C32 regimes).
5. **vLLM runs several processes.** The model runs in worker processes, one
   per GPU under tensor parallelism. The runtime must be loaded in each, write
   its own run directory, and the ranks' traces must be aligned in time.
   (Each process writes its own directory; export merges them and checks
   that step ids and fingerprints agree across ranks.)

The plan was [`docs/plan-m2.md`](../../docs/plan-m2.md). A plugin was enough:
every point is reached by wrapping methods after import, with no change to
vLLM. What M2 found on the way (graph handles reused by PyTorch, vLLM's warmup
batches, servers killed rather than finalized) is in
[`docs/feasibility-m2.md`](../../docs/feasibility-m2.md).
