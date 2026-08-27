# Profiler / capture entry points and env vars (sglang)

Static analysis of `sglang` @ b4fe0246c.

## HTTP endpoints

```python
# python/sglang/srt/entrypoints/http_server.py:948
@app.api_route("/start_profile", methods=["GET", "POST"])
async def start_profile_async(obj: Optional[ProfileReqInput] = None):
    if obj is None:
        obj = ProfileReqInput()
    await _global_state.tokenizer_manager.start_profile(
        output_dir=obj.output_dir, start_step=obj.start_step,
        num_steps=obj.num_steps, activities=obj.activities,
        with_stack=obj.with_stack, record_shapes=obj.record_shapes,
        profile_by_stage=obj.profile_by_stage,
        merge_profiles=obj.merge_profiles,
        profile_prefix=obj.profile_prefix,
        profile_stages=obj.profile_stages)
    return Response(content="Start profiling.\n", status_code=200)

# :973  POST /stop_profile → tokenizer_manager.stop_profile()
```

gRPC sidecar exposes the same (`entrypoints/grpc_server.py:86, 135, 152, 153`).
In-process equivalents: `Engine.start_profile` / `stop_profile`
(`entrypoints/engine.py:848, 851`).

## Request body shape (`ProfileReqInput`)

```python
# python/sglang/srt/managers/io_struct.py:1673
@dataclass
class ProfileReqInput(BaseReq):
    output_dir: Optional[str] = None            # → $SGLANG_TORCH_PROFILER_DIR or "/tmp"
    start_step: Optional[int] = None            # wait until step N
    num_steps: Optional[int] = None             # auto-stop after N steps
    activities: Optional[List[str]] = None      # default ["CPU","GPU"]; accepts
                                                # CPU/GPU/XPU/MEM/RPD/CUDA_PROFILER
    profile_by_stage: bool = False              # split prefill / decode
    with_stack: Optional[bool] = None           # None → effective True
    record_shapes: Optional[bool] = None        # None → effective False
    merge_profiles: bool = False                # merge TP-rank traces
    profile_prefix: Optional[str] = None        # filename prefix
    profile_stages: Optional[List[str]] = None  # list of stage names
```

## Backend wiring (`managers/scheduler_profiler_mixin.py`)

`init_profile` (`:67`): if `SGLANG_PROFILE_V2.get()` returns early to V2
manager (`:81`). Otherwise stores config; at `:105` it defaults the dir:
```python
if output_dir is None:
    output_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")  # :106
if activities is None:
    activities = ["CPU", "GPU"]                                  # :108
```

`start_profile` (`:138`) constructs and launches torch.profiler:
```python
# scheduler_profiler_mixin.py:192-205 (torch path)
elif torchprof_activities:
    self.torch_profiler = torch.profiler.profile(
        activities=torchprof_activities,
        with_stack=with_stack if with_stack is not None else True,    # :195
        record_shapes=record_shapes if record_shapes is not None else False,  # :196
        on_trace_ready=(None if not _is_npu else
            torch_npu.profiler.tensorboard_trace_handler(str(self.torch_profiler_output_dir))),
    )
    self.torch_profiler.start()  # :205
```
NB: `with_stack` / `record_shapes` here come from the request, not from
`SGLANG_PROFILE_WITH_STACK`. The latter is declared in `environ.py` but
not consulted by V1.

ROCm RPD branch:
```python
# scheduler_profiler_mixin.py:163-189
if "RPD" in activities:
    from rpdTracerControl import rpdTracerControl
    rpdTracerControl.skipCreate()
    self.rpd_profile_path = os.path.join(
        self.torch_profiler_output_dir,
        "rpd-" + str(time.time()) + f"-TP-{self.tp_rank}" + ".trace.json.gz")  # :168-170
    if self.tp_rank == 0:
        from rocpd.schema import RocpdSchema
        if os.path.exists("trace.rpd"): os.unlink("trace.rpd")
        schema = RocpdSchema()
        connection = sqlite3.connect("trace.rpd")
        schema.writeSchema(connection); connection.commit()
    torch.distributed.barrier(self.dp_tp_cpu_group)
    self.rpd_profiler = rpdTracerControl()
    self.rpd_profiler.setPythonTrace(True); self.rpd_profiler.start()
    self.rpd_profiler.rangePush("", "rpd profile range", "")
```

Other activity flags:
- `"MEM"` (`:208`): `torch.cuda.memory._record_memory_history(max_entries=100000)`.
- `"CUDA_PROFILER"` (`:212`): on `gpu_id == base_gpu_id` only, calls
  `torch.cuda.cudart().cudaProfilerStart()`.

`stop_profile` (`:251`): exports a chrome trace with this filename rule
(`:275-294`):
```python
filename_parts = [self.profile_id, f"TP-{self.tp_rank}"]   # :276
if dp_size > 1: filename_parts.append(f"DP-{dp_rank}")     # :279
if pp_size > 1: filename_parts.append(f"PP-{pp_rank}")     # :281
if moe_ep_size > 1: filename_parts.append(f"EP-{moe_ep_rank}")  # :283
filename = (stage_prefix + "-".join(filename_parts)
            + stage_suffix + ".trace.json.gz")
self.torch_profiler.export_chrome_trace(
    os.path.join(self.torch_profiler_output_dir, filename))  # :293
torch.distributed.barrier(self.dp_tp_cpu_group)              # :296
```
RPD branch (`:298-307`): `rangePop`, `stop`, `flush`, then on rank 0:
`rpd_to_chrome_trace("trace.rpd", self.rpd_profile_path)`.

## Profiling env vars (`python/sglang/srt/environ.py`)

```python
# environ.py:198-208
SGLANG_PROFILE_WITH_STACK    = EnvBool(True)    # :198 — declared, V1 ignores
SGLANG_PROFILE_RECORD_SHAPES = EnvBool(True)    # :199 — declared, V1 ignores
SGLANG_PROFILE_V2            = EnvBool(False)   # :200 — routes to ProfileManager
SGLANG_RECORD_STEP_TIME      = EnvBool(False)   # :201 — CPU step-time log
SGLANG_TORCH_PROFILER_DIR    = EnvStr("/tmp")   # :208 — output dir fallback
```

Caveats:
- `SGLANG_PROFILE_WITH_STACK` / `SGLANG_PROFILE_RECORD_SHAPES`: V1 path
  reads only the per-request fields (see `:195-196` above). V2
  (`SGLANG_PROFILE_V2=True`) routes via `_profile_manager.configure(...)`
  at mixin `:82` — V2 may consume the env vars but is not audited here.
- `SGLANG_PROFILE_V2` rerouting sites: `scheduler_profiler_mixin.py:39, 81, 141, 254`.

## Not present in environ — controlled elsewhere

- `CUDAGRAPH_MODE` is not a sglang env. Eager vs piecewise vs breakable
  is selected by server args. Decision is in
  `model_executor/model_runner.py:2910-2914`:
  ```python
  if self.server_args.enable_breakable_cuda_graph:
      self.piecewise_cuda_graph_runner = BreakableCudaGraphRunner(self)
  else:
      self.piecewise_cuda_graph_runner = PiecewiseCudaGraphRunner(self)
  ```
  Disable graphs via CLI: `--cuda-graph-bs 0` (or similar disable flag on
  the runner). **Not investigated**: exact CLI knob mapping for
  production-graph-on vs eager.

## Capture recipe (copy-paste once you have a node)

```bash
export SGLANG_TORCH_PROFILER_DIR=/workspace/traces/<run-id>
export SGLANG_PROFILE_WITH_STACK=False   # safety net for V2 path; V1 ignores
# Launch server; once up:

curl -X POST http://127.0.0.1:30000/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"activities":["CPU","GPU"], "with_stack": false, "record_shapes": false, "num_steps": 5, "merge_profiles": false}'

# Run small bench (4-8 prompts, conc=2) ...

curl -X POST http://127.0.0.1:30000/stop_profile
# wait 30-60s for all TP-rank traces to flush — barriers per rank
ls $SGLANG_TORCH_PROFILER_DIR
# expect <prefix>-<profile_id>-TP-<r>.trace.json.gz per rank
```

RPD on ROCm:
```bash
curl -X POST http://127.0.0.1:30000/start_profile -H 'Content-Type: application/json' \
  -d '{"activities":["RPD"], "num_steps": 5}'
# output: rpd-<ts>-TP-<rank>.trace.json.gz (after rpd_to_chrome_trace conversion)
```

## Not investigated

- The V2 `ProfileManager` (`utils/profile_utils.py`,
  `utils/profile_merger.py`) — only the V1 path was traced here.
- Stage-split (`profile_by_stage=True`) interaction with cudagraphs.
- Exact `with_stack=False` requirement on TP=8 V1 engine — claimed in
  charter, not verified in code.
