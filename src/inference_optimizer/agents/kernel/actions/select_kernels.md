# select_kernels — Pick optimization candidates from a profiler trace

**Trigger**: `topic="request"` envelope with `payload.kind="select_kernels"`.

## Inputs (request payload)

| field | required | example |
|---|:-:|---|
| `params.trace_path` | yes | `/path/to/results/<task_id>/traces/filtered-TP-0.trace.json.gz` |
| `params.top_n` | no (default 5) | `5` |
| `params.min_gpu_pct` | no (default 3.0) | `3.0` |

## Procedure

1. Validate `params.trace_path` exists. If not, emit `response{status=failed, result.reason="trace_not_found", trace_path=...}`.
2. Run `bash $AGENT_PKG_DIR/scripts/trace_summary.sh <trace_path> <top_n>` to extract hot kernels.
3. Cross-reference kernel names with known source paths (see
   `reference/geak_guide.md` "Kernel identification from TraceLens"
   table). Only emit candidates we can actually optimize:
   - SGLang Triton kernels (source available)
   - torch.compile inductor kernels (in `/tmp/torchinductor_root/`)
   - Custom HIP `__global__` kernels (user-provided)
   - Skip vendor BLAS like `Cijk_Ailk_*` (no source)
4. For each candidate, resolve a `source_path` if possible. Use Read
   to peek at the file to confirm it's a kernel (not a wrapper).

## Output (RESPONSE payload)

```json
{
  "intent_type": "response",
  "payload": {
    "in_reply_to": "<the request msg_id from your inbox>",
    "kind": "select_kernels_done",
    "status": "succeeded",
    "result": {
      "candidates": [
        {
          "name": "triton_red_fused_sum_42",
          "framework": "triton",
          "source_path": "/tmp/torchinductor_root/abc/123.py",
          "gpu_pct": 12.4,
          "rationale": "top hot kernel; inductor source available"
        },
        ...
      ],
      "trace_path": "<echoed back>",
      "top_n": 5,
      "skipped": [
        {"name": "Cijk_Ailk_Bljk_HSS", "reason": "vendor BLAS, no source"}
      ]
    }
  }
}
```

## Failure modes

| Symptom | Recovery |
|---|---|
| `trace_path` doesn't exist | `response{status=failed, result.reason="trace_not_found"}` |
| `trace_summary.sh` returns rc != 0 | `response{status=failed, result.reason="trace_parse_failed", log_excerpt=...}` |
| All candidates filtered out (no source available) | `response{status=succeeded, result.candidates=[], note="no optimizable kernels"}` — let executor decide |

## Soft rules

- **IR-2** (recommended): do NOT modify any candidate's source file
  during selection — selection is read-only. The actual rewrite happens
  in `run_optimization` via GEAK.

## Discipline

- Top-N defaults to 5 because GEAK_TOP_CANDIDATES=5 is the prior; emit
  more if executor explicitly asks (`params.top_n=10`) but flag in
  `result.note` that the per-candidate budget shrinks.
- Always include `gpu_pct` and `rationale` so executor's selection step
  has signal to choose between them.
