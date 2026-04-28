# backends — try sglang vs vllm

**Family**: `shallow` · **Cost**: ~15‑30 min · **Risk**: 10% accuracy drift

Run the baseline workload under each candidate backend (sglang, vllm,
optionally TRT‑LLM) and compare `tok/s/GPU`. Per IR‑4 every backend
switch begins with `kill_server` + `check_gpu_memory`.

Constraints:

- Use `vllm_flag_translator` from `process_management` when invoking vllm
  with sglang‑style flags.
- Accuracy‑gate (gsm8k) must pass with ≤1% drop or the result is
  REVERTed (DESIGN §7.5).

Outputs:

- `update_state` `current_tput=...` only after gate passes
- `propose_action` topic=`proposal` if a clear winner emerges (delta > 3%)
