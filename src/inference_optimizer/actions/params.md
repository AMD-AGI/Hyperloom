# params — server-flag tuning

**Family**: `shallow` · **Cost**: ~10‑25 min · **Risk**: 0% baseline,
+0.30 if quantization (kv‑cache fp8) is among the params being swept.

Tunes safe runtime params: `--max-running-requests`, `--mem-fraction-static`,
`--enable-torch-compile`, `--chunked-prefill-size` etc.

When tuning ventures into `--kv-cache-dtype fp8` / `--quantization fp8`
override `accuracy_risk` to `0.30` for that specific candidate (the gate
will trigger).

Outputs:

- `propose_action` topic=`proposal` for each tested param combo
- after `update_after_action` records result, scheduler may follow up
  with `report` once `cumulative_gain_plateau` triggers.
