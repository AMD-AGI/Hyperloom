# Phase 0 · Step 03 — Golden smoke capture notes

## Environment limitation (honest record)

A full CLI `optimize` run to CLOSE requires a **real GPU + real backends**.
The CLI exposes `--critic-mock` and `--robustness-mock`, but there is **no
mock kernel / mock baseline workload flag**, so a fully GPU-free end-to-end
`optimize` run is **not possible** in this environment. Per step 03's
fallback guidance, golden artifact shapes were captured **without
fabrication** by other means (below), and the §1 contract tests in the
keep-list are the primary functional-equivalence net.

## What was captured

1. **CLI flag-surface golden** — `golden_cli_help.txt`
   - `inference_optimizer.cli --help` + `optimize --help`, both exit 0.
   - This is the §1 CLI-flags contract snapshot. Phase A/B/D diff this file;
     outward flags must not disappear/rename (retired flags excepted, and
     must be recorded in Phase A when removed).

2. **Artifact key-shape golden** — captured authentically by running
   `inference_optimizer.breakdown.build()` over the realistic synthetic
   session tree from `test_breakdown_smoke._build_fixture` (manifest + state
   + runs/ + agent workdirs), no GPU required:
   - `golden_breakdown_keys.txt` — 30 top-level keys, schema
     `hyperloom.session_breakdown.v2`. Confirms the v1<->v2 alias pairs
     (`param_search`==`explore_search`, `phase_timeline`==`action_timeline`)
     are BOTH emitted — the one preserved back-compat point (§1 / §10.3).
   - `golden_state_keys.txt` — state.json top-level keys (resume contract).
   - `golden_manifest_keys.txt` — manifest.json top-level keys (resume
     contract).

3. **Sub-process JSON bridge envelopes** — not snapshotted as separate JSON
   files; their shapes are guarded directly by keep-list envelope/CLI tests:
   - critic: `test_intent_envelope.py`, `test_cli.py`
   - robustness: `test_role_envelope.py`, `test_role_contract.py`,
     `test_runtime_cli.py`
   - framework: `test_phase_flow_cli.py` (fa phase-discover)

## How to re-verify golden artifact keys later

```bash
python - <<'PY'
import json, tempfile, pathlib
from inference_optimizer.tests import test_breakdown_smoke as t
from inference_optimizer.breakdown import build
with tempfile.TemporaryDirectory() as d:
    sd = pathlib.Path(d) / "session"; t._build_fixture(sd)
    bd = build(sd)
    print(sorted(bd.keys()))
    print(sorted(json.loads((sd/"manifest.json").read_text()).keys()))
    print(sorted(json.loads((sd/"state.json").read_text()).keys()))
PY
```

Compare against the `golden_*_keys.txt` files. Comparison is on
**keys/shape/exit-code**, not byte-for-byte values (numbers change).
