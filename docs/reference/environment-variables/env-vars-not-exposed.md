---
myst:
    html_meta:
        "description": "Variables intentionally not exposed in Hyperloom: internal-only environment variables such as host-stall probe handoffs and test suite toggles that should not be set by hand."
        "keywords": "Hyperloom, environment variables, internal variables, HYPERLOOM_HOST_PROBE, INFERENCE_OPTIMIZER_INTERNAL, not exposed, diagnostic"
---
# Variables intentionally not exposed

These are read by `os.environ` somewhere in the codebase but are
internal-only — do not set them by hand:

* `HYPERLOOM_HOST_PROBE`, `HYPERLOOM_HOST_PROBE_DEEP`,
  `HYPERLOOM_HOST_PROBE_DIR`, `HYPERLOOM_HOST_PROBE_ROOTS` (and the
  `..._MAX_SITES` / `..._ARG_SAMPLES` caps): internal handoff to the host-stall
  evidence probe, armed by the profile leg that collects the evidence. The deep
  tier inflates host time by design, so setting it by hand distorts any trace
  collected alongside it.
* Any `_INFERENCE_OPTIMIZER_*_INTERNAL_*` symbol: internal toggles for
  the test suite.

If you find one of these in a log message, treat it as diagnostic
detail rather than something you should tune.
