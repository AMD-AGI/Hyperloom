---
myst:
    html_meta:
        "description": "Kernel source resolution reference for Hyperloom: resolution ladder (curated dictionary, trace-derived launcher frame, name grep) and the kernel_source_resolution.json output schema."
        "keywords": "Hyperloom, kernel source resolution, kernel_source_resolution.json, TraceLens, kernel candidates, source file, resolution tier"
---
# Hyperloom Kernel source resolution

A kernel candidate must resolve to a real source file before any backend can
rewrite it. Resolution runs as a ladder: curated dictionary, then the
trace-derived launcher frame, then a name grep. All three are deterministic and
require no configuration.

Every run writes `kernel_source_resolution.json` next to the candidate report.
It answers one question per hot kernel — which file defines it, and which tier
decided that — in a versioned schema (`schema_version`, currently `1.2.0`), so
consumers and triage read a contract rather than candidate internals.
