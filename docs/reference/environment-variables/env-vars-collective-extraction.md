---
myst:
    html_meta:
        "description": "Collective candidate extraction variable for Hyperloom: opt-in switch to allow a source-resolved collective to borrow shapes from the trace's sole all-reduce workload family."
        "keywords": "Hyperloom, environment variables, collective extraction, HYPERLOOM_COLLECTIVE_ALLOW_INFERRED_SHAPES, NCCL, RCCL, TraceLens, kernel candidates"
---
# Hyperloom Collective candidate extraction

Communication operators are optimized by the rewrite controller like any other
operator. TraceLens still resolves them deterministically: it maps a mangled
NCCL/RCCL kernel name back to the framework device source that issued it and
publishes the row in `kernel_candidates.json` with
`candidate_source: nccl_summary`. Vendor RCCL/NCCL symbols are opaque binaries
and never resolve to an editable source, so they are never published.

| Variable                       | Default                       | Description                                                                                                                                                                                       |
|--------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `HYPERLOOM_COLLECTIVE_ALLOW_INFERRED_SHAPES` | Unset (disabled) | Truthy allows a source-resolved collective to borrow shapes from the trace's sole all-reduce workload family. The default rejects this inference because those shapes were not observed on that device symbol. |
