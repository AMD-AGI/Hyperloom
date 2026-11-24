---
myst:
    html_meta:
        "description": "TraceLens analysis route variable for Hyperloom: selects agent (LLM-driven) or non-llm (TraceLens-only) analysis route for trace_analyze requests."
        "keywords": "Hyperloom, environment variables, TraceLens, HYPERLOOM_TRACE_ANALYSIS_ROUTE, trace_analyze, non-llm, agent, analysis route"
---
# TraceLens analysis route

| Variable | Default | Description |
|---|---|---|
| `HYPERLOOM_TRACE_ANALYSIS_ROUTE` | `agent` | `agent` runs the TraceLens skill under a model. `non-llm` runs TraceLens without a model and writes `analysis.json` from the TraceLens report; it shares the gates, candidates and reports with `agent` but adds no collective candidates and writes no prose, and the report's trace-quality warnings appear in `trace_health_warnings`. A `trace_analyze` request's `analysis_route` key overrides it. Any other value fails the request. |
