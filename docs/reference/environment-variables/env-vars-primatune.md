---
myst:
    html_meta:
        "description": "PrimaTune predictor variables for Hyperloom: service endpoint URL, operating mode (off/shadow/active), and per-request timeout. Set by --primatune-endpoint and --primatune-mode."
        "keywords": "Hyperloom, environment variables, PrimaTune, HYPERLOOM_PREDICTOR_ENDPOINT, HYPERLOOM_PREDICTOR_MODE, HYPERLOOM_PREDICTOR_TIMEOUT_SEC, predictor"
---
# Hyperloom PrimaTune variables

Set by `--primatune-endpoint` and `--primatune-mode`, or directly in the shell. See
[PrimaTune predictor](primatune-predictor.md).

| Variable | Default | Description |
|---|---|---|
| `HYPERLOOM_PREDICTOR_ENDPOINT` | Unset (off) | Base URL of a PrimaTune service; the client posts to `<endpoint>/v1/predict`. |
| `HYPERLOOM_PREDICTOR_MODE` | `shadow` | `off`, `shadow` (ask and log, queue nothing) or `active` (queue the answer for benchmarking). An invalid value falls back to `shadow`. |
| `HYPERLOOM_PREDICTOR_TIMEOUT_SEC` | `900` | Per-request timeout in seconds, at least `1`. The request runs off the tick loop, so a long timeout costs no Coordinator time. |
