# Summary

- Clarify that `paths.py` is the source of truth for Hyperloom path handling and launcher agents should rely on `install.sh` instead of hand-building runtime paths.
- Strengthen guidance for `model_arch.json`, session-dir discovery, and GEAK mirror recovery to avoid corrupting shared `source-mirrors` state.
- Align kernel-agent documentation with the installer-owned runtime mirror model.

# Test Plan

- `ReadLints` on updated skill files: no diagnostics.
- `python3 -m pytest`

Result: `4621 passed, 1 skipped, 9 warnings in 200.60s`
