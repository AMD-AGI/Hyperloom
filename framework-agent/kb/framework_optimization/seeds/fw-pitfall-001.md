# fw-pitfall-001 block_manager refactor easily triggers OOM

Framework:
Tags: pitfall, kvcache, oom

Refactoring `block_manager.py` to change allocation order or buddy
strategy frequently triggers spurious OOMs at high CONC because the
fragmentation profile changes. Always pair such a patch with a
`--gpu-memory-utilization 0.85` (or lower) guard on the *same* PR,
and require a full Magpie 320-prompt sweep at the target CONC before
KEEP. PR-H accuracy gate alone is insufficient -- OOMs surface as
crashes mid-bench, not as accuracy drops.
