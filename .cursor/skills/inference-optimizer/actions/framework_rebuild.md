# framework_rebuild — rebuild framework against latest aiter / sgl-kernel

**Family**: `long` · **Cost**: ~60‑90 min · **Risk**: 15% accuracy

Marathon‑only, runs at most once per session. Pulls the latest
aiter / sgl-kernel commit and rebuilds. Heavy enough that it's gated by
the `cumulative_gain_plateau` predicate so we don't burn this on a
session that's already winning.
