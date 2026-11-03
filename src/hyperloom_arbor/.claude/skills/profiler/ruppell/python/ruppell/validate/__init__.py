"""Checks every output carries: attributed share of GPU time, unattributed
kernels by name, registry-versus-replay sequence agreement, steps, the
cross-rank merge, the workload schema, and comparison with an independent
witness (rocprofv3) on the same work."""

from ruppell.engine import analyze_run

__all__ = ["analyze_run"]
