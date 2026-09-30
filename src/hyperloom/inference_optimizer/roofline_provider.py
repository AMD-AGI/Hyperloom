"""RooflineProvider — one seam for analytical roofline data (native compute vs external CSV).

Consumers ask a provider for a value; the provider decides whether to compute it natively or
read it from an external (MAIDAS-authored) CSV. This collapses the mode-selection logic that was
otherwise duplicated at every consume site (arch peak / ceiling / model-meta / per-kernel).

The five read methods mirror the shapes the native math already exposes, so ``NativeRooflineProvider``
is a thin delegate — the default provider runs the identical stock computation with no CSV, which
keeps a no-external-CSV run byte-identical by construction.

Milestone 1: this module is additive — nothing is wired to it yet, so it changes no behavior. The
choke points are switched to ``make_roofline_provider(state).X(...)`` in Milestone 2, and
``CsvRooflineProvider`` / ``CsvWritingProvider`` land there too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # avoid import cycles at module load; these are only type hints
    from pathlib import Path

    from .roofline_ceiling import ModelMeta, RooflineBreakdown


@runtime_checkable
class RooflineProvider(Protocol):
    """Read-only facade over analytical roofline data. Every method is fail-soft (missing -> None)."""

    def arch_peak(self, device: str, dtype: str) -> float | None:
        """Max-achievable matrix TFLOP/s for ``device``/``dtype`` (None on miss)."""
        ...

    def mem_bw(self, device: str) -> float | None:
        """HBM bandwidth GB/s for ``device`` (None on miss)."""
        ...

    def ceiling(self, state: Any, *, arm: str | None = None) -> "RooflineBreakdown | None":
        """Composed tok/s ceiling (mem/cmp/peak + bound) for the run's ``arm``."""
        ...

    def model_meta(
        self, state: Any, model_path: "str | Path", *, precision_hint: str = ""
    ) -> "ModelMeta | None":
        """Model memory sizes + geometry."""
        ...

    def kernel(self, name: str) -> dict | None:
        """Per-kernel analytical roofline row by kernel name (None when computed inline / absent)."""
        ...


class NativeRooflineProvider:
    """Default provider: every value comes from Hyperloom's stock native computation, no CSV.

    A thin delegate over the existing ``roofline_ceiling`` functions — imported lazily so this
    module has no import-time dependency on the (heavier) ceiling module.
    """

    def arch_peak(self, device: str, dtype: str) -> float | None:
        from .roofline_ceiling import _resolve_achievable_tflops, _resolve_peak_tflops

        ach = _resolve_achievable_tflops(device or "", dtype or "")
        if ach and ach > 0:
            return float(ach)
        vendor = _resolve_peak_tflops(device or "", dtype or "")
        return float(vendor) if vendor and vendor > 0 else None

    def mem_bw(self, device: str) -> float | None:
        from .roofline_ceiling import HW_SPECS

        spec = HW_SPECS.get((device or "").strip().lower())
        bw = spec.get("hbm_bw_gbps") if spec else None
        return float(bw) if isinstance(bw, (int, float)) and bw > 0 else None

    def ceiling(self, state: Any, *, arm: str | None = None) -> "RooflineBreakdown | None":
        from .roofline_ceiling import _compute_roofline_breakdown_native

        return _compute_roofline_breakdown_native(state, arm=arm)

    def model_meta(
        self, state: Any, model_path: "str | Path", *, precision_hint: str = ""
    ) -> "ModelMeta | None":
        # The pure-native model meta comes from the HF dir; the CSV-aware path is CsvRooflineProvider.
        from .roofline_ceiling import load_model_meta

        return load_model_meta(model_path, precision_hint=precision_hint)

    def kernel(self, name: str) -> dict | None:
        # Native per-kernel magnitudes are computed inline by the trace tools; there is no native
        # per-kernel *row* to return — the consumer keeps its inline computation when this is None.
        return None


def make_roofline_provider(state: Any) -> RooflineProvider:
    """Pick the provider for this run from the roofline-csv flags (the single decision point).

    Milestone 1 returns the native provider for every mode. Milestone 2 adds the external branch:
    when ``--roofline-csv-dir`` is set, return ``CsvRooflineProvider(csv_dir, fallback=native,
    strict=state.roofline_csv_strict)``; otherwise (default or ``--no-roofline-csv``) stay pure native.

    CSV *persistence* (durable artifacts) is a separate opt-in ``CsvWritingProvider`` wrap applied only
    where a downstream reader needs the files; it is never required for correctness in the default run.
    """
    return NativeRooflineProvider()
