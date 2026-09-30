"""RooflineProvider — one seam for analytical roofline data (native compute vs external CSV).

Consumers ask a provider for a value; the provider decides, per read, whether to compute it natively
or read it from an external (MAIDAS-authored) CSV — the seam that collapses that external-vs-native
selection. It is read-only by design: Hyperloom's existing producers own CSV *writes*.

The five read methods mirror the shapes the native math already exposes, so ``NativeRooflineProvider``
is a thin delegate — the default provider runs the identical stock computation with no CSV, which
keeps a no-external-CSV run byte-identical by construction.

The ceiling read (:func:`compute_roofline_breakdown_from_state`) is wired through the provider today;
the other read methods (arch peak / model-meta / per-kernel) are the stable read surface the remaining
consume sites migrate onto.
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


class CsvRooflineProvider:
    """Read-through provider over an external (MAIDAS-authored) CSV dir, with native fallback.

    A per-value miss falls through to ``fallback`` (native). Used in external mode
    (``--roofline-csv-dir``); it wraps the leaf ``RooflineResolver`` and the module-level external
    readers rather than re-implementing any read logic.

    ``--roofline-csv-strict`` is enforced where it is contractually scoped — the ceiling read, inside
    :func:`roofline_ceiling._external_ceiling_breakdown` (which raises on a missing arm row). The other
    readers here are lenient by design (they fall back), matching their live seams; a strict guard is
    added deliberately, with a test, only if/when one of those sites needs it.
    """

    def __init__(self, csv_dir: "str | Path", *, fallback: "RooflineProvider"):
        from .roofline_csv import RooflineResolver

        self._resolver = RooflineResolver(csv_dir)
        self._fallback = fallback

    def arch_peak(self, device: str, dtype: str) -> float | None:
        v = self._resolver.arch_peak(device, dtype)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
        return self._fallback.arch_peak(device, dtype)

    def mem_bw(self, device: str) -> float | None:
        v = self._resolver.mem_bw(device)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
        return self._fallback.mem_bw(device)

    def ceiling(self, state: Any, *, arm: str | None = None) -> "RooflineBreakdown | None":
        # _external_ceiling_breakdown reads the external ceiling arm and owns the strict-raise /
        # trust guards; None means "not in the external CSV" -> native fallback.
        from .roofline_ceiling import _external_ceiling_breakdown

        bd = _external_ceiling_breakdown(state, arm)
        if bd is not None:
            return bd
        return self._fallback.ceiling(state, arm=arm)

    def model_meta(
        self, state: Any, model_path: "str | Path", *, precision_hint: str = ""
    ) -> "ModelMeta | None":
        row = self._resolver.model_meta()
        if row:
            from .roofline_ceiling import _model_meta_from_row

            return _model_meta_from_row(row)
        return self._fallback.model_meta(state, model_path, precision_hint=precision_hint)

    def kernel(self, name: str) -> dict | None:
        # A per-kernel miss is not an error: the consumer keeps its inline computation for kernels the
        # external author did not supply.
        return self._resolver.kernel(name)


def make_roofline_provider(state: Any) -> RooflineProvider:
    """Pick the provider for this run from the roofline-csv flags.

    - ``--roofline-csv-dir`` set -> external: read the MAIDAS CSVs, native fallback (``CsvRooflineProvider``).
    - otherwise (default or ``--no-roofline-csv``) -> pure native.

    The provider is read-only: Hyperloom's existing ``publish_*_csv`` producers own CSV persistence, so
    consuming through the provider never writes.
    """
    native = NativeRooflineProvider()
    csv_dir = str(getattr(state, "roofline_csv_dir", "") or "").strip()
    if csv_dir:
        return CsvRooflineProvider(csv_dir, fallback=native)
    return native
