# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""On-node hardware probes backing the roofline ceiling.

Both terms of the roofline are hardware capabilities, and both are currently
read from hand-maintained per-SKU tables. That is avoidable: the hardware can be
asked directly, and asking is more accurate than the tables.

For compute, the vendor peak is exactly ``ISA_rate x CUs x boost_clock`` -- every
published entry inverts onto the architectural MFMA rate with no residual. Two of
those three come straight from ``hipDeviceProp_t``, and the third is measurable:
a back-to-back MFMA kernel with enough independent accumulator chains to cover
instruction latency recovers the documented rate to within a few percent on every
precision (measured on gfx950: bf16 93%, fp16 96%, fp8 99%, fp4 97%). Because the
probe is raw MFMA rather than a library GEMM, nothing about kernel quality enters
the number, so the resulting roof stays a genuine upper bound that a tuned kernel
cannot beat -- unlike a microbenchmark-derived "achievable" figure, which bakes
one kernel's inefficiency into the ceiling and hides the very headroom the kernel
agent exists to recover.

For memory the probe reports absolute achievable GB/s rather than a fraction of
a theoretical peak, because the theoretical peak is *not* derivable at runtime:
``hipDeviceProp_t`` describes MI355X as 8192-bit at 2000 MHz, which the usual
double-data-rate formula turns into 4096 GB/s against an actual 8000 GB/s, since
HBM3E clocks its pins at four times the reported rate and that multiplier moves
with the HBM generation.

Everything here is best-effort and fails soft. Probing needs ``hipcc`` and a
visible GPU; when either is missing, or a compile or run fails, every entry point
returns ``None`` and the caller falls back through the existing chain
(max-achievable table, then vendor peak) exactly as before.

Probing and reading are deliberately separate calls. :func:`probe_and_cache`
compiles and runs, and is meant to be invoked at a known point such as
environment setup; the resolver-facing readers only ever touch the cache, so a
roofline computation can never trigger a surprise compile in the middle of a
measurement window.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.session.paths import deps_cache_root

from ._hw_probe_src import BANDWIDTH_PROBE_SRC, MFMA_PROBE_SRC

log = logging.getLogger(__name__)

#: Set truthy to skip probing entirely; every reader then returns ``None`` and
#: callers keep their table-derived values.
DISABLE_ENV = "HYPERLOOM_GPU_PROBE_DISABLE"

#: Overrides the per-subprocess timeout (seconds) for compiles and probe runs.
TIMEOUT_ENV = "HYPERLOOM_GPU_PROBE_TIMEOUT_SEC"

#: Compiling the probe dominates; the runs themselves are ~1-3 s.
_DEFAULT_TIMEOUT_SEC = 180.0

#: Bumped whenever the cached payload's meaning changes, so a stale cache from
#: an older build is ignored rather than misread.
_SCHEMA_VERSION = 1

#: Floor, as a fraction of boost, on a sampled probe clock we are willing to
#: normalize by. The sampled clock is a *divisor*, so under-reading it inflates
#: the derived rate and therefore the ceiling -- the one direction this module
#: must never fail in. Below this the sample is treated as unusable and boost is
#: used instead, which can only understate the rate.
_MIN_PROBE_SCLK_FRACTION = 0.5

#: Lead time before the synchronized bandwidth window opens, covering
#: allocation, buffer fill, and warm-up on every concurrent instance.
_BANDWIDTH_START_LEAD_SEC = 8.0

#: Length of the synchronized bandwidth window.
_BANDWIDTH_MEASURE_SEC = 2.0


@dataclass(frozen=True)
class DeviceInfo:
    """Runtime-discovered device identity behind a probe result.

    Attributes:
        arch (str): Normalized architecture, e.g. ``"gfx950"`` (feature suffixes
            such as ``:sramecc+`` are stripped).
        cu_count (int): Compute units reported by the runtime. This is what
            makes a cut-down part correct without a table entry.
        boost_sclk_mhz (float): Peak engine clock reported by the runtime.
    """

    arch: str
    cu_count: int
    boost_sclk_mhz: float


@dataclass(frozen=True)
class MfmaRate:
    """Measured matrix-core issue rate for one precision.

    Attributes:
        precision (str): Precision tag (``bf16``, ``fp16``, ``fp8``, ``fp4``).
        variant (str): Winning instruction variant, recorded so a surprising
            rate can be traced to the opcode that produced it.
        flops_per_clk_per_cu (float): Measured architectural rate.
    """

    precision: str
    variant: str
    flops_per_clk_per_cu: float


@dataclass(frozen=True)
class ProbeResult:
    """Everything one node's probe run established.

    Attributes:
        schema (int): Payload schema version.
        device (DeviceInfo): Runtime-discovered device identity.
        rocm_version (str): ROCm version the probe was built against; part of
            the cache key, since a toolchain change can move the numbers.
        mfma_rates (dict[str, MfmaRate]): Winning rate per precision.
        bandwidth_gb_per_sec (dict[int, float]): Per-GPU achievable streaming
            bandwidth, keyed by how many GPUs were loaded concurrently.
        probe_sclk_mhz (float): Engine clock sampled during the MFMA probe, used
            to convert its FLOP/s into a per-clock rate.
        probed_at (float): Unix timestamp of the run.
    """

    schema: int
    device: DeviceInfo
    rocm_version: str
    mfma_rates: dict[str, MfmaRate]
    bandwidth_gb_per_sec: dict[int, float]
    probe_sclk_mhz: float
    probed_at: float


def _disabled() -> bool:
    """Whether probing has been switched off by the environment.

    Returns:
        ``True`` when the disable flag is set to a truthy token.
    """
    return os.environ.get(DISABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _timeout_sec() -> float:
    """Per-subprocess timeout for compiles and probe runs.

    Returns:
        The configured timeout, or the default when unset or unparseable.
    """
    raw = os.environ.get(TIMEOUT_ENV, "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_SEC
    return value if value > 0 else _DEFAULT_TIMEOUT_SEC


def normalize_arch(arch: str | None) -> str:
    """Strip target-feature suffixes from a gfx architecture string.

    ``hipDeviceProp_t`` reports ``gfx950:sramecc+:xnack-``; the features do not
    change the matrix-core rate and would fragment the cache key.

    Args:
        arch: Raw architecture string.

    Returns:
        The bare architecture (``"gfx950"``), or ``""`` when unavailable.
    """
    return (arch or "").split(":", 1)[0].strip().lower()


def rocm_version() -> str:
    """Installed ROCm version, for the cache key.

    Returns:
        The version string, or ``"unknown"`` when it cannot be read.
    """
    for path in (Path("/opt/rocm/.info/version"), Path("/opt/rocm/.info/version-dev")):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            return text.split("-", 1)[0]
    return "unknown"


@functools.lru_cache(maxsize=1)
def detect_arch() -> str:
    """Architecture of the local GPU, without running a probe.

    Needed to find the cache entry. Tries torch first (cheapest where it is
    installed), then the ``GFX Version`` column of ``rocm-smi
    --showproductname``. Note that ``--showhw`` refuses CSV output, so it is not
    a usable source here.

    Returns:
        The normalized architecture, or ``""`` when undetectable.
    """
    try:
        import torch  # noqa: PLC0415  (optional, and import cost is real)

        if torch.cuda.is_available():
            return normalize_arch(torch.cuda.get_device_properties(0).gcnArchName)
    except Exception:  # noqa: BLE001  (torch absent, no driver, no device)
        pass
    if not shutil.which("rocm-smi"):
        return ""
    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname", "--csv"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    for token in out.replace(",", " ").split():
        if token.lower().startswith("gfx") and any(ch.isdigit() for ch in token):
            return normalize_arch(token)
    return ""


def probe_cache_dir() -> Path:
    """Directory holding compiled probes and their cached results.

    Returns:
        The probe cache directory (not created).
    """
    return deps_cache_root() / "gpu_probes"


def _cache_path(arch: str, rocm: str) -> Path:
    """Path of the cached probe payload for an architecture and toolchain.

    Args:
        arch: Normalized architecture.
        rocm: ROCm version string.

    Returns:
        The JSON cache path.
    """
    return probe_cache_dir() / f"probe-{arch}-rocm{rocm}.json"


def _sample_sclk_once() -> float:
    """Highest engine clock currently reported across visible GPUs.

    Reads the ``sclk clock speed:`` column by name. Scanning the row for any
    plausible number instead would silently return the memory clock, which sits
    in the same range and never droops -- and an under-read clock *inflates* the
    derived per-clock rate, so this has to be exact rather than approximate.

    The maximum rather than the mean: the probe loads one GPU while its peers
    idle near 95 MHz, and averaging those in would understate the clock the
    probe actually ran at.

    Returns:
        The clock in MHz, or ``0.0`` when unavailable.
    """
    if not shutil.which("rocm-smi"):
        return 0.0
    try:
        out = subprocess.run(
            ["rocm-smi", "--showclocks", "--csv"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0
    import csv  # noqa: PLC0415  (only needed on this path)

    best = 0.0
    try:
        rows = list(csv.DictReader(out.splitlines()))
    except csv.Error:
        return 0.0
    for row in rows:
        for key, value in row.items():
            if not key or "sclk" not in key.lower() or "speed" not in key.lower():
                continue
            digits = "".join(ch for ch in str(value or "") if ch.isdigit())
            if digits:
                best = max(best, float(digits))
    return best


class _SclkSampler:
    """Samples engine clock in the background for the duration of a probe."""

    def __init__(self, interval_sec: float = 0.25) -> None:
        """Initialize the sampler.

        Args:
            interval_sec: Delay between samples.
        """
        self._interval = interval_sec
        self._stop = threading.Event()
        self._samples: list[float] = []
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        """Poll until stopped, keeping positive samples."""
        while not self._stop.is_set():
            value = _sample_sclk_once()
            if value > 0:
                self._samples.append(value)
            self._stop.wait(self._interval)

    def __enter__(self) -> _SclkSampler:
        """Start sampling.

        Returns:
            This sampler.
        """
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        """Stop sampling and join the thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @property
    def mean_mhz(self) -> float:
        """Mean of the collected samples.

        Returns:
            The mean engine clock, or ``0.0`` when nothing was sampled.
        """
        return sum(self._samples) / len(self._samples) if self._samples else 0.0


def _compile(src: str, stem: str, arch: str, cache_dir: Path) -> Path | None:
    """Compile one embedded probe source, reusing a current binary.

    Args:
        src: HIP source text.
        stem: Base name for the source and binary.
        arch: Normalized offload architecture.
        cache_dir: Directory to build in.

    Returns:
        Path to the executable, or ``None`` when the toolchain is missing or the
        compile fails.
    """
    hipcc = shutil.which("hipcc")
    if not hipcc:
        log.debug("hw_probe: hipcc not found; skipping %s", stem)
        return None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        src_path = cache_dir / f"{stem}.hip"
        binary = cache_dir / f"{stem}-{arch}"
        if not src_path.exists() or src_path.read_text(encoding="utf-8") != src:
            src_path.write_text(src, encoding="utf-8")
        elif binary.exists() and binary.stat().st_mtime >= src_path.stat().st_mtime:
            return binary
        proc = subprocess.run(
            [hipcc, "-O3", f"--offload-arch={arch}", str(src_path), "-o", str(binary)],
            capture_output=True,
            text=True,
            timeout=_timeout_sec(),
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("hw_probe: compiling %s failed: %r", stem, exc)
        return None
    if proc.returncode != 0 or not binary.exists():
        log.debug("hw_probe: compiling %s failed: %s", stem, (proc.stderr or "")[:400])
        return None
    return binary


def _run(binary: Path, args: list[str], *, env: dict[str, str] | None = None) -> str:
    """Run a compiled probe and capture stdout.

    Args:
        binary: Probe executable.
        args: Command-line arguments.
        env: Optional environment overlay (used to mask visible devices).

    Returns:
        Captured stdout, or ``""`` on any failure.
    """
    merged = {**os.environ, **(env or {})}
    try:
        proc = subprocess.run(
            [str(binary), *args],
            capture_output=True,
            text=True,
            timeout=_timeout_sec(),
            check=False,
            env=merged,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("hw_probe: running %s failed: %r", binary.name, exc)
        return ""
    if proc.returncode != 0:
        log.debug("hw_probe: %s exited %d: %s", binary.name, proc.returncode, (proc.stderr or "")[:400])
        return ""
    return proc.stdout or ""


def _parse_json_lines(text: str) -> list[dict[str, Any]]:
    """Parse the probe's JSON-lines stdout, skipping unparseable lines.

    Args:
        text: Raw stdout.

    Returns:
        The decoded objects.
    """
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def detect_gpu_count() -> int:
    """Number of GPUs visible to this process.

    Returns:
        The device count, or ``0`` when it cannot be determined.
    """
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        pass
    if not shutil.which("rocm-smi"):
        return 0
    try:
        out = subprocess.run(
            ["rocm-smi", "--showid", "--csv"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in out.splitlines() if line.strip().lower().startswith("card"))


def _probe_mfma(binary: Path) -> tuple[DeviceInfo | None, dict[str, MfmaRate], float]:
    """Run the matrix-core probe and reduce it to one rate per precision.

    The probe emits every instruction variant the device supports; the fastest
    per precision wins. That matters because the wide and narrow forms of the
    same precision differ by 2x, and on gfx950 the fp8 and fp4 rates come from a
    shared ``f8f6f4`` opcode that a naive per-precision opcode guess misses
    entirely.

    Rates are normalized by the clock sampled *during* the run rather than by
    boost, so a probe that ran below boost is not misreported as a slower part.

    Args:
        binary: Compiled MFMA probe.

    Returns:
        ``(device, rates_by_precision, probe_sclk_mhz)``; the device is ``None``
        and the mapping empty when the probe produced nothing usable.
    """
    with _SclkSampler() as sampler:
        text = _run(binary, ["0", "20000"])
    records = _parse_json_lines(text)
    if not records:
        return None, {}, 0.0

    device: DeviceInfo | None = None
    for rec in records:
        if rec.get("kind") == "device":
            device = DeviceInfo(
                arch=normalize_arch(str(rec.get("arch", ""))),
                cu_count=int(rec.get("cus", 0) or 0),
                boost_sclk_mhz=float(rec.get("boost_mhz", 0.0) or 0.0),
            )
            break
    if device is None or device.cu_count <= 0:
        return None, {}, 0.0

    # Reject a missing or implausibly low sample and use boost instead. A probe
    # this compute-dense sits at or near boost, so the substitution costs little
    # accuracy, and it errs toward understating the rate rather than inflating
    # the ceiling.
    sclk = sampler.mean_mhz
    if sclk < device.boost_sclk_mhz * _MIN_PROBE_SCLK_FRACTION:
        log.debug(
            "hw_probe: sampled sclk %.1f MHz implausible against %.1f MHz boost; using boost",
            sclk,
            device.boost_sclk_mhz,
        )
        sclk = device.boost_sclk_mhz
    if sclk <= 0:
        return device, {}, 0.0

    rates: dict[str, MfmaRate] = {}
    for rec in records:
        if rec.get("kind") != "mfma":
            continue
        precision = str(rec.get("precision", "")).strip().lower()
        flops_per_sec = float(rec.get("flops_per_sec", 0.0) or 0.0)
        if not precision or flops_per_sec <= 0:
            continue
        rate = flops_per_sec / (sclk * 1e6) / device.cu_count
        current = rates.get(precision)
        if current is None or rate > current.flops_per_clk_per_cu:
            rates[precision] = MfmaRate(
                precision=precision,
                variant=str(rec.get("variant", "")),
                flops_per_clk_per_cu=rate,
            )
    return device, rates, sclk


def _probe_bandwidth_at(binary: Path, devices: list[int]) -> float:
    """Mean per-GPU streaming bandwidth with *devices* loaded concurrently.

    Every instance is handed the same wall-clock start instant so the
    measurement windows genuinely overlap; letting them free-run makes an
    all-GPU probe silently report the solo figure.

    Measuring under load rather than assuming: on MI355X the per-GPU figure
    turns out to be flat at ~7130 GB/s whether one GPU streams or all eight do
    (verified at 100% utilization on every card, ~990 W each), because each GPU
    owns its HBM stacks and there is no shared path to contend for. Parts that
    do share a memory path would show it here instead of going unnoticed.

    Args:
        binary: Compiled bandwidth probe.
        devices: Device indices to load simultaneously.

    Returns:
        Mean per-GPU GB/s, or ``0.0`` when no device reported.
    """
    results: dict[int, float] = {}
    lock = threading.Lock()
    # Enough lead time for every instance to allocate, fill, and warm up before
    # the shared window opens.
    start_at = time.time() + _BANDWIDTH_START_LEAD_SEC

    def _one(index: int) -> None:
        """Run the probe on one device and record its bandwidth."""
        text = _run(
            binary,
            [str(index), "16", f"{start_at:.3f}", str(_BANDWIDTH_MEASURE_SEC)],
        )
        for rec in _parse_json_lines(text):
            if rec.get("kind") != "bandwidth":
                continue
            value = float(rec.get("gb_per_sec", 0.0) or 0.0)
            if value > 0:
                with lock:
                    results[index] = value

    threads = [threading.Thread(target=_one, args=(d,)) for d in devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=_timeout_sec() + 30.0)
    return sum(results.values()) / len(results) if results else 0.0


def probe_and_cache(*, force: bool = False) -> ProbeResult | None:
    """Run both probes on this node and persist the result.

    Compiles and executes, so call it from setup rather than from anything on a
    measurement path. Results are cached per architecture and ROCm version; an
    existing cache short-circuits the whole thing unless *force* is set.

    Bandwidth is measured twice, with one GPU loaded and with all of them, since
    the attainable per-GPU figure depends on how many peers are streaming.

    Args:
        force: Re-probe and overwrite even when a cache entry exists.

    Returns:
        The probe result, or ``None`` when probing is disabled, unsupported, or
        failed at any step.
    """
    if _disabled():
        log.debug("hw_probe: disabled via %s", DISABLE_ENV)
        return None
    arch = detect_arch()
    if not arch:
        log.debug("hw_probe: no AMD GPU architecture detected")
        return None
    rocm = rocm_version()
    if not force:
        cached = load_cached(arch=arch, rocm=rocm)
        if cached is not None:
            return cached

    cache_dir = probe_cache_dir()
    mfma_bin = _compile(MFMA_PROBE_SRC, "mfma_probe", arch, cache_dir)
    if mfma_bin is None:
        return None
    device, rates, sclk = _probe_mfma(mfma_bin)
    if device is None or not rates:
        log.debug("hw_probe: matrix-core probe produced no rates")
        return None

    bandwidth: dict[int, float] = {}
    bw_bin = _compile(BANDWIDTH_PROBE_SRC, "bw_probe", arch, cache_dir)
    if bw_bin is not None:
        count = detect_gpu_count()
        single = _probe_bandwidth_at(bw_bin, [0])
        if single > 0:
            bandwidth[1] = single
        if count > 1:
            allgpu = _probe_bandwidth_at(bw_bin, list(range(count)))
            if allgpu > 0:
                bandwidth[count] = allgpu

    result = ProbeResult(
        schema=_SCHEMA_VERSION,
        device=device,
        rocm_version=rocm,
        mfma_rates=rates,
        bandwidth_gb_per_sec=bandwidth,
        probe_sclk_mhz=sclk,
        probed_at=time.time(),
    )
    _write_cache(result)
    return result


def _write_cache(result: ProbeResult) -> None:
    """Persist a probe result, tolerating an unwritable cache directory.

    Args:
        result: The result to store.
    """
    path = _cache_path(result.device.arch, result.rocm_version)
    payload = {
        "schema": result.schema,
        "device": asdict(result.device),
        "rocm_version": result.rocm_version,
        "mfma_rates": {k: asdict(v) for k, v in result.mfma_rates.items()},
        # JSON object keys are strings; readers coerce back to int.
        "bandwidth_gb_per_sec": {str(k): v for k, v in result.bandwidth_gb_per_sec.items()},
        "probe_sclk_mhz": result.probe_sclk_mhz,
        "probed_at": result.probed_at,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        log.debug("hw_probe: could not write cache %s: %r", path, exc)
    clear_caches()


def clear_caches() -> None:
    """Drop memoized detection and cache reads.

    Needed after a fresh probe writes new results, and by tests that swap the
    cache directory or stub the detection path underneath a warm memo. Tolerates
    either function having been replaced by a plain callable, which is exactly
    what a test that stubs detection does.
    """
    for fn in (detect_arch, _load_cached_impl):
        clear = getattr(fn, "cache_clear", None)
        if callable(clear):
            clear()


def load_cached(*, arch: str | None = None, rocm: str | None = None) -> ProbeResult | None:
    """Read a cached probe result without touching the GPU.

    This is the resolver-facing entry point: it sits behind every ceiling
    computation, so it must stay cheap. The disk read and the architecture
    detection behind it are both memoized -- without that, resolving a ceiling
    would shell out to ``rocm-smi`` and stat the cache every single call.

    Args:
        arch: Architecture to look up; detected when omitted.
        rocm: ROCm version to look up; detected when omitted.

    Returns:
        The cached result, or ``None`` when probing is disabled or no usable
        entry exists.
    """
    if _disabled():
        return None
    resolved_arch = normalize_arch(arch) or detect_arch()
    if not resolved_arch:
        return None
    return _load_cached_impl(resolved_arch, rocm or rocm_version())


@functools.lru_cache(maxsize=8)
def _load_cached_impl(arch: str, rocm: str) -> ProbeResult | None:
    """Memoized cache read for one architecture and toolchain.

    Args:
        arch: Normalized architecture.
        rocm: ROCm version string.

    Returns:
        The cached result, or ``None`` when absent or unusable.
    """
    path = _cache_path(arch, rocm)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA_VERSION:
        return None
    try:
        device = DeviceInfo(**payload["device"])
        rates = {k: MfmaRate(**v) for k, v in (payload.get("mfma_rates") or {}).items()}
        bandwidth = {int(k): float(v) for k, v in (payload.get("bandwidth_gb_per_sec") or {}).items()}
        return ProbeResult(
            schema=int(payload["schema"]),
            device=device,
            rocm_version=str(payload.get("rocm_version", "")),
            mfma_rates=rates,
            bandwidth_gb_per_sec=bandwidth,
            probe_sclk_mhz=float(payload.get("probe_sclk_mhz", 0.0) or 0.0),
            probed_at=float(payload.get("probed_at", 0.0) or 0.0),
        )
    except (KeyError, TypeError, ValueError) as exc:
        log.debug("hw_probe: malformed cache %s: %r", path, exc)
        return None


def expected_arch_for(gpu_type: str | None) -> str:
    """Architecture a GPU type key is expected to run on.

    Args:
        gpu_type: GPU type key such as ``"mi300x"``.

    Returns:
        The normalized architecture, or ``""`` when the key is unknown or
        absent.
    """
    if not gpu_type:
        return ""
    try:
        from hyperloom.inference_optimizer.gpu_types import (  # noqa: PLC0415
            amd_gpu_dispatch_identity,
        )

        identity = amd_gpu_dispatch_identity(gpu_type)
    except Exception:  # noqa: BLE001  (unknown key must not be fatal)
        return ""
    return normalize_arch(identity[0]) if identity else ""


def _probe_for(gpu_type: str | None, result: ProbeResult | None) -> ProbeResult | None:
    """Cached probe result, but only when it describes *gpu_type*.

    A roofline is frequently computed for a GPU that is not the one running the
    code -- comparing against another part, or replaying a recorded session. The
    cache only ever describes the local device, so handing it to a caller asking
    about a different part silently substitutes one part's hardware for
    another's. That is worse than having no measurement at all, since it moves
    the ceiling in an unpredictable direction rather than falling back.

    Args:
        gpu_type: GPU type the caller is asking about; ``None`` skips the check.
        result: Caller-supplied result, returned as-is when present.

    Returns:
        A probe result safe to use for *gpu_type*, or ``None``.
    """
    if result is not None:
        return result
    probe = load_cached()
    if probe is None:
        return None
    expected = expected_arch_for(gpu_type)
    if expected and expected != probe.device.arch:
        log.debug(
            "hw_probe: ignoring %s probe for %s (expects %s)",
            probe.device.arch,
            gpu_type,
            expected,
        )
        return None
    return probe


def probe_compute_peak_tflops(
    precision: str | None,
    *,
    gpu_type: str | None = None,
    sustained_sclk_mhz: float = 0.0,
    result: ProbeResult | None = None,
) -> float | None:
    """Measured compute roof in TFLOPS for one precision.

    Evaluates ``measured_MFMA_rate x CUs x clock``, where the clock is the
    workload's sustained engine clock when telemetry supplied one and the
    device's boost otherwise. Because the rate came from raw MFMA rather than a
    library GEMM, this is a hardware limit that a tuned kernel cannot exceed.

    Args:
        precision: Precision tag (``bf16``, ``fp16``, ``fp8``, ``fp4``).
        gpu_type: GPU the ceiling is being built for. The probe is ignored
            unless it describes this part.
        sustained_sclk_mhz: Measured sustained engine clock; ``0`` falls back to
            the device boost clock.
        result: Pre-loaded probe result; read from cache when omitted.

    Returns:
        The compute roof in TFLOPS, or ``None`` when the precision was never
        probed, the cache describes a different part, or no cache exists.
    """
    probe = _probe_for(gpu_type, result)
    if probe is None:
        return None
    rate = probe.mfma_rates.get((precision or "").strip().lower())
    if rate is None or rate.flops_per_clk_per_cu <= 0:
        return None
    clock = sustained_sclk_mhz if sustained_sclk_mhz > 0 else probe.device.boost_sclk_mhz
    if clock <= 0 or probe.device.cu_count <= 0:
        return None
    return rate.flops_per_clk_per_cu * probe.device.cu_count * clock * 1e6 / 1e12


def probe_hbm_bandwidth_gb_per_sec(
    *,
    gpu_type: str | None = None,
    active_gpus: int = 1,
    result: ProbeResult | None = None,
) -> float | None:
    """Measured per-GPU achievable streaming bandwidth.

    Picks the measurement taken with the closest number of GPUs loaded. On
    MI355X the two measurements agree to within 0.2%, since per-GPU HBM is
    private; the keying exists so a part that *does* share a memory path is
    described correctly rather than assumed away.

    Args:
        gpu_type: GPU the ceiling is being built for. The probe is ignored
            unless it describes this part.
        active_gpus: How many GPUs the workload actually loads.
        result: Pre-loaded probe result; read from cache when omitted.

    Returns:
        Per-GPU GB/s, or ``None`` when bandwidth was never probed or the cache
        describes a different part.
    """
    probe = _probe_for(gpu_type, result)
    if probe is None or not probe.bandwidth_gb_per_sec:
        return None
    target = max(int(active_gpus), 1)
    closest = min(probe.bandwidth_gb_per_sec, key=lambda k: (abs(k - target), k))
    value = probe.bandwidth_gb_per_sec[closest]
    return value if value > 0 else None
