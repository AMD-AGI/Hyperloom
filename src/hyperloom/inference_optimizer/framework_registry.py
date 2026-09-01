# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Single source of truth for inference-framework capabilities.

A *framework* is one serving/execution backend a session optimizes (see the
``FRAMEWORKS`` table below for the current set). This module centralizes the
allowed set and per-framework behavior (extra-args env name, repo URL,
server-reuse eligibility) so adding a framework is a single-table edit.

It also introduces ``kind`` to split the two execution models the loop must
support:

* ``serving``    — a long-lived OpenAI-compatible server benchmarked by a
  client (``benchmark_serving.py``); throughput is tokens/sec; accuracy is a
  numeric eval (GSM8K). This is sglang/vllm/atom.
* ``scriptable`` — a server-less single-command workload (e.g. xDiT diffusion)
  whose bench script writes a ``benchmark_report.json`` directly; throughput is
  images/sec; "accuracy" is an image-quality gate (LPIPS/SSIM/MSE). Server
  reuse, the ``benchmark_serving`` client, and the GSM8K gate do not apply.

The module has no third-party imports so it is safe to import from the CLI,
the executors, and standalone tools alike.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameworkSpec:
    """Static capabilities of one inference framework.

    Attributes:
        name: Canonical lowercase framework name.
        kind: Execution model — ``"serving"`` or ``"scriptable"``.
        extra_args_env: Env var Magpie scripts expand to append backend args.
        repo_url: Canonical upstream git URL (FRAMEWORK), or ``None``.
        supports_server_reuse: Whether the Magpie ``server_lifecycle`` reuse
            protocol applies (always ``False`` for ``scriptable``).
        throughput_unit: Human-readable throughput unit for reports.
        has_denoiser_config: Whether a session's model directory carries a
            diffusers denoiser config Hyperloom can read. This is what the
            analytic diffusion ceiling needs, and it is not implied by ``kind``:
            ``custom`` is scriptable but its model arrives from the operator, so
            nothing about it can be read ahead of the run.
        magpie_script_fmt: Format string for the Magpie benchmark script name,
            resolved with ``framework`` and ``runner_type``.
    """

    name: str
    kind: str
    extra_args_env: str
    repo_url: str | None
    supports_server_reuse: bool
    throughput_unit: str
    has_denoiser_config: bool = False
    magpie_script_fmt: str = "{framework}_{runner_type}.sh"


SERVING = "serving"
SCRIPTABLE = "scriptable"


# The single registry (one entry per framework). Order is the canonical order
# surfaced in CLI help.
FRAMEWORKS: dict[str, FrameworkSpec] = {
    "sglang": FrameworkSpec(
        name="sglang",
        kind=SERVING,
        extra_args_env="EXTRA_SGLANG_ARGS",
        repo_url="https://github.com/sgl-project/sglang.git",
        supports_server_reuse=True,
        throughput_unit="tok/s",
    ),
    "vllm": FrameworkSpec(
        name="vllm",
        kind=SERVING,
        extra_args_env="EXTRA_VLLM_ARGS",
        repo_url="https://github.com/ROCm/vllm.git",
        supports_server_reuse=True,
        throughput_unit="tok/s",
    ),
    "atom": FrameworkSpec(
        name="atom",
        kind=SERVING,
        extra_args_env="EXTRA_ATOM_ARGS",
        repo_url="https://github.com/ROCm/ATOM.git",
        supports_server_reuse=False,
        throughput_unit="tok/s",
    ),
    "xdit": FrameworkSpec(
        name="xdit",
        kind=SCRIPTABLE,
        extra_args_env="EXTRA_XDIT_ARGS",
        repo_url="https://github.com/xdit-project/xDiT.git",
        supports_server_reuse=False,
        throughput_unit="img/s",
        # A diffusers pipeline: transformer/ + vae/ configs are on disk.
        has_denoiser_config=True,
    ),
    # An operator's own workload. Everything the entries above hardcode — the
    # checkout, the entrypoint, the knobs the script reads — arrives at launch
    # instead: ``--framework-path`` and ``--benchmark-scripts-dir`` (or their
    # env forms), plus ``--extra-env`` for whatever the script itself reads.
    # There is no upstream repo to discover PRs from, and the throughput unit
    # is deliberately neutral: only the operator's own report knows whether the
    # number it produced counts frames, images or anything else.
    "custom": FrameworkSpec(
        name="custom",
        kind=SCRIPTABLE,
        extra_args_env="EXTRA_CUSTOM_ARGS",
        repo_url=None,
        supports_server_reuse=False,
        throughput_unit="unit/s",
    ),
}

DEFAULT_FRAMEWORK = "sglang"


def names() -> tuple[str, ...]:
    """Return the canonical tuple of supported framework names.

    Returns:
        tuple[str, ...]: All registered framework names, in registry order.
    """
    return tuple(FRAMEWORKS)


def is_supported(framework: str | None) -> bool:
    """Return whether ``framework`` is a registered framework.

    Args:
        framework (str | None): Candidate name; matched case-insensitively.

    Returns:
        bool: ``True`` when the name is registered.
    """
    return str(framework or "").strip().lower() in FRAMEWORKS


def _spec_or_default(framework: str | None) -> FrameworkSpec:
    """Return the spec for ``framework`` or the default's spec when unknown.

    Args:
        framework (str | None): Framework name; matched case-insensitively.

    Returns:
        FrameworkSpec: The matching spec, or the default framework's spec.
    """
    key = str(framework or "").strip().lower()
    return FRAMEWORKS.get(key, FRAMEWORKS[DEFAULT_FRAMEWORK])


def is_scriptable(framework: str | None) -> bool:
    """Return whether ``framework`` is a server-less scriptable workload.

    Args:
        framework (str | None): Framework name; matched case-insensitively.

    Returns:
        bool: ``True`` for ``scriptable`` frameworks (e.g. xDiT).
    """
    return _spec_or_default(framework).kind == SCRIPTABLE


def has_denoiser_config(framework: str | None) -> bool:
    """Return whether ``framework``'s model can be read as a diffusers denoiser.

    The predicate the analytic diffusion ceiling needs. Deliberately separate
    from :func:`is_scriptable`: ``custom`` is scriptable yet its model arrives
    from the operator at launch, so no geometry can be resolved ahead of the run
    and a guessed one would be worse than none.

    Args:
        framework (str | None): Framework name; matched case-insensitively.

    Returns:
        bool: ``True`` only for frameworks shipping a readable denoiser config.
    """
    return _spec_or_default(framework).has_denoiser_config


def extra_args_env(framework: str | None) -> str:
    """Return the Magpie env var used to append backend args.

    Args:
        framework (str | None): Framework name; matched case-insensitively.

    Returns:
        str: The ``EXTRA_*_ARGS`` env name (default framework's when unknown).
    """
    return _spec_or_default(framework).extra_args_env


def server_args_env_name(framework: str | None) -> str:
    """Return the Magpie env var used to append backend server args.

    Resolution is exact (registry-keyed) with a substring fallback so a
    framework string carrying a version suffix (e.g. ``"vllm@0.21"``) still
    maps correctly. Unknown names fall back to the default framework's env.

    Args:
        framework (str | None): Framework name; matched case-insensitively.

    Returns:
        str: The ``EXTRA_*_ARGS`` env name for the framework (e.g.
        ``"EXTRA_XDIT_ARGS"`` for xDiT, ``"EXTRA_SGLANG_ARGS"`` default).
    """
    name = str(framework or "").strip().lower()
    if is_supported(name):
        return extra_args_env(name)
    for fw in names():
        if fw in name:
            return extra_args_env(fw)
    return extra_args_env(DEFAULT_FRAMEWORK)


def throughput_unit(framework: str | None) -> str:
    """Return the throughput unit string for ``framework``.

    Args:
        framework (str | None): Framework name; matched case-insensitively.

    Returns:
        str: The unit the entry declares, e.g. ``"tok/s"`` (serving),
        ``"img/s"`` (xDiT) or ``"unit/s"`` (``custom``, deliberately neutral).
    """
    return _spec_or_default(framework).throughput_unit


def primary_metric_unit(framework: str | None) -> str:
    """Return the human-readable unit for a session's primary display metric.

    Serving frameworks display token throughput (``tok/s/GPU``). Scriptable
    image frameworks (e.g. xDiT) display the per-image end-to-end latency
    ``e2el_mean_ms`` in milliseconds instead of a reciprocal-of-latency value.

    Args:
        framework (str | None): Framework name; matched case-insensitively.

    Returns:
        str: ``"ms"`` for scriptable xDiT, else ``"tok/s/GPU"``.
    """
    return "ms" if is_scriptable(framework) else "tok/s/GPU"


def primary_metric_name(framework: str | None) -> str:
    """Return the state field name holding a session's primary result metric.

    Serving frameworks are ranked by token throughput
    (``throughput_tok_s_per_gpu``); scriptable image frameworks (e.g. xDiT) are
    ranked by per-image end-to-end latency (``e2el_mean_ms``). Consumers use
    this to pick the correct headline/result field per framework.

    Args:
        framework (str | None): Framework name; matched case-insensitively.

    Returns:
        str: ``"e2el_mean_ms"`` for scriptable xDiT, else
        ``"throughput_tok_s_per_gpu"``.
    """
    return "e2el_mean_ms" if is_scriptable(framework) else "throughput_tok_s_per_gpu"


def primary_metric_value(framework: str | None, tput_per_gpu: float | int | None) -> float | None:
    """Convert stored per-GPU throughput into the value shown for ``framework``.

    Scriptable image frameworks (xDiT) store throughput as ``img/s``
    (``1 / latency``); the displayed metric is the equivalent per-image latency
    ``e2el_mean_ms = 1000 / img_per_s``. Serving frameworks display the stored
    throughput unchanged.

    Args:
        framework (str | None): Framework name; matched case-insensitively.
        tput_per_gpu (float | int | None): Per-GPU throughput as stored in
            state (``tok/s`` for serving, ``img/s`` for scriptable xDiT).

    Returns:
        float | None: The value to display, or ``None`` when it is undefined
        (non-positive throughput for a scriptable framework).
    """
    tput = float(tput_per_gpu or 0.0)
    if is_scriptable(framework):
        return (1000.0 / tput) if tput > 0 else None
    return tput


def format_primary_metric(framework: str | None, tput_per_gpu: float | int | None, *, precision: int = 1) -> str:
    """Format a session's primary performance metric for human-readable display.

    Serving frameworks report token throughput, so the value is shown as
    ``"<tput> tok/s/GPU"``. Scriptable image frameworks (e.g. xDiT) are
    server-less and measure a single-stream, per-image end-to-end latency; their
    ``output_throughput`` is merely ``1 / latency`` (img/s). Rendering that
    reciprocal as ``tok/s/GPU`` is misleading (there is no comparable token
    stream), so instead surface the equivalent per-image latency
    ``e2el_mean_ms`` — derived exactly as ``1000 / img_per_s`` — in
    milliseconds.

    Args:
        framework (str | None): Session framework name; matched
            case-insensitively.
        tput_per_gpu (float | int | None): Per-GPU throughput as stored in
            state (``tok/s`` for serving, ``img/s`` for scriptable xDiT).
        precision (int): Number of decimals to render (default 1).

    Returns:
        str: A display string such as ``"123.4 tok/s/GPU"`` (serving) or
        ``"6440.0 ms"`` (xDiT ``e2el_mean_ms``). Returns ``"n/a ms"`` /
        ``"0.0 tok/s/GPU"`` for non-positive inputs.
    """
    unit = primary_metric_unit(framework)
    value = primary_metric_value(framework, tput_per_gpu)
    if value is None:
        return f"n/a {unit}"
    return f"{value:.{precision}f} {unit}"
