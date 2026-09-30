# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU pin resolution helpers for the GEAK handoff."""

from __future__ import annotations

import os
import shlex
from collections.abc import Mapping
from typing import Any

from hyperloom.common.env import env_float as _env_float, env_int as _env_int
from hyperloom.common.visible_devices import (
    HIP_LEVEL_VARS,
    VISIBLE_DEVICE_VARS,
    effective_mask_tokens,
    is_rocr_level,
    mask_tokens,
    parse_device_list,
)

#: Visible-device env masks, in the repo's ROCm precedence order.
#: The pin-resolution chain, imported rather than re-declared: the same tuple
#: and the same parser had five copies in this repo (``bus/gpu_pool``,
#: ``policy/gate``, ``actions/executors/_ray_serving``, ``common/env_safety``,
#: and ``loop/coordinator_helpers``) and their empty-mask semantics had
#: already drifted apart. ``hyperloom.common.visible_devices`` is now the single
#: definition and is dependency-free, so this pure-helper layer can use it
#: without dragging in the SQLite connection ``gpu_pool`` owns.
#:
#: Note this resolver uses the FULL chain, not the three vars the
#: capacity-counting layers read: it answers "where is this run pinned", and a
#: run pinned with ``HSA_VISIBLE_DEVICES`` or ``GPU_DEVICE_ORDINAL`` is really
#: pinned. Those layers keep their narrower :data:`COUNTING_VISIBLE_DEVICE_VARS`
#: because widening them would change GPU accounting repo-wide.
_VISIBLE_DEVICE_VARS: tuple[str, ...] = VISIBLE_DEVICE_VARS

_mask_tokens = mask_tokens
_parse_device_list = parse_device_list


def _is_autofilled_rocr(*, value: str, recipe_envs: Mapping[str, Any]) -> bool:
    """Is this recipe's ROCR mask the materializer's autofill rather than a pin?

    ``materialize_config_with_envs`` unconditionally writes
    ``ROCR_VISIBLE_DEVICES=0..tp-1`` into ``benchmark.envs`` whenever the mask
    is absent or narrower than TP (``_workload_envs.py``). Every materialized
    recipe therefore carries the key, so a recipe ROCR value that is
    byte-identical to that default carries no information about where the run
    is actually pinned — treating it as a pin is what made this resolver
    override a real ``HIP_VISIBLE_DEVICES`` and re-pin GEAK to cards ``0..tp-1``.

    A hand-authored ``ROCR_VISIBLE_DEVICES: "0,1"`` at ``TP=2`` is
    indistinguishable from the autofill and is also treated as "not a pin";
    that is harmless, because the unpinned path emits the same ``gpu_ids`` and
    merely omits ``gpu_pin``.

    When the recipe carries no usable ``TP`` — a hand-written or pre-clamp YAML
    — there is no width to compare against, so the test falls back to the SHAPE
    the materializer always produces: a mask that is exactly ``0..n-1`` for its
    own length. Returning ``False`` there instead would let the synthetic mask
    pose as a pin for precisely the recipes that never recorded a TP, which is
    the hole this function exists to close.

    Args:
        value: The recipe's ROCR mask, already stripped.
        recipe_envs: The recipe's ``benchmark.envs`` (read for its resolved TP).

    Returns:
        ``True`` when the value equals the ``0..tp-1`` the materializer would
        have synthesized — or, absent a recipe TP, the ``0..n-1`` shape of one.
    """
    tokens = _mask_tokens(value)
    if not tokens:
        return False
    try:
        tp = int(str(recipe_envs.get("TP") or 0))
    except (TypeError, ValueError):
        tp = 0
    if tp <= 0:
        tp = len(tokens)
    return tokens == [str(i) for i in range(tp)]


def _mask_value(raw: Any) -> str:
    """Normalize a raw mask (string or YAML sequence) to its string form.

    A YAML ``ROCR_VISIBLE_DEVICES: [4, 5]`` reaches us as a list, and
    ``str([4, 5])`` would produce ``"[4, 5]"`` — a value no consumer can export.

    Args:
        raw: The value as read from the env mapping or the recipe.

    Returns:
        The comma-joined, stripped mask; ``""`` for an empty or blank one.
    """
    if isinstance(raw, (list, tuple)):
        return ",".join(str(p).strip() for p in raw if str(p).strip())
    return str(raw if raw is not None else "").strip()


def _resolve_inner_hip_mask(
    *,
    var: str,
    env: Mapping[str, str],
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    """The HIP-level mask nested inside a winning ROCr-level pin, if any.

    ``ROCR_VISIBLE_DEVICES=4,5,6,7`` with ``HIP_VISIBLE_DEVICES=2,3`` does not
    mean "cards 2 and 3": HIP indexes INTO what ROCr exposed, so the run is on
    absolute cards 6 and 7. Dropping the inner mask and advertising
    ``0..tp-1`` would move the servers to cards 4 and 5 — a quieter version of
    the same #1312 bug, so the inner mask travels with the pin.

    Args:
        var: The winning mask variable.
        env: Process environment mapping.
        recipe: The baseline recipe's ``benchmark.envs``.

    Returns:
        ``{"var", "value", "ids", "count", "source"}`` for the innermost
        HIP-level mask, or ``{}`` when the winner is not ROCr-level or no
        HIP-level mask is set.
    """
    if not is_rocr_level(var):
        return {}
    for hip_var in HIP_LEVEL_VARS:
        for source, table in (("process_env", env), ("baseline_recipe", recipe)):
            raw = table.get(hip_var)
            if raw is None:
                continue
            value = _mask_value(raw)
            if not value:
                # Set but empty is terminal here too, for the same reason it is
                # in :func:`_resolve_gpu_pin`: ``HIP="" + CUDA=4,5`` exposes
                # zero devices, so the CUDA mask must not be picked up as the
                # inner one. Reported as a zero-device inner mask, which drives
                # the whole pin to ``count == 0``.
                return {"var": hip_var, "value": "", "ids": [], "count": 0, "source": source}
            return {
                "var": hip_var,
                "value": value,
                "ids": _parse_device_list(value),
                "count": len(effective_mask_tokens(value)),
                "source": source,
            }
    return {}


def _resolve_gpu_pin(
    *,
    recipe_envs: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve the run's ACTUAL GPU pin for the geak handoff.

    GEAK launches full servers out-of-process and re-writes a visible-devices
    mask for each one. Without the pin it can only guess, and the guess
    (``0..tp-1``) silently lands on physical GPU 0 — see issue #1312, where a
    run pinned elsewhere collided with a foreign tenant on card 0. Forwarding
    the pin lets the consumer compose masks instead of clobbering them.

    Precedence is VARIABLE-major: ``ROCR_VISIBLE_DEVICES`` before ``HIP``
    before ``CUDA`` — the repo-wide order — and within each variable the
    process env before the baseline recipe. Source-major ordering was wrong in
    both directions: a leftover recipe ``CUDA_VISIBLE_DEVICES`` would outrank a
    real process ROCR pin, and the recipe's autofilled ROCR (see
    :func:`_is_autofilled_rocr`) would outrank everything.

    A mask that is SET BUT EMPTY ends the walk where it stands. It is not a
    weaker pin that a real mask further down can beat: it hides every device,
    and a HIP-level mask indexes into what ROCr left visible rather than
    restoring it. Treating it as a fallback is what let ``ROCR="" +
    HIP=4,5`` report two cards for a run that ROCm refuses to start at all.

    Args:
        recipe_envs: The baseline recipe's ``benchmark.envs`` mapping (may be
            ``None`` when no recipe is materialized yet).
        environ: Environment mapping to read; defaults to ``os.environ``.

    Returns:
        ``{"var", "value", "ids", "count", "source"}`` for the winning mask.
        ``ids`` are the ABSOLUTE NUMERIC device ids and ``count`` is how many
        devices the mask exposes; both derive from
        :func:`effective_mask_tokens`, so ``count >= len(ids)`` always, and
        they differ only when the mask is (partly) non-numeric — a UUID mask
        gives ``ids == []`` with a non-zero ``count``. ``source`` is
        ``"process_env"`` or ``"baseline_recipe"``.
        A mask that is SET BUT EMPTY yields ``count == 0`` (zero devices
        visible) rather than ``{}``, and wins outright over anything below it
        in the chain; when the winner is a ROCr-level mask and a HIP-level mask
        is also in force, the latter travels under ``"inner"`` because it
        selects a subset *within* the ROCr-visible set — and an EMPTY inner
        mask drives the pin's own ``count`` to 0, since it leaves nothing
        usable however many cards the ROCr mask exposes.
        ``{}`` only when no mask is set anywhere — meaning "whole machine
        visible", not "pinned to 0".
    """
    env = os.environ if environ is None else environ
    recipe = dict(recipe_envs or {})
    for var in _VISIBLE_DEVICE_VARS:
        for source, table in (("process_env", env), ("baseline_recipe", recipe)):
            raw = table.get(var)
            if raw is None:
                continue
            value = _mask_value(raw)
            if not value:
                # Present but empty: TERMINAL, not a fallback. An empty mask
                # hides every device, and nothing further down the chain can
                # re-expose one — a HIP-level mask can only index INTO what
                # ROCr left visible. Measured on ROCm 7.2 / MI350X, reading the
                # ROCr agent count out of ``rocminfo`` rather than
                # ``torch.cuda.device_count()`` (which reports a lazy ``1``
                # here and only raises on first use):
                #
                #   ROCR=""                -> 0 agents
                #   ROCR="" + HIP=0        -> 0 agents
                #   ROCR="" + CUDA=4,5     -> 0 agents
                #   ROCR="" + HSA=4,5      -> 0 agents
                #   HIP=""  + CUDA=4,5     -> 0 usable devices
                #
                # ``ROCR="" + HIP=4,5`` does not even reach a device count: HIP
                # aborts with "HIP_VISIBLE_DEVICES contains more devices than
                # ROCR_VISIBLE_DEVICES". Letting the HIP mask win here put two
                # cards that cannot exist into the handoff, and the consumer
                # died on that abort at server start.
                return {"var": var, "value": "", "ids": [], "count": 0, "source": source}
            if (
                source == "baseline_recipe"
                and is_rocr_level(var)
                and _is_autofilled_rocr(value=value, recipe_envs=recipe)
            ):
                continue
            pin: dict[str, Any] = {
                "var": var,
                "value": value,
                "ids": _parse_device_list(value),
                "count": len(effective_mask_tokens(value)),
                "source": source,
            }
            inner = _resolve_inner_hip_mask(var=var, env=env, recipe=recipe)
            if inner:
                pin["inner"] = inner
                if int(inner.get("count") or 0) <= 0:
                    # An empty HIP mask nested in a ROCr pin still leaves the
                    # run with nothing usable: ``ROCR=4,5 + HIP=""`` keeps two
                    # ROCr agents but exposes zero devices to HIP (measured, as
                    # above). The pin keeps the ROCr mask as its ``value`` for
                    # diagnostics, but its device count is the effective one, so
                    # the handoff reports the coordinate space as ``"none"``
                    # instead of advertising the two cards ROCr still shows.
                    pin["count"] = 0
            return pin
    return {}


def _resolve_handoff_gpu_ids(*, gpu_pin: Mapping[str, Any] | None, tp: int) -> str:
    """Resolve the handoff's ``gpu_ids`` in the coordinate system GEAK applies it in.

    ``gpu_ids`` is a HIP-level device list: the consumer exports it as
    ``HIP_VISIBLE_DEVICES``/``CUDA_VISIBLE_DEVICES`` for the servers it
    launches, and HIP indexes into the ROCr-visible set. So:

      * pinned with a ROCr-level mask, from the process env or from the
        baseline recipe — either way it is in force for the servers GEAK
        launches and renumbers their devices, so the ids must be LOGICAL
        positions inside it
        (``ROCR=6`` → ``"0"``), capped at ``tp`` (``ROCR=4,5,6,7`` with
        ``tp=2`` → ``"0,1"``) and at the mask width when ``tp`` overshoots it.
        Counted from :func:`effective_mask_tokens`, so a UUID mask resolves to
        the right number of logical slots and a repeated ordinal does not
        invent one. A HIP-level mask nested inside the ROCr slice is already in
        logical coordinates and is forwarded instead (``ROCR=4,5,6,7`` +
        ``HIP=2,3`` is cards 6 and 7, so ``"2,3"``);
      * any other pin — ROCr still shows every card, so the mask's own tokens
        pass through uncapped (``HIP=4,5`` → ``"4,5"``). They come from
        :func:`effective_mask_tokens`, the same list ``gpu_pin["count"]`` is
        derived from, so whitespace is normalized without the id list and the
        advertised device count ever disagreeing. A NON-NUMERIC mask (a UUID
        list) is forwarded token for token rather than collapsed to
        ``0..tp-1``, which would silently move the servers onto cards
        ``0..tp-1`` — the #1312 failure this resolver exists to prevent;
      * not pinned — ``0..tp-1``, unchanged.

    The absolute pin travels separately in ``handoff["gpu_pin"]``, and
    :func:`_resolve_handoff_gpu_ids_space` says which of the two coordinate
    systems the result is in. Because EVERY ROCr-level pin now yields logical
    ids, both consumer styles agree: exporting the result as
    ``HIP_VISIBLE_DEVICES`` is correct, and so is re-applying
    ``gpu_pin["value"]`` as ``ROCR_VISIBLE_DEVICES`` and then these ids as the
    inner HIP mask. No case is left in which the consumer has to switch which
    field it reads.

    Args:
        gpu_pin: The :func:`_resolve_gpu_pin` result (``{}``/``None`` = unpinned).
        tp: Tensor-parallel size; ``<= 1`` is treated as 1.

    Returns:
        A comma-separated device list, never empty.
    """
    width = max(int(tp or 1), 1)
    pin = gpu_pin or {}
    ids = list(pin.get("ids") or [])
    # Logical remapping applies to any ROCr-level pin, from either source: a
    # process-env mask reaches the servers through GEAK, and a recipe mask
    # reaches them directly as ``handoff["launch_recipe"]``. Either way the
    # servers see a renumbered set, so absolute ids would index out of it.
    if _pin_renumbers_devices(pin):
        # Token count, not len(ids): a UUID mask parses to zero numeric ids but
        # still exposes that many cards to the child. Defaulted rather than
        # ``or``-chained, so an explicit ``count == 0`` (an empty mask, or an
        # empty HIP mask nested in this pin) stays zero instead of falling back
        # to the ids of a mask that exposes nothing.
        visible = int(pin.get("count", len(ids)))
        if visible > 0:
            # A HIP-level mask nested inside the ROCr pin is ALREADY expressed
            # in the child's logical coordinates, so it is forwarded as-is
            # rather than overwritten with ``0..n-1``. Out-of-range entries are
            # dropped: they name devices the ROCr mask never exposed.
            # Effective, not literal: ``-1`` names no device and a repeated
            # ordinal is not a second one, and either would otherwise travel
            # into ``gpu_ids`` and inflate the ``tp`` derived from it.
            inner = effective_mask_tokens((pin.get("inner") or {}).get("value"))
            kept = [tok for tok in inner if not tok.isdigit() or int(tok) < visible]
            if kept:
                return ",".join(kept[:width])
            return ",".join(str(i) for i in range(min(visible, width)))
    # Forward the EFFECTIVE tokens, not a re-serialization of the parsed ints:
    # a UUID mask has no ints to re-serialize and would otherwise collapse to
    # ``0..tp-1`` (the #1312 failure), and ``pin["count"]`` is derived from this
    # same list, so the id list and the advertised device count cannot disagree.
    tokens = effective_mask_tokens(pin.get("value"))
    if tokens:
        return ",".join(tokens)
    return ",".join(str(i) for i in range(width))


def _pin_renumbers_devices(pin: Mapping[str, Any] | None) -> bool:
    """Will this pin's ROCr slice be in force for the servers GEAK launches?

    Only then are the handoff's ``gpu_ids`` logical -- and the question is about
    the SERVERS, not about the GEAK process. An earlier version asked whether
    GEAK itself inherits the mask (``source == "process_env"``), which is true
    of the process env and false of the recipe. That was the wrong level: GEAK
    starts its servers from ``handoff["launch_recipe"]``, and a recipe-sourced
    ``ROCR_VISIBLE_DEVICES`` is applied to exactly those servers. The
    renumbering still happens, one level down, so calling those ids absolute
    made a mask index out of its own slice -- ``ROCR=4,5,6,7`` re-exported as
    ``HIP=4,5,6,7`` indexes 4..7 into a four-element set and the server dies on
    an invalid ordinal. Both sources renumber; only the LEVEL of the mask
    decides.

    Args:
        pin: The :func:`_resolve_gpu_pin` result.

    Returns:
        ``True`` for any ROCr-level pin, whatever its source.
    """
    return is_rocr_level(str((pin or {}).get("var") or ""))


def _resolve_handoff_gpu_ids_space(*, gpu_pin: Mapping[str, Any] | None) -> str:
    """Which coordinate system the handoff's ``gpu_ids`` are expressed in.

    ``gpu_ids`` alone is ambiguous: ``"0,1"`` is either "the first two cards of
    the in-force ROCr mask" or "absolute cards 0 and 1", and a consumer that
    guesses wrong re-pins the servers onto physical GPU 0 — issue #1312. This
    field makes the distinction explicit so a consumer that composes masks
    itself (rather than exporting ``gpu_ids`` into HIP) can tell which it was
    handed. Consumers that ignore it keep the old, correct behaviour of
    exporting ``gpu_ids`` as ``HIP_VISIBLE_DEVICES``, which is a HIP-level
    variable in both spaces.

    ``"none"`` is the third case and the reason this is a tri-state rather
    than a boolean: the mask is SET BUT EMPTY, so the run has no visible
    devices and NO id list can be truthful. ``gpu_ids`` still carries
    ``0..tp-1`` because the consumer reads a falsy ``gpu_ids`` as "unset" and
    falls back to exactly those ids anyway (``interface/run_e2e.py``) — an
    empty string would buy nothing and lose the ability to say why. The ids are
    placeholders in that case and a consumer must not launch on them.

    Args:
        gpu_pin: The :func:`_resolve_gpu_pin` result (``{}``/``None`` = unpinned).

    Returns:
        ``"none"`` when the pin exposes zero devices, ``"logical"`` when the
        ids index into a ROCr mask that is in force for the launched servers,
        ``"absolute"`` otherwise (including unpinned).
    """
    pin = gpu_pin or {}
    if pin and int(pin.get("count") or 0) <= 0:
        return "none"
    return "logical" if _pin_renumbers_devices(pin) else "absolute"


def _coerce_tp(*args: Any, default: int = 1) -> int:
    """First positional that parses as a positive int, else ``default``.

    Every candidate is guarded, so no caller has to wrap ``int()`` in a
    ``try`` whose handler then calls ``int()`` again on a value that can raise
    the same exception it is handling.

    Args:
        *args: Candidate TP values in precedence order (``None``/blank skipped).
        default: Returned when nothing parses; floored at 1.

    Returns:
        A TP of at least 1.
    """
    for cand in args:
        text = str(cand if cand is not None else "").strip()
        if not text:
            continue
        try:
            val = int(text)
        except (TypeError, ValueError):
            continue
        if val > 0:
            return val
    return max(int(default), 1)


def _resolve_handoff_tp(*, gpu_ids: str, tp: int) -> int:
    """Clamp ``tp`` to the number of devices the handoff actually advertises.

    ``gpu_ids`` is capped at the pin's mask width, so a run whose ``$TP``
    overshoots its pin (``ROCR=6`` with ``TP=2``, or a stale ``TP=8`` against a
    materializer-clamped 4-card recipe) would otherwise ship ``tp`` and
    ``gpu_ids`` that disagree — and GEAK would launch ``--tp N`` against fewer
    visible cards and fail to load weights. Deriving both from the same resolved
    mask makes that state unrepresentable.

    Args:
        gpu_ids: The resolved handoff ``gpu_ids`` string.
        tp: The TP resolved from the recipe/process env.

    Returns:
        ``min(tp, len(gpu_ids))``, never below 1.
    """
    advertised = len(_mask_tokens(gpu_ids))
    if advertised <= 0:
        return max(int(tp or 1), 1)
    return max(min(int(tp or 1), advertised), 1)


def _parse_server_arg_value(server_args: str, flag: str) -> str | None:
    """Extract a CLI flag's value from a server-args string."""
    if not server_args or not flag:
        return None
    try:
        toks = shlex.split(server_args)
    except ValueError:
        toks = server_args.split()
    prefix = flag + "="
    for i, tok in enumerate(toks):
        if tok == flag:
            return toks[i + 1] if i + 1 < len(toks) else None
        if tok.startswith(prefix):
            return tok[len(prefix) :]
    return None


def _resolve_serving_fidelity(
    *,
    baseline_server_args: str,
    state_max_model_len: int = 0,
) -> dict[str, Any]:
    """Resolve serving-fidelity knobs to forward in the geak handoff."""
    out: dict[str, Any] = {}

    mml = int(state_max_model_len or 0)
    if mml <= 0:
        v = _parse_server_arg_value(baseline_server_args, "--max-model-len")
        try:
            mml = int(v) if v else 0
        except (TypeError, ValueError):
            mml = 0
    if mml <= 0:
        mml = _env_int("MAX_MODEL_LEN", default=0)
    if mml > 0:
        out["max_model_len"] = mml

    v = _parse_server_arg_value(baseline_server_args, "--gpu-memory-utilization")
    try:
        mem = float(v) if v else 0.0
    except (TypeError, ValueError):
        mem = 0.0
    if mem <= 0:
        mem = _env_float("GPU_MEMORY_UTILIZATION", default=0.0)
    if mem > 0:
        out["mem_fraction"] = mem

    return out
