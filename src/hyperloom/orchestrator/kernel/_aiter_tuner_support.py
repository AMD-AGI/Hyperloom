# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What aiter's CK tuners can actually take: supported dtype pairs, M lookup, MoE inputs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from hyperloom.common.env import env_bool

log = logging.getLogger(__name__)


# Forge tuner families whose deliverable is an aiter tuned-GEMM CSV, i.e. the
# ones whose rows are resolved through aiter's padded (M, N, K) lookup.
_AITER_CSV_TUNER_FRAMEWORKS = ("sglang", "vllm-aiter")

#: Wall-clock the aiter CK tuner needs per shape, measured on gfx950 at
#: ``--mp 1`` (12 shapes / 1462s in production, ~135s each). Used to size the
#: shape budget so a wider ladder cannot push the tuner past its timeout and
#: return nothing at all.
_AITER_TUNE_SEC_PER_SHAPE = 150


def _align_forge_shapes_for_aiter(
    shapes_json: str,
    *,
    forge_framework: str,
    workspace: Path,
    budget_sec: int = 0,
    mp: int = 1,
) -> tuple[str, dict[str, Any] | None]:
    """Re-key profiled GEMM shapes onto the M values aiter actually looks up.

    Captured shapes carry the raw runtime M, which for prefill is the
    data-dependent scheduled-token count and so never recurs between runs. aiter
    resolves a tuned row by trying the raw M and then two padded M variants, so a
    CSV keyed on raw M is unreachable and the tuner's micro win never reaches the
    server. Padding the shapes first makes each tuned row serve the whole bucket
    that pads onto it.

    Returns the shapes-JSON path to hand forge plus an alignment report, or the
    input path and ``None`` when alignment does not apply.
    """
    if forge_framework not in _AITER_CSV_TUNER_FRAMEWORKS:
        return shapes_json, None
    if not env_bool("HYPERLOOM_GEMM_ALIGN_SHAPES", True):
        return shapes_json, None

    from .gemm_shape_coverage import align_shapes_to_aiter_keys, load_shapes_json, write_shapes_json

    observed = load_shapes_json(shapes_json)
    if not observed:
        return shapes_json, None
    try:
        max_shapes = int(os.environ.get("HYPERLOOM_GEMM_ALIGN_MAX_SHAPES") or 64)
    except (TypeError, ValueError):
        max_shapes = 64
    max_shapes = max(1, max_shapes)
    if budget_sec > 0:
        try:
            per_shape = int(os.environ.get("HYPERLOOM_GEMM_TUNE_SEC_PER_SHAPE") or _AITER_TUNE_SEC_PER_SHAPE)
        except (TypeError, ValueError):
            per_shape = _AITER_TUNE_SEC_PER_SHAPE
        # Reserve a third of the window for JIT builds and the report step.
        affordable = int(budget_sec * 0.66 * max(1, mp) // max(1, per_shape))
        max_shapes = min(max_shapes, max(len(observed), affordable))
    aligned, report = align_shapes_to_aiter_keys(observed, max_shapes=max_shapes)
    if not aligned or report.get("unchanged"):
        return shapes_json, {**report, "applied": False, "source_shapes_json": shapes_json}
    try:
        out = write_shapes_json(aligned, workspace / "forge_shapes.aiter_aligned.json")
    except OSError:
        return shapes_json, {**report, "applied": False, "source_shapes_json": shapes_json}
    log.info(
        "Forge GEMM shapes: re-keyed %d observed shape(s) onto %d aiter lookup key(s) (observed M=%s -> aligned M=%s)",
        report.get("observed"),
        report.get("aligned"),
        report.get("observed_m"),
        report.get("aligned_m"),
    )
    return out, {**report, "applied": True, "source_shapes_json": shapes_json}


#: Which dtypes fall in each of aiter's width buckets. Mirrors ``bit16_list`` /
#: ``bit8_list`` / ``bit4_list`` in
#: ``csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py``.
_AITER_BIT16_DTYPES = frozenset({"bfloat16", "float16"})

_AITER_BIT8_DTYPES = frozenset({"float8_e4m3fn", "float8_e4m3fnuz", "int8"})

_AITER_BIT4_DTYPES = frozenset({"float4_e2m1fn_x2", "uint32", "int4"})


def _aiter_moe_dtype_pair_supported(q_dtype_a: str, q_dtype_w: str) -> bool:
    """Return whether aiter's CK MoE codegen has a kernel family for this pair.

    ``get_gemm1_kernels_list`` / ``get_gemm2_kernels_list`` pick a family from the
    activation/weight widths and raise ``Unsupported data type combination`` for
    anything else. Notably a BF16 activation against FP4 weights -- which the
    serving path runs happily -- matches no family, so handing it to the tuner
    trades a silent no-op for a hard error.
    """
    act = q_dtype_a.replace("torch.", "")
    weight = q_dtype_w.replace("torch.", "")
    if act in _AITER_BIT16_DTYPES and weight in _AITER_BIT16_DTYPES:
        return True
    if act in _AITER_BIT8_DTYPES and weight in _AITER_BIT8_DTYPES:
        return True
    # The a8w4 family is FP8-only on the activation side; INT8 does not qualify.
    if act.startswith("float8") and weight in _AITER_BIT4_DTYPES:
        return True
    return act in _AITER_BIT4_DTYPES and weight in _AITER_BIT4_DTYPES


def _aiter_fused_moe_dispatch_keys(server_log: str) -> list[dict[str, Any]]:
    """Return the distinct MoE problems a server log shows aiter dispatching.

    Deduplicated on everything but the token count, most-missed first. One model
    routinely yields several problems -- the same checkpoint dispatches both a
    BF16-activation and an FP8-activation variant, and the EP path appends a
    masked fake-expert slot so ``expert``/``topk`` arrive one higher than the
    model config states. Neither is derivable from the config, which is why the
    log is the authoritative source for what to tune.
    """
    if not server_log:
        return []
    from kernelforge.gemm_tune import moe_dispatch_keys, parse_log_file

    return moe_dispatch_keys(parse_log_file(server_log))


def _aiter_ck_moe_tuner_supports(server_log: str) -> bool:
    """Return whether aiter's CK MoE tuner can tune anything the server dispatched.

    The tuner builds its kernel candidates from the activation/weight dtype pair
    and rejects some combinations the serving path happily runs. Measured on
    gpt-oss-120b at TP=1, a BF16-activation / FP4-weight MoE (the
    ``AITER_MXFP4_BF16`` backend) benchmarks fine but fails candidate generation
    with ``Unsupported data type combination: b16, fp4x2``, so routing it to
    ``fmoe_ck`` would only trade silent no-op for a hard tuner error.

    A single checkpoint can dispatch several dtype pairs at once, so this asks
    whether *any* of them is tunable; per-problem filtering happens where the
    tuning input is written.
    """
    if not server_log:
        return False
    keys = _aiter_fused_moe_dispatch_keys(server_log)
    if not keys:
        # MoE evidence without a parseable problem tuple: let Forge decide.
        return True
    return any(_aiter_moe_dtype_pair_supported(key["q_dtype_a"], key["q_dtype_w"]) for key in keys)


def _fmoe_token_list(tokens: Any) -> list[int]:
    """Positive token counts to sweep, keyed off whatever the caller sends.

    Accepts forge's comma-separated string (what :func:`_normalize_tokens`
    produces) or a sequence. Unparseable and non-positive entries are dropped
    rather than raising -- one bad entry is not worth the run -- and ``[1]`` is
    the floor so there is always a token to key on.
    """
    if isinstance(tokens, str):
        raw: list[str] = [part.strip() for part in tokens.split(",")]
    elif isinstance(tokens, (list, tuple, set, frozenset)):
        raw = [str(item).strip() for item in tokens]
    elif tokens is None:
        raw = []
    else:
        raw = [str(tokens).strip()]

    out: set[int] = set()
    for item in raw:
        if not item:
            continue
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value > 0:
            out.add(value)
    return sorted(out) or [1]


def _write_fmoe_untuned_csv_from_log(
    server_log: str,
    tokens: Any,
    workspace: Path,
) -> tuple[str, dict[str, Any]]:
    """Turn the MoE problems observed in ``server_log`` into a tuning input CSV.

    Returns ``(csv_path, report)``; ``csv_path`` is "" when nothing tunable was
    observed. Writing the observed tuple verbatim is the whole point: the
    quantisation pair, the per-partition ``inter_dim`` and the EP-inflated
    expert/topk counts are all properties of what the serving framework chose,
    and every attempt to re-derive them from the model config is a guess that has
    already produced tables no runtime lookup could reach.

    Problems whose dtype pair aiter's codegen rejects are dropped rather than
    passed through, because one unsupported row aborts the whole tuner run.
    """
    from kernelforge.gemm_tune import MOE_SHAPE_FIELDS, moe_untuned_csv_text

    report: dict[str, Any] = {
        "observed": 0,
        "tunable": 0,
        "dropped_unsupported": [],
        "keys": [],
    }
    keys = _aiter_fused_moe_dispatch_keys(server_log)
    report["observed"] = len(keys)
    if not keys:
        return "", report

    tunable: list[dict[str, Any]] = []
    for key in keys:
        pair = (key["q_dtype_a"], key["q_dtype_w"])
        if _aiter_moe_dtype_pair_supported(*pair):
            tunable.append(key)
            report["keys"].append({name: key[name] for name in MOE_SHAPE_FIELDS})
        else:
            combo = f"{pair[0]}/{pair[1]}"
            if combo not in report["dropped_unsupported"]:
                report["dropped_unsupported"].append(combo)
    report["tunable"] = len(tunable)
    if not tunable:
        return "", report

    token_list = _fmoe_token_list(tokens)
    rendered = [moe_untuned_csv_text(key, tokens=token_list).splitlines() for key in tunable]
    lines = rendered[0][:1] + [row for key_lines in rendered for row in key_lines[1:]]

    csv_path = workspace / "untuned_fmoe_from_runtime.csv"
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        # A full disk or a read-only workspace must cost the MoE tuner its input,
        # not the whole tuning run: the dense tuners take their shapes from
        # elsewhere and can still produce something useful.
        report["write_error"] = f"{type(exc).__name__}: {exc}"
        log.warning("Forge GEMM shapes: cannot write %s: %s", csv_path, exc)
        return "", report
    log.info(
        "Forge GEMM shapes: derived %d MoE problem(s) x %d token(s) from %s%s",
        len(tunable),
        len(token_list),
        server_log,
        (f"; dropped {report['dropped_unsupported']} as untunable by aiter" if report["dropped_unsupported"] else ""),
    )
    return str(csv_path), report
