# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Workload renderer — model / framework / GPU / shape / objective."""

from __future__ import annotations

from typing import Any

from ..base import RenderedSection, md_kv_list, register_renderer


@register_renderer("workload")
def render(breakdown: dict[str, Any]) -> RenderedSection:
    """Render the workload section: model / framework / GPU / shape / objective.

    Surfaces the model and framework identity, GPU type, request shape
    (tp / conc / isl / osl / max_model_len / precision) and objective,
    warning when the GPU type is missing. Skipped when neither model nor
    framework is present.

    Args:
        breakdown (dict[str, Any]): The full ``session_breakdown.json`` dict.

    Returns:
        RenderedSection: The rendered workload section.
    """
    w = breakdown.get("workload") or {}
    facts: list[str] = []
    warnings: list[str] = []

    model = w.get("model_name") or ""
    fw = w.get("framework") or ""
    fw_v = w.get("framework_version") or ""
    gpu = w.get("gpu_type") or ""
    tp = w.get("tp")
    conc = w.get("conc")
    isl = w.get("isl")
    osl = w.get("osl")
    mml = w.get("max_model_len")
    prec = w.get("precision") or ""
    obj = w.get("objective") or {}

    if model:
        facts.append(f"Model: `{model}` (path={w.get('model_path') or '?'}).")
    if fw:
        facts.append(f"Framework: `{fw}` {f'(v={fw_v})' if fw_v else ''}".strip())
    if gpu:
        facts.append(f"GPU: `{gpu}`.")
    else:
        warnings.append(
            "gpu_type missing — manifest did not stamp GPU_TYPE; downstream "
            "dashboards may show empty hardware fields."
        )
    facts.append(
        f"Shape: tp={tp}, conc={conc}, isl={isl}, osl={osl}, max_model_len={mml}, "
        f"precision={prec or '?'}."
    )
    if obj:
        facts.append(
            f"Objective: {obj.get('kind') or '?'}={obj.get('value')}."
        )

    md = md_kv_list([
        ("model_name",      model),
        ("model_path",      w.get("model_path")),
        ("model_class",     w.get("model_class")),
        ("framework",       fw),
        ("framework_version", fw_v or None),
        ("gpu_type",        gpu or None),
        ("tp",              tp),
        ("conc",            conc),
        ("isl",             isl),
        ("osl",             osl),
        ("max_model_len",   mml),
        ("precision",       prec or None),
        ("objective",       obj or None),
    ])
    return RenderedSection(
        section_id="workload",
        title="Workload",
        key_facts=facts,
        markdown_block=md,
        warnings=warnings,
        skipped=not (model or fw),
    )
