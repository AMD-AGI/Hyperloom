"""Drop candidate variants a trained ranker says will not help.

Every EXPLORE variant costs a benchmark slot -- minutes of GPU each -- and on the
corpus this project trained against, most candidates for a given identity do not
help: roughly 12% of them carry a measured gain. A model that recognises the
other 88% is worth a slot even if it cannot rank the winners, which is the job
this does. It is a filter, not a recommender: it decides what not to measure and
leaves the ordering of what survives to the measurements themselves.

Measured behaviour at the default threshold, on 1,028 held-out identities of the
sglang 0.5.12 corpus: 44.5% of candidates dropped, 98.9% of each identity's best
available gain still reachable, and the single best candidate lost for 1.5% of
identities. On a hand-built pool of seven configurations measured end to end on
Qwen3-14B-FP8, it kept all seven including the +327.7% winner.

Three limits are enforced rather than documented, because each one has been
measured to matter:

  * The model is valid for exactly one framework version. Preferences invert
    across a single sglang minor bump -- a 0.5.11 model scores 0.4844 on 0.5.12,
    below chance -- so a mismatch disables filtering instead of degrading it.
  * The grid is never emptied. A filter that discards everything converts a
    slow search into no search.
  * Everything is opt-in via ``HYPERLOOM_DELTA_FILTER_DIR``. Unset, or with the
    artifact or lightgbm unavailable, behaviour is byte-identical to no filter.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ENV_DIR = "HYPERLOOM_DELTA_FILTER_DIR"
ENV_THRESHOLD = "HYPERLOOM_DELTA_FILTER_THRESHOLD"
DEFAULT_THRESHOLD = 0.030

FLAG_RE = re.compile(r"--[a-z0-9][a-z0-9-]*")
NUM_RE = re.compile(r"--[a-z0-9-]+\s+(\d+(?:\.\d+)?)")

IDENTITY_KEYS = (
    "model_type", "architectures", "hardware", "framework", "framework_version",
    "precision", "param_b", "n_sessions", "hidden_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "vocab_size",
    "intermediate_size", "head_dim", "num_experts", "num_experts_per_tok",
    "gqa_ratio", "ffn_ratio", "is_moe", "size_source",
)


def _threshold() -> float:
    """Score below which a variant is dropped."""
    raw = os.environ.get(ENV_THRESHOLD, "").strip()
    if not raw:
        return DEFAULT_THRESHOLD
    try:
        return float(raw)
    except ValueError:
        log.warning("delta_filter: %s=%r is not a number; using %.3f",
                    ENV_THRESHOLD, raw, DEFAULT_THRESHOLD)
        return DEFAULT_THRESHOLD


def _identity_features(model_path: str, precision: str, hardware: str,
                       framework: str, framework_version: str) -> dict:
    """Build the identity half of the feature row from the checkpoint's config."""
    cfg: dict = {}
    try:
        from hyperloom.inference_optimizer.cli.model_gate import _load_model_config_dict

        cfg = _load_model_config_dict(model_path) or {}
    except Exception:  # noqa: BLE001 -- an unreadable config only costs features
        log.debug("delta_filter: could not read config of %s", model_path, exc_info=True)

    hidden = cfg.get("hidden_size") or 0
    inter = cfg.get("intermediate_size") or 0
    heads = cfg.get("num_attention_heads") or 0
    kv_heads = cfg.get("num_key_value_heads") or heads
    layers = cfg.get("num_hidden_layers") or 0
    vocab = cfg.get("vocab_size") or 0
    # Same rough estimate the training features used: embeddings plus per-layer
    # attention and MLP. The ranker was fitted on this definition, so it has to
    # be reproduced here rather than improved.
    per_layer = 4 * hidden * hidden + 3 * hidden * inter
    param_b = round((vocab * hidden * 2 + layers * per_layer) / 1e9, 3) if hidden else None
    arch = (cfg.get("architectures") or [""])
    return {
        "model_type": cfg.get("model_type"),
        "architectures": str(arch[0]).lower() if arch else None,
        "hardware": hardware or None,
        "framework": framework or None,
        "framework_version": framework_version or None,
        "precision": precision or None,
        "param_b": param_b,
        "n_sessions": 1,
        "hidden_size": hidden or None,
        "num_hidden_layers": layers or None,
        "num_attention_heads": heads or None,
        "num_key_value_heads": kv_heads or None,
        "vocab_size": vocab or None,
        "intermediate_size": inter or None,
        "head_dim": cfg.get("head_dim"),
        "num_experts": cfg.get("num_experts"),
        "num_experts_per_tok": cfg.get("num_experts_per_tok"),
        "gqa_ratio": round(heads / kv_heads, 3) if kv_heads else None,
        "ffn_ratio": round(inter / hidden, 3) if hidden else None,
        "is_moe": bool(cfg.get("num_experts")),
        "size_source": "config" if cfg else None,
    }


def _delta_features(args: str, envs: dict[str, str] | None) -> dict:
    """Flag presence, env presence and shape features for one candidate."""
    flags = sorted(set(FLAG_RE.findall(args or "")))
    nums = [float(x) for x in NUM_RE.findall(args or "")]
    env_keys = sorted((envs or {}).keys())
    text = "%s | env: %s" % (args or "", " ".join(
        "%s=%s" % (k, (envs or {})[k]) for k in env_keys)) if env_keys else (args or "")
    feats: dict[str, Any] = {
        "shape_n_flags": len(flags),
        "shape_n_chars": len(text),
        "shape_n_num": len(nums),
        "shape_max_num": max(nums) if nums else 0.0,
        "shape_has_env": int(bool(env_keys)),
        "shape_n_env": len(env_keys),
    }
    for f in flags:
        feats["flag%s" % f.replace("-", "_")] = 1
    for k in env_keys:
        feats["env_%s" % k] = 1
    return feats


def filter_variants(
    variants: list,
    *,
    framework: str,
    framework_version: str,
    model_path: str,
    precision: str,
    hardware: str = "",
) -> tuple[list, dict]:
    """Return (kept variants, info). Falls back to keeping everything.

    Args:
        variants: GridVariant-likes exposing ``name`` / ``extra_server_args`` /
            ``extra_envs``.
        framework: Inference framework; only sglang has a trained model.
        framework_version: The running version, matched against the artifact's.
        model_path: Checkpoint directory, read for identity features.
        precision: Workload precision.
        hardware: GPU label, used as a feature when known.

    Returns:
        The surviving variants and a dict describing what happened, suitable for
        logging or for a task result field.
    """
    info: dict[str, Any] = {"applied": False, "reason": "", "dropped": []}
    model_dir = os.environ.get(ENV_DIR, "").strip()
    if not model_dir:
        info["reason"] = "disabled"
        return variants, info
    if (framework or "").strip().lower() != "sglang":
        info["reason"] = "framework_not_sglang"
        return variants, info
    if not variants:
        info["reason"] = "empty_input"
        return variants, info

    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError:
        # An undeclared optional dependency must never break EXPLORE.
        info["reason"] = "lightgbm_unavailable"
        return variants, info

    d = Path(model_dir)
    try:
        schema = json.loads((d / "schema.json").read_text())
        manifest = json.loads((d / "manifest.json").read_text())
        booster = lgb.Booster(model_file=str(d / "ranker.txt"))
    except Exception as exc:  # noqa: BLE001
        # Broad on purpose. Naming lightgbm's own exception type here is a trap:
        # it lives at lightgbm.basic.LightGBMError, and referencing the
        # non-existent lgb.LightGBMError made the handler itself raise, which
        # defeated the fallback this whole function exists to provide.
        log.warning("delta_filter: cannot load ranker from %s: %s: %s",
                    d, type(exc).__name__, exc)
        info["reason"] = "artifact_unreadable"
        return variants, info

    trained_for = str(manifest.get("valid_only_for_version") or "").strip()
    running = str(framework_version or "").strip()
    if trained_for and running and trained_for != running:
        # Section 16: a preference learned on one sglang version scores below
        # chance on the next, so a stale model is worse than no model.
        log.warning("delta_filter: ranker is for %s %s but this run is %s; not filtering",
                    framework, trained_for, running)
        info["reason"] = "version_mismatch"
        return variants, info

    ident = _identity_features(model_path, precision, hardware, framework, running)
    cols, cat, vocab = schema["columns"], set(schema["categorical"]), schema["vocab"]
    X = np.zeros((len(variants), len(cols)), dtype=np.float32)
    for i, v in enumerate(variants):
        row = dict(ident)
        row.update(_delta_features(getattr(v, "extra_server_args", "") or "",
                                   getattr(v, "extra_envs", None)))
        for j, c in enumerate(cols):
            val = row.get(c)
            if c in cat:
                X[i, j] = vocab[c].get(str(val), -1)
            elif isinstance(val, bool):
                X[i, j] = int(val)
            elif isinstance(val, (int, float)):
                X[i, j] = val

    try:
        scores = [float(s) for s in booster.predict(X)]
    except Exception as exc:  # noqa: BLE001 -- a scoring failure must not block EXPLORE
        log.warning("delta_filter: scoring failed (%s: %s); not filtering",
                    type(exc).__name__, exc)
        info["reason"] = "scoring_failed"
        return variants, info
    th = _threshold()
    kept = [v for v, s in zip(variants, scores) if s >= th]
    dropped = [(getattr(v, "name", "?"), round(s, 4))
               for v, s in zip(variants, scores) if s < th]

    if not kept:
        # Dropping every candidate turns a slow search into no search, so keep
        # the best-scoring one and say so.
        best = max(range(len(scores)), key=lambda i: scores[i])
        log.warning("delta_filter: every variant scored below %.3f; keeping the "
                    "highest (%s at %.4f) rather than emptying the grid",
                    th, getattr(variants[best], "name", "?"), scores[best])
        info.update(applied=True, reason="all_below_threshold_kept_best",
                    threshold=th, dropped=dropped, kept=1, total=len(variants))
        return [variants[best]], info

    info.update(applied=True, reason="ok", threshold=th, dropped=dropped,
                kept=len(kept), total=len(variants),
                ranker_version=trained_for)
    if dropped:
        log.info("delta_filter: dropped %d/%d variants below %.3f: %s",
                 len(dropped), len(variants), th,
                 ", ".join("%s=%.4f" % kv for kv in dropped[:6]))
    return kept, info
