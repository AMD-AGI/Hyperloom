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

Four limits are enforced rather than documented, because each one has been
measured to matter:

  * The model is valid for exactly one framework version. Preferences invert
    across a single sglang minor bump -- a 0.5.11 model scores 0.4844 on 0.5.12,
    below chance -- so a mismatch disables filtering instead of degrading it.
    Measured on 1,467 models paired across 0.5.11/0.5.12, whether a flag helps
    agrees only 78.4% of the time against 74.5% by chance, a kappa of 0.15, and
    ``--kv-cache-dtype`` inverts outright: never useful on 0.5.11, useful in
    half of 0.5.12's identities. Version is a partition key, not a feature.
  * The model is valid only for GPUs its corpus covered, which the schema's
    ``hardware`` vocabulary states exactly. An unseen GPU would encode to the
    out-of-vocabulary sentinel the trees never split on, so it declines instead.
  * The grid is never emptied. A filter that discards everything converts a
    slow search into no search.
  * Everything is opt-in via ``HYPERLOOM_DELTA_FILTER_DIR``. Unset, or with the
    artifact or lightgbm unavailable, behaviour is byte-identical to no filter.

Because version and GPU partition rather than generalise, one deployment
serving several of both needs one artifact per cell. ``HYPERLOOM_DELTA_FILTER_DIR``
therefore accepts either a single artifact directory or a root of them, resolved
per session against the running framework, version and GPU; see
:func:`_resolve_artifact`. The GPU axis is narrower than it looks, since the
corpus is keyed on Magpie's runner label and mi325x/mi308x collapse into
mi300x.

A served-LLM backend is also available. It orders candidates better than the
GBDT (95.5% vs 91.4% precision@1) but prunes worse, because an ordering has no
absolute scale to threshold: it can only keep a fixed share of the list, and at
a matched 1.5% loss rate that share leaves 22.6% of slots saved against the
GBDT's 44.5%. Training it to emit its own cut did not close the gap.

Configure both and they combine, which measured better than either alone --
threshold, plus the ranker's top pick regardless of its score. See
:func:`_keep_union` for the numbers and for the tightening that does not work.

The LLM backend carries one constraint this module cannot enforce: it needs a
GPU of its own. Concurrent GPU work on a single node was measured to slow a
benchmark by more than 4x -- a 57 GB MoE that loads in 3m38s alone did not
finish in 15 minutes beside another job -- so hosting the ranker next to the
benchmarks corrupts the very measurements the optimizer decides on. Serve it on
a separate node, or take it down while EXPLORE measures. The GBDT backend has no
such conflict: it is microseconds of CPU.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ENV_DIR = "HYPERLOOM_DELTA_FILTER_DIR"
ENV_THRESHOLD = "HYPERLOOM_DELTA_FILTER_THRESHOLD"
DEFAULT_THRESHOLD = 0.030

# A fixed cut is well calibrated across the corpus but not within an identity:
# score bins track their true positive rate closely overall (0.5% at 0.00-0.01
# rising to 83% above 0.50), yet an identity whose candidates all score low
# loses its best one to the cut while an identity that scores high keeps junk.
# Scaling the cut by the identity's own top score removes that offset, and it
# dominates the fixed cut on both held-out splits:
#
#            val: saved / lost      test: saved / lost
#   fixed .030    42.8% / 1.6%          42.1% / 1.5%
#   max * .04     41.6% / 0.8%          41.7% / 0.4%   same saving, half the risk
#   max * .08     62.8% / 1.6%          61.9% / 1.5%   same risk, +20pp saving
#
# The top-scoring candidate always clears a relative cut, so this cannot empty
# the grid. Unset to keep the fixed cut.
#
# This assumes the artifact scores a calibrated probability in [0,1], which the
# shipped pointwise model does. A ranking objective does not: swapping in the
# lambdarank variant, whose scores run -4.6 to 3.5 with most of them negative,
# makes a fraction of the maximum meaningless and the loss rate jumps from 1.5%
# to 7.9%. Retrain that as a classifier before pointing this at it.
ENV_RELATIVE = "HYPERLOOM_DELTA_FILTER_RELATIVE"

# LLM backend. The ranker returns an ordering rather than per-candidate scores,
# so it prunes by keeping a fraction of the list instead of by thresholding.
ENV_LLM_URL = "HYPERLOOM_DELTA_FILTER_LLM_URL"
ENV_LLM_MODEL = "HYPERLOOM_DELTA_FILTER_LLM_MODEL"
ENV_LLM_LORA = "HYPERLOOM_DELTA_FILTER_LLM_LORA"
ENV_LLM_VERSION = "HYPERLOOM_DELTA_FILTER_LLM_VERSION"
# One endpoint can hold several adapters, so a deployment serving several
# versions or GPUs maps each cell to its own: "0.5.12/mi300x=ranker-0512,
# 0.5.11/mi300x=ranker-0511". Takes precedence over the single-cell
# ENV_LLM_LORA + ENV_LLM_VERSION pair, which stays valid for one cell.
ENV_LLM_ADAPTERS = "HYPERLOOM_DELTA_FILTER_LLM_ADAPTERS"
ENV_KEEP_FRACTION = "HYPERLOOM_DELTA_FILTER_KEEP_FRACTION"
# Measured over 200 held-out identities on the served adapter: keeping half the
# list drops 48.4% of candidates while 98.4% of the best available gain survives
# and the single best candidate is lost for 2.5%. Keeping three quarters loses it
# for 1.5% but only drops 22.6%.
DEFAULT_KEEP_FRACTION = 0.5
LLM_TIMEOUT_S = 60

# How many of the ranker's top picks survive the union regardless of their GBDT
# score. One is the measured knee: it costs 0.1pp of dropped candidates and
# takes the lost-best rate from 1.5% to 1.1%. Two and three keep helping (0.9%
# and 0.7%) but give back 4pp and 9pp of savings for it.
ENV_BACKSTOP = "HYPERLOOM_DELTA_FILTER_LLM_BACKSTOP"
DEFAULT_BACKSTOP = 1

RANKER_SYSTEM = (
    "You tune SGLang inference servers. Given a model and a list of candidate "
    "configuration changes, order the candidates from most to least likely to "
    "improve throughput on that model. Reply with the candidate numbers only, "
    "best first, comma separated."
)

FLAG_RE = re.compile(r"--[a-z0-9][a-z0-9-]*")
NUM_RE = re.compile(r"--[a-z0-9-]+\s+(\d+(?:\.\d+)?)")

IDENTITY_KEYS = (
    "model_type", "architectures", "hardware", "framework", "framework_version",
    "precision", "param_b", "n_sessions", "hidden_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "vocab_size",
    "intermediate_size", "head_dim", "num_experts", "num_experts_per_tok",
    "gqa_ratio", "ffn_ratio", "is_moe", "size_source",
)


def _backstop() -> int:
    """How many of the ranker's top picks the union keeps regardless of score."""
    raw = os.environ.get(ENV_BACKSTOP, "").strip()
    if not raw:
        return DEFAULT_BACKSTOP
    try:
        return max(0, int(raw))
    except ValueError:
        log.warning("delta_filter: %s=%r is not an integer; using %d",
                    ENV_BACKSTOP, raw, DEFAULT_BACKSTOP)
        return DEFAULT_BACKSTOP


def _runner_label(gpu_type: str) -> str:
    """Collapse a GPU model to the runner label the corpus is keyed on.

    The corpus records ``hardware`` as Magpie's runner label, where mi325x and
    mi308x are mi300x -- same gfx942, same 304 CUs, same benchmark script. The
    caller may hand over either spelling, since it reads ``runner_type`` from the
    session config and falls back to ``GPU_TYPE``, so normalising here is what
    keeps a mi325x session from encoding as out-of-vocabulary against a corpus
    that never spells mi325x. Collapsing does discard the HBM difference, which
    is not nothing for memory-fraction flags, but that is the corpus's own
    assumption and this only stops the filter from disagreeing with it.
    """
    try:
        from hyperloom.inference_optimizer.gpu_types import _gpu_runner_type

        return _gpu_runner_type(gpu_type)
    except Exception:  # noqa: BLE001 -- a missing helper must not disable the filter
        return str(gpu_type or "").strip().lower()


def _hardware_known(schema: dict, hardware: str) -> bool:
    """Whether the artifact's corpus covered this GPU runner.

    The schema's vocabulary is the authoritative record of what the model saw:
    an absent value encodes to -1, a sentinel no tree ever split on, so scoring
    would be confident nonsense. Returns True when there is no basis to refuse --
    an unknown running GPU, or an artifact predating the column -- which mirrors
    the version guard's rule of only comparing when both sides are known.
    """
    hw = str(hardware or "").strip()
    if not hw:
        return True
    vocab = (schema.get("vocab") or {}).get("hardware")
    if not isinstance(vocab, dict) or not vocab:
        return True
    return hw in vocab


def _cell_matches(manifest: dict, schema: dict, framework: str,
                  version: str, hardware: str) -> bool:
    """Whether one registry entry was trained for this session's cell."""
    if str(manifest.get("framework") or "").strip().lower() != str(framework or "").strip().lower():
        return False
    trained_for = str(manifest.get("valid_only_for_version") or "").strip()
    if trained_for and version and trained_for != version:
        return False
    return _hardware_known(schema, hardware)


def _resolve_artifact(root: str, framework: str, version: str,
                      hardware: str) -> tuple[Path | None, str]:
    """Pick this session's artifact directory under ``root``.

    ``root`` holding a ``manifest.json`` is a single artifact and is returned
    unchanged, so an existing single-cell deployment behaves exactly as before
    and the guards downstream still decide whether it applies. Otherwise its
    immediate subdirectories are read as a registry and matched on framework,
    version and GPU.

    Two matches is an operator error, not a choice to make silently: picking one
    arbitrarily is the same class of bug as scoring a mi355x run with a mi300x
    model, so it declines and says so.

    Returns:
        ``(directory, reason)``; the directory is None when no single entry
        applies, and ``reason`` is empty on success.
    """
    d = Path(root)
    if (d / "manifest.json").is_file():
        return d, ""
    if not d.is_dir():
        # Neither an artifact nor a registry. Reported as an unreadable artifact
        # because that is what it is from the caller's side, and because the
        # single-artifact path said exactly this before the registry existed.
        log.warning("delta_filter: %s is neither an artifact nor a registry", d)
        return None, "artifact_unreadable"
    try:
        subs = sorted(p for p in d.iterdir() if p.is_dir())
    except OSError as exc:
        log.warning("delta_filter: cannot list registry %s: %r", d, exc)
        return None, "registry_unreadable"

    matches: list[Path] = []
    for sub in subs:
        try:
            manifest = json.loads((sub / "manifest.json").read_text())
            schema = json.loads((sub / "schema.json").read_text())
        except Exception:  # noqa: BLE001 -- a malformed entry is skipped, not fatal
            log.debug("delta_filter: registry entry %s unreadable", sub, exc_info=True)
            continue
        if _cell_matches(manifest, schema, framework, version, hardware):
            matches.append(sub)

    if not matches:
        log.info("delta_filter: no ranker for %s %s on %s under %s; not filtering",
                 framework, version or "?", hardware or "?", d)
        return None, "registry_miss"
    if len(matches) > 1:
        log.warning("delta_filter: %d rankers claim %s %s on %s (%s); refusing to "
                    "choose one, not filtering", len(matches), framework,
                    version or "?", hardware or "?",
                    ", ".join(p.name for p in matches))
        return None, "registry_ambiguous"
    return matches[0], ""


def _llm_adapter_for(version: str, hardware: str) -> tuple[str, str]:
    """Resolve the served adapter for this cell.

    Returns:
        ``(adapter, reason)``. An empty adapter with an empty reason means the
        base model is served with no adapter; an empty adapter with a reason
        means this cell has none and ranking must be skipped.
    """
    raw = os.environ.get(ENV_LLM_ADAPTERS, "").strip()
    if not raw:
        # Single-cell form: one adapter, valid for the version it declares.
        declared = os.environ.get(ENV_LLM_VERSION, "").strip()
        if declared and version and declared != version:
            return "", "version_mismatch"
        return os.environ.get(ENV_LLM_LORA, "").strip(), ""

    table: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        key, name = entry.split("=", 1)
        table[key.strip()] = name.strip()
    adapter = table.get("%s/%s" % (version, hardware)) or table.get(version)
    if not adapter:
        log.info("delta_filter: no served adapter for %s/%s in %s; not ranking",
                 version or "?", hardware or "?", ENV_LLM_ADAPTERS)
        return "", "adapter_cell_miss"
    return adapter, ""


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


def _relative_factor() -> float:
    """Fraction of the identity's top score to cut at; 0 keeps the fixed cut."""
    raw = os.environ.get(ENV_RELATIVE, "").strip()
    if not raw:
        return 0.0
    try:
        f = float(raw)
    except ValueError:
        log.warning("delta_filter: %s=%r is not a number; using the fixed cut",
                    ENV_RELATIVE, raw)
        return 0.0
    if not 0.0 < f < 1.0:
        log.warning("delta_filter: %s=%r is outside (0,1); using the fixed cut",
                    ENV_RELATIVE, raw)
        return 0.0
    return f


def _cut_for(scores: list) -> tuple[float, str]:
    """Return the score cut for one identity and a label for the log."""
    f = _relative_factor()
    if f and scores:
        top = max(scores)
        if top > 0:
            return f * top, "max*%.3f" % f
    return _threshold(), "%.3f" % _threshold()


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


def _describe_identity(model_path: str, ident: dict) -> str:
    """Render the identity as the prose the ranker was fine-tuned on.

    Must match build_sft_dataset.describe_identity: the model was trained on this
    exact phrasing, and a paraphrase is a distribution shift for no benefit.
    """
    bits = []
    if ident.get("architectures"):
        bits.append("architecture %s" % ident["architectures"])
    if ident.get("param_b"):
        bits.append("%.1fB parameters" % float(ident["param_b"]))
    if ident.get("is_moe"):
        n = ident.get("num_experts")
        bits.append("mixture of experts%s" % (" (%s experts)" % n if n else ""))
    else:
        bits.append("dense")
    if ident.get("hidden_size") and ident.get("num_hidden_layers"):
        bits.append("%s hidden x %s layers" % (ident["hidden_size"], ident["num_hidden_layers"]))
    if ident.get("gqa_ratio"):
        bits.append("GQA ratio %s" % ident["gqa_ratio"])
    for key, label in (("precision", "precision"), ("hardware", "GPU"),
                       ("framework_version", "sglang")):
        if ident.get(key):
            bits.append("%s %s" % (label, ident[key]))
    return ", ".join(bits)


def _keep_fraction() -> float:
    """Share of the candidate list the LLM backend keeps."""
    raw = os.environ.get(ENV_KEEP_FRACTION, "").strip()
    if not raw:
        return DEFAULT_KEEP_FRACTION
    try:
        v = float(raw)
    except ValueError:
        log.warning("delta_filter: %s=%r is not a number; using %.2f",
                    ENV_KEEP_FRACTION, raw, DEFAULT_KEEP_FRACTION)
        return DEFAULT_KEEP_FRACTION
    return min(max(v, 0.0), 1.0)


def _rank_with_llm(variants: list, identity_text: str, lora: str = "") -> list[int] | None:
    """Ask the served ranker for an ordering. None on any failure.

    Args:
        variants: Candidates to order.
        identity_text: The identity rendered as the prose the adapter was tuned on.
        lora: Served adapter name for this cell; empty serves the base model.
    """
    import json as _json
    import urllib.error
    import urllib.request

    url = os.environ.get(ENV_LLM_URL, "").strip().rstrip("/")
    listing = "\n".join("%d. %s" % (i + 1, _delta_text(v)) for i, v in enumerate(variants))
    payload = {
        "model": os.environ.get(ENV_LLM_MODEL, "").strip() or "default",
        "messages": [
            {"role": "system", "content": RANKER_SYSTEM},
            {"role": "user", "content": "Model: %s\n\nCandidates:\n%s"
                                        % (identity_text, listing)},
        ],
        # Roughly four tokens per "12, " plus slack. A fixed budget silently
        # truncates the tail of a long ordering, and a truncated ordering is not
        # an error the caller can see -- the unranked remainder just keeps its
        # original position.
        "max_tokens": max(32, 5 * len(variants) + 16),
        "temperature": 0,
        # The fine-tune was done with thinking off. Left on, the reply carries an
        # empty <think></think> pair that eats the token budget for nothing, and
        # a model or sampling change that made those blocks non-empty would eat
        # all of it.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if lora:
        payload["lora_path"] = lora

    req = urllib.request.Request(
        "%s/v1/chat/completions" % url,
        data=_json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_S) as resp:
            body = _json.load(resp)
        text = (body["choices"][0]["message"].get("content") or "")
    except Exception as exc:  # noqa: BLE001 -- a ranking call must never break EXPLORE
        log.warning("delta_filter: ranker call failed (%s: %s); not filtering",
                    type(exc).__name__, exc)
        return None

    n = len(variants)
    order: list[int] = []
    for tok in re.findall(r"\d+", text):
        i = int(tok) - 1
        if 0 <= i < n and i not in order:
            order.append(i)
    if not order:
        log.warning("delta_filter: ranker reply had no usable candidate numbers: %r",
                    text[:120])
        return None
    # Whatever the reply omitted keeps its original relative position, after
    # everything it did rank.
    return order + [i for i in range(n) if i not in order]


def _delta_text(v) -> str:
    """The candidate as the ranker saw it during training: args, then envs."""
    args = (getattr(v, "extra_server_args", "") or "").strip()
    envs = getattr(v, "extra_envs", None) or {}
    if not envs:
        return args
    return "%s | env: %s" % (args, " ".join("%s=%s" % (k, envs[k]) for k in sorted(envs)))


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
    model_root = os.environ.get(ENV_DIR, "").strip()
    llm_url = os.environ.get(ENV_LLM_URL, "").strip()
    if not model_root and not llm_url:
        info["reason"] = "disabled"
        return variants, info
    if (framework or "").strip().lower() != "sglang":
        info["reason"] = "framework_not_sglang"
        return variants, info
    if not variants:
        info["reason"] = "empty_input"
        return variants, info

    running = str(framework_version or "").strip()
    # Normalise before anything reads it: both the identity feature and the
    # registry lookup have to agree with the corpus's runner-label spelling.
    hardware = _runner_label(hardware)
    ident = _identity_features(model_path, precision, hardware, framework, running)
    info["cell"] = "%s/%s" % (running or "?", hardware or "?")

    scores: list[float] | None = None
    gbdt_version = ""
    if model_root:
        model_dir, resolve_reason = _resolve_artifact(
            model_root, framework, running, hardware)
        if model_dir is None:
            info["gbdt_reason"] = resolve_reason
        else:
            info["gbdt_artifact"] = model_dir.name
            scores, gbdt_reason, gbdt_version = _gbdt_scores(
                str(model_dir), variants, ident, framework, running)
            if scores is None:
                info["gbdt_reason"] = gbdt_reason

    order: list[int] | None = None
    llm_version = os.environ.get(ENV_LLM_VERSION, "").strip()
    llm_lora, adapter_reason = _llm_adapter_for(running, hardware)
    if llm_url:
        # Same version guard as the GBDT: the adapter was fine-tuned on one
        # sglang version's corpus, and preferences invert across a bump. With
        # several cells served from one endpoint the guard becomes a lookup.
        if adapter_reason:
            info["llm_reason"] = adapter_reason
        elif not ident.get("hidden_size"):
            # Without the checkpoint the identity collapses to a couple of
            # labels, which is a prompt the ranker never saw. Ranking from it
            # would still return a confident order, so refuse instead: the GBDT
            # backend degrades gracefully on missing features, this one does not.
            log.warning("delta_filter: no readable config at %r; not ranking",
                        model_path)
            info["llm_reason"] = "identity_unavailable"
        else:
            info["llm_adapter"] = llm_lora or None
            order = _rank_with_llm(variants, _describe_identity(model_path, ident),
                                   llm_lora)
            if order is None:
                info["llm_reason"] = "llm_unavailable"

    if scores is not None and order is not None:
        return _keep_union(variants, scores, order, info,
                           gbdt_version or llm_version)
    if scores is not None:
        return _keep_by_threshold(variants, scores, info, gbdt_version)
    if order is not None:
        return _keep_by_fraction(variants, order, info, llm_version)

    info["reason"] = info.get("gbdt_reason") or info.get("llm_reason") or "unavailable"
    return variants, info


def _keep_union(variants, scores, order, info, version) -> tuple[list, dict]:
    """Keep everything the threshold keeps, plus the ranker's first picks.

    The two models fail in different places, and combining them was measured
    over 1,028 held-out identities to be better than either alone. The threshold
    on its own dropped 41.7% of candidates and lost the best one for 1.5% of
    identities; adding the ranker's single top pick held the drop at 41.6% and
    cut the loss to 1.1%. One extra benchmark slot buys roughly a quarter off
    the risk, which is a good trade when a slot is five minutes and a lost
    winner is the whole round.

    The obvious next move -- tighten the threshold and let the ranker cover the
    risk -- does not work. Between 0.030 and 0.060 the loss rate jumps from 1.5%
    to 4.0%, and three ranker picks only pull it back to 2.2%; the threshold sits
    against a cliff with no margin to trade. Nothing measured saves more than
    41.7% while holding the loss at or under 1.5%.

    ``HYPERLOOM_DELTA_FILTER_LLM_BACKSTOP`` widens the ranker's contribution for
    callers wanting more safety: 2 picks measured 37.6% dropped at 0.9% lost,
    3 picks 32.7% at 0.7%.
    """
    th, cut_label = _cut_for(scores)
    n_back = _backstop()
    above = {i for i, s in enumerate(scores) if s >= th}
    backstop = set(order[:n_back])
    kept_idx = above | backstop
    if not kept_idx:
        kept_idx = {max(range(len(scores)), key=lambda i: scores[i])}

    kept = [v for i, v in enumerate(variants) if i in kept_idx]
    dropped = [(getattr(v, "name", "?"), round(scores[i], 4))
               for i, v in enumerate(variants) if i not in kept_idx]
    added = sorted(backstop - above)
    info.update(applied=True, reason="ok_union", backend="union",
                threshold=th, cut=cut_label, backstop=n_back, kept=len(kept),
                total=len(variants), dropped=dropped,
                ranker_version=version or None,
                backstop_added=[getattr(variants[i], "name", "?") for i in added])
    log.info("delta_filter(union): kept %d/%d (cut %s + top-%d); "
             "ranker rescued %s",
             len(kept), len(variants), cut_label, n_back,
             ", ".join(getattr(variants[i], "name", "?") for i in added) or "nothing")
    return kept, info


def _keep_by_threshold(variants, scores, info, version) -> tuple[list, dict]:
    """Keep the candidates the GBDT scores at or above the cut."""
    th, cut_label = _cut_for(scores)
    kept = [v for v, s in zip(variants, scores) if s >= th]
    dropped = [(getattr(v, "name", "?"), round(s, 4))
               for v, s in zip(variants, scores) if s < th]

    if not kept:
        # Dropping every candidate turns a slow search into no search, so keep
        # the best-scoring one and say so. Unreachable under a relative cut,
        # where the top scorer clears it by construction.
        best = max(range(len(scores)), key=lambda i: scores[i])
        log.warning("delta_filter: every variant scored below %s; keeping the "
                    "highest (%s at %.4f) rather than emptying the grid",
                    cut_label, getattr(variants[best], "name", "?"), scores[best])
        info.update(applied=True, reason="all_below_threshold_kept_best",
                    backend="gbdt", threshold=th, cut=cut_label, dropped=dropped,
                    kept=1, total=len(variants))
        return [variants[best]], info

    info.update(applied=True, reason="ok", backend="gbdt", threshold=th,
                cut=cut_label, dropped=dropped, kept=len(kept),
                total=len(variants), ranker_version=version)
    if dropped:
        log.info("delta_filter: dropped %d/%d variants below %s: %s",
                 len(dropped), len(variants), cut_label,
                 ", ".join("%s=%.4f" % kv for kv in dropped[:6]))
    return kept, info


def _keep_by_fraction(variants, order, info, version) -> tuple[list, dict]:
    """Keep the head of the ranker's ordering, with no scores to threshold on."""
    frac = _keep_fraction()
    # At least one survivor: an empty grid turns a slow search into no search.
    k = max(1, min(len(variants), math.ceil(len(variants) * frac)))
    kept_idx = set(order[:k])
    kept = [v for i, v in enumerate(variants) if i in kept_idx]
    dropped = [(getattr(v, "name", "?"), order.index(i))
               for i, v in enumerate(variants) if i not in kept_idx]
    info.update(applied=True, reason="ok_llm", backend="llm",
                keep_fraction=frac, kept=len(kept), total=len(variants),
                dropped=dropped, ranker_version=version or None)
    if dropped:
        log.info("delta_filter(llm): kept %d/%d (fraction %.2f); dropped %s",
                 len(kept), len(variants), frac,
                 ", ".join("%s@rank%d" % kv for kv in dropped[:6]))
    return kept, info

# Shape measures that get a within-identity counterpart. Must stay in step with
# claw-dev/docs-zh/post-training/recipe-rank/try_context_features.py: a context
# feature computed differently here than during training scores silently wrong
# rather than failing.
_CONTEXT_SHAPE_KEYS = ("shape_n_flags", "shape_n_num", "shape_max_num",
                       "shape_n_chars", "shape_n_env", "shape_has_env")


def _add_context_features(rows: list[dict]) -> None:
    """Add each candidate's standing within this grid, in place.

    Every other feature describes a candidate in isolation, so the model cannot
    tell a strong candidate in a weak field from a weak one in a strong field --
    which is the same blind spot the relative cut works around downstream.
    Giving it the comparison directly bought about 5 points of extra candidates
    dropped at an unchanged loss rate, on both held-out splits.
    """
    n = len(rows)
    if not n:
        return
    for key in _CONTEXT_SHAPE_KEYS:
        vals = [float(r.get(key) or 0.0) for r in rows]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        mean = sum(vals) / n
        for r, v in zip(rows, vals):
            r["ctx_%s_rel" % key] = (v - lo) / span
            r["ctx_%s_dev" % key] = v - mean
    for r in rows:
        r["ctx_n_candidates"] = n
        # A flag every candidate carries says nothing about this one; a flag
        # only it carries might.
        flags = [k for k in r if k.startswith("flag__") and r.get(k)]
        if flags:
            share = [sum(1 for o in rows if o.get(f)) / n for f in flags]
            r["ctx_flag_rarity"] = 1.0 - (sum(share) / len(share))
        else:
            r["ctx_flag_rarity"] = 0.0


def _gbdt_scores(
    model_dir: str,
    variants: list,
    ident: dict,
    framework: str,
    framework_version: str,
) -> tuple[list[float] | None, str, str]:
    """Score every variant with the trained GBDT.

    Returns ``(scores, reason, trained_for)``; ``scores`` is None when the model
    could not be applied, and ``reason`` says why so the caller can report it.
    Every failure path returns rather than raises: this runs inside EXPLORE, and
    a filter that cannot score must degrade to not filtering.
    """
    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError:
        # An undeclared optional dependency must never break EXPLORE.
        return None, "lightgbm_unavailable", ""

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
        return None, "artifact_unreadable", ""

    trained_for = str(manifest.get("valid_only_for_version") or "").strip()
    running = str(framework_version or "").strip()
    if trained_for and running and trained_for != running:
        # Section 16: a preference learned on one sglang version scores below
        # chance on the next, so a stale model is worse than no model.
        log.warning("delta_filter: ranker is for %s %s but this run is %s; not filtering",
                    framework, trained_for, running)
        return None, "version_mismatch", trained_for

    if not _hardware_known(schema, ident.get("hardware")):
        # An unseen GPU encodes to -1, a value no tree ever split on, so the
        # scores would be confident and meaningless. The version axis learned
        # this the expensive way; the hardware axis inherits the guard.
        log.warning("delta_filter: ranker's corpus covers %s but this run is on "
                    "%s; not filtering",
                    sorted((schema.get("vocab") or {}).get("hardware") or {}),
                    ident.get("hardware"))
        return None, "hardware_mismatch", trained_for

    cols, cat, vocab = schema["columns"], set(schema["categorical"]), schema["vocab"]
    rows = []
    for v in variants:
        row = dict(ident)
        row.update(_delta_features(getattr(v, "extra_server_args", "") or "",
                                   getattr(v, "extra_envs", None)))
        rows.append(row)
    # Only artifacts trained with them ask for context columns, and computing
    # them needs the whole grid, so this runs once over all rows rather than
    # per variant.
    if any(c.startswith("ctx_") for c in cols):
        _add_context_features(rows)

    X = np.zeros((len(rows), len(cols)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, c in enumerate(cols):
            val = row.get(c)
            if c in cat:
                X[i, j] = vocab[c].get(str(val), -1)
            elif isinstance(val, bool):
                X[i, j] = int(val)
            elif isinstance(val, (int, float)):
                X[i, j] = val

    try:
        return [float(s) for s in booster.predict(X)], "", trained_for
    except Exception as exc:  # noqa: BLE001 -- a scoring failure must not block EXPLORE
        log.warning("delta_filter: scoring failed (%s: %s); not filtering",
                    type(exc).__name__, exc)
        return None, "scoring_failed", trained_for
