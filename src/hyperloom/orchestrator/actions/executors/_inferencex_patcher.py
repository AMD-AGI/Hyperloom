# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Idempotent, backward-compatible patchers for the InferenceX checkout.

Each ``ensure_*`` function rewrites one upstream file in place: ``$NUM_PROMPTS``
support and ``PROFILE_EXTRA_BODY`` consumption for profiling, the eval-artifact
redirect to ``$RESULT_DIR``, the ``HYPERLOOM_EVAL_START`` phase marker, and the
generation-pathology probe plus per-request generation bounds and model-derived
terminators appended to lm-eval's ``lm_eval_sitecustomize.py``.

Applied in place, once: idempotent via a sentinel substring, serialized across
processes via ``fcntl.flock``, written atomically.

The four line-replacement patches are gated on locating exact upstream text, so a
``False`` return is ambiguous on its own -- it reads the same whether there was
nothing to patch or the anchor rotted. :func:`verify_patch_anchors` reports that
distinction without touching the checkout, so callers can assert the contract
instead of inferring it. The probe needs none of this: it is appended to a real
file, and :func:`eval_probe_targets_exist` already separates the two cases.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

from ._file_lock import best_effort_file_lock
from ._magpie_patcher import atomic_write_text
from ._patch_sentinel import file_contains_sentinel

log = logging.getLogger(__name__)


# Exact upstream line, whitespace-anchored so we don't match an unrelated
# ``num_prompts`` reference elsewhere in the file.
_LEGACY_LINE = '        num_prompts="$max_concurrency"'
_PATCHED_LINE = '        num_prompts="${NUM_PROMPTS:-$max_concurrency}"'
# "Already patched?" sentinel.
_PATCH_SENTINEL = "${NUM_PROMPTS:-$max_concurrency}"

# System-wide lock; cross-reboot persistence is not needed.
_LOCK_PATH = str(Path(tempfile.gettempdir()) / "hyperloom_benchmark_lib_patcher.lock")


# ``benchmark_serving.py`` hardcodes the ``/start_profile`` ``extra_body`` and
# never reads Hyperloom's ``PROFILE_EXTRA_BODY`` env. Single-line replacement
# gated on the exact legacy text; sentinel is ``PROFILE_EXTRA_BODY``.
_BENCH_SERVING_LEGACY = (
    '                                         extra_body={"num_steps": 1, '
    '"merge_profiles": True, "profile_by_stage": True},'
)
# JSON fallback uses lowercase ``true``; ``json.loads`` maps it back so the
# dict matches the upstream literal byte-for-byte.
_BENCH_SERVING_PATCHED = (
    "                                         extra_body=__import__('json')."
    "loads(__import__('os').environ.get('PROFILE_EXTRA_BODY') or "
    '\'{"num_steps": 1, "merge_profiles": true, "profile_by_stage": true}\'),'
)
_BENCH_SERVING_SENTINEL = "PROFILE_EXTRA_BODY"
_BENCH_SERVING_LOCK_PATH = str(Path(tempfile.gettempdir()) / "hyperloom_benchmark_serving_patcher.lock")

# ``append_lm_eval_summary`` does ``mv ./`` — eval artifacts land in the process
# cwd (the InferenceX checkout), escaping the session. Redirect to ``$RESULT_DIR``
# (Hyperloom's session dir), falling back to ``.`` when unset.
_EVAL_DEST_LEGACY = 'mv -f "$jf" ./ || echo "WARN: failed to move ${jf}" >&2'
_EVAL_DEST_PATCHED = 'mv -f "$jf" "${RESULT_DIR:-.}/" || echo "WARN: failed to move ${jf}" >&2'
_EVAL_DEST_SENTINEL = '"${RESULT_DIR:-.}/"'
_EVAL_DEST_LOCK_PATH = str(Path(tempfile.gettempdir()) / "hyperloom_benchmark_lib_eval_dest_patcher.lock")

# The explore overtime kill bounds the throughput phase only, but benchmark and
# eval share one Magpie subprocess, so Hyperloom cannot see the boundary. Emit a
# sentinel on the last line before ``lm_eval`` starts; the soft-deadline watcher
# retires the deadline when it appears. Anchored on the unique
# ``EVAL_RESULT_DIR`` export — ``set -x`` occurs three times in the file.
_EVAL_START_LEGACY = '    export EVAL_RESULT_DIR="$results_dir"'
_EVAL_START_PATCHED = '    export EVAL_RESULT_DIR="$results_dir"\n    echo "HYPERLOOM_EVAL_START" >&2'
_EVAL_START_SENTINEL = "HYPERLOOM_EVAL_START"
_EVAL_START_LOCK_PATH = str(Path(tempfile.gettempdir()) / "hyperloom_benchmark_lib_eval_start_patcher.lock")

# Two independent answers to the same budget, injected together. InferenceX runs
# lm-eval with ``--gen_kwargs max_tokens=min(16384, ctx-4096)``, so a model that
# does not terminate burns that budget on every one of GSM8K's 1319 docs and
# takes the whole baseline timeout with it.
#
# The probe handles a model that never emits EOS at all: it short-circuits the
# remaining requests, which voids the eval (~0 score) and is only safe because it
# waits for a decisive ratio. The bounds handle the ordinary case that ratio can
# never catch -- a healthy model whose hardest samples do not converge -- by
# capping each request so those samples are truncated individually and the rest
# of the measurement survives. The bounds also supply the terminators the model
# declares, which lm-eval structurally cannot: eos_string holds one value and the
# concurrent path never sends even that.
#
# Appended to ``lm_eval_sitecustomize.py``, so both land after InferenceX's own
# patches and need no anchor line.
_EVAL_PROBE_PY = """
# --- HYPERLOOM_EVAL_PROBE (early-exit probe + per-request bounds) ------------
import json as _hl_json
import os as _hl_os
import sys as _hl_sys


def _hl_eval_probe_install():
    if (_hl_os.environ.get("HYPERLOOM_EVAL_PROBE") or "1").strip().lower() in ("0", "false", "no", "off"):
        return

    def _num(name, default, cast, ok):
        try:
            val = cast((_hl_os.environ.get(name) or "").strip())
        except (TypeError, ValueError):
            return default
        return val if ok(val) else default

    # Out of range falls back to the default, not to the nearest legal value:
    # RATIO=0 means "turn the probe off", and clamping would do the opposite.
    min_samples = _num("HYPERLOOM_EVAL_PROBE_MIN_SAMPLES", 128, int, lambda v: v >= 8)
    ratio_limit = _num("HYPERLOOM_EVAL_PROBE_LENGTH_RATIO", 0.75, float, lambda v: 0.0 < v <= 1.0)

    import asyncio as _hl_asyncio
    from lm_eval.models import api_models as _hl_api
    from lm_eval.models.openai_completions import LocalChatCompletion as _hl_lcc

    # The imports above prove this is lm-eval, not one of the other python3
    # invocations sitecustomize runs in, so any sidecar here is a stale one from
    # the attempt that reused this $RESULT_DIR.
    _hl_dir = (_hl_os.environ.get("RESULT_DIR") or "").strip()
    if _hl_dir:
        try:
            _hl_os.remove(_hl_os.path.join(_hl_dir, "hyperloom_eval_probe.json"))
        except OSError:
            pass

    state = {"observed": 0, "length": 0, "max_tokens_seen": 0, "cap_hits": 0, "tripped": False}
    # completion_tokens -> count, over responses the server stopped on length.
    capped = {}

    def _emit():
        record = {
            "reason": "model_not_terminating",
            "observed_samples": state["observed"],
            "finish_reason_length": state["length"],
            "cap_hits": state["cap_hits"],
            "max_completion_tokens_seen": state["max_tokens_seen"],
            "min_samples": min_samples,
            "cap_hit_ratio_threshold": ratio_limit,
        }
        blob = _hl_json.dumps(record, sort_keys=True)
        print("HYPERLOOM_EVAL_PROBE_TRIPPED " + blob, file=_hl_sys.stderr, flush=True)
        # $RESULT_DIR, never $EVAL_RESULT_DIR: append_lm_eval_summary rm -rf's
        # the latter. The name must not match results*.json -- that glob is how
        # parse_eval_results finds the accuracy score.
        out_dir = (_hl_os.environ.get("RESULT_DIR") or "").strip()
        if not out_dir:
            # The cwd is InferenceX's checkout; stderr above already has it all.
            return
        _hl_os.makedirs(out_dir, exist_ok=True)
        with open(_hl_os.path.join(out_dir, "hyperloom_eval_probe.json"), "w", encoding="utf-8") as fh:
            fh.write(blob)

    def _observe(outputs):
        for out in outputs if isinstance(outputs, list) else [outputs]:
            seen = int((out.get("usage") or {}).get("completion_tokens") or 0)
            state["max_tokens_seen"] = max(state["max_tokens_seen"], seen)
            for choice in out.get("choices") or []:
                state["observed"] += 1
                if choice.get("finish_reason") == "length":
                    state["length"] += 1
                    capped[seen] = capped.get(seen, 0) + 1
        if state["observed"] < min_samples:
            return
        # A model that never terminates piles every capped response onto the
        # same ceiling; cap 0 means no usage was reported, so it is unknown.
        cap = max(capped) if capped else 0
        state["cap_hits"] = capped.get(cap, 0)
        if cap > 0 and float(state["cap_hits"]) / state["observed"] >= ratio_limit:
            state["tripped"] = True
            _emit()

    # Wrap whatever is installed now so InferenceX's own parse_generations
    # patch (appended just above) stays in effect. Observation must never break
    # the eval it is watching, hence the guard.
    _hl_prev_parse = _hl_lcc.parse_generations

    def _hl_probe_parse_generations(outputs, **kwargs):
        if not state["tripped"]:
            try:
                _observe(outputs)
            except Exception:
                pass
        return _hl_prev_parse(outputs, **kwargs)

    _hl_lcc.parse_generations = staticmethod(_hl_probe_parse_generations)

    # get_batched_requests creates one task per request up front, and
    # amodel_call builds its payload BEFORE awaiting the inner semaphore, so
    # every payload already carries the large max_tokens by the time the probe
    # trips. Park the tasks in an equally sized outer gate instead. asyncio.run
    # builds a fresh loop per batch and a Semaphore binds to the first loop
    # that awaits it, so the gate is loop-keyed.
    _hl_prev_amodel_call = _hl_api.TemplateAPI.amodel_call
    gate = {"loop": None, "sem": None}

    async def _hl_probe_amodel_call(self, session, sem, messages, **kwargs):
        loop = _hl_asyncio.get_running_loop()
        if gate["loop"] is not loop:
            gate["loop"] = loop
            gate["sem"] = _hl_asyncio.Semaphore(max(1, getattr(self, "_concurrent", 1) or 1))
        async with gate["sem"]:
            if not (state["tripped"] and kwargs.get("generate", True)):
                return await _hl_prev_amodel_call(self, session, sem, messages, **kwargs)
            answers = [""] * len(messages)
            for answer, cache_key in zip(answers, kwargs.get("cache_keys") or []):
                self.cache_hook.add_partial("generate_until", cache_key, answer)
            return answers

    _hl_api.TemplateAPI.amodel_call = _hl_probe_amodel_call


def _hl_eval_model_dir():
    # Upstream's own precedence (get_native_max_context_length): prefer
    # MODEL_PATH, because the served model name may be neither a repo id nor a
    # path. Hyperloom's CLI exports MODEL_PATH and it survives the benchmark env
    # scrub, so it is here.
    raw = (_hl_os.environ.get("MODEL_PATH") or "").strip()
    if not raw:
        return None
    if _hl_os.path.isdir(raw):
        return raw
    # A repo id that was still uncached when Hyperloom assembled this
    # subprocess's env is cached by the time this runs: the server had to
    # download the weights to boot, and boot precedes the eval. Deriving here
    # rather than in the parent is what makes that ordering work for us.
    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(repo_id=raw, filename="config.json")
    except Exception:
        return None
    if isinstance(hit, str) and _hl_os.path.isfile(hit):
        return _hl_os.path.dirname(hit)
    return None


def _hl_eval_read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = _hl_json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _hl_eval_token_text(value):
    # A special token is serialized either bare or as an AddedToken mapping.
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return None


def _hl_eval_derive_terminators():
    # Returns (stop_strings, stop_token_ids) read from the model's own metadata.
    #
    # This exists because lm-eval cannot express what the model declares.
    # eos_string carries exactly one terminator, while generation_config may list
    # several: Qwen3 declares eos_token_id [151645, 151643] and eos_token is only
    # the first, so <|endoftext|> never becomes a stop condition. Worse, the
    # concurrent path (amodel_call, which is the one InferenceX drives) does not
    # pass eos at all, so even the single value never reaches the payload.
    if (_hl_os.environ.get("HYPERLOOM_EVAL_DERIVE_STOP") or "1").strip().lower() in ("0", "false", "no", "off"):
        return [], []
    model_dir = _hl_eval_model_dir()
    if not model_dir:
        return [], []
    tok = _hl_eval_read_json(_hl_os.path.join(model_dir, "tokenizer_config.json"))
    gen = _hl_eval_read_json(_hl_os.path.join(model_dir, "generation_config.json"))
    decoder = tok.get("added_tokens_decoder")
    decoder = decoder if isinstance(decoder, dict) else {}

    ids = gen.get("eos_token_id")
    if isinstance(ids, int) and not isinstance(ids, bool):
        ids = [ids]
    out_ids = []
    out_stops = []
    for tid in ids if isinstance(ids, list) else []:
        # bool is an int in Python, and JSON can carry a true here.
        if not isinstance(tid, int) or isinstance(tid, bool) or tid in out_ids:
            continue
        out_ids.append(tid)
        # added_tokens_decoder is keyed by the id as a string.
        text = _hl_eval_token_text(decoder.get(str(tid)))
        if text and text not in out_stops:
            out_stops.append(text)
    # generation_config is authoritative for what generation stops on, but it is
    # optional; tokenizer_config's eos_token is the fallback and yields no id.
    text = _hl_eval_token_text(tok.get("eos_token"))
    if text and text not in out_stops:
        out_stops.append(text)
    return out_stops, out_ids


def _hl_eval_bounds_install():
    # Bound each request. Distinct from the probe above: the probe answers a
    # model that NEVER terminates, and short-circuits the whole eval to a ~0
    # score. A healthy model whose few hardest reasoning samples do not converge
    # must not be scored that way -- those samples have to be truncated
    # individually so the rest of the measurement survives. The probe's ratio
    # threshold is what makes it safe, and also what makes it unable to help
    # here.
    #
    # The default lives here rather than in the caller's environment on purpose.
    # The gate is differential (baseline_accuracy - new_accuracy <= 0.05), so the
    # ceiling is only sound if every arm shares it; defaulting inside the shim
    # makes that structural instead of a plumbing invariant the baseline and grid
    # call sites each have to remember. HYPERLOOM_EVAL_MAX_TOKENS overrides it;
    # 0 disables the clamp.
    default_cap = 4096
    raw_cap = (_hl_os.environ.get("HYPERLOOM_EVAL_MAX_TOKENS") or "").strip()
    if not raw_cap:
        cap = default_cap
    else:
        try:
            cap = int(raw_cap)
        except (TypeError, ValueError):
            cap = default_cap
        else:
            if cap < 0:
                cap = default_cap
    raw_stop = (_hl_os.environ.get("HYPERLOOM_EVAL_STOP_STRINGS") or "").strip()
    extra_stop = [s for s in raw_stop.split("\\x1f") if s]
    derived_stop, derived_ids = _hl_eval_derive_terminators()
    if cap <= 0 and not extra_stop and not derived_stop and not derived_ids:
        return

    import atexit as _hl_atexit

    from lm_eval.models.openai_completions import LocalChatCompletion as _hl_lcc

    # Truncation is only defensible while it stays rare, so the run has to say
    # how rare it actually was. Without this the ceiling is unfalsifiable: too
    # low silently depresses both arms' scores, too high leaves the tail in
    # place, and neither shows up anywhere.
    counts = {"generations": 0, "truncated": 0}

    def _hl_emit_bounds_summary():
        if counts["generations"] <= 0:
            return
        record = {
            "max_tokens": cap,
            "stop_prefix": extra_stop,
            "derived_stop": derived_stop,
            "derived_stop_token_ids": derived_ids,
            "generations": counts["generations"],
            "truncated": counts["truncated"],
        }
        blob = _hl_json.dumps(record, sort_keys=True)
        print("HYPERLOOM_EVAL_BOUNDS_SUMMARY " + blob, file=_hl_sys.stderr, flush=True)
        out_dir = (_hl_os.environ.get("RESULT_DIR") or "").strip()
        if not out_dir:
            return
        # Must not match results*.json -- that glob is how parse_eval_results
        # finds the accuracy score.
        _hl_os.makedirs(out_dir, exist_ok=True)
        with open(_hl_os.path.join(out_dir, "hyperloom_eval_bounds.json"), "w", encoding="utf-8") as fh:
            fh.write(blob)

    _hl_prev_parse = _hl_lcc.parse_generations

    def _hl_bounds_parse_generations(outputs, **kwargs):
        # Counting must never break the eval it is measuring, hence the guard --
        # same reason the probe guards its own observation above. The payload
        # hook below needs none: it reads defensively instead.
        try:
            for out in outputs if isinstance(outputs, list) else [outputs]:
                for choice in out.get("choices") or []:
                    counts["generations"] += 1
                    if choice.get("finish_reason") == "length":
                        counts["truncated"] += 1
        except Exception:
            pass
        return _hl_prev_parse(outputs, **kwargs)

    _hl_lcc.parse_generations = staticmethod(_hl_bounds_parse_generations)
    _hl_atexit.register(_hl_emit_bounds_summary)

    _hl_prev_create_payload = _hl_lcc._create_payload
    announced = {"done": False}

    def _hl_bounded_create_payload(self, messages, **kwargs):
        payload = _hl_prev_create_payload(self, messages, **kwargs)
        # Only generation carries max_tokens/stop; loglikelihood scoring shares
        # this seam and must pass through untouched.
        if not kwargs.get("generate", False):
            return payload
        if cap > 0:
            current = payload.get("max_tokens")
            current = current if isinstance(current, int) else 0
            if current <= 0 or current > cap:
                payload["max_tokens"] = cap
        if derived_ids:
            # The exact mechanism, and the reason the string list below does not
            # have to fight for room: both frameworks this repo drives (vLLM and
            # SGLang) accept stop_token_ids, it takes token ids rather than text,
            # and it has no length limit.
            ids = list(derived_ids)
            for item in payload.get("stop_token_ids") or []:
                if item not in ids:
                    ids.append(item)
            payload["stop_token_ids"] = ids
        if extra_stop or derived_stop:
            # Order encodes priority under a 4-entry ceiling upstream enforces.
            # An operator who named terminators explicitly outranks everything.
            # The task's own ``until`` comes next: its answer extraction depends
            # on that list, so silently dropping an entry would change what is
            # being scored. Derived terminators go last -- they are the fallback
            # for a server that ignores stop_token_ids, and where that field
            # works they are already covered exactly.
            merged = list(extra_stop)
            for group in (payload.get("stop") or [], derived_stop):
                for item in group:
                    if item not in merged:
                        merged.append(item)
            payload["stop"] = merged[:4]
        if not announced["done"]:
            announced["done"] = True
            print(
                "HYPERLOOM_EVAL_BOUNDS max_tokens=%s stop=%s stop_token_ids=%s"
                % (
                    payload.get("max_tokens"),
                    _hl_json.dumps(payload.get("stop")),
                    _hl_json.dumps(payload.get("stop_token_ids")),
                ),
                file=_hl_sys.stderr,
                flush=True,
            )
        return payload

    _hl_lcc._create_payload = _hl_bounded_create_payload


_hl_eval_probe_install()
_hl_eval_bounds_install()
# --- end HYPERLOOM_EVAL_PROBE -----------------------------------------------
"""

_EVAL_PROBE_SENTINEL = "HYPERLOOM_EVAL_PROBE"
_EVAL_PROBE_LOCK_PATH = str(Path(tempfile.gettempdir()) / "hyperloom_eval_probe_patcher.lock")
# Appending needs no anchor, but it does need this file: upstream renaming or
# moving it puts the probe and the bounds back to warn-only, and the eval runs
# unbounded again. That is what the anchor contract pins in its place.
EVAL_PROBE_TARGET_PARTS = ("utils", "evals", "patches", "lm_eval_sitecustomize.py")


def _discover_inferencex_roots(
    inferencex_path: Path | str | None,
) -> list[Path]:
    """Return every InferenceX checkout root Hyperloom should patch.

    Magpie loads its bundled ``$MAGPIE_PATH/InferenceX`` at runtime, not
    ``$INFERENCEX_PATH``, so all discovered roots (deduped by resolved path)
    are patched: ``inferencex_path`` arg, ``$INFERENCEX_PATH``,
    ``$MAGPIE_PATH/InferenceX``. Returns ``[]`` when none resolve.

    Args:
        inferencex_path: Caller-provided override root to include in the scan.

    Returns:
        A deduped list of resolved InferenceX checkout directories, or ``[]``
        when none resolve.
    """
    roots: list[Path] = []
    seen: set[Path] = set()

    def _add(candidate: Path | str | None) -> None:
        """Resolve and append a candidate root if it is a new directory.

        Args:
            candidate (Path | str | None): A candidate InferenceX root.

        Returns:
            None: Mutates the enclosing ``roots``/``seen`` collections.
        """
        if not candidate:
            return
        try:
            resolved = Path(candidate).expanduser().resolve()
        except OSError:
            return
        if not resolved.is_dir():
            return
        if resolved in seen:
            return
        seen.add(resolved)
        roots.append(resolved)

    _add(inferencex_path)
    _add(os.environ.get("INFERENCEX_PATH", "").strip() or None)
    magpie_dir = (os.environ.get("MAGPIE_PATH") or "").strip()
    if magpie_dir:
        _add(Path(magpie_dir) / "InferenceX")
    return roots


def _resolve_inferencex_files(
    inferencex_path: Path | str | None,
    *rel_parts: str,
) -> list[Path]:
    """Return every existing ``<root>/<*rel_parts>`` across discovered roots.

    One entry per :func:`_discover_inferencex_roots` root whose joined relative
    path is an existing file. ``[]`` = skip patching.

    Args:
        inferencex_path: Caller-provided override root to include in the scan.
        *rel_parts: Relative path components joined onto each discovered root.

    Returns:
        A list of existing files, or ``[]`` when none exist.
    """
    out: list[Path] = []
    for root in _discover_inferencex_roots(inferencex_path):
        candidate = root.joinpath(*rel_parts)
        if candidate.is_file():
            out.append(candidate)
    return out


def _resolve_benchmark_lib_paths(
    inferencex_path: Path | str | None,
) -> list[Path]:
    """Return every existing ``<root>/benchmarks/benchmark_lib.sh`` to patch
    (one per :func:`_discover_inferencex_roots` root). ``[]`` = skip patching.

    Args:
        inferencex_path: Caller-provided override root to include in the scan.

    Returns:
        A list of existing ``benchmark_lib.sh`` paths, or ``[]`` when none
        exist.
    """
    return _resolve_inferencex_files(inferencex_path, "benchmarks", "benchmark_lib.sh")


def _is_patched(src: Path) -> bool:
    """Return whether ``benchmark_lib.sh`` already carries the patch.

    Args:
        src (Path): The ``benchmark_lib.sh`` file to inspect.

    Returns:
        bool: ``True`` if the patch sentinel is present; ``False`` on a
        miss or read error.
    """
    return file_contains_sentinel(src, _PATCH_SENTINEL, log, "_inferencex_patcher")


def _apply_line_replacement_atomic(
    src: Path,
    legacy: str,
    patched_line: str,
    *,
    tmp_prefix: str,
    missing_msg: str,
    success_msg: str,
) -> bool:
    """Replace a single exact ``legacy`` line with ``patched_line`` in ``src``
    via temp-file + atomic rename so a crash mid-write cannot leave a corrupt
    file.

    Shared by both InferenceX patches (``benchmark_lib.sh`` and
    ``benchmark_serving.py``); they differ only in the legacy/patched text,
    temp-file prefix, and log messages.

    Args:
        src: The file to patch in place.
        legacy: Exact legacy line that must be present to patch.
        patched_line: Replacement text for ``legacy`` (first occurrence).
        tmp_prefix: Temp-file prefix for the atomic write.
        missing_msg: Warning (one ``%s`` for ``src``) when ``legacy`` is absent.
        success_msg: Info (one ``%s`` for ``src``) logged on a successful write.

    Returns:
        bool: ``True`` when the patched bytes were written; ``False`` when the
        legacy line is missing or any IO step fails.
    """
    try:
        original = src.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("_inferencex_patcher: cannot read %s: %s", src, e)
        return False

    if legacy not in original:
        log.warning(missing_msg, src)
        return False

    patched = original.replace(legacy, patched_line, 1)
    if patched == original:
        return False

    if not atomic_write_text(
        src,
        patched,
        tmp_prefix=tmp_prefix,
        log_prefix="_inferencex_patcher",
    ):
        return False

    log.info(success_msg, src)
    return True


def _ensure_patched(
    sources: list[Path],
    is_patched: Callable[[Path], bool],
    apply_patch: Callable[[Path], bool],
    lock_path: str,
    *,
    empty_msg: str,
    failure_msg: str,
) -> bool:
    """Drive a set of discovered files to patched state.

    Empty fast-path: ``log.info(empty_msg)`` + ``False``. All-already-patched
    fast-path skips the lock. Otherwise, under the lock, each source is
    re-checked and patched; a failed apply emits ``log.warning(failure_msg,
    src)`` and the remaining roots are still attempted.

    Args:
        sources: Discovered files to patch.
        is_patched: "Already patched?" predicate for one file.
        apply_patch: In-place atomic patcher for one file (True on success).
        lock_path: Cross-process lock file path.
        empty_msg: Info message logged when ``sources`` is empty.
        failure_msg: Warning message (one ``%s`` for ``src``) on apply failure.

    Returns:
        True when at least one source is patched (or already patched), False
        when none could be patched.
    """
    if not sources:
        log.info(empty_msg)
        return False

    # Patch every discovered InferenceX root, not just the first.
    if all(is_patched(s) for s in sources):
        return True  # all already patched, fast-path no lock

    any_patched = False
    with best_effort_file_lock(lock_path, label="_inferencex_patcher"):
        for src in sources:
            # Re-check under the lock (another process may have patched).
            if is_patched(src):
                any_patched = True
                continue
            if apply_patch(src):
                any_patched = True
            else:
                log.warning(failure_msg, src)
    return any_patched


def ensure_benchmark_lib_patched(
    inferencex_path: Path | str | None = None,
) -> bool:
    """Ensure InferenceX ``benchmark_lib.sh`` honours ``$NUM_PROMPTS``.

    Returns ``True`` when patched at exit, ``False`` (non-fatal) when the file
    is missing or the legacy line is absent. Concurrency-safe (flock +
    atomic rename; already-patched fast-path skips the lock).

    Args:
        inferencex_path: Caller-provided override root; defaults to env-based
            discovery when ``None``.

    Returns:
        True when at least one discovered ``benchmark_lib.sh`` is patched (or
        already patched), False when none could be patched.
    """
    return _ensure_patched(
        _resolve_benchmark_lib_paths(inferencex_path),
        _is_patched,
        # Preserve perms so the patched file stays runnable as a sourced lib.
        partial(
            _apply_line_replacement_atomic,
            legacy=_LEGACY_LINE,
            patched_line=_PATCHED_LINE,
            tmp_prefix=".benchmark_lib.sh.hyperloom_",
            missing_msg=(
                "_inferencex_patcher: expected legacy line not found in %s; "
                "the file may already have been hand-patched to a "
                "different shape, or the upstream layout has changed. "
                "Manual review needed."
            ),
            success_msg=("_inferencex_patcher: applied NUM_PROMPTS-respecting patch to %s (Hyperloom issue #194 §2)"),
        ),
        _LOCK_PATH,
        empty_msg=(
            "_inferencex_patcher: no InferenceX root discovered "
            "(checked $INFERENCEX_PATH, $MAGPIE_PATH/InferenceX) or "
            "benchmark_lib.sh missing — skipping patch (this is fine "
            "for tests and dry-runs without a real InferenceX tree)"
        ),
        failure_msg=("_inferencex_patcher: failed to patch %s; other discovered roots will still be attempted"),
    )


# =====================================================================
# PROFILE_EXTRA_BODY consumer patch for benchmark_serving.py
# =====================================================================
def _resolve_benchmark_serving_paths(
    inferencex_path: Path | str | None,
) -> list[Path]:
    """Return every existing
    ``<root>/utils/bench_serving/benchmark_serving.py`` to patch (one per
    :func:`_discover_inferencex_roots` root, including Magpie's bundled copy).
    Independent of the benchmark_lib.sh resolver.

    Args:
        inferencex_path: Caller-provided override root to include in the scan.

    Returns:
        A list of existing ``benchmark_serving.py`` paths, or ``[]`` when none
        exist.
    """
    return _resolve_inferencex_files(inferencex_path, "utils", "bench_serving", "benchmark_serving.py")


def _is_benchmark_serving_patched(src: Path) -> bool:
    """Return whether ``benchmark_serving.py`` already carries the patch.

    Args:
        src (Path): The ``benchmark_serving.py`` file to inspect.

    Returns:
        bool: ``True`` if the ``PROFILE_EXTRA_BODY`` sentinel is present;
        ``False`` on a miss or read error.
    """
    return file_contains_sentinel(src, _BENCH_SERVING_SENTINEL, log, "_inferencex_patcher")


def ensure_benchmark_serving_patched(
    inferencex_path: Path | str | None = None,
) -> bool:
    """Ensure InferenceX ``benchmark_serving.py`` reads ``PROFILE_EXTRA_BODY``
    on ``/start_profile``.

    Returns ``True`` when patched at exit, ``False`` (non-fatal) when missing.
    Concurrency-safe; independent lock file from
    :func:`ensure_benchmark_lib_patched` so the two patches don't serialize.

    Args:
        inferencex_path: Caller-provided override root; defaults to env-based
            discovery when ``None``.

    Returns:
        True when at least one discovered ``benchmark_serving.py`` is patched
        (or already patched), False when none could be patched.
    """
    return _ensure_patched(
        _resolve_benchmark_serving_paths(inferencex_path),
        _is_benchmark_serving_patched,
        partial(
            _apply_line_replacement_atomic,
            legacy=_BENCH_SERVING_LEGACY,
            patched_line=_BENCH_SERVING_PATCHED,
            tmp_prefix=".benchmark_serving.py.hyperloom_",
            missing_msg=(
                "_inferencex_patcher: expected legacy `extra_body=` line not "
                "found in %s; InferenceX layout may have changed and Hyperloom "
                "needs an updated patch. PROFILE_EXTRA_BODY env var will be "
                "ignored — TraceLens shape_discovery / roofline_annotations / "
                "steady-state start_step won't reach the server. Manual review "
                "needed."
            ),
            success_msg=(
                "_inferencex_patcher: patched %s to consume PROFILE_EXTRA_BODY env "
                "var (PR-D §2: fixes silently-ignored shape_discovery / "
                "roofline_annotations / steady-state start_step from "
                "_workload_envs.py)"
            ),
        ),
        _BENCH_SERVING_LOCK_PATH,
        empty_msg=(
            "_inferencex_patcher: no InferenceX root discovered "
            "(checked $INFERENCEX_PATH, $MAGPIE_PATH/InferenceX) or "
            "benchmark_serving.py missing — skipping PROFILE_EXTRA_BODY "
            "patch (this is fine for tests and dry-runs without a real "
            "InferenceX tree)"
        ),
        failure_msg=(
            "_inferencex_patcher: failed to PROFILE_EXTRA_BODY-patch %s; other discovered roots will still be attempted"
        ),
    )


def _is_eval_dest_patched(src: Path) -> bool:
    """Return whether ``benchmark_lib.sh`` already redirects eval artifacts to
    ``$RESULT_DIR`` (the eval-dest sentinel is present).

    Args:
        src (Path): The ``benchmark_lib.sh`` file to inspect.

    Returns:
        bool: ``True`` if the eval-dest sentinel is present; ``False`` on a
        miss or read error.
    """
    return file_contains_sentinel(src, _EVAL_DEST_SENTINEL, log, "_inferencex_patcher")


def ensure_benchmark_lib_eval_dest_patched(
    inferencex_path: Path | str | None = None,
) -> bool:
    """Ensure ``append_lm_eval_summary`` moves eval artifacts to ``$RESULT_DIR``
    instead of the process cwd (the InferenceX checkout).

    Returns ``True`` when patched at exit, ``False`` (non-fatal) when the file
    is missing or the legacy line is absent (falls back to the scan-side
    salvage in :mod:`benchmark_result`). Concurrency-safe; independent lock so
    it does not serialize with the NUM_PROMPTS patch on the same file.

    Args:
        inferencex_path: Caller-provided override root; defaults to env-based
            discovery when ``None``.

    Returns:
        True when at least one discovered ``benchmark_lib.sh`` is patched (or
        already patched), False when none could be patched.
    """
    return _ensure_patched(
        _resolve_benchmark_lib_paths(inferencex_path),
        _is_eval_dest_patched,
        partial(
            _apply_line_replacement_atomic,
            legacy=_EVAL_DEST_LEGACY,
            patched_line=_EVAL_DEST_PATCHED,
            tmp_prefix=".benchmark_lib.sh.eval_dest_",
            missing_msg=(
                "_inferencex_patcher: expected eval-artifact ``mv ./`` line not "
                "found in %s; upstream layout may have changed. Eval artifacts "
                "will land in the process cwd (InferenceX checkout) and be "
                "recovered by the benchmark_result scan-side salvage instead."
            ),
            success_msg=("_inferencex_patcher: redirected eval artifacts to $RESULT_DIR in %s"),
        ),
        _EVAL_DEST_LOCK_PATH,
        empty_msg=(
            "_inferencex_patcher: no InferenceX root discovered "
            "(checked $INFERENCEX_PATH, $MAGPIE_PATH/InferenceX) or "
            "benchmark_lib.sh missing — skipping eval-dest patch (fine for "
            "tests and dry-runs without a real InferenceX tree)"
        ),
        failure_msg=(
            "_inferencex_patcher: failed to eval-dest-patch %s; other discovered roots will still be attempted"
        ),
    )


def _is_eval_start_patched(src: Path) -> bool:
    """Return whether ``benchmark_lib.sh`` already emits the eval-start sentinel.

    Args:
        src (Path): The ``benchmark_lib.sh`` file to inspect.

    Returns:
        bool: ``True`` if the eval-start sentinel is present; ``False`` on a
        miss or read error.
    """
    return file_contains_sentinel(src, _EVAL_START_SENTINEL, log, "_inferencex_patcher")


def ensure_benchmark_lib_eval_start_patched(
    inferencex_path: Path | str | None = None,
) -> bool:
    """Ensure ``run_eval`` announces the benchmark→eval boundary on stderr.

    The explore overtime kill bounds the throughput phase only; without this
    marker the eval's wall-clock is charged against a throughput-only anchor and
    every gated variant is killed. Returns ``True`` when patched at exit,
    ``False`` (non-fatal) when the file is missing or the anchor line is absent —
    the deadline then behaves as before. Concurrency-safe; independent lock so it
    does not serialize with the other patches on the same file.

    Args:
        inferencex_path: Caller-provided override root; defaults to env-based
            discovery when ``None``.

    Returns:
        True when at least one discovered ``benchmark_lib.sh`` is patched (or
        already patched), False when none could be patched.
    """
    return _ensure_patched(
        _resolve_benchmark_lib_paths(inferencex_path),
        _is_eval_start_patched,
        partial(
            _apply_line_replacement_atomic,
            legacy=_EVAL_START_LEGACY,
            patched_line=_EVAL_START_PATCHED,
            tmp_prefix=".benchmark_lib.sh.eval_start_",
            missing_msg=(
                "_inferencex_patcher: expected EVAL_RESULT_DIR export not found "
                "in %s; upstream layout may have changed. The overtime kill will "
                "keep charging accuracy-eval time against the throughput anchor."
            ),
            success_msg=("_inferencex_patcher: added eval-start marker to %s"),
        ),
        _EVAL_START_LOCK_PATH,
        empty_msg=(
            "_inferencex_patcher: no InferenceX root discovered "
            "(checked $INFERENCEX_PATH, $MAGPIE_PATH/InferenceX) or "
            "benchmark_lib.sh missing — skipping eval-start patch (fine for "
            "tests and dry-runs without a real InferenceX tree)"
        ),
        failure_msg=(
            "_inferencex_patcher: failed to eval-start-patch %s; other discovered roots will still be attempted"
        ),
    )


def _resolve_eval_sitecustomize_paths(
    inferencex_path: Path | str | None,
) -> list[Path]:
    """Return every existing ``<root>/utils/evals/patches/lm_eval_sitecustomize.py``."""
    return _resolve_inferencex_files(inferencex_path, *EVAL_PROBE_TARGET_PARTS)


def eval_probe_targets_exist(inferencex_path: Path | str | None = None) -> bool:
    """Whether any discovered InferenceX root carries the probe target file.

    Separates "present and unpatchable" from "laid out somewhere we do not look".
    """
    return bool(_resolve_eval_sitecustomize_paths(inferencex_path))


def _is_eval_probe_patched(src: Path) -> bool:
    """Return whether ``lm_eval_sitecustomize.py`` already carries the eval probe."""
    return file_contains_sentinel(src, _EVAL_PROBE_SENTINEL, log, "_inferencex_patcher")


def _apply_eval_probe_atomic(src: Path) -> bool:
    """Append the eval probe to ``src`` via temp-file + atomic rename."""
    try:
        original = src.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("_inferencex_patcher: cannot read %s: %s", src, e)
        return False
    patched = original + _EVAL_PROBE_PY
    if not atomic_write_text(
        src,
        patched,
        tmp_prefix=".lm_eval_sitecustomize.eval_probe_",
        log_prefix="_inferencex_patcher",
    ):
        return False
    log.info("_inferencex_patcher: appended eval generation-pathology probe to %s", src)
    return True


def ensure_eval_probe_patched(
    inferencex_path: Path | str | None = None,
) -> bool:
    """Ensure the early-exit probe is appended to ``lm_eval_sitecustomize.py``.

    The probe watches ``finish_reason`` on completed responses; once the pattern
    says the model never terminates it short-circuits the remaining requests, so
    lm-eval finishes in seconds with a ~0 score instead of burning the budget.

    Args:
        inferencex_path: Caller-provided override root; defaults to env-based
            discovery when ``None``.

    Returns:
        True when at least one target is patched (or already patched); False when
        none were found or none could be patched. :func:`eval_probe_targets_exist`
        tells the two apart.
    """
    return _ensure_patched(
        _resolve_eval_sitecustomize_paths(inferencex_path),
        _is_eval_probe_patched,
        _apply_eval_probe_atomic,
        _EVAL_PROBE_LOCK_PATH,
        empty_msg=(
            "_inferencex_patcher: no InferenceX root discovered "
            "(checked $INFERENCEX_PATH, $MAGPIE_PATH/InferenceX) or "
            "utils/evals/patches/lm_eval_sitecustomize.py missing — "
            "skipping eval-probe patch"
        ),
        failure_msg=(
            "_inferencex_patcher: failed to append eval-probe to %s; other discovered roots will still be attempted"
        ),
    )


# =====================================================================
# Anchor contract
# =====================================================================
# The probe no longer has an anchor to rot: it is appended to a real file, and
# ``eval_probe_targets_exist`` already separates "present and unpatchable" from
# "laid out somewhere we do not look". The four patches below are still gated on
# locating exact upstream text, which makes a cosmetic upstream edit
# indistinguishable from "nothing to patch": the ``ensure_*`` call returns False,
# no caller reads it, and the run proceeds with the patch silently absent -- the
# same failure mode that took the probe offline before it was re-homed.
# Verification is therefore separate from patching, so a caller can assert the
# contract instead of inferring it from a boolean nobody reads.


@dataclass(frozen=True)
class AnchorStatus:
    """Whether one patch can still find its place in one resolved file."""

    name: str
    path: Path
    patched: bool
    hits: int

    @property
    def ok(self) -> bool:
        """True when the patch is applied, or applicable exactly once.

        Two or more hits is a failure, not a success: every patch here rewrites
        a single site, so an ambiguous anchor means the file drifted into a
        shape the patcher was never written for.
        """
        return self.patched or self.hits == 1

    def describe(self) -> str:
        """Return a one-line human summary for logs and preflight output."""
        if self.patched:
            state = "already patched"
        elif self.hits == 1:
            state = "anchor found"
        elif self.hits == 0:
            state = "ANCHOR MISSING — upstream text changed"
        else:
            state = f"ANCHOR AMBIGUOUS — matched {self.hits} sites, expected 1"
        return f"{self.name}: {state} ({self.path})"


# name -> (relative path parts, sentinel, anchor)
_ANCHOR_CONTRACT: tuple[tuple[str, tuple[str, ...], str, str], ...] = (
    ("num_prompts", ("benchmarks", "benchmark_lib.sh"), _PATCH_SENTINEL, _LEGACY_LINE),
    ("eval_dest", ("benchmarks", "benchmark_lib.sh"), _EVAL_DEST_SENTINEL, _EVAL_DEST_LEGACY),
    ("eval_start", ("benchmarks", "benchmark_lib.sh"), _EVAL_START_SENTINEL, _EVAL_START_LEGACY),
    (
        "profile_extra_body",
        ("utils", "bench_serving", "benchmark_serving.py"),
        _BENCH_SERVING_SENTINEL,
        _BENCH_SERVING_LEGACY,
    ),
)


def count_anchor_hits(text: str, anchor: str) -> int:
    """Return how many sites in ``text`` the given anchor would rewrite.

    Args:
        text: Full contents of the file the patch targets.
        anchor: The literal upstream text the patch replaces.

    Returns:
        The number of matching sites.
    """
    return text.count(anchor)


def verify_patch_anchors(
    inferencex_path: Path | str | None = None,
) -> list[AnchorStatus]:
    """Report, per patch and per discovered file, whether the anchor still holds.

    Read-only: this never writes to the checkout, so it is safe to call before
    patching, after patching, or from preflight. Files that do not exist are
    omitted rather than reported as failures -- a tree without
    ``benchmark_serving.py`` has nothing to patch, which the ``ensure_*``
    functions already treat as a skip.

    Args:
        inferencex_path: Caller-provided override root; defaults to env-based
            discovery when ``None``.

    Returns:
        One :class:`AnchorStatus` per (patch, existing file) pair, in
        ``_ANCHOR_CONTRACT`` order. Empty when no InferenceX tree resolves.
    """
    out: list[AnchorStatus] = []
    for name, rel_parts, sentinel, anchor in _ANCHOR_CONTRACT:
        for path in _resolve_inferencex_files(inferencex_path, *rel_parts):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                log.warning("_inferencex_patcher: cannot read %s for anchor check: %s", path, exc)
                continue
            out.append(
                AnchorStatus(
                    name=name,
                    path=path,
                    patched=sentinel in text,
                    hits=count_anchor_hits(text, anchor),
                )
            )
    return out


def failed_patch_anchors(
    inferencex_path: Path | str | None = None,
) -> list[AnchorStatus]:
    """Return only the anchors that no longer hold. Empty means the contract is intact.

    Args:
        inferencex_path: Caller-provided override root; defaults to env-based
            discovery when ``None``.

    Returns:
        The failing subset of :func:`verify_patch_anchors`.
    """
    return [status for status in verify_patch_anchors(inferencex_path) if not status.ok]


__all__ = [
    "AnchorStatus",
    "count_anchor_hits",
    "ensure_benchmark_lib_patched",
    "ensure_benchmark_lib_eval_dest_patched",
    "ensure_benchmark_lib_eval_start_patched",
    "ensure_benchmark_serving_patched",
    "ensure_eval_probe_patched",
    "failed_patch_anchors",
    "verify_patch_anchors",
]
