# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The golden acceptance InferenceX records for a draft a candidate launches, read from the staged InferenceX.

A base's golden acceptance is read once, when the base is resolved (:mod:`inference_suite.recipe.golden_curve`). A
candidate that turns speculative decoding on is composed later, often on a base that drafts nothing, so
:func:`candidate_golden` answers for its draft from the InferenceX the tool cache stages at its pin: the model prefix
from the staged master config, and InferenceX's own ``golden_length`` and ``curve_name`` run in the staged client's
interpreter, so which curve a draft selects stays InferenceX's decision. Nothing is fetched.

The draft is read with its acceptance simulation taken out (:func:`~inference_suite.golden.real_acceptance`), as
InferenceX reads a draft before it forces a golden length on it, so a candidate that already forces one is asked about
the same curve as one that does not. vLLM's forcing replaces the draft's ``rejection_sample_method`` with the
synthetic sampler, so a forced vLLM draft no longer shows the verifier its curve is selected by: it is asked about
each verifier the sampler can stand in for, and answered with the one whose recorded length it forces. A model's HF
base model is not known here, so a model InferenceX names only through its base model has no prefix.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from inference_suite.argv import argv_flag_occurrences, speculative_config
from inference_suite.contracts.base_launch import GoldenCurve
from inference_suite.golden import real_acceptance
from inference_suite.pins import INFERENCEX_TOOL, PINNED_TOOLS
from inference_suite.recipe.golden_curve import GoldenLength, GoldenRequest, draft_request, launch_draft
from inference_suite.recipe.inferencex import GOLDEN_DIR, INFERENCEX_REPO, MASTER
from inference_suite.recipe.model import RecipeSourceError, RepoBundle, SourceGolden
from inference_suite.recipe.sources import GOLDEN_ANSWER_SOURCE
from inference_suite.staging import AGENTX_PYTHON, GOLDEN_PACKAGE, MASTER_CONFIG, inferencex_root

_TIMEOUT_S = 120
#: The verifiers vLLM's synthetic rejection sampler takes the place of in a forced draft, as InferenceX names them.
_HIDDEN_VERIFIERS = ("standard", "block")
_OUTPUT_TAIL = 2000
# Run as ``python -I -c`` by the staged client's interpreter: reads ``{"root", "request"}`` on stdin and writes the
# answer on stdout; InferenceX's own output goes to stderr.
_CHILD = (
    """
import json
import sys

asked = json.load(sys.stdin)
answer, sys.stdout = sys.stdout, sys.stderr
sys.path.insert(0, asked["root"])
from infx.golden_al_distribution import GOLDEN_DIR, curve_name, golden_length
"""
    + GOLDEN_ANSWER_SOURCE
    + """
answer.write(json.dumps(golden(asked["request"])))
"""
)


class GoldenLookupError(RuntimeError):
    """The staged InferenceX could not be asked: a file it needs is absent, or its interpreter failed."""


def _ask(root: Path, request: GoldenRequest) -> GoldenLength:
    document = json.dumps({"root": str(root), "request": request.to_document()})
    try:
        done = subprocess.run(
            [str(root / AGENTX_PYTHON), "-I", "-c", _CHILD],
            input=document,
            capture_output=True,
            text=True,
            cwd=root,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GoldenLookupError(f"the staged InferenceX at {root} could not be asked: {exc}") from exc
    if done.returncode != 0:
        raise GoldenLookupError(
            f"the staged InferenceX at {root} exited with code {done.returncode}: {done.stderr[-_OUTPUT_TAIL:]}"
        )
    try:
        return GoldenLength.from_document(json.loads(done.stdout))
    except ValueError as exc:
        raise GoldenLookupError(f"the staged InferenceX at {root} answered {done.stdout[-_OUTPUT_TAIL:]!r}") from exc


def _staged_text(root: Path, relative: str) -> str:
    try:
        return (root / relative).read_text(encoding="utf-8")
    except OSError as exc:
        raise GoldenLookupError(f"the staged InferenceX at {root} has no readable {relative}: {exc}") from exc


def _forced_vllm_length(framework: str, args: Sequence[str]) -> float | None:
    """The length ``args`` force under vLLM's synthetic sampler, ``None`` when they force none there."""
    spec = speculative_config(argv_flag_occurrences(args))
    if framework != "vllm" or spec.get("rejection_sample_method") != "synthetic":
        return None
    try:
        return float(spec["synthetic_acceptance_length"])
    except (KeyError, TypeError, ValueError):
        return None


def candidate_golden(tool_cache: Path, framework: str, model_id: str, args: Sequence[str]) -> SourceGolden | None:
    """The golden acceptance InferenceX records for the draft ``args`` launch on ``framework`` for ``model_id``, or why
    it records none; ``None`` when ``args`` draft nothing.

    Raises:
        GoldenLookupError: The tool cache stages no InferenceX the lookup can read, or its interpreter failed.
    """
    root = inferencex_root(tool_cache)
    commit = PINNED_TOOLS[INFERENCEX_TOOL]
    bundle = RepoBundle(INFERENCEX_REPO, commit, {MASTER: _staged_text(root, MASTER_CONFIG)})
    drafted, _ = real_acceptance(args, {})
    try:
        draft = launch_draft(framework, drafted, {})
    except RecipeSourceError as exc:
        return SourceGolden(None, reason=str(exc))
    request = draft_request(draft, model_id, bundle, None)
    if not isinstance(request, GoldenRequest):
        return request
    forced = _forced_vllm_length(framework, args)
    verifiers = _HIDDEN_VERIFIERS if forced is not None else (request.verify,)
    answers = [_ask(root, replace(request, verify=verify)) for verify in verifiers]
    answer = next((asked for asked in answers if asked.value == forced), answers[0])
    if answer.value is None or answer.curve is None:
        return SourceGolden(None, reason=f"InferenceX has no golden curve for it: {answer.error}")
    name = f"{answer.curve}.yaml"
    sha256 = hashlib.sha256(_staged_text(root, f"{GOLDEN_PACKAGE}/{name}").encode()).hexdigest()
    curve = GoldenCurve(repo=INFERENCEX_REPO, commit=commit, path=f"{GOLDEN_DIR}/{name}", sha256=sha256)
    return SourceGolden(answer.value, curve=curve)


def runs_a_draft(framework: str, args: Sequence[str]) -> bool:
    """Whether a ``framework`` launch of ``args`` runs a draft, read as InferenceX keys a curve by it: an SGLang launch
    only once ``--speculative-algorithm`` names one, whatever draft tuning flags it also carries.

    Raises:
        GoldenLookupError: InferenceX reads no drafts of ``framework``.
    """
    try:
        return bool(launch_draft(framework, args, {}).spec)
    except RecipeSourceError as exc:
        raise GoldenLookupError(str(exc)) from exc


__all__ = ["GoldenLookupError", "candidate_golden", "runs_a_draft"]
