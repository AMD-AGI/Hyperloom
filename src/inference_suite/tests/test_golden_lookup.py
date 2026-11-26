# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A candidate's draft is answered with the golden acceptance the staged InferenceX records for it.

What is pinned: the value comes with the staged curve file it was read from (InferenceX's repository, the pin and the
file's sha256); InferenceX is asked about the draft with its acceptance simulation taken out; a model the staged master
config gives no prefix, a draft InferenceX refuses, and a framework it keys no curve by each come back as a reason,
not a value; args that draft nothing come back ``None``; and a staging the lookup cannot read, or an interpreter that
fails, raises :class:`GoldenLookupError` rather than answering.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from inference_suite import GoldenLookupError, SourceGolden, candidate_golden
from inference_suite.pins import INFERENCEX_CLIENT_COMMIT
from inference_suite.recipe.inferencex import GOLDEN_DIR, INFERENCEX_REPO
from inference_suite.staging import AGENTX_PYTHON, GOLDEN_PACKAGE, MASTER_CONFIG, inferencex_root

_MODEL = "org/model-fp8"
_CURVE = "mp__mtp__native__default-default"
_CURVE_TEXT = "mp:\n  thinking_on:\n    3: 2.32\n"
#: Stands in for InferenceX's golden_al_distribution: one MTP curve, and a record of what it was asked.
_PACKAGE = f'''
import json
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent


def curve_name(model, spec, golden_dir, verify=None):
    if spec.get("method") != "mtp":
        raise ValueError(f"no golden curve for {{model}} {{spec.get('method')}}")
    return "{_CURVE}"


def golden_length(model, spec, thinking, golden_dir, verify=None):
    asked = {{"model": model, "spec": spec, "thinking": thinking, "verify": verify}}
    (golden_dir / "asked.json").write_text(json.dumps(asked))
    return {{1: 1.65, 3: 2.32}}[spec["num_speculative_tokens"]]
'''
_MTP3 = ("--speculative-config", '{"method": "mtp", "num_speculative_tokens": 3}')


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    """A tool cache staging the lookup's part of InferenceX at the pin, run by this interpreter."""
    root = inferencex_root(tmp_path)
    files = {
        "infx/__init__.py": "",
        f"{GOLDEN_PACKAGE}/__init__.py": _PACKAGE,
        f"{GOLDEN_PACKAGE}/{_CURVE}.yaml": _CURVE_TEXT,
        MASTER_CONFIG: f"point-a:\n  model: {_MODEL}\n  model-prefix: mp\n",
    }
    for relative, text in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text)
    (root / AGENTX_PYTHON).parent.mkdir(parents=True)
    (root / AGENTX_PYTHON).symlink_to(sys.executable)
    return tmp_path


def _asked(cache: Path) -> dict[str, object]:
    return json.loads((inferencex_root(cache) / GOLDEN_PACKAGE / "asked.json").read_text())


def test_a_draft_inferencex_records_is_answered_with_its_value_and_the_staged_curve_it_was_read_from(cache) -> None:
    golden = candidate_golden(cache, "vllm", _MODEL, _MTP3)

    sha256 = hashlib.sha256(_CURVE_TEXT.encode()).hexdigest()
    assert golden is not None and golden.value == 2.32 and golden.reason is None
    assert golden.curve is not None
    assert (golden.curve.repo, golden.curve.commit) == (INFERENCEX_REPO, INFERENCEX_CLIENT_COMMIT)
    assert (golden.curve.path, golden.curve.sha256) == (f"{GOLDEN_DIR}/{_CURVE}.yaml", sha256)
    assert _asked(cache) == {
        "model": "mp",
        "spec": {"method": "mtp", "num_speculative_tokens": 3},
        "thinking": "thinking_on",
        "verify": "standard",
    }


def test_a_draft_spelled_with_per_key_flags_is_asked_about_as_its_json_spelling_is(cache) -> None:
    dotted = ("--speculative-config.method", "mtp", "--speculative-config.num_speculative_tokens", "3")

    golden = candidate_golden(cache, "vllm", _MODEL, dotted)

    assert golden is not None and golden.value == 2.32
    assert _asked(cache)["spec"] == {"method": "mtp", "num_speculative_tokens": 3}


def test_a_draft_that_already_forces_an_acceptance_is_asked_about_without_it(cache) -> None:
    spec = {"method": "mtp", "num_speculative_tokens": 3, "rejection_sample_method": "synthetic"}
    forced = ("--speculative-config", json.dumps({**spec, "synthetic_acceptance_length": 9.9}))

    golden = candidate_golden(cache, "vllm", _MODEL, forced)

    assert golden is not None and golden.value == 2.32
    assert _asked(cache)["spec"] == {"method": "mtp", "num_speculative_tokens": 3}
    assert _asked(cache)["verify"] == "standard"


@pytest.mark.parametrize(
    ("framework", "model", "args", "reason"),
    [
        ("vllm", "org/unknown", _MTP3, "InferenceX names no model prefix for org/unknown"),
        ("sglang", _MODEL, ("--speculative-algorithm", "EAGLE", "--speculative-num-steps", "3"), "no golden curve"),
        ("trtllm", _MODEL, ("--speculative-config", "{}"), "drafts nothing InferenceX keys a golden curve by"),
    ],
)
def test_a_draft_inferencex_records_nothing_for_is_answered_with_why(cache, framework, model, args, reason) -> None:
    golden = candidate_golden(cache, framework, model, args)

    assert isinstance(golden, SourceGolden) and golden.value is None and golden.curve is None
    assert reason in str(golden.reason)


def test_args_that_draft_nothing_are_answered_with_none(cache) -> None:
    assert candidate_golden(cache, "vllm", _MODEL, ("--max-num-seqs", "64")) is None


def test_a_staging_without_the_master_config_raises(cache) -> None:
    (inferencex_root(cache) / MASTER_CONFIG).unlink()

    with pytest.raises(GoldenLookupError, match=MASTER_CONFIG):
        candidate_golden(cache, "vllm", _MODEL, _MTP3)


def test_an_interpreter_that_fails_raises_with_its_error(cache) -> None:
    (inferencex_root(cache) / GOLDEN_PACKAGE / "__init__.py").write_text("raise RuntimeError('broken staging')\n")

    with pytest.raises(GoldenLookupError, match="broken staging"):
        candidate_golden(cache, "vllm", _MODEL, _MTP3)
