# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Two identical-recipe accuracy tests on a GPU machine: how far their greedy decode outputs agree (W55).

Needs the same setup as ``test_toolbox_kinds_gpu`` (``HL_W13_MODEL``, ``HL_W13_INFERENCEX``, ``HL_W13_OUT``). The
test prints, per pair of runs, how many of the prompts decode token for token alike and where the others first differ;
it asserts only that both runs carry a pass and that the pass identities agree.
"""

from __future__ import annotations


from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom.contracts.recipe import RecipeDelta
from hyperloom.measure.decode import read_decode_pass
from hyperloom_multibranch.toolbox.tests.helpers import req
from hyperloom_multibranch.toolbox.tests.test_toolbox_kinds_gpu import (  # noqa: F401  (workload_environment is a fixture)
    _probe,
    _run_root,
    _submit,
    _toolbox,
    pytestmark,
    workload_environment,
)

_PROMPT_SET = 32


def _first_difference(left: tuple[int, ...], right: tuple[int, ...]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


async def _baseline(name: str):
    root = _run_root(name)
    toolbox, notebook, *_ = _toolbox(root, f"w55-{name}")
    try:
        request = req(kind=TestKind.ACCURACY, purpose=Purpose.BASELINE, accuracy=True, change=RecipeDelta())
        _, result, landed = await _submit(toolbox, notebook, request, root)
        return result, root / "session"
    finally:
        await toolbox.close()


async def test_two_identical_recipe_baselines_decode_alike_up_to_batch_noise() -> None:
    first, first_session = await _baseline("decode-a")
    second, second_session = await _baseline("decode-b")

    assert first.decode_outputs and second.decode_outputs, (first.nonfatal_warnings, second.nonfatal_warnings)
    left, right = read_decode_pass(first, first_session), read_decode_pass(second, second_session)
    assert left.identity == right.identity and len(left.outputs) == len(right.outputs) == _PROMPT_SET
    differences = [_first_difference(a, b) for a, b in zip(left.outputs, right.outputs, strict=True)]
    _probe(
        "decode",
        identity=left.identity,
        identical=sum(d is None for d in differences),
        of=_PROMPT_SET,
        first_divergence=[d for d in differences if d is not None],
        lengths=[len(t) for t in left.outputs],
        warnings=[first.nonfatal_warnings, second.nonfatal_warnings],
    )
