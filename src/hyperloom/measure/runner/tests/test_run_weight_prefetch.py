# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner leaves the checkpoint prefetch as its sandbox's environment sets it: off unless the template turns it on."""

from __future__ import annotations

import pytest

from hyperloom.measure import local
from hyperloom.measure.runner.tests.test_run_jit import _arm, env, key, rig  # noqa: F401 - fixtures
from hyperloom_common.weight_prefetch import WEIGHT_PREFETCH_ENV


@pytest.mark.parametrize(("given", "seen"), [(None, None), ("0", "0"), ("1", "1")])
def test_the_runner_leaves_the_prefetch_as_the_sandbox_sets_it(rig, monkeypatch, given, seen) -> None:
    observed: list[str | None] = []
    original_start = rig._start

    def start(*args, **kwargs):
        run, environ = original_start(*args, **kwargs)
        grid = local.run_grid

        async def serve(**grid_kwargs):
            observed.append(environ.get(WEIGHT_PREFETCH_ENV))
            return await grid(**grid_kwargs)

        monkeypatch.setattr(local, "run_grid", serve)
        return run, environ

    monkeypatch.setattr(rig, "_start", start)
    rig.run("t", _arm(rig), cache=False, extra_env=None if given is None else {WEIGHT_PREFETCH_ENV: given})

    assert observed and set(observed) == {seen}
