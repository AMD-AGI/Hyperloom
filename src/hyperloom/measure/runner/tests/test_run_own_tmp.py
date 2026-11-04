# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A test whose server names no temporary directory is given one of its own, so newer aiter's merged tables
(``tempfile.gettempdir()/aiter_configs``) are the test's and not shared through ``/tmp`` with other tests on the pod."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from hyperloom.measure import local
from hyperloom.measure.runner.own_tmp import OWN_TMP_PREFIX, own_tmpdir
from hyperloom.measure.runner.tests.test_run_jit import Rig, _arm, env, key, rig  # noqa: F401 - fixtures
from hyperloom.measure.socket import TestCall

MERGED = "aiter_configs/merged.csv"


def _call(tmp_path: Path, envs: dict[str, str]) -> TestCall:
    point = tmp_path / "point.yaml"
    point.write_text(yaml.safe_dump({"benchmark": {"envs": envs}}), encoding="utf-8")
    return TestCall(
        points=(point,), output_dir=tmp_path, timeout_sec=1, cwd=str(tmp_path), magpie_python="", task_id="t"
    )


def test_each_run_of_a_test_writes_below_a_temporary_directory_of_its_own_removed_afterwards(rig, monkeypatch) -> None:
    seen: list[tuple[str, bool]] = []
    original_start = rig._start

    def start(*args, **kwargs):
        run, environ = original_start(*args, **kwargs)
        grid = local.run_grid

        async def serve(**grid_kwargs):
            own = environ.get("TMPDIR", "")
            if own:  # what aiter would do; never into the machine's own /tmp
                (Path(own) / MERGED).parent.mkdir(parents=True)
                (Path(own) / MERGED).write_text(run.name, encoding="utf-8")
            seen.append((own, (Path(own) / MERGED).is_file() if own else False))
            return await grid(**grid_kwargs)

        monkeypatch.setattr(local, "run_grid", serve)
        return run, environ

    monkeypatch.setattr(rig, "_start", start)
    rig.run("first", _arm(rig), cache=False)
    rig.run("second", _arm(rig), cache=False)

    (first, wrote_first), (second, wrote_second) = seen
    assert wrote_first and wrote_second
    assert Path(first).name.startswith(OWN_TMP_PREFIX) and Path(second).name.startswith(OWN_TMP_PREFIX)
    assert first != second, "two tests on one pod do not share the merged view"
    assert not Path(first).exists() and not Path(second).exists(), "removed when the test ends"


@pytest.mark.parametrize(
    ("inherited", "envs", "overrides"),
    [({"TMPDIR": "/inherited"}, {}, {}), ({}, {"TEMP": "/point"}, {}), ({}, {}, {"TMP": "/sheet"})],
)
def test_a_temporary_directory_the_server_is_already_given_is_left_as_it_is(
    tmp_path, inherited, envs, overrides
) -> None:
    call = dataclasses.replace(_call(tmp_path, envs), env_overrides=overrides)
    environ = dict(inherited)

    with own_tmpdir(call, environ) as own:
        assert own is None and environ == inherited
    assert environ == inherited


def test_the_runners_environment_is_put_back_when_the_test_fails(tmp_path) -> None:
    environ = {"TMPDIR": "  "}

    with pytest.raises(RuntimeError, match="server died"), own_tmpdir(_call(tmp_path, {}), environ) as own:
        assert own is not None and environ["TMPDIR"] == str(own) and own.is_dir()
        raise RuntimeError("server died")
    assert environ == {"TMPDIR": "  "} and not own.exists()
