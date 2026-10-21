# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The local runner template: the command names the sandbox's own directories, never the global ones."""

from __future__ import annotations

import subprocess
import sys

import pytest

from hyperloom.measure import local_templates
from hyperloom.measure.framework_base import FRAMEWORK_WATCH_ENV, SESSION_DIGEST_ENV, tree_digest
from hyperloom.measure.local_templates import image_env, local_runner_command, local_runner_template


@pytest.fixture(autouse=True)
def _no_framework_here(monkeypatch):
    """What these tests assert must not depend on whether this machine has the image's framework installed."""
    monkeypatch.setattr(local_templates, "DEFAULT_FRAMEWORK_ROOTS", ())
    monkeypatch.setattr(local_templates, "DEFAULT_FRAMEWORK_WATCH", ())


def _expanded(command: tuple[str, ...], root: str) -> str:
    """What the shell would run, with ``exec`` replaced by ``printf`` so nothing starts."""
    script = command[2].replace('exec "$0"', 'printf "%s\\n" "$0"')
    done = subprocess.run(
        [command[0], "-c", script, *command[3:]],
        env={"HYPERLOOM_SANDBOX_ROOT": root, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout


def test_the_command_runs_the_runner_in_the_sandbox_directories() -> None:
    command = local_runner_command("/opt/py", spec_wait_sec=30)
    printed = _expanded(command, "/sb/1")
    assert printed.split() == [
        "/opt/py",
        "-m",
        "hyperloom.measure.runner",
        "run",
        "--in",
        "/sb/1/in",
        "--out",
        "/sb/1/out",
        "--scratch",
        "/sb/1/work/scratch",
        "--spec-wait-sec",
        "30",
    ]


def test_no_global_directory_is_named() -> None:
    script = local_runner_command()[2]
    for forbidden in (" /in", " /out", " /scratch", '"/in', '"/out', '"/scratch'):
        assert forbidden not in script


def test_the_default_python_is_this_interpreter() -> None:
    assert local_runner_command()[3] == sys.executable


def test_the_template_is_a_sealed_signing_measurement_template() -> None:
    template = local_runner_template("r", gpus=2, envs={"A": "b"})
    assert (template.role, template.sealed, template.signing, template.gpus) == ("measurement", True, True, 2)
    assert template.envs == {"A": "b"}
    assert template.command == local_runner_command()


def test_a_non_positive_wait_is_refused() -> None:
    with pytest.raises(ValueError, match="spec_wait_sec"):
        local_runner_command(spec_wait_sec=0)


IMAGE = {
    "SGLANG_USE_AITER": "1",
    "HIP_FORCE_DEV_KERNARG": "1",
    "HSA_NO_SCRATCH_RECLAIM": "1",
    "AITER_COMMIT": "abc",
    "ROCM_QUICK_REDUCE_QUANTIZATION": "INT4",
    "PYTORCH_ROCM_ARCH": "gfx950",
    "TORCHINDUCTOR_MAX_AUTOTUNE": "1",
    "TRITON_COMMIT": "def",
    "LD_LIBRARY_PATH": "/opt/rocm/lib",
}


def test_the_image_variables_are_taken_by_name_and_nothing_else() -> None:
    environ = {**IMAGE, "HOME": "/root", "PATH": "/bin", "KUBERNETES_SERVICE_HOST": "10.0.0.1", "AWS_REGION": "x"}
    assert image_env(environ) == IMAGE


def test_a_credential_shaped_name_is_left_out_even_when_a_prefix_matches() -> None:
    secrets = {
        "SGLANG_API_KEY": "k",
        "HIP_ACCESS_TOKEN": "t",
        "AITER_SECRET": "s",
        "TRITON_PASSWORD": "p",
        "ROCM_CUSTOM_HEADERS": "h",
    }
    assert image_env({**IMAGE, **secrets}) == IMAGE


def test_a_template_without_envs_carries_the_image_variables_of_this_process(monkeypatch) -> None:
    for name, value in IMAGE.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("SGLANG_API_KEY", "k")
    template = local_runner_template()
    assert {k: v for k, v in template.envs.items() if k in IMAGE} == IMAGE
    assert "SGLANG_API_KEY" not in template.envs


def test_explicit_envs_replace_the_image_variables(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_USE_AITER", "1")
    assert local_runner_template(envs={"A": "b"}).envs == {"A": "b"}
    assert local_runner_template(envs={}).envs == {}


def test_the_image_framework_installs_that_exist_here_are_the_roots(tmp_path, monkeypatch) -> None:
    present = tmp_path / "sgl-workspace"
    present.mkdir()
    monkeypatch.setattr(local_templates, "DEFAULT_FRAMEWORK_ROOTS", (str(present), str(tmp_path / "absent")))
    assert local_runner_template().framework_roots == (str(present),)


def test_the_tuned_tables_are_hashed_as_the_session_starts_and_named_for_the_runner(tmp_path, monkeypatch) -> None:
    tables = tmp_path / "configs"
    tables.mkdir()
    (tables / "tuned.csv").write_text("M,N,K\n")
    monkeypatch.setattr(local_templates, "DEFAULT_FRAMEWORK_WATCH", (str(tables),))
    envs = local_runner_template(envs={}).envs
    assert envs == {FRAMEWORK_WATCH_ENV: str(tables), SESSION_DIGEST_ENV: tree_digest([str(tables)])}
    (tables / "tuned.csv").write_text("M,N,K\n64,5120,5120\n")
    assert local_runner_template(envs={}).envs[SESSION_DIGEST_ENV] != envs[SESSION_DIGEST_ENV]


def test_the_digest_follows_names_and_contents_and_counts_a_missing_path_as_absent(tmp_path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.csv").write_text("1")
    paths = [str(tmp_path / "a"), str(tmp_path / "missing")]
    first = tree_digest(paths)
    assert tree_digest(paths) == first
    (tmp_path / "a" / "x.csv").write_text("2")
    assert tree_digest(paths) != first
    (tmp_path / "a" / "x.csv").write_text("1")
    (tmp_path / "a" / "x.csv").rename(tmp_path / "a" / "y.csv")
    assert tree_digest(paths) != first
