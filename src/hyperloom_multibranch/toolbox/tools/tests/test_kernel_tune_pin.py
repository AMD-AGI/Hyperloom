# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which aiter ``kernel_tune`` pins the tuner to: the install whose ``aiter/configs`` the toolbox's tests read, found by its
layout and never guessed (an install that cannot be found is a refusal, not a default)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom_multibranch.toolbox.tools.base import ToolFailure
from hyperloom_multibranch.toolbox.tools.kernel_tune import serving_aiter_root


def install(root: Path, *, tuner_scripts: str | None = "csrc") -> Path:
    """An aiter install in ``root``: ``aiter/configs`` (returned), ``aiter/jit/core.py`` and the tuner scripts dir."""
    configs = root / "aiter" / "configs"
    configs.mkdir(parents=True)
    (root / "aiter" / "jit").mkdir()
    (root / "aiter" / "jit" / "core.py").write_text("", encoding="utf-8")
    if tuner_scripts:
        (root / tuner_scripts).mkdir(parents=True)
    return configs


def test_the_root_is_the_directory_around_the_package_whose_configs_the_tests_read(tmp_path: Path) -> None:
    assert serving_aiter_root(install(tmp_path / "image")) == tmp_path / "image"


def test_a_packaged_install_keeps_its_tuner_scripts_in_aiter_meta_and_that_is_the_root(tmp_path: Path) -> None:
    configs = install(tmp_path / "site", tuner_scripts="aiter_meta/csrc")

    assert serving_aiter_root(configs) == tmp_path / "site" / "aiter_meta"


def test_a_relative_path_names_the_same_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install(tmp_path / "image")
    monkeypatch.chdir(tmp_path / "image")

    assert serving_aiter_root(Path("aiter/configs")) == tmp_path / "image"


@pytest.mark.parametrize("flaw", ["not-called-configs", "no-core-py", "no-tuner-scripts", "no-configs-dir"])
def test_a_directory_that_is_not_the_configs_of_an_install_is_refused_naming_the_pin(tmp_path: Path, flaw: str) -> None:
    configs = install(tmp_path / "image", tuner_scripts=None if flaw == "no-tuner-scripts" else "csrc")
    if flaw == "not-called-configs":
        configs = configs.rename(configs.with_name("tables"))
    elif flaw == "no-core-py":
        (configs.parent / "jit" / "core.py").unlink()
    elif flaw == "no-configs-dir":
        configs.rmdir()

    with pytest.raises(ToolFailure, match="the aiter the tests' servers run cannot be found") as refused:
        serving_aiter_root(configs)

    assert str(configs) in str(refused.value) and "AITER_ROOT_DIR" in str(refused.value)
