# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The registry derives its launch facts from ``inference_suite.frameworks`` and answers as it did before."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer import framework_registry as fr
from inference_suite.frameworks import Framework

# The answers the registry gave while it held these facts itself.
_PACKAGES = {"sglang": "sglang", "vllm": "vllm", "atom": "atom", "xdit": "xfuser", "custom": None}
_SCRIPTABLE = {"sglang": False, "vllm": False, "atom": False, "xdit": True, "custom": True}
_SGLANG_EP = ("--ep-size", ("--ep-size", "--expert-parallel-size", "--ep"), True)
_VLLM_EP = ("--enable-expert-parallel", ("--enable-expert-parallel", "-ep"), False)
_ATOM_EP = ("--enable-expert-parallel", ("--enable-expert-parallel",), False)
_EP = {
    "sglang": _SGLANG_EP,
    "vllm": _VLLM_EP,
    "atom": _ATOM_EP,
    "xdit": None,
    "custom": None,
    None: _SGLANG_EP,
    "": _SGLANG_EP,
    "tensorrt": _SGLANG_EP,
    "vllm-rocm": _VLLM_EP,
    " ATOM ": _ATOM_EP,
}


def test_the_registry_holds_one_entry_per_framework():
    assert set(fr.FRAMEWORKS) == {framework.value for framework in Framework}
    assert all(spec.name == name for name, spec in fr.FRAMEWORKS.items())


@pytest.mark.parametrize("name", sorted(_PACKAGES))
def test_package_and_kind_are_unchanged(name):
    assert fr.python_package(name) == _PACKAGES[name]
    assert fr.is_scriptable(name) is _SCRIPTABLE[name]


@pytest.mark.parametrize("framework", list(_EP), ids=repr)
def test_expert_parallel_flag_is_unchanged(framework):
    assert fr.expert_parallel_flag(framework) == _EP[framework]


def test_an_unknown_framework_reads_as_the_default_serving_one():
    assert fr.is_scriptable("tensorrt") is False
    with pytest.raises(KeyError):
        fr.python_package("tensorrt")


def test_the_kind_names_keep_their_values():
    assert fr.SERVING == "serving"
    assert fr.SCRIPTABLE == "scriptable"
