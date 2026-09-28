# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the framework agent's shared library surface (repo_map, framework_registry)."""

from __future__ import annotations

from hyperloom.agents.framework import repo_map as _repo_map


def test_repo_url_for_framework_known_and_unknown() -> None:
    assert _repo_map.repo_url_for_framework("sglang").endswith("sglang.git")
    assert _repo_map.repo_url_for_framework("xdit") == "https://github.com/xdit-project/xDiT.git"
    assert _repo_map.repo_url_for_framework("nope") == ""


def test_scriptable_framework_registry_specs() -> None:
    from hyperloom.inference_optimizer import framework_registry

    xdit_spec = framework_registry.FRAMEWORKS["xdit"]
    assert xdit_spec.repo_url == "https://github.com/xdit-project/xDiT.git"
    assert xdit_spec.kind == framework_registry.SCRIPTABLE
    assert xdit_spec.extra_args_env == "EXTRA_XDIT_ARGS"
    assert xdit_spec.throughput_unit == "img/s"
    assert framework_registry.primary_metric_name("xdit") == "e2el_mean_ms"
