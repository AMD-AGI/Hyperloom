# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``grid_server_args`` exposes none of the argv semantics that ``inference_suite.argv`` owns."""

from __future__ import annotations

import inspect

import pytest

from hyperloom.inference_optimizer import grid_server_args
from inference_suite import argv

_PRE_MOVE_PRIVATE_NAMES = ("_split_args_preserving_json", "_reserialize_json_blobs", "_parse_removal_specs")
# multibranch-dev's re-exports, pinned by test_multibranch_dev_compat; removed with 5.12.
_MULTIBRANCH_DEV_SHIMS = frozenset({"remove_server_args", "tokenize_server_args_preserving_json"})


@pytest.mark.parametrize("name", sorted({*argv.__all__, *_PRE_MOVE_PRIVATE_NAMES} - _MULTIBRANCH_DEV_SHIMS))
def test_moved_name_is_not_on_grid_server_args(name):
    assert not hasattr(grid_server_args, name)


def test_no_grid_server_args_attribute_aliases_an_argv_function():
    owned = {id(obj) for name in argv.__all__ if callable(obj := getattr(argv, name))}
    aliases = [
        name
        for name, obj in vars(grid_server_args).items()
        if not inspect.ismodule(obj) and callable(obj) and id(obj) in owned and name not in _MULTIBRANCH_DEV_SHIMS
    ]
    assert aliases == []
