# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A ``site-packages`` or ``dist-packages`` directory is told apart from a source checkout."""

from __future__ import annotations


from hyperloom.common.env_safety import is_python_package_root


class TestIsPythonPackageRoot:
    def test_site_packages_is_package_root(self) -> None:
        assert is_python_package_root("/opt/venv/lib/python3.12/site-packages")

    def test_dist_packages_is_package_root(self) -> None:
        assert is_python_package_root("/usr/lib/python3/dist-packages")

    def test_trailing_slash_still_detected(self) -> None:
        assert is_python_package_root("/opt/venv/lib/python3.12/site-packages/")

    def test_source_checkout_is_not_package_root(self) -> None:
        assert not is_python_package_root("/data/.cache/checkout@abc1234")

    def test_empty_is_not_package_root(self) -> None:
        assert not is_python_package_root("")
        assert not is_python_package_root(None)
