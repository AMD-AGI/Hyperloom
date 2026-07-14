# Copyright Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import re

from hyperloom.common.timeutil import iso_z, now_iso, utc_now_compact


def test_now_iso_z_suffix():
    assert now_iso(z_suffix=True).endswith("Z")
    assert "+00:00" in now_iso()


def test_utc_now_compact_shape():
    ts = utc_now_compact()
    assert re.fullmatch(r"\d{8}T\d{6}Z", ts), ts


class TestIsoZ:
    def test_none_and_empty(self):
        assert iso_z(None) == ""
        assert iso_z("") == ""
        assert iso_z("   ") == ""

    def test_z_suffix_second_precision(self):
        assert iso_z("2021-01-01T00:00:00.123456Z") == "2021-01-01T00:00:00Z"

    def test_offset_converted_to_utc(self):
        assert iso_z("2021-01-01T01:00:00+01:00") == "2021-01-01T00:00:00Z"

    def test_naive_assumed_utc(self):
        assert iso_z("2021-01-01T00:00:00") == "2021-01-01T00:00:00Z"

    def test_unparseable_returned_unchanged(self):
        assert iso_z("not-a-timestamp") == "not-a-timestamp"

