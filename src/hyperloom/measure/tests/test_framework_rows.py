# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tuned rows laid over a default table, on lines of aiter's own tables (see ``aiter_excerpts``).

The key a shape is told apart by is read from the image's untuned sibling of the table, as aiter's merge reads it; the
shapes a rows file shares with a default table are the ones aiter would rewrite the files over and raise.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from hyperloom.measure.framework_rows import RowsOnlyRefused, lay_rows_over
from hyperloom.measure.framework_tables import table_key
from hyperloom.measure.tests.aiter_excerpts import (
    A8W8_HEADER,
    A8W8_ROWS,
    A8W8_UNTUNED,
    BF16_HEADER,
    BF16_ROWS,
    BF16_UNTUNED,
    csv_text,
)

A8W8, BF16 = "a8w8_bpreshuffle_tuned_gemm", "bf16_tuned_gemm"


@pytest.fixture
def configs(tmp_path: Path) -> Path:
    directory = tmp_path / "configs"
    directory.mkdir()
    (directory / "a8w8_bpreshuffle_untuned_gemm.csv").write_text(A8W8_UNTUNED, encoding="utf-8")
    (directory / "bf16_untuned_gemm.csv").write_text(BF16_UNTUNED, encoding="utf-8")
    return directory


def table(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def records(data: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(data.decode())))


def test_the_key_is_the_untuned_header_plus_what_aiter_adds_to_it(configs: Path) -> None:
    a8w8_columns = A8W8_HEADER.split(",")
    bf16_columns = BF16_HEADER.split(",")

    assert table_key(configs, A8W8, a8w8_columns) == ("M", "N", "K", "q_dtype_w", "cu_num", "gfx")
    assert table_key(configs, BF16, bf16_columns) == (
        "M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle", "cu_num", "gfx",
    )  # fmt: skip
    assert table_key(configs, A8W8, [*a8w8_columns, "_tag"])[-1] == "_tag", (
        "a tag column joins the key when there is one"
    )
    assert "gfx" not in table_key(configs, A8W8, [c for c in a8w8_columns if c != "gfx"]), "only the columns present"
    assert table_key(configs, "no_such_tuned_table", a8w8_columns) is None, "no untuned sibling, no check by aiter"


def test_a_rows_file_takes_out_of_the_default_table_the_shapes_it_tunes_and_nothing_else(
    tmp_path: Path, configs: Path
) -> None:
    default = table(configs, f"{A8W8}.csv", csv_text(A8W8_HEADER, *A8W8_ROWS))
    faster = A8W8_ROWS[1].replace(",17.9602,", ",12.5,")
    new_shape = A8W8_ROWS[1].replace(",128,1536,5120,", ",96,1536,5120,")
    rows = table(tmp_path, "rows.csv", csv_text(A8W8_HEADER, faster, new_shape))

    over = lay_rows_over([default], [rows], configs, A8W8)

    assert over.key == ("M", "N", "K", "q_dtype_w", "cu_num", "gfx")
    assert (over.rows, over.overridden, over.dropped) == (2, 1, {f"{A8W8}.csv": 1})
    left = records(over.kept[default])
    assert left[0] == A8W8_HEADER.split(",")
    assert [",".join(row) for row in left[1:]] == [A8W8_ROWS[0], A8W8_ROWS[2], A8W8_ROWS[3]], (
        "the same shape on the other chip (cu_num 80) stays, and so does everything the rows do not name"
    )


def test_a_rows_file_that_shares_no_shape_leaves_the_default_table_as_it_is(tmp_path: Path, configs: Path) -> None:
    default = table(configs, f"{BF16}.csv", csv_text(BF16_HEADER, *BF16_ROWS))
    rows = table(tmp_path, "rows.csv", csv_text(BF16_HEADER, BF16_ROWS[0].replace(",64,256,5120,", ",32,256,5120,")))

    over = lay_rows_over([default], [rows], configs, BF16)

    assert over.kept == {} and over.overridden == 0 and over.rows == 1


def test_the_rows_are_compared_as_aiter_reads_them_not_as_text(tmp_path: Path, configs: Path) -> None:
    default = table(configs, f"{BF16}.csv", csv_text(BF16_HEADER, *BF16_ROWS))
    spelled = BF16_ROWS[2].replace(",128,256,5120,False,", ",128.0,256,5120,false,")
    rows = table(tmp_path, "rows.csv", csv_text(BF16_HEADER, spelled))

    over = lay_rows_over([default], [rows], configs, BF16)

    assert over.dropped == {f"{BF16}.csv": 1}, "128.0 is 128 and false is False to pandas"


def test_a_rows_file_without_gfx_is_matched_by_the_chip_it_names(tmp_path: Path, configs: Path) -> None:
    default = table(configs, f"{A8W8}.csv", csv_text(A8W8_HEADER, *A8W8_ROWS))
    without_gfx = ",".join(A8W8_HEADER.split(",")[1:])
    rows = table(tmp_path, "rows.csv", csv_text(without_gfx, A8W8_ROWS[0].split(",", 1)[1]))

    over = lay_rows_over([default], [rows], configs, A8W8)

    assert over.dropped == {f"{A8W8}.csv": 1}
    assert [row[1] for row in records(over.kept[default])[1:]] == ["304", "304", "256"], "the cu_num 80 row is gone"


def test_every_default_table_loses_the_shapes_the_rows_tune(tmp_path: Path, configs: Path) -> None:
    default = table(configs, f"{BF16}.csv", csv_text(BF16_HEADER, *BF16_ROWS[:3]))
    model = table(configs, f"model_configs/dsv3_{BF16}.csv", csv_text(BF16_HEADER, *BF16_ROWS[1:]))
    rows = table(tmp_path, "rows.csv", csv_text(BF16_HEADER, BF16_ROWS[2]))

    over = lay_rows_over([default, model], [rows], configs, BF16)

    assert over.dropped == {f"{BF16}.csv": 1, f"dsv3_{BF16}.csv": 1} and over.overridden == 2
    assert set(over.kept) == {default, model}


def test_rows_that_lack_a_column_that_identifies_a_shape_are_refused_naming_it(tmp_path: Path, configs: Path) -> None:
    default = table(configs, f"{A8W8}.csv", csv_text(A8W8_HEADER, *A8W8_ROWS))
    header = ",".join(c for c in A8W8_HEADER.split(",") if c != "cu_num")
    rows = table(tmp_path, "rows.csv", csv_text(header, A8W8_ROWS[1].replace(",304,", ",")))

    with pytest.raises(RowsOnlyRefused, match=r"lack the column\(s\) \['cu_num'\]"):
        lay_rows_over([default], [rows], configs, A8W8)


def test_rows_that_name_a_shape_twice_are_refused_naming_it(tmp_path: Path, configs: Path) -> None:
    default = table(configs, f"{A8W8}.csv", csv_text(A8W8_HEADER, *A8W8_ROWS))
    again = A8W8_ROWS[1].replace(",17.9602,", ",12.5,")
    rows = table(tmp_path, "rows.csv", csv_text(A8W8_HEADER, A8W8_ROWS[1], again))

    with pytest.raises(RowsOnlyRefused, match=r"'M': '128'.*more than once"):
        lay_rows_over([default], [rows], configs, A8W8)


def test_where_aiter_checks_no_key_nothing_is_taken_out(tmp_path: Path, configs: Path) -> None:
    default = table(configs, "other_tuned_gemm.csv", csv_text(A8W8_HEADER, *A8W8_ROWS))
    rows = table(tmp_path, "rows.csv", csv_text(A8W8_HEADER, A8W8_ROWS[1]))

    over = lay_rows_over([default], [rows], configs, "other_tuned_gemm")

    assert over.key is None and over.kept == {} and over.rows == 1
    assert lay_rows_over([default], [rows], configs, None).key is None, "a variable the install does not declare"
