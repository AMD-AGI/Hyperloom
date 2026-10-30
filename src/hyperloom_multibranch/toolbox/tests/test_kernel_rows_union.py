# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``union_rows``: the rows of two tuned tables of one variable as one table, on the rows of run ``cc2bf0ef``.

The run's merges were refused by shape: b09's table held every row of the head's and was refused, b01/b07's and b06's
shared two shapes and one with the head, tuned to other rows. The tables here are those rows (``kernel_rows_cc2bf0ef``),
read against a stand-in for the image's aiter, whose untuned sibling of the table gives the shape key.
"""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import KernelKind
from hyperloom_multibranch.toolbox.kernel_artifacts import (
    KernelRowsConflict,
    KernelRowsError,
    KernelRowsMixed,
    KernelRowsUnmergeable,
    mint_kernel_artifact,
    union_rows,
)
from hyperloom_multibranch.toolbox.tests.kernel_rows_cc2bf0ef import (
    B01_HEAD,
    B01_NEW,
    B01_SHARED,
    B06_HEAD,
    B06_NEW,
    B06_SHARED,
    B09_NEW,
    ENV,
    HEADER,
    SHARED,
    csv_text,
    install,
)

HEAD = csv_text(SHARED, B01_HEAD, B06_HEAD)
B09 = csv_text(SHARED, B01_HEAD, B06_HEAD, B09_NEW)
B01 = csv_text(B01_SHARED, B01_NEW)
B06 = csv_text(B06_SHARED, SHARED[:3], B06_NEW)
#: b06's rows that no other table has, and three the head has the same.
B06_AND_HEAD_ROWS = csv_text(SHARED[:3], B06_NEW)
#: The union of the head's and b09's rows, as the bytes a merge stores.
B09_UNION_SHA256 = "457c705743cac8fd6276d500fd7905893bfe85998f940cff5306d151796a4a52"
#: b01's two shapes the head also tuned, and b06's one.
B01_SHAPES = {("64", "34816", "5120"), ("64", "5120", "17408")}
B06_SHAPES = {("20480", "7168", "2304")}


@pytest.fixture
def configs(tmp_path: Path) -> Path:
    return install(tmp_path / "image")


def rows(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode())))


def mnk(conflict: KernelRowsConflict) -> set[tuple[str, str, str]]:
    return {(shape["M"], shape["N"], shape["K"]) for shape in conflict.shapes}


def test_a_table_that_holds_the_other_gives_its_rows(configs: Path) -> None:
    united = union_rows(HEAD, B09, ENV, configs=configs)

    assert sorted(map(tuple, map(dict.items, rows(united)))) == sorted(map(tuple, map(dict.items, rows(B09))))
    assert len(rows(united)) == 12


def test_two_tables_of_other_shapes_give_all_their_rows(configs: Path) -> None:
    united = rows(union_rows(csv_text(B01_NEW), csv_text(B06_NEW), ENV, configs=configs))

    assert len(united) == len(B01_NEW) + len(B06_NEW)
    assert {r["kernelName"] for r in united} == {line.split(",")[9] for line in (*B01_NEW, *B06_NEW)}


def test_a_shape_both_tuned_to_one_row_is_kept_once(configs: Path) -> None:
    united = rows(union_rows(HEAD, B06_AND_HEAD_ROWS, ENV, configs=configs))

    assert len(united) == len(rows(HEAD)) + len(B06_NEW)
    assert len({(r["M"], r["N"], r["K"]) for r in united}) == len(united)


@pytest.mark.parametrize(
    ("other", "shapes"),
    [(B01, B01_SHAPES), (B06, B06_SHAPES), (csv_text(B01_SHARED, B06_SHARED), B01_SHAPES | B06_SHAPES)],
    ids=["b01-b07", "b06", "both"],
)
@pytest.mark.parametrize("held", [HEAD, B09], ids=["head", "b09"])
def test_a_shape_both_tuned_differently_is_a_conflict_naming_only_it(
    configs: Path, held: bytes, other: bytes, shapes: set[tuple[str, str, str]]
) -> None:
    with pytest.raises(KernelRowsConflict) as caught:
        union_rows(held, other, ENV, configs=configs)

    assert mnk(caught.value) == shapes
    assert len(caught.value.shapes) == len(shapes)
    assert all(f"M={m} N={n} K={k}" in str(caught.value) for m, n, k in shapes)


def test_a_conflict_is_the_same_whichever_table_is_first(configs: Path) -> None:
    with pytest.raises(KernelRowsConflict) as caught:
        union_rows(B01, HEAD, ENV, configs=configs)

    assert mnk(caught.value) == B01_SHAPES


def test_rows_only_sets_make_a_rows_only_set_and_a_whole_table_with_a_rows_only_set_is_refused(configs: Path) -> None:
    united = union_rows(HEAD, B09, ENV, rows_only=(True, True), configs=configs)

    assert united == union_rows(HEAD, B09, ENV, configs=configs)
    assert mint_kernel_artifact(KernelKind.GEMM, "gemm_a8w8", ENV, united, rows_only=True).sha256
    for first_only, second_only in [(True, False), (False, True)]:
        with pytest.raises(KernelRowsMixed) as mixed:
            union_rows(HEAD, B09, ENV, rows_only=(first_only, second_only), configs=configs)
        assert not isinstance(mixed.value, KernelRowsConflict)
        assert isinstance(mixed.value, KernelRowsError)


def test_a_mixed_pair_is_refused_before_its_rows_are_read(configs: Path) -> None:
    with pytest.raises(KernelRowsMixed):
        union_rows(b"not a table", HEAD, ENV, rows_only=(True, False), configs=configs)


def test_the_result_is_one_content_reference_in_any_order(configs: Path) -> None:
    first, second = csv_text(B01_NEW), csv_text(B06_NEW)
    united = union_rows(first, second, ENV, configs=configs)
    swapped = union_rows(second, first, ENV, configs=configs)
    shuffled = union_rows(csv_text(B01_NEW[::-1]), csv_text(B06_NEW[::-1]), ENV, configs=configs)

    assert united == swapped == shuffled == union_rows(first, second, ENV, configs=configs)
    assert hashlib.sha256(union_rows(HEAD, B09, ENV, configs=configs)).hexdigest() == B09_UNION_SHA256
    refs = {
        mint_kernel_artifact(KernelKind.GEMM, "gemm_a8w8", ENV, data, rows_only=True).ref
        for data in (united, swapped, shuffled)
    }
    assert len(refs) == 1


def test_rows_come_in_the_numeric_order_of_their_key(configs: Path) -> None:
    united = rows(union_rows(csv_text(B06_NEW), csv_text(B01_NEW, B09_NEW), ENV, configs=configs))

    order = [(float(r["M"]), float(r["N"]), float(r["K"])) for r in united]
    assert order == sorted(order)
    assert [r["M"] for r in united][:2] == ["1", "1"]
    assert united[-1]["M"] == "12288"


def test_the_header_and_the_line_ending_are_the_inputs(configs: Path) -> None:
    crlf = [csv_text(B01_NEW).replace(b"\n", b"\r\n"), csv_text(B06_NEW).replace(b"\n", b"\r\n")]

    united = union_rows(*crlf, ENV, configs=configs)

    assert united.startswith(HEADER.encode() + b"\r\n")
    assert united.count(b"\r\n") == united.count(b"\n") == 1 + len(B01_NEW) + len(B06_NEW)
    assert b"\r\n" not in union_rows(crlf[0], csv_text(B06_NEW), ENV, configs=configs)


def test_the_same_columns_in_another_order_are_one_table_in_either_order(configs: Path) -> None:
    other = HEADER.split(",")
    reordered = [other[0], *reversed(other[1:])]
    cells = [dict(zip(other, line.split(","), strict=True)) for line in B09_NEW]
    text = ",".join(reordered) + "\n" + "".join(",".join(cell[name] for name in reordered) + "\n" for cell in cells)

    united = union_rows(HEAD, text.encode(), ENV, configs=configs)

    assert united == union_rows(text.encode(), HEAD, ENV, configs=configs)
    assert len(rows(united)) == len(rows(HEAD)) + len(B09_NEW)
    assert {r["kernelName"] for r in rows(united)} >= {line.split(",")[9] for line in B09_NEW}


def test_columns_that_are_not_one_set_are_refused_not_filled(configs: Path) -> None:
    narrower = b"M,N,K,cu_num,gfx,us\n1,34816,5120,256,gfx950,9.5\n"

    with pytest.raises(KernelRowsUnmergeable, match="different columns"):
        union_rows(HEAD, narrower, ENV, configs=configs)


def test_a_variable_the_install_has_no_key_for_is_refused(configs: Path, tmp_path: Path) -> None:
    with pytest.raises(KernelRowsUnmergeable, match="no shape key for AITER_CONFIG_OTHER"):
        union_rows(HEAD, B09, "AITER_CONFIG_OTHER", configs=configs)
    with pytest.raises(KernelRowsUnmergeable, match="no shape key"):
        union_rows(HEAD, B09, ENV, configs=tmp_path / "no-aiter" / "configs")
    (configs / "a8w8_blockscale_bpreshuffle_untuned_gemm.csv").unlink()
    with pytest.raises(KernelRowsUnmergeable, match="no shape key"):
        union_rows(HEAD, B09, ENV, configs=configs)


def test_cells_are_compared_as_aiter_reads_them(configs: Path) -> None:
    same_number = csv_text(SHARED[:2]).replace(b",64,", b",64.0,").replace(b",16,512,", b",16.0,512.0,")

    united = rows(union_rows(csv_text(SHARED[:2]), same_number, ENV, configs=configs))

    assert len(united) == 2


def test_the_same_sizes_on_another_chip_are_another_shape(configs: Path) -> None:
    """aiter's key is the untuned header plus ``cu_num`` and ``gfx``: a row tuned on a gfx942 chip is not b01's."""
    cells = SHARED[0].split(",")
    other_chip = ",".join(["gfx942", "304", *cells[2:9], "9.99", *cells[10:]])

    united = rows(union_rows(csv_text(SHARED[:1]), csv_text((other_chip,)), ENV, configs=configs))

    assert [(r["gfx"], r["cu_num"], r["M"]) for r in united] == [("gfx950", "256", "16"), ("gfx942", "304", "16")]


def test_whole_numbers_beyond_a_float_are_other_shapes(configs: Path) -> None:
    """pandas reads an integer column as integers: 2**53 and 2**53 + 1 are two shapes, not one row lost."""
    cells = SHARED[0].split(",")
    sizes = ["9007199254740992", "9007199254740993"]
    tables = [csv_text((",".join([*cells[:2], size, *cells[3:]]),)) for size in sizes]

    united = rows(union_rows(*tables, ENV, configs=configs))

    assert [r["M"] for r in united] == sizes


def test_a_table_that_lists_a_shape_twice_is_refused(configs: Path) -> None:
    twice = csv_text(SHARED[:1], SHARED[:1])

    with pytest.raises(KernelRowsUnmergeable, match="first table lists the shape"):
        union_rows(twice, HEAD, ENV, configs=configs)
    with pytest.raises(KernelRowsUnmergeable, match="second table lists the shape"):
        union_rows(HEAD, twice, ENV, configs=configs)


@pytest.mark.parametrize("data", [b"", b"\xff\xfe", b"M,N,M\n1,2,3\n", HEADER.encode() + b"\n1,2\n"])
def test_a_table_that_is_not_a_table_is_refused(configs: Path, data: bytes) -> None:
    with pytest.raises(KernelRowsUnmergeable):
        union_rows(data, HEAD, ENV, configs=configs)
