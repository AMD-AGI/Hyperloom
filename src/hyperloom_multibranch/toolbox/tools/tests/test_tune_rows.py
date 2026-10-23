# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The tuned rows a Forge tuner left, handed over as a rows-only CSV: each checked against the reference, keyed as aiter keys
the table, and none handed over unchecked. The rows are rows of the image's own ``a8w8_blockscale_bpreshuffle`` table."""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import KernelKind
from hyperloom_multibranch.toolbox.kernel_artifacts import mint_kernel_artifact
from hyperloom_multibranch.toolbox.tests.kernel_rows_cc2bf0ef import ENV, HEADER, SHARED, install
from hyperloom_multibranch.toolbox.tests.kernel_tune_rig import table_text
from hyperloom_multibranch.toolbox.tools.tune_rows import (
    ERR_RATIO_LIMIT,
    RowsRefused,
    dropped_by_forge,
    from_artifact,
)

COLUMNS = HEADER.split(",")


def row(m: int, n: int, k: int, *, us: float = 10.0, err: str = "0.0") -> str:
    """A row of the table for the shape, as aiter writes one (``cktile`` kernel, gfx950)."""
    cells = dict.fromkeys(COLUMNS, "0") | {
        "gfx": "gfx950",
        "cu_num": "256",
        "M": str(m),
        "N": str(n),
        "K": str(k),
        "libtype": "ck",
        "us": str(us),
        "kernelName": "a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256",
        "errRatio": err,
    }
    return ",".join(cells[name] for name in COLUMNS)


@pytest.fixture
def configs(tmp_path: Path) -> Path:
    return install(tmp_path / "image")


def table(tmp_path: Path, *rows: str, header: str = HEADER) -> Path:
    path = tmp_path / "candidate.csv"
    path.write_text(table_text(header, rows), encoding="utf-8")
    return path


def shapes(text: str) -> list[tuple[str, str, str]]:
    return [(r["M"], r["N"], r["K"]) for r in csv.DictReader(io.StringIO(text))]


def test_rows_within_the_limit_are_handed_over_each_with_its_figure_and_verified(tmp_path: Path, configs: Path) -> None:
    rows = from_artifact(table(tmp_path, *SHARED[:2]), ENV, configs)

    assert [(r["M"], r["verified"], r["err_ratio"]) for r in rows.verified] == [
        ("16", True, 0.0029),
        ("20480", True, 0.0),
    ]
    assert rows.refused == () and shapes(rows.csv) == [("16", "512", "7168"), ("20480", "512", "7168")]
    assert rows.target == "gemm_a8w8_blockscale_bpreshuffle"


def test_a_row_whose_output_is_outside_the_references_tolerance_is_refused_and_not_in_the_csv(
    tmp_path: Path, configs: Path
) -> None:
    """The 69% case: the fastest fused row of an outside package had 69% of its elements outside the tolerance."""
    wrong = row(64, 5120, 5120, us=1.0, err="0.69")
    rows = from_artifact(table(tmp_path, row(16, 512, 7168), wrong), ENV, configs)

    assert shapes(rows.csv) == [("16", "512", "7168")]
    (refused,) = rows.refused
    assert (refused["verified"], refused["err_ratio"], refused["M"]) == (False, 0.69, "64")
    assert "outside the reference's tolerance" in refused["reason"] and f"{ERR_RATIO_LIMIT:g}" in refused["reason"]


@pytest.mark.parametrize("figure", ["", "Null", "nan", "inf", "-", "-0.5"])
def test_a_row_with_no_usable_figure_cannot_be_checked_and_is_refused_by_name(
    tmp_path: Path, configs: Path, figure: str
) -> None:
    rows = from_artifact(table(tmp_path, row(16, 512, 7168), row(64, 5120, 5120, err=figure)), ENV, configs)

    assert shapes(rows.csv) == [("16", "512", "7168")]
    (refused,) = rows.refused
    assert refused["verified"] is False and refused["err_ratio"] is None
    assert "is not a ratio from 0 up: the row was not checked" in refused["reason"]


def test_a_table_without_an_accuracy_column_has_every_row_refused_as_unchecked(tmp_path: Path, configs: Path) -> None:
    bare = ",".join(name for name in COLUMNS if name != "errRatio")
    lines = [
        ",".join(cell for cell, name in zip(r.split(","), COLUMNS, strict=True) if name != "errRatio") for r in SHARED
    ]
    rows = from_artifact(table(tmp_path, *lines, header=bare), ENV, configs)

    assert rows.verified == () and len(rows.refused) == len(SHARED)
    assert all("has no accuracy column" in r["reason"] and r["verified"] is False for r in rows.refused)
    assert shapes(rows.csv) == []


def test_the_bf16_tables_err_ratio_column_is_read_too(tmp_path: Path, configs: Path) -> None:
    header = HEADER.replace("errRatio", "err_ratio")
    ok_row, bad_row = (r.replace("0.0029", "0.0") for r in SHARED[:1]), row(64, 5120, 5120, err="0.5")
    rows = from_artifact(table(tmp_path, *ok_row, bad_row, header=header), ENV, configs)

    assert shapes(rows.csv) == [("16", "512", "7168")] and [r["M"] for r in rows.refused] == ["64"]


def test_the_limit_is_forges_own_and_a_row_exactly_at_it_passes(tmp_path: Path, configs: Path) -> None:
    pytest.importorskip("kernelforge.gemm_tune.tuners._aiter_dense_common")
    from kernelforge.gemm_tune.tuners import _aiter_dense_common as forge

    assert ERR_RATIO_LIMIT == forge._MAX_ERR_RATIO, "Forge changed the limit it drops rows at: change ERR_RATIO_LIMIT"
    rows = from_artifact(table(tmp_path, row(16, 512, 7168, err=str(ERR_RATIO_LIMIT))), ENV, configs)
    assert len(rows.verified) == 1 and rows.refused == ()


def test_a_shape_the_tuner_left_twice_keeps_its_fastest_row_and_the_csv_is_in_the_numeric_order_of_the_key(
    tmp_path: Path, configs: Path
) -> None:
    slow, fast = row(128, 4096, 1280, us=9.0), row(128, 4096, 1280, us=6.0)
    rows = from_artifact(table(tmp_path, row(20480, 512, 7168), slow, row(16, 512, 7168), fast), ENV, configs)

    assert shapes(rows.csv) == [("16", "512", "7168"), ("128", "4096", "1280"), ("20480", "512", "7168")]
    assert [float(r["us"]) for r in rows.verified if r["M"] == "128"] == [6.0]


def test_the_same_rows_in_another_order_are_the_same_bytes_so_the_same_artifact(tmp_path: Path, configs: Path) -> None:
    forward = from_artifact(table(tmp_path, *SHARED), ENV, configs)
    backward = from_artifact(table(tmp_path, *reversed(SHARED)), ENV, configs)

    assert forward.csv == backward.csv
    one = mint_kernel_artifact(KernelKind.GEMM, forward.target, ENV, forward.csv.encode(), rows_only=True)
    two = mint_kernel_artifact(KernelKind.GEMM, backward.target, ENV, backward.csv.encode(), rows_only=True)
    assert one.sha256 == two.sha256


def test_the_csv_is_accepted_by_put_kernel_artifact_as_rows_only(tmp_path: Path, configs: Path) -> None:
    rows = from_artifact(table(tmp_path, *SHARED), ENV, configs)

    artifact = mint_kernel_artifact(KernelKind.GEMM, rows.target, ENV, rows.csv.encode(), rows_only=True)

    assert artifact.target == "gemm_a8w8_blockscale_bpreshuffle" and artifact.metadata["load"]["rows_only"] is True


def test_a_variable_the_install_declares_no_shape_key_for_is_refused(tmp_path: Path, configs: Path) -> None:
    with pytest.raises(RowsRefused, match="declares no shape key for AITER_CONFIG_GEMM_BF16"):
        from_artifact(table(tmp_path, *SHARED), "AITER_CONFIG_GEMM_BF16", configs)


def test_a_table_that_cannot_be_read_is_refused(tmp_path: Path, configs: Path) -> None:
    with pytest.raises(RowsRefused, match="cannot be read"):
        from_artifact(tmp_path / "absent.csv", ENV, configs)


def test_what_forge_dropped_as_wrong_is_reported_as_rejected_with_its_figure() -> None:
    run = {
        "tuner": "a8w8_blockscale_bpreshuffle",
        "dropped_inaccurate": [
            {"M": 64, "N": 5120, "K": 5120, "libtype": "ck", "splitK": 0, "us": 1.0, "err_ratio": 0.69}
        ],
    }

    (dropped,) = dropped_by_forge(run)

    assert (dropped["M"], dropped["verified"], dropped["err_ratio"]) == (64, False, 0.69)
    assert "Forge removed it" in dropped["reason"]
    assert dropped_by_forge({"tuner": "x"}) == () and dropped_by_forge({"dropped_inaccurate": "no"}) == ()


def test_no_row_is_ever_both_in_the_csv_and_refused(tmp_path: Path, configs: Path) -> None:
    lines = [row(m, 512, 7168, err=str(0.002 * m)) for m in range(1, 12)]
    rows = from_artifact(table(tmp_path, *lines), ENV, configs)

    in_csv = {r["M"] for r in rows.verified}
    assert in_csv and {r["M"] for r in rows.refused} and not in_csv & {r["M"] for r in rows.refused}
    assert all(r["err_ratio"] <= ERR_RATIO_LIMIT and math.isfinite(r["err_ratio"]) for r in rows.verified)
