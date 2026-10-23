# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How many rows of a kernel table a test's server hit, from the server's log, and what a table nothing hit is called.

The rows of the a8w8 tables are those of run ``cc2bf0ef`` (``kernel_rows_cc2bf0ef``), the bf16 rows are the four ``torch``
rows of the rows-only test (W115) at Qwen3-8B's linear shapes, and the log lines are aiter's own: the bf16 ones are the
lookups of the W116 server of that test (paths shortened), the a8w8 ones follow the line aiter writes for its CK GEMM, with
the rows' real kernel names. Each outcome is its own test: a table with hits, a table with none although hits were
logged ("not exercised"), a table the logs cannot speak for ("not observable"), and no table at all.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import (
    TABLE_EXERCISED,
    TABLE_NOT_EXERCISED,
    TABLE_NOT_OBSERVABLE,
    TableHits,
)
from hyperloom.contracts.recipe import KernelArtifact, KernelKind, Recipe
from hyperloom.measure.compile import CompiledEnvironment, LocalBlobStore
from hyperloom_multibranch.toolbox import table_hits
from hyperloom_multibranch.toolbox.kernel_artifacts import mint_kernel_artifact
from hyperloom_multibranch.toolbox.table_hits import describe, read_table_hits, runtime_findings
from hyperloom_multibranch.toolbox.tests.kernel_rows_cc2bf0ef import B01_HEAD, B01_NEW, ENV, csv_text

IMAGE = "sha256:" + "a" * 64
BF16_ENV = "AITER_CONFIG_GEMM_BF16"
FMOE_ENV = "AITER_CONFIG_FMOE"
MERGED = "/work/hl-framework-x/merged/a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
BF16_MERGED = "/work/hl-framework-x/merged/bf16_tuned_gemm.csv"
HITS_ON = "AITER_LOG_TUNED_CONFIG"

#: The kernel names of the rows of ``B01_NEW`` (``gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,...``).
_KERNEL = {tuple(int(cell) for cell in row.split(",")[2:5]): row.split(",")[9] for row in (*B01_NEW, *B01_HEAD)}
ROW_1_34816 = (1, 34816, 5120)
ROW_16_34816 = (16, 34816, 5120)
ROW_16_5120 = (16, 5120, 17408)
HEAD_ROW_64_5120 = (64, 5120, 17408)

#: From the W116 server of Qwen3-8B (bf16, TP1): lookups that resolved in the image's own rows (``flydsl``, ``triton``,
#: and ``torch`` rows whose kernel is named ``native``), and one that resolved in a row of the rows put, which are ``torch``
#: rows whose kernel is named ``torch``.
_BF16_PREFIX = "[aiter] shape is M:{asked}, N:{n}, K:{k} dtype='torch.bfloat16' otype='torch.bfloat16' bias=False, "
_BF16_FOUND = (
    f"scaleAB=False, bpreshuffle=False found padded_M: {{m}}, N:{{n}}, K:{{k}} is tuned on cu_num = 256 in {BF16_MERGED}, "
    "libtype is {libtype}, kernel name is {kernel}"
)
IMAGE_ROW_HIT = (_BF16_PREFIX + _BF16_FOUND).format(
    asked=512,
    m=512,
    n=4096,
    k=4096,
    libtype="flydsl",
    kernel="flydsl_hgemm_abf16_wbf16_bf16_t128x64x64x4_ks1_w4x2x1_bias0_ktail0_gm0_pft_gfx950",
)
IMAGE_TORCH_ROW_HIT = (_BF16_PREFIX + _BF16_FOUND).format(
    asked=192, m=192, n=4096, k=4096, libtype="torch", kernel="native"
)
PUT_ROW_HIT = (_BF16_PREFIX + _BF16_FOUND).format(asked=4, m=4, n=6144, k=4096, libtype="torch", kernel="torch")
MISS = (
    "[aiter] shape is M:512, N:6144, K:4096 dtype='torch.bfloat16' otype='torch.bfloat16' bias=False, scaleAB=False, "
    f"bpreshuffle=False, not found tuned config in {BF16_MERGED}, will use default config! using torch solution:0"
)
BF16_HEADER = (
    "gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw"
)
#: The (N, K) of the linear layers of Qwen3-8B at TP1 that the rows-only test overrides, as ``torch`` rows at M=4.
QWEN3_8B_SHAPES = ((6144, 4096), (4096, 4096), (24576, 4096), (4096, 12288))


def a8w8_hit(shape: tuple[int, int, int], path: str, *, asked_m: int | None = None) -> str:
    m, n, k = shape
    kernel = _KERNEL[shape]
    return (
        f"[aiter] shape is M:{asked_m or m}, N:{n}, K:{k}, found padded_M: {m}, N:{n}, K:{k} is tuned on "
        f"cu_num = 256 in {path} , kernel name is {kernel}!"
    )


def bf16_row(n: int, k: int, m: int = 4) -> str:
    return f"gfx950,256,{m},{n},{k},False,torch.bfloat16,torch.bfloat16,False,False,torch,0,0,0.0,torch,0.0,0.0,0.0"


def bf16_torch_hit(n: int, k: int) -> str:
    return PUT_ROW_HIT.replace("N:6144", f"N:{n}").replace("K:4096", f"K:{k}")


class Session:
    """A blob store, the recipe's kernels and the directory a test landed in."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.blobs = LocalBlobStore(root / "blobs")
        self.output = root / "out"
        self.output.mkdir()
        self.kernels: list[KernelArtifact] = []

    def put(self, data: bytes, env: str = ENV, target: str = "gemm") -> KernelArtifact:
        artifact = mint_kernel_artifact(KernelKind.GEMM, target, env, data)
        self.blobs.put(data)
        self.kernels.append(artifact)
        return artifact

    def log(self, *lines: str, name: str = "variant_00_c128/benchmark_sglang_0/server.log") -> None:
        path = self.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def point(self, hits: object = None, name: str = "point.yaml") -> Path:
        path = self.root / name
        envs = "" if hits is None else f"    {HITS_ON}: '{hits}'\n"
        path.write_text(f"benchmark:\n  envs:\n    TP: '1'\n{envs}", encoding="utf-8")
        return path

    def read(self, *points: Path) -> tuple[TableHits, ...]:
        environment = CompiledEnvironment(
            Recipe(IMAGE, kernels=tuple(self.kernels)), IMAGE, (), (), {}, self.blobs.root
        )
        return read_table_hits(environment, points, self.output)


@pytest.fixture
def session(tmp_path: Path) -> Session:
    return Session(tmp_path)


def test_a_table_whose_rows_the_server_hit_says_how_many_rows_and_lines(session: Session) -> None:
    artifact = session.put(csv_text(B01_NEW))
    session.log(
        MISS,
        a8w8_hit(ROW_16_34816, MERGED, asked_m=12),
        a8w8_hit(ROW_16_34816, MERGED, asked_m=12),
        a8w8_hit(ROW_16_5120, MERGED),
    )
    (hits,) = session.read()
    assert hits == TableHits(ENV, "gemm", artifact.sha256, TABLE_EXERCISED, rows=4, rows_hit=2, hit_lines=3)
    assert "exercised, 2 of 4 rows hit in 3 log lines; 2 never hit" in describe(hits)


def test_the_row_a_lookup_found_is_the_padded_shape_not_the_shape_it_asked_for(session: Session) -> None:
    session.put(csv_text(B01_NEW))
    session.log(a8w8_hit(ROW_16_34816, MERGED, asked_m=12))
    (hits,) = session.read()
    assert hits.rows_hit == 1, "the row of M=16 is hit by the lookup of M=12, and the row of M=1 is not"


def test_a_table_with_no_row_hit_although_the_server_logged_hits_is_not_exercised_and_says_so(session: Session) -> None:
    artifact = session.put(csv_text(B01_NEW))
    session.log(MISS, a8w8_hit(HEAD_ROW_64_5120, MERGED), a8w8_hit(HEAD_ROW_64_5120, MERGED))
    (hits,) = session.read()
    assert hits == TableHits(
        ENV,
        "gemm",
        artifact.sha256,
        TABLE_NOT_EXERCISED,
        rows=4,
        rows_hit=0,
        hit_lines=0,
        reason="2 aiter lookup(s) hit rows of other tables and none hit a row of this one",
    )
    (finding,) = runtime_findings([hits])
    assert "NOT EXERCISED, none of its 4 rows was hit" in finding and "flat result says nothing" in finding


def test_a_server_launched_to_log_hits_that_logged_none_did_not_exercise_the_table(session: Session) -> None:
    session.put(csv_text(B01_NEW))
    session.log(MISS)
    (hits,) = session.read(session.point(hits=1))
    assert (hits.status, hits.rows_hit, hits.reason) == (TABLE_NOT_EXERCISED, 0, "aiter logged no hit at all")


def test_with_no_hit_logging_a_table_with_no_hit_is_not_observable_not_not_exercised(session: Session) -> None:
    session.put(csv_text(B01_NEW))
    session.log(MISS)
    for point in (session.point(), session.point(hits=0)):
        (hits,) = session.read(point)
        assert hits.status == TABLE_NOT_OBSERVABLE and hits.rows == 4
        assert hits.rows_hit is None and hits.hit_lines is None, "an unknown count is not a zero"
        assert "AITER_LOG_TUNED_CONFIG=1" in hits.reason and "logs a hit only under" in hits.reason
    assert "not observable" in describe(hits) and "unknown" in describe(hits)


def test_a_hit_line_of_any_table_shows_that_hits_were_logged_whatever_the_points_say(session: Session) -> None:
    session.put(csv_text(B01_NEW))
    session.log(IMAGE_ROW_HIT)
    (hits,) = session.read()
    assert hits.status == TABLE_NOT_EXERCISED


def test_one_server_that_logged_no_hit_keeps_the_table_not_observable_whatever_another_logged(session: Session) -> None:
    """The first point's server may have hit the table unseen: another's hit line of another table does not say it did not."""
    session.put(csv_text(B01_NEW))
    session.log(MISS, name="variant_00_c32/server.log")
    session.log(MISS, IMAGE_ROW_HIT, name="variant_01_c64/server.log")
    mixed = (session.point(hits=0, name="point-0.yaml"), session.point(hits=1, name="point-1.yaml"))
    (hits,) = session.read(*mixed)
    assert hits.status == TABLE_NOT_OBSERVABLE and hits.rows_hit is None
    assert "AITER_LOG_TUNED_CONFIG=1" in hits.reason
    (hits,) = session.read()
    assert hits.status == TABLE_NOT_OBSERVABLE, "no point asks for hits and one server's log shows none"
    session.log(MISS, IMAGE_ROW_HIT, name="variant_00_c32/server.log")
    (hits,) = session.read()
    assert hits.status == TABLE_NOT_EXERCISED, (
        "every server's log shows hits of some table, and none is a row of this one"
    )


def test_a_test_that_landed_no_server_log_says_nothing_of_its_table(session: Session) -> None:
    session.put(csv_text(B01_NEW))
    (hits,) = session.read(session.point(hits=1))
    assert (hits.status, hits.reason) == (TABLE_NOT_OBSERVABLE, "the test landed no server log")


def test_hit_lines_this_reader_cannot_read_keep_a_table_with_no_hit_from_being_called_not_exercised(
    session: Session,
) -> None:
    session.put(csv_text(B01_NEW))
    # aiter's quantised GEMM words its hit as "is tuned, in <file>!"; the rule pack's hit line has no such form.
    session.log(
        "[aiter] shape M:128, N:512, K:7168 q_dtype_w:torch.float8_e4m3fn, found padded_M: 128, N:512, K:7168 is tuned, in /x/t.csv!"
    )
    (hits,) = session.read(session.point(hits=1))
    assert hits.status == TABLE_NOT_OBSERVABLE
    assert hits.reason == "1 log line(s) report a tuned lookup in a form this reader cannot read"


def test_an_unreadable_log_keeps_a_table_with_no_hit_from_being_called_not_exercised(session: Session) -> None:
    session.put(csv_text(B01_NEW))
    session.log(MISS)
    (session.output / "variant_01" / "server.log").mkdir(parents=True)  # a path that matches and cannot be read
    (hits,) = session.read(session.point(hits=1))
    assert hits.status == TABLE_NOT_OBSERVABLE and hits.reason == "1 server log(s) could not be read"


def test_a_hit_is_a_fact_even_when_other_lines_were_not_read(session: Session) -> None:
    session.put(csv_text(B01_NEW))
    session.log(
        "[aiter] shape M:1, N:2, K:3 q, found padded_M: 1, N:2, K:3 is tuned, in /x/t.csv!",
        a8w8_hit(ROW_16_5120, MERGED),
    )
    (hits,) = session.read()
    assert (hits.status, hits.rows_hit) == (TABLE_EXERCISED, 1)


def test_the_lines_of_a_server_with_several_ranks_count_once_a_row_and_once_a_line_each(session: Session) -> None:
    session.put(csv_text(B01_NEW))
    ranks = [f"(Worker_TP{rank} pid={rank}) {a8w8_hit(ROW_1_34816, MERGED)}" for rank in range(8)]
    session.log(*ranks)
    (hits,) = session.read()
    assert (hits.rows_hit, hits.hit_lines) == (1, 8)


def test_a_hit_in_the_private_copy_of_another_table_is_not_a_row_of_this_one(session: Session) -> None:
    artifact = session.put(csv_text(B01_NEW))
    own = f"/work/hl-framework-x/tables/{ENV}/{artifact.sha256[:16]}/k-{artifact.sha256[:16]}"
    other_variable = f"/work/hl-framework-x/tables/{BF16_ENV}/{artifact.sha256[:16]}/k-{artifact.sha256[:16]}"
    other_table = f"/work/hl-framework-x/tables/{ENV}/{'0' * 16}/k-{'0' * 16}"
    session.log(a8w8_hit(ROW_16_34816, other_variable), a8w8_hit(ROW_16_34816, other_table))
    (hits,) = session.read(session.point(hits=1))
    assert (hits.status, hits.rows_hit) == (TABLE_NOT_EXERCISED, 0)
    session.log(a8w8_hit(ROW_16_34816, own))
    (hits,) = session.read()
    assert (hits.status, hits.rows_hit, hits.hit_lines) == (TABLE_EXERCISED, 1, 1)


def test_a_row_with_the_shape_of_the_hit_but_another_kernel_is_not_the_row_that_was_hit(session: Session) -> None:
    session.put(csv_text((B01_NEW[0].replace(_KERNEL[ROW_1_34816], "kernel_of_another_tuner"),)))
    session.log(a8w8_hit(ROW_1_34816, MERGED))
    (hits,) = session.read(session.point(hits=1))
    assert (hits.status, hits.rows_hit) == (TABLE_NOT_EXERCISED, 0)


def test_rows_put_over_the_image_table_are_hit_only_by_lines_that_name_a_row_of_theirs(session: Session) -> None:
    table = "\n".join([BF16_HEADER, *(bf16_row(n, k) for n, k in QWEN3_8B_SHAPES)]).encode() + b"\n"
    session.put(table, env=BF16_ENV, target="gemm_bf16")
    session.log(MISS, IMAGE_ROW_HIT)
    (hits,) = session.read()
    assert (hits.status, hits.rows, hits.rows_hit) == (TABLE_NOT_EXERCISED, 4, 0), "the image's own row is not ours"
    session.log(MISS, IMAGE_ROW_HIT, bf16_torch_hit(*QWEN3_8B_SHAPES[0]), bf16_torch_hit(*QWEN3_8B_SHAPES[2]))
    (hits,) = session.read()
    assert (hits.status, hits.rows, hits.rows_hit, hits.hit_lines) == (TABLE_EXERCISED, 4, 2, 2)


def test_an_image_row_of_the_same_libtype_and_shape_but_another_kernel_is_not_a_row_that_was_put(
    session: Session,
) -> None:
    """The W116 server's own ``torch`` rows are named ``native``; a count by libtype alone took them for the rows put."""
    table = "\n".join([BF16_HEADER, bf16_row(4096, 4096, m=192)]).encode() + b"\n"
    session.put(table, env=BF16_ENV, target="gemm_bf16")
    session.log(IMAGE_TORCH_ROW_HIT)
    (hits,) = session.read()
    assert (hits.status, hits.rows_hit, hits.hit_lines) == (TABLE_NOT_EXERCISED, 0, 0)
    session.log(IMAGE_TORCH_ROW_HIT, IMAGE_TORCH_ROW_HIT.replace("kernel name is native", "kernel name is torch"))
    (hits,) = session.read()
    assert (hits.status, hits.rows_hit, hits.hit_lines) == (TABLE_EXERCISED, 1, 1)


def test_a_hit_that_prints_other_dtypes_or_flags_than_the_row_is_not_that_row(session: Session) -> None:
    table = "\n".join([BF16_HEADER, bf16_row(*QWEN3_8B_SHAPES[0])]).encode() + b"\n"
    session.put(table, env=BF16_ENV, target="gemm_bf16")
    biased = bf16_torch_hit(*QWEN3_8B_SHAPES[0]).replace("bias=False", "bias=True")
    session.log(biased)
    (hits,) = session.read()
    assert (hits.status, hits.rows_hit) == (TABLE_NOT_EXERCISED, 0)


def test_a_table_of_another_variable_than_aiters_is_not_observable_by_this_reader(session: Session) -> None:
    artifact = session.put(
        b"Validator,x\nGemmTunableOp_float_TN,1\n", env="PYTORCH_TUNABLEOP_FILENAME", target="tunableop"
    )
    session.log(MISS)
    (hits,) = session.read(session.point(hits=1))
    assert (hits.status, hits.rows, hits.rows_hit, hits.sha256) == (TABLE_NOT_OBSERVABLE, None, None, artifact.sha256)
    assert "PYTORCH_TUNABLEOP_FILENAME are not read" in hits.reason


def test_an_aiter_table_that_is_not_keyed_by_shape_is_not_observable_by_this_reader(session: Session) -> None:
    session.put(b"token,model_dim,inter_dim\n16,4096,1536\n", env=FMOE_ENV, target="fmoe")
    session.log(MISS)
    (hits,) = session.read(session.point(hits=1))
    assert hits.status == TABLE_NOT_OBSERVABLE and hits.rows == 1
    assert hits.reason == "the table has no M/N/K columns, so its hits cannot be told: only a dense GEMM table is read"


def test_a_table_whose_blob_is_gone_is_not_observable_not_an_error(session: Session) -> None:
    artifact = session.put(csv_text(B01_NEW))
    (session.blobs.root / artifact.sha256).unlink()
    session.log(a8w8_hit(ROW_16_34816, MERGED))
    (hits,) = session.read()
    assert hits.status == TABLE_NOT_OBSERVABLE and hits.reason == "the table cannot be read as rows: BlobError"


def test_every_table_of_the_recipe_has_its_own_entry_in_the_recipes_order(session: Session) -> None:
    first = session.put(csv_text(B01_NEW))
    second = session.put(
        "\n".join([BF16_HEADER, bf16_row(*QWEN3_8B_SHAPES[1])]).encode() + b"\n", env=BF16_ENV, target="gemm_bf16"
    )
    session.log(a8w8_hit(ROW_16_5120, MERGED))
    one, two = session.read()
    assert (one.sha256, one.status) == (first.sha256, TABLE_EXERCISED)
    assert (two.sha256, two.status) == (second.sha256, TABLE_NOT_EXERCISED)
    assert [text.split(":")[0] for text in runtime_findings([one, two])] == [
        f"kernel table gemm ({ENV})",
        f"kernel table gemm_bf16 ({BF16_ENV})",
    ]


def test_a_recipe_that_loads_no_table_has_nothing_to_report(session: Session) -> None:
    session.log(a8w8_hit(ROW_16_5120, MERGED))
    assert session.read() == ()
    assert read_table_hits(object(), (), session.output) == ()
    assert runtime_findings(()) == ()


def test_a_pack_whose_hit_line_has_no_group_for_the_shape_says_nothing_of_any_table_instead_of_failing_the_test(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    session.put(csv_text(B01_NEW))
    session.log(a8w8_hit(ROW_16_5120, MERGED))
    monkeypatch.setattr(table_hits, "_HIT", re.compile(r"found padded_M: (?P<M>\d+), N:(?P<N>\d+) is tuned"))
    (hits,) = session.read(session.point(hits=1))
    assert hits.status == TABLE_NOT_OBSERVABLE and hits.rows_hit is None
    assert hits.reason == "the rule pack's aiter_tuned_hit pattern has no group K or table, so no hit line can be read"
