"""Unit tests for the operator-agnostic (BYOD) forge-rewrite-by-flydsl layer.

These are pure-Python (no GPU, no LLM, no FlyDSL): they pin the rewrite spec,
the source-entry discovery heuristic, the unresolved-entry path, the generic
seed skeleton, and the speedup math. GPU/agent behavior (and driver measurement
primitives) is covered by the L1/L2/L3 integration ladder, not here.
"""

from __future__ import annotations

import textwrap
from types import SimpleNamespace

import pytest

from hyperloom.agents.kernel.tools.backends import forge_submit
from kernelforge.mcp_server.tools.bench import calculate_mean_case_speedup, parse_case_timings
from kernelforge.rewrite_by_flydsl import driver_contract, ingest, kb, report, seed
from kernelforge.rewrite_by_flydsl.port_loop import (
    _validation_error_tail,
    check_flydsl_port,
)
from kernelforge.rewrite_by_flydsl.spec import RewriteSpec


# ── spec ─────────────────────────────────────────────────────────────────────


def test_builder_symbol_and_rewrite_shapes_are_preserved():
    s = RewriteSpec(
        op_name="rmsnorm", source_kernel="/w/r.py", target_functions=[], shapes=[{"M": 4, "N": 4, "dtype": "fp16"}]
    )
    assert s.builder_symbol == "build_rmsnorm_module"
    assert s.shapes == [{"M": 4, "N": 4, "dtype": "fp16"}]


# ── ingest: source-entry discovery is a best-effort hint (no fail-fast) ───────

_TRITON_SRC = textwrap.dedent("""
    import triton
    @triton.jit
    def softmax_kernel_online(o, i, s, n): ...
    def softmax(x):
        y = x
        softmax_kernel_online[(1,)](y, x, x.stride(0), x.shape[0])
        return y
""")


def test_discover_source_entry_finds_wrapper(tmp_path):
    src = tmp_path / "softmax.py"
    src.write_text(_TRITON_SRC)
    assert ingest.discover_source_entry(str(src), ["softmax_kernel_online"]) == "softmax"


def test_discover_source_entry_empty_targets_returns_empty():
    assert ingest.discover_source_entry("whatever.py", []) == ""


def test_discover_source_entry_unparseable_returns_empty(tmp_path):
    src = tmp_path / "broken.py"
    src.write_text("def (:\n")  # syntax error
    assert ingest.discover_source_entry(str(src), ["k"]) == ""


def test_discover_source_entry_plain_call_and_bare_name(tmp_path):
    # `wrap` launches the kernel via a plain call; `alias` references it as a bare
    # name — exercises both the Call and Name discovery branches.
    src = tmp_path / "s.py"
    src.write_text(
        "def k(x):\n    return x\ndef alias():\n    fn = k\n    return fn\ndef wrap(x):\n    k(x)\n    return x\n"
    )
    # Both reference k; the simplest wrapper (fewest positional args) wins -> alias.
    assert ingest.discover_source_entry(str(src), ["k"]) in {"alias", "wrap"}


def test_build_spec_autodiscovers_entry_when_absent(tmp_path):
    src = tmp_path / "softmax.py"
    src.write_text("def _softmax_kernel(): ...\ndef softmax(x):\n    _softmax_kernel[(1,)](x)\n    return x\n")
    spec = ingest.build_spec(
        op_name="softmax",
        source_kernel=str(src),
        flydsl_kernel=str(tmp_path / "kernel.py"),
        workspace=str(tmp_path),
        target_functions=["_softmax_kernel"],
    )
    assert spec.source_entry == "softmax"


def test_build_spec_does_not_raise_on_unresolved_entry(tmp_path):
    # BYOD: the driver owns the oracle, so an unresolved entry must NOT fail-fast.
    src = tmp_path / "mystery.py"
    src.write_text("def unrelated():\n    return 1\n")
    spec = ingest.build_spec(
        op_name="op",
        source_kernel=str(src),
        flydsl_kernel=str(tmp_path / "kernel.py"),
        workspace=str(tmp_path),
        target_functions=["nonexistent_kernel"],
    )
    assert spec.source_entry == ""  # unresolved, but no exception


def test_candidate_outside_the_workspace_reads_as_its_bare_name(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    src = workspace / "softmax.py"
    src.write_text("def softmax(x):\n    return x\n")
    outside = tmp_path / "elsewhere" / "kernel.py"
    spec = ingest.build_spec(
        op_name="softmax",
        source_kernel=str(src),
        flydsl_kernel=str(outside),
        workspace=str(workspace),
        target_functions=["softmax"],
    )

    assert spec.flydsl_kernel_relpath == "kernel.py"


def test_discover_source_entry_unreadable_c_source_returns_empty(tmp_path):
    missing = tmp_path / "gone.hip"
    assert ingest.discover_source_entry(str(missing), ["k"], source_language="hip") == ""


def test_discover_source_entry_c_source_without_a_launch_returns_empty(tmp_path):
    src = tmp_path / "plain.hip"
    src.write_text("__global__ void k(float* o) {}\nvoid host(float* o) { }\n")
    assert ingest.discover_source_entry(str(src), ["k"], source_language="hip") == ""


def test_resolve_source_language_unreadable_python_reads_as_unknown(tmp_path):
    assert ingest.resolve_source_language(str(tmp_path / "gone.py")) == ""


# ── ingest: the source language decides how the source is read ───────────────

_HIP_SRC = textwrap.dedent("""
    #include <hip/hip_runtime.h>

    __global__ void attention_kernel(const float* q, float* o, int n) {
        o[0] = q[0];
    }

    void attention(const float* q, float* o, int n) {
        attention_kernel<<<dim3(1), dim3(64)>>>(q, o, n);
    }
""")


@pytest.mark.parametrize(
    ("declared", "filename", "expected"),
    [
        # A caller's declaration wins: it came from a profiler that saw the kernel
        # run, which the file itself cannot tell us.
        ("triton", "kernel.py", "triton"),
        # A curated kind naming a language this producer reads is mapped onto it.
        ("hip_cpp", "kernel.cpp", "hip"),
        ("", "attention.hip", "hip"),
        ("", "attention.cu", "cuda"),
        ("", "attention.cpp", "cpp"),
        # Nothing to port: refused by reporting no language rather than defaulted.
        ("aiter_asm", "kernel.s", ""),
    ],
)
def test_resolve_source_language(tmp_path, declared, filename, expected):
    source = tmp_path / filename
    source.write_text("// stub\n")

    assert ingest.resolve_source_language(str(source), declared) == expected


def test_a_python_file_without_triton_is_not_assumed_to_be_triton(tmp_path):
    src = tmp_path / "helper.py"
    src.write_text("def helper(x):\n    return x\n")

    assert ingest.resolve_source_language(str(src)) == ""


def test_entry_discovery_reads_a_c_like_source_instead_of_parsing_it(tmp_path):
    """``ast.parse`` only raises ``SyntaxError`` on HIP.

    Left on the Python path, every C-like kernel reported no entry at all and the
    port prompt silently lost the one hint it had about how to call the source.
    """
    src = tmp_path / "attention.hip"
    src.write_text(_HIP_SRC)

    entry = ingest.discover_source_entry(
        str(src),
        ["attention_kernel"],
        source_language="hip",
    )

    assert entry == "attention"


def test_build_spec_resolves_the_language_and_the_c_like_entry(tmp_path):
    src = tmp_path / "attention.hip"
    src.write_text(_HIP_SRC)

    spec = ingest.build_spec(
        op_name="attention",
        source_kernel=str(src),
        flydsl_kernel=str(tmp_path / "kernel.py"),
        workspace=str(tmp_path),
        target_functions=["attention_kernel"],
    )

    assert spec.source_language == "hip"
    assert spec.source_entry == "attention"


# ── seed: generic (operator-agnostic) skeleton ───────────────────────────────


def test_seed_defines_builder_symbol_generically(tmp_path):
    s = RewriteSpec(
        op_name="gemm", source_kernel="/w/gemm.py", target_functions=[], flydsl_kernel=str(tmp_path / "kernel.py")
    )
    seed.generate_seed(s, s.flydsl_kernel)
    text = (tmp_path / "kernel.py").read_text()
    assert "def build_gemm_module(*args, **kwargs):" in text  # no fixed (M,N,dtype)
    # The stub imports and exposes the symbol; calling launch raises NotImplemented.
    ns: dict = {}
    exec(compile(text, "kernel.py", "exec"), ns)
    launch = ns["build_gemm_module"](1, 2, 3, foo="bar")
    with pytest.raises(NotImplementedError):
        launch(object(), object())


# ── port_loop: FlyDSL-only gate (a correct-but-cheating port is not a rewrite) ─


def _spec_with_kernel(tmp_path, kernel_src: str) -> RewriteSpec:
    (tmp_path / "softmax.py").write_text("def softmax(x):\n    return x\n")
    (tmp_path / "kernel.py").write_text(kernel_src)
    return RewriteSpec(
        op_name="softmax",
        source_kernel=str(tmp_path / "softmax.py"),
        target_functions=["softmax"],
        flydsl_kernel=str(tmp_path / "kernel.py"),
        workspace=str(tmp_path),
    )


def test_flydsl_gate_accepts_a_real_flydsl_port(tmp_path):
    s = _spec_with_kernel(
        tmp_path,
        "import flydsl.expr as fx\n"
        "def build_softmax_module(M, N, dt):\n"
        "    def launch(A, C, m, stream=None): ...\n"
        "    return launch\n",
    )
    assert check_flydsl_port(s) == ""


def test_flydsl_gate_rejects_missing_flydsl(tmp_path):
    s = _spec_with_kernel(tmp_path, "import torch\ndef build_softmax_module(*a): ...\n")
    assert "import `flydsl`" in check_flydsl_port(s)


def test_flydsl_gate_rejects_triton_reimplementation(tmp_path):
    s = _spec_with_kernel(tmp_path, "import flydsl\nimport triton\ndef build_softmax_module(*a): ...\n")
    assert "triton" in check_flydsl_port(s)


def test_flydsl_gate_bans_triton_whatever_the_source_language_was(tmp_path):
    # Triton ships alongside FlyDSL, so deriving the ban from the source language
    # would hand a HIP port a free pass to reimplement the op in Triton.
    s = _spec_with_kernel(tmp_path, "import flydsl\nimport triton\ndef build_softmax_module(*a): ...\n")
    s.source_language = "hip"

    assert "triton" in check_flydsl_port(s)


def test_flydsl_gate_rejects_calling_the_source_module(tmp_path):
    # The sneakiest cheat: import flydsl for show, but re-call the source oracle.
    s = _spec_with_kernel(tmp_path, "import flydsl\nfrom softmax import softmax\ndef build_softmax_module(*a): ...\n")
    assert "source module" in check_flydsl_port(s)


def test_flydsl_gate_rejects_relative_source_import(tmp_path):
    # `from . import softmax` binds the source name with node.module=None.
    s = _spec_with_kernel(tmp_path, "import flydsl\nfrom . import softmax\ndef build_softmax_module(*a): ...\n")
    assert "source module" in check_flydsl_port(s)


def test_flydsl_gate_rejects_dynamic_import_of_source_or_triton(tmp_path):
    s = _spec_with_kernel(
        tmp_path,
        "import flydsl, importlib\nm = importlib.import_module('softmax')\ndef build_softmax_module(*a): ...\n",
    )
    assert "dynamically imports" in check_flydsl_port(s)
    s2 = _spec_with_kernel(tmp_path, "import flydsl\nt = __import__('triton')\ndef build_softmax_module(*a): ...\n")
    assert "dynamically imports" in check_flydsl_port(s2)


def test_flydsl_gate_rejects_nonliteral_dynamic_import(tmp_path):
    s = _spec_with_kernel(
        tmp_path,
        "import flydsl, importlib\n"
        "name = 'soft' + 'max'\n"
        "m = importlib.import_module(name)\n"
        "def build_softmax_module(*a): ...\n",
    )
    assert "non-literal" in check_flydsl_port(s)


def test_flydsl_gate_allows_benign_calls(tmp_path):
    # A normal (non-import) call must pass through the dynamic-import scan.
    s = _spec_with_kernel(tmp_path, "import flydsl\nprint('building')\ndef build_softmax_module(*a): ...\n")
    assert check_flydsl_port(s) == ""


def test_flydsl_gate_reports_unparseable_kernel(tmp_path):
    s = _spec_with_kernel(tmp_path, "def build_softmax_module(:\n")  # syntax error
    assert "could not parse" in check_flydsl_port(s)


def test_validation_error_tail_empty_when_passed():
    class _Passed:
        all_passed = True

    assert _validation_error_tail(_Passed()) == ""


# ── report: cross-language speedup math ──────────────────────────────────────


def test_rewrite_uses_forge_loop_result_sentinel():
    assert report.SENTINEL == "__FORGE_RESULT__"


def test_speedup_only_when_port_ok_and_both_times():
    ok = report.build_result(
        op_name="op", port_ok=True, port_attempts=1, source_ms=2.0, optimize_result={"best_ms": 1.0}
    )
    assert ok.speedup == pytest.approx(2.0)
    assert ok.compiled and ok.correct and ok.target_language == "flydsl"

    no_base = report.build_result(
        op_name="op", port_ok=True, port_attempts=1, source_ms=None, optimize_result={"best_ms": 1.0}
    )
    assert no_base.speedup is None

    failed = report.build_result(op_name="op", port_ok=False, port_attempts=3, source_ms=2.0, optimize_result={})
    assert failed.speedup is None and not failed.correct


def test_case_timings_are_read_off_the_driver_not_just_case_ids():
    # The aggregate cannot reconstruct them: a ratio of two sums is dominated by
    # the largest case, so the per-case values have to survive parsing.
    reading = driver_contract.read_driver_output(
        "case_ms: m_1 0.012818\ncase_ms: m_4096 0.229021\nmean_ms: 0.120920\n"
    )
    assert reading.case_ids == ("m_1", "m_4096")
    assert reading.case_times == {"m_1": pytest.approx(0.012818), "m_4096": pytest.approx(0.229021)}
    assert reading.timing_ms == pytest.approx(0.120920)


def test_case_timings_survive_a_driver_that_prefixes_its_output():
    # A collectives driver prefixes every line with its rank. Anchoring the
    # pattern to the start of a line reads no cases at all there, and the result
    # silently degrades to the aggregate ratio instead of failing.
    reading = driver_contract.read_driver_output(
        "[rank0] case_ms: n_8 1.500\n[rank0] case_ms: n_16 3.000\n[rank0] mean_ms: 2.250\n"
    )
    assert reading.case_times == {"n_8": pytest.approx(1.5), "n_16": pytest.approx(3.0)}


def test_unscored_cases_are_kept_out_of_the_score():
    # forge-loop excludes them because their run-to-run spread swamps a real
    # change; averaging them in lets noise, or a win on a case nobody is
    # optimizing, carry the verdict.
    reading = driver_contract.read_driver_output(
        "case_ms: a 1.000\ncase_ms: b 2.000 unscored\nmean_ms: 1.500\n"
    )
    assert reading.unscored_cases == ("b",)
    assert reading.scored_case_times == {"a": pytest.approx(1.0)}


def test_duplicate_case_timings_are_refused_like_the_bench_tool_refuses_them():
    # bench.py fails a measurement with duplicate case timings. Accepting it
    # here would let the rewrite score a suite the loop rejects.
    reading = driver_contract.read_driver_output(
        "case_ms: a 1.000\ncase_ms: a 9.000\nmean_ms: 5.000\n"
    )
    assert reading.duplicate_case_ids == ("a",)
    report_out = driver_contract._timing_report(reading)
    assert not report_out.ok
    assert report_out.failure_class == driver_contract.DUPLICATE_CASE_TIMINGS


def test_one_parser_serves_the_bench_tool_and_the_rewrite_contract():
    # Two regexes that agree today drift; the ids a coverage check accepts and
    # the times a KEEP score is built from have to come from one place.
    text = "[rank0] case_ms: a 1.000\ncase_ms: b 2.000 unscored\ncase_ms: a 3.000\n"
    shared = parse_case_timings(text)
    reading = driver_contract.read_driver_output(text)
    assert reading.case_times == shared.case_times
    assert list(reading.unscored_cases) == shared.unscored
    assert list(reading.duplicate_case_ids) == shared.duplicates


def test_rewrite_speedup_is_the_mean_over_cases_not_the_ratio_of_aggregates():
    # Measured aiter a16w16 baseline at n=k=6144, whose per-case times span 18x.
    source = {
        "m_1": 0.012818, "m_2": 0.013138, "m_4": 0.013352, "m_8": 0.013153,
        "m_16": 0.013757, "m_32": 0.015211, "m_64": 0.018329, "m_128": 0.020830,
        "m_256": 0.039681, "m_512": 0.045687, "m_1024": 0.075407,
        "m_2048": 0.114414, "m_4096": 0.229021,
    }
    # A candidate 5x faster on the eight cheap cases and 20% slower on the five
    # expensive ones: the shape a FlyDSL port takes when it beats aiter's own
    # small-M kernels but not hipBLASLt at large M.
    cheap = ("m_1", "m_2", "m_4", "m_8", "m_16", "m_32", "m_64", "m_128")
    candidate = {
        case: (ms / 5.0 if case in cheap else ms / 0.8) for case, ms in source.items()
    }
    source_ms = sum(source.values()) / len(source)
    candidate_ms = sum(candidate.values()) / len(candidate)

    result = report.build_result(
        op_name="op",
        port_ok=True,
        port_attempts=1,
        source_ms=source_ms,
        optimize_result={"best_ms": candidate_ms},
        source_case_times=source,
        flydsl_best_case_times=candidate,
    )
    assert result.speedup_basis == report.SPEEDUP_BASIS_MEAN_CASE
    assert result.speedup == pytest.approx(3.385, abs=1e-3)
    # The aggregate ratio calls the same candidate a regression, which is the
    # verdict this pipeline used to publish.
    assert source_ms / candidate_ms == pytest.approx(0.955, abs=1e-3)
    # Both readings are true and the contradiction is named, but the mean is
    # this layer's authoritative statistic: letting the aggregate withdraw
    # `improved` would restore it as the gate, and every consumer that reads
    # `improved is True` would reject the candidate exactly as before.
    assert result.aggregate_regression
    assert result.improved is True
    assert result.source_case_times == source


def test_rewrite_speedup_matches_the_statistic_forge_loop_keeps_on():
    # The reported number has to be the one the loop optimized, or the pipeline
    # publishes a verdict on a different objective than it searched.
    source = {"a": 4.0, "b": 1.0}
    candidate = {"a": 2.0, "b": 0.25}
    result = report.build_result(
        op_name="op",
        port_ok=True,
        port_attempts=1,
        source_ms=2.5,
        optimize_result={"best_ms": 1.125},
        source_case_times=source,
        flydsl_best_case_times=candidate,
    )
    assert result.speedup == pytest.approx(calculate_mean_case_speedup(candidate, source))
    assert result.speedup == pytest.approx(3.0)  # mean(2x, 4x), not 2.222x


def test_the_submission_gate_grades_on_the_mean_not_the_aggregate():
    # The PR's own example: a candidate the mean scores at 3.385x and the
    # aggregate calls a 0.955x regression. Grading on the aggregate rejects it,
    # so publishing the mean in the result while the gate still divides the
    # aggregates would change nothing that matters.
    applyback = {
        "baseline_ms": 0.048061,
        "best_ms": 0.050326,  # slower in total
        "mean_case_speedup": 3.385,
        "best_commit": "c" * 40,
    }
    assert forge_submit._rewrite_micro_speedup(applyback) == pytest.approx(3.385)
    assert forge_submit._rewrite_micro_speedup(applyback) > 1.0


def test_the_submission_gate_falls_back_to_the_aggregate_without_a_mean():
    aggregate_only = {"baseline_ms": 2.0, "best_ms": 1.0, "best_commit": "c" * 40}
    assert forge_submit._rewrite_micro_speedup(aggregate_only) == pytest.approx(2.0)

    unusable = {"baseline_ms": 0.0, "best_ms": 1.0, "best_commit": "c" * 40}
    assert forge_submit._rewrite_micro_speedup(unusable) is None


def test_the_kb_improvement_gate_grades_on_the_mean_not_the_aggregate(monkeypatch, tmp_path):
    # kb.py refused to record anything whose aggregate ratio was <= 1.0, which
    # discards exactly the ports this route is looking for. The store and the
    # gpu_type are stubbed so the improvement gate is the check under test.
    monkeypatch.setattr(kb, "create_rewrite_record_store", lambda config: object())
    config = SimpleNamespace(gpu_type="gfx950")
    spec = RewriteSpec(
        op_name="gemm",
        source_kernel="/w/gemm.py",
        target_functions=[],
        flydsl_kernel=str(tmp_path / "kernel.py"),
    )

    # Slower in aggregate (0.048 -> 0.050) but 3.385x on the mean over cases.
    kept = kb.write_flydsl_kb_solution(
        spec, "driver.py", config, source_ms=0.048061, flydsl_best_ms=0.050326, mean_case_speedup=3.385
    )
    assert kept["reason"] != "no_improvement"

    # A mean at or below 1.0 is still refused; the change is which statistic
    # decides, not whether there is a gate.
    refused = kb.write_flydsl_kb_solution(
        spec, "driver.py", config, source_ms=0.048061, flydsl_best_ms=0.010000, mean_case_speedup=0.9
    )
    assert refused["reason"] == "no_improvement"

    # No mean available -> the aggregate ratio still gates, unchanged.
    legacy = kb.write_flydsl_kb_solution(
        spec, "driver.py", config, source_ms=2.0, flydsl_best_ms=4.0, mean_case_speedup=None
    )
    assert legacy["reason"] == "no_improvement"


def test_aggregate_ratio_is_used_and_named_when_a_driver_reports_no_cases():
    result = report.build_result(
        op_name="op",
        port_ok=True,
        port_attempts=1,
        source_ms=2.0,
        optimize_result={"best_ms": 1.0},
        source_case_times={},
        flydsl_best_case_times={},
    )
    assert result.speedup == pytest.approx(2.0)
    assert result.speedup_basis == report.SPEEDUP_BASIS_AGGREGATE
    # A ratio of aggregates cannot contradict itself, so nothing to report.
    assert result.aggregate_regression == ""


def test_a_case_set_mismatch_refuses_to_score_rather_than_comparing_subsets():
    result = report.build_result(
        op_name="op",
        port_ok=True,
        port_attempts=1,
        source_ms=2.0,
        optimize_result={"best_ms": 1.0},
        source_case_times={"a": 2.0, "b": 2.0},
        flydsl_best_case_times={"a": 1.0},
    )
    # No number at all, not a quiet fall back to the aggregate: the two paths
    # timed different work, so their aggregates compare different work too.
    # Scoring 'a' alone would publish a partial workload as the whole suite.
    assert result.speedup is None
    assert result.speedup_basis == report.SPEEDUP_BASIS_NONE
    assert result.speedup_unavailable_reason == driver_contract.CASE_SCORE_INCOMPARABLE
    assert result.improved is False


def test_applyback_is_required_only_for_framework_repositories():
    legacy = report.build_result(
        op_name="op",
        port_ok=True,
        port_attempts=1,
        source_ms=2.0,
        optimize_result={"best_ms": 1.0},
        applyback_result={"ok": False, "error": "no git base"},
        applyback_required=False,
    )
    assert legacy.success is True

    framework = report.build_result(
        op_name="op",
        port_ok=True,
        port_attempts=1,
        source_ms=2.0,
        optimize_result={"best_ms": 1.0},
        applyback_result={"ok": False, "error": "agent failed"},
        applyback_required=True,
    )
    assert framework.success is False


def test_rewrite_result_exposes_canonical_forge_patch_contract():
    result = report.build_result(
        op_name="op",
        port_ok=True,
        port_attempts=1,
        source_ms=2.0,
        optimize_result={"best_ms": 1.0, "best_commit": "flydsl-best"},
        applyback_result={
            "ok": True,
            "best_commit": "framework-best",
            "manifest_path": ("/workspace/forge_experiments/rewrite_applyback/best/manifest.json"),
            "patch_path": ("/workspace/forge_experiments/rewrite_applyback/best/iter_001/forge.patch"),
            "canonical_patch_path": ("/workspace/forge_experiments/rewrite_applyback/best/iter_001/forge.patch"),
            "canonical_files_root": ("/workspace/forge_experiments/rewrite_applyback/best/iter_001/files"),
            "canonical_result_path": ("/workspace/forge_experiments/rewrite_applyback/result.json"),
            "forge_workspace": "/workspace",
            "artifacts": ["/workspace/forge_experiments/rewrite_applyback/best/iter_001/forge.patch"],
            "changed_files": ["framework/op.py"],
        },
        applyback_required=True,
    )

    assert result.success is True
    assert result.best_commit == "framework-best"
    assert result.flydsl_best_commit == "flydsl-best"
    assert result.artifact_kind == "framework_applyback"
    assert result.artifact_schema_version == 2
    assert result.canonical_patch_path == result.patch_path
    assert result.canonical_files_root.endswith("/files")
    assert result.canonical_result_path.endswith("/rewrite_applyback/result.json")
    assert result.forge_workspace == "/workspace"
    assert result.artifacts == [result.patch_path]


def test_rewrite_result_reports_the_logical_identity_and_its_symbol():
    result = report.build_result(
        op_name="vllm::softmax",
        port_ok=True,
        port_attempts=1,
        source_ms=2.0,
        optimize_result={"best_ms": 1.0},
    )

    assert result.logical_op_name == "vllm::softmax"
    assert result.builder_symbol == f"build_{result.operator_slug}_module"
    assert result.builder_symbol.isidentifier()


def test_rewrite_result_never_reports_the_standalone_best_as_framework_best():
    failed = report.build_result(
        op_name="op",
        port_ok=True,
        port_attempts=1,
        source_ms=2.0,
        optimize_result={"best_ms": 1.0, "best_commit": "flydsl-best"},
        applyback_result={"ok": False, "error": "agent failed"},
        applyback_required=True,
    )

    assert failed.success is False
    assert failed.best_commit == ""
    assert failed.flydsl_best_commit == "flydsl-best"
    assert failed.canonical_result_path == ""
    # No published bundle means no artifact to name.
    assert failed.artifact_kind == ""
    assert failed.artifact_schema_version == 0

    standalone = report.build_result(
        op_name="op",
        port_ok=True,
        port_attempts=1,
        source_ms=2.0,
        optimize_result={"best_ms": 1.0, "best_commit": "flydsl-best"},
    )

    assert standalone.success is True
    assert standalone.best_commit == "flydsl-best"
