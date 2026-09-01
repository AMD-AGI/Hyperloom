"""Unit tests for the operator-agnostic (BYOD) forge-rewrite-by-flydsl layer.

These are pure-Python (no GPU, no LLM, no FlyDSL): they pin the rewrite spec,
the source-entry discovery heuristic, the unresolved-entry path, the generic
seed skeleton, and the speedup math. GPU/agent behavior (and driver measurement
primitives) is covered by the L1/L2/L3 integration ladder, not here.
"""

from __future__ import annotations

import textwrap

import pytest

from kernelforge.rewrite_by_flydsl import ingest, report, seed
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
