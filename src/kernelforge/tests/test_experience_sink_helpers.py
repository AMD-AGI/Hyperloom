"""Unit tests for experience_sink pure helpers (identity, signature, summary)."""

from __future__ import annotations

from kernelforge.knowledge import experience_sink as sink


# --------------------------------------------------------------------------- #
# resolve_operation
# --------------------------------------------------------------------------- #
def test_resolve_operation_prefers_compute_kernel(monkeypatch):
    import kernelforge.mcp_server.tools.pmc as pmc

    monkeypatch.setattr(pmc, "derive_kernel_names", lambda _src: ["launch_wrapper", "my_gemm"])
    assert sink.resolve_operation("src", "/p/f.py") == "my_gemm"


def test_resolve_operation_uses_first_when_all_launchers(monkeypatch):
    import kernelforge.mcp_server.tools.pmc as pmc

    monkeypatch.setattr(pmc, "derive_kernel_names", lambda _src: ["launch_a", "main"])
    assert sink.resolve_operation("src", "/p/f.py") == "launch_a"


def test_resolve_operation_falls_back_to_target_then_stem(monkeypatch):
    import kernelforge.mcp_server.tools.pmc as pmc

    monkeypatch.setattr(pmc, "derive_kernel_names", lambda _src: [])
    assert sink.resolve_operation("", "/p/f.py", target_functions=["", " op_x "]) == "op_x"
    assert sink.resolve_operation("", "/p/my_file.py", target_functions=[]) == "my_file"


def test_resolve_operation_fallback_is_order_independent(monkeypatch):
    # A producer and consumer may hand the same target-function set in different
    # orders; the resolved op (and thus the slug) must not depend on that order.
    import kernelforge.mcp_server.tools.pmc as pmc

    monkeypatch.setattr(pmc, "derive_kernel_names", lambda _src: [])
    a = sink.resolve_operation("", "/p/f.py", target_functions=["gemm_kernel", "epilogue_kernel"])
    b = sink.resolve_operation("", "/p/f.py", target_functions=["epilogue_kernel", "gemm_kernel"])
    assert a == b
    # launchers/wrappers are de-prioritized even when they sort first
    c = sink.resolve_operation("", "/p/f.py", target_functions=["launch_gemm", "gemm_kernel"])
    assert c == "gemm_kernel"


def test_resolve_operation_survives_derive_exception(monkeypatch):
    import kernelforge.mcp_server.tools.pmc as pmc

    def boom(_src):
        raise RuntimeError("derive failed")

    monkeypatch.setattr(pmc, "derive_kernel_names", boom)
    assert sink.resolve_operation("", "/p/stem.py") == "stem"


# --------------------------------------------------------------------------- #
# detect_backend_language
# --------------------------------------------------------------------------- #
def test_detect_backend_language_kernel_backend_wins():
    assert sink.detect_backend_language("flydsl") == "flydsl"


def test_detect_backend_language_requires_kernel_backend():
    assert sink.detect_backend_language("") == "unknown"


def test_detect_framework_standalone_is_unknown():
    assert sink.detect_framework("/tmp/standalone/k.py") == "unknown"


def test_detect_framework_from_path():
    assert sink.detect_framework("/repo/aiter/csrc/k.hip") == "aiter"
    assert sink.detect_framework("/x/sglang/y/k.py") == "sglang"


def test_detect_framework_explicit_override_wins_over_path():
    # A flattened/scratch workspace can drop the 'vllm/' dir from the path; an
    # explicit --framework must still yield the right framework so the slug does
    # not diverge between producer and consumer.
    assert (
        sink.detect_framework(
            "/tmp/scratch/k.py",
            framework_override="vllm",
        )
        == "vllm"
    )


def test_detect_framework_canonicalizes_aiter_meta_owner():
    assert (
        sink.detect_framework(
            "/tmp/flattened/kernel.py",
            framework_override="aiter_meta",
        )
        == "aiter"
    )


def test_detect_framework_standalone_sentinel_is_unknown():
    # Explicit 'standalone' == a framework-less file == undetected path.
    assert (
        sink.detect_framework(
            "/x/vllm/y/k.py",
            framework_override="standalone",
        )
        == "unknown"
    )


# --------------------------------------------------------------------------- #
# find_defining_source
# --------------------------------------------------------------------------- #
def test_find_defining_source_empty_op_returns_anchor():
    assert sink.find_defining_source("", "/a.py", "anchor body", None) == "anchor body"


def test_find_defining_source_prefers_anchor_when_it_defines():
    anchor = "def my_op(x):\n    return x\n"
    assert sink.find_defining_source("my_op", "/a.py", anchor, ["/other.py"]) == anchor


def test_find_defining_source_scans_other_files(tmp_path):
    other = tmp_path / "impl.py"
    other.write_text("def real_op(a, b):\n    return a\n")
    got = sink.find_defining_source("real_op", "/a.py", "wrapper only", [str(other)])
    assert "def real_op" in got


def test_find_defining_source_falls_back_to_anchor(tmp_path):
    missing = tmp_path / "nope.py"
    got = sink.find_defining_source("absent", "/a.py", "anchor", [str(missing)])
    assert got == "anchor"


def test_find_defining_source_matches_global_kernel():
    anchor = "__global__ void my_kernel(float* a) {}\n"
    assert sink.find_defining_source("my_kernel", "/a.cu", anchor, None) == anchor


# --------------------------------------------------------------------------- #
# signature -> dtype parsing
# --------------------------------------------------------------------------- #
def test_extract_input_dtypes_python():
    src = "def f(x: float, y: int = 3, *args, self_unused=1):\n    pass\n"
    dt = sink.extract_input_dtypes(src, "f", "triton")
    assert dt["x"] == "float"
    assert dt["y"] == "int"
    assert "args" not in dt


def test_extract_input_dtypes_python_untyped_is_unknown():
    dt = sink.extract_input_dtypes("def g(a, b):\n    pass\n", "g", "torch")
    assert dt == {"a": "unknown", "b": "unknown"}


def test_extract_input_dtypes_c_pointer_folding():
    src = "__global__ void k(const float* a, int n) { }"
    dt = sink.extract_input_dtypes(src, "k", "hip")
    assert dt["a"] == "const float*"
    assert dt["n"] == "int"


def test_extract_input_dtypes_c_array_subscript():
    src = "void k(float a[16], int n) {\n  return;\n}"
    dt = sink.extract_input_dtypes(src, "k", "cuda")
    assert dt["a"] == "float[16]"


def test_extract_input_dtypes_empty_on_missing_signature():
    assert sink.extract_input_dtypes("no such func here", "ghost", "hip") == {}
    assert sink.extract_input_dtypes("", "f", "hip") == {}
    assert sink.extract_input_dtypes("def f(x):pass", "", "hip") == {}


def test_extract_input_dtypes_strips_comments():
    src = "def f(\n    x: float,  # the input\n    y: int,  # count\n):\n    pass\n"
    dt = sink.extract_input_dtypes(src, "f", "triton")
    assert dt == {"x": "float", "y": "int"}


def test_extract_input_dtypes_c_comment_strip():
    src = "void k(float* a /* NxM */, int n /* rows */) { }"
    dt = sink.extract_input_dtypes(src, "k", "hip")
    assert set(dt) == {"a", "n"}


def test_signature_params_skips_call_site_finds_definition():
    src = "some_call(1, 2);\nvoid some_call(int a, int b) {\n  return;\n}"
    assert sink._signature_params(src, "some_call") == "int a, int b"


def test_signature_params_none_when_absent():
    assert sink._signature_params("nothing", "f") is None


def test_split_top_level_respects_brackets():
    assert sink._split_top_level("a, b[1, 2], c") == ["a", "b[1, 2]", "c"]


def test_parse_param_c_void_and_empty():
    assert sink._parse_param_c("void") == ("", "")
    assert sink._parse_param_c("   ") == ("", "")


def test_balanced_parens_unbalanced_returns_sentinel():
    inner, close = sink._balanced_parens("f(a, b", 1)
    assert close == -1


def test_signature_params_skips_unbalanced_candidate():
    # First call-like match is unbalanced (no closing paren) -> skipped; the real
    # definition below is returned.
    src = "k(a, b\nvoid k(int a) {\n  return;\n}"
    assert sink._signature_params(src, "k") == "int a"


def test_parse_param_c_no_identifier_returns_empty():
    assert sink._parse_param_c("float*") == ("", "")


def test_parse_param_c_reference_marker():
    name, typ = sink._parse_param_c("Tensor& t")
    assert name == "t"
    assert "&" in typ


# --------------------------------------------------------------------------- #
# LLM summary helpers
# --------------------------------------------------------------------------- #
def test_extract_json_from_code_fence():
    text = 'noise\n```json\n{"category": "GEMM", "strategy": "x"}\n```\ntail'
    assert sink._extract_json(text) == {"category": "GEMM", "strategy": "x"}


def test_extract_json_bare_object():
    assert sink._extract_json('prefix {"a": 1} suffix') == {"a": 1}


def test_extract_json_bad_inputs():
    assert sink._extract_json("") == {}
    assert sink._extract_json("no json here") == {}
    assert sink._extract_json("{not valid}") == {}
    assert sink._extract_json("[1,2,3]") == {}


def test_normalize_summary_defaults_and_category_clamp():
    assert sink._normalize_summary({}) == {
        "category": "others",
        "strategy": "",
        "recipe": "",
        "lessons": "",
    }
    out = sink._normalize_summary({"category": "GEMM", "strategy": " s ", "recipe": 5, "lessons": ""})
    assert out["category"] == "gemm"
    assert out["strategy"] == "s"
    assert out["recipe"] == ""


def test_summarize_run_returns_defaults_on_llm_failure(monkeypatch):
    """Return deterministic defaults when the provider summary fails."""

    async def boom(*_a, **_k):
        raise RuntimeError("no sdk")

    monkeypatch.setattr(sink, "_query_llm", boom)
    out = sink.summarize_run(config=object(), workspace="/w", op="op", digest="d", kernel_source="s")
    assert out == {"category": "others", "strategy": "", "recipe": "", "lessons": ""}


def test_summarize_run_parses_reply(monkeypatch):
    """Parse the provider reply and forward the shared usage accumulator."""
    captured = {}

    async def reply(*_a, **kwargs):
        captured.update(kwargs)
        return '{"category": "attention", "strategy": "tile"}'

    monkeypatch.setattr(sink, "_query_llm", reply)
    usage = object()
    out = sink.summarize_run(config=object(), workspace="/w", op="op", digest="d", kernel_source="s", usage=usage)
    assert out["category"] == "attention"
    assert out["strategy"] == "tile"
    assert captured["usage"] is usage


def test_summary_prompt_truncates_inputs():
    prompt = sink._summary_prompt("op", "d" * 20000, "s" * 20000)
    assert prompt.count("d") <= sink._MAX_DIGEST_CHARS + 50
    assert "Operator under optimization: op" in prompt


# --------------------------------------------------------------------------- #
# diff parsing
# --------------------------------------------------------------------------- #
def test_changed_files_from_diff_dedups():
    diff = "diff --git a/x.py b/x.py\ndiff --git a/y.c b/y.c\ndiff --git a/x.py b/x.py\n"
    assert sink._changed_files_from_diff(diff) == ["x.py", "y.c"]
    assert sink._changed_files_from_diff("") == []
