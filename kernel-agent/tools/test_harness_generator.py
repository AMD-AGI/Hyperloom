# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for harness_generator.py (AST analysis + harness synthesis)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parent


def _load_module():
    """Load harness_generator.py as an isolated module."""
    spec = importlib.util.spec_from_file_location(
        "harness_generator_under_test",
        _TOOLS_DIR / "harness_generator.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec: dataclasses with PEP 563 annotations resolve their
    # module via sys.modules during class creation.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


hg = _load_module()


BENCH_SRC = """\
import torch
import torch.nn.functional as F

torch.set_default_device("cuda")
torch.manual_seed(0)

@perftest
def torch_ref(x, weight, eps):
    y = torch.randn(256, 128, dtype=torch.bfloat16)
    return F.rms_norm(x, weight, eps)

@perftest
def triton_kernel(x, weight, eps):
    return my_kernel(x, weight, eps)

@benchmark
def test_main():
    x = torch.randn(M, N, dtype=torch.bfloat16)
    w = torch.randn(N, dtype=torch.bfloat16)
    torch_ref(x, w, 1e-6)
    triton_kernel(x, w, 1e-6)
"""


# ---- BenchmarkAnalyzer ----


def test_get_imports():
    a = hg.BenchmarkAnalyzer(BENCH_SRC)
    imports = a.get_imports()
    assert any("import torch" in i for i in imports)


def test_get_imports_dedents_function_local():
    """Regression: ast.walk yields imports nested inside functions with their
    source indentation; emitted verbatim at the harness module top they raise
    'unexpected indent'. get_imports must dedent them to valid top-level imports.
    """
    import ast as _ast

    src = (
        "import torch\n"
        "def f():\n"
        "    import math\n"
        "    return math.pi\n"
    )
    a = hg.BenchmarkAnalyzer(src)
    imports = a.get_imports()
    assert "import math" in imports, imports
    # Joined imports must form valid top-level Python (no IndentationError).
    _ast.parse("\n".join(imports))


def test_aiter_harness_recognizes_perftest(tmp_path):
    """Regression: aiter op_tests that time the op with @perftest (not
    @benchmark) must be picked up by the aiter idiom path and emit a valid
    harness (previously they fell through to the weaker generic path)."""
    import ast as _ast

    src = (
        "import torch\n"
        "import aiter\n"
        "from aiter.test_common import perftest, run_perftest\n"
        "@perftest\n"
        "def bench_op(m, n, k, dtype):\n"
        "    x = torch.randn(m, k, dtype=dtype)\n"
        "    return run_perftest(aiter.gemm, x)\n"
    )
    bench = tmp_path / "test_op.py"
    bench.write_text(src, encoding="utf-8")
    a = hg.BenchmarkAnalyzer(src)
    out = hg._try_generate_aiter_harness(
        analyzer=a,
        decorated=a.get_decorated_functions(),
        candidate={"input_shapes": {"M": 64, "N": 64, "K": 64}, "precision": "bf16"},
        source_file=str(bench),
        benchmark_path=bench,
        out_dir=tmp_path,
        log=lambda _m: None,
    )
    assert out is not None
    _ast.parse(Path(out.harness_path).read_text(encoding="utf-8"))


def test_aiter_harness_maps_tracelens_list_shapes(tmp_path):
    """AITer idiom generation must understand production TraceLens list shapes."""
    import ast as _ast

    src = (
        "import torch\n"
        "import aiter\n"
        "from aiter.test_common import perftest, run_perftest\n"
        "@perftest\n"
        "def bench_op(m, n, k, dtype):\n"
        "    x = torch.randn(m, k, dtype=dtype)\n"
        "    return run_perftest(aiter.gemm, x)\n"
    )
    bench = tmp_path / "test_list_shape_op.py"
    bench.write_text(src, encoding="utf-8")
    a = hg.BenchmarkAnalyzer(src)
    out = hg._try_generate_aiter_harness(
        analyzer=a,
        decorated=a.get_decorated_functions(),
        candidate={
            "input_shapes": [
                {"call_num": 1, "shape": "(32, 32) bf16"},
                {"call_num": 9, "shape": "(64, 128, 256) bf16"},
            ],
            "precision": "bf16",
        },
        source_file=str(bench),
        benchmark_path=bench,
        out_dir=tmp_path,
        log=lambda _m: None,
    )
    assert out is not None
    code = Path(out.harness_path).read_text(encoding="utf-8")
    _ast.parse(code)
    assert "'m': 64" in code
    assert "'n': 128" in code
    assert "'k': 256" in code


def test_aiter_harness_rejects_incomplete_shape_kwargs(tmp_path):
    """Do not emit dtype-only/partial calls when required m/n/k values are absent."""
    src = (
        "import torch\n"
        "import aiter\n"
        "from aiter.test_common import perftest, run_perftest\n"
        "@perftest\n"
        "def bench_op(m, n, k, dtype):\n"
        "    x = torch.randn(m, k, dtype=dtype)\n"
        "    return run_perftest(aiter.gemm, x)\n"
    )
    bench = tmp_path / "test_bad_shape_op.py"
    bench.write_text(src, encoding="utf-8")
    a = hg.BenchmarkAnalyzer(src)
    logs: list[str] = []
    out = hg._try_generate_aiter_harness(
        analyzer=a,
        decorated=a.get_decorated_functions(),
        candidate={"input_shapes": [{"call_num": 5, "shape": "(64, 128) bf16"}]},
        source_file=str(bench),
        benchmark_path=bench,
        out_dir=tmp_path,
        log=logs.append,
    )
    assert out is None
    assert any("missing values" in msg for msg in logs)


def test_get_decorated_functions():
    a = hg.BenchmarkAnalyzer(BENCH_SRC)
    dec = a.get_decorated_functions()
    assert set(dec) == {"torch_ref", "triton_kernel", "test_main"}
    assert dec["torch_ref"].decorator == "perftest"
    assert dec["test_main"].decorator == "benchmark"
    assert dec["torch_ref"].params == ["x", "weight", "eps"]


def test_decorator_name_variants():
    import ast

    a = hg.BenchmarkAnalyzer("x = 1")
    assert a._decorator_name(ast.parse("@deco\ndef f():pass").body[0].decorator_list[0]) == "deco"
    assert a._decorator_name(ast.parse("@deco()\ndef f():pass").body[0].decorator_list[0]) == "deco"
    assert a._decorator_name(ast.parse("@m.deco\ndef f():pass").body[0].decorator_list[0]) == "deco"
    assert a._decorator_name(ast.parse("@m.deco()\ndef f():pass").body[0].decorator_list[0]) == "deco"


def test_classify_functions_ref_and_kernel():
    a = hg.BenchmarkAnalyzer(BENCH_SRC)
    ref, kernel = a.classify_functions(a.get_decorated_functions())
    assert ref.name == "torch_ref"
    assert kernel.name == "triton_kernel"


def test_classify_functions_single_perftest_is_kernel():
    src = "@perftest\ndef do_thing(a):\n    return a\n"
    a = hg.BenchmarkAnalyzer(src)
    ref, kernel = a.classify_functions(a.get_decorated_functions())
    assert ref is None
    assert kernel.name == "do_thing"


def test_classify_functions_two_plain_perftests():
    src = "@perftest\ndef alpha(a):\n    return a\n@perftest\ndef beta(a):\n    return a\n"
    a = hg.BenchmarkAnalyzer(src)
    ref, kernel = a.classify_functions(a.get_decorated_functions())
    # Both unclassified -> both become kernel candidates; first wins as kernel.
    assert kernel is not None


def test_classify_with_source_module():
    src = "@perftest\ndef run_it(a):\n    return mypkg.kernel(a)\n"
    a = hg.BenchmarkAnalyzer(src, source_file_module="mypkg.ops")
    ref, kernel = a.classify_functions(a.get_decorated_functions())
    assert kernel.name == "run_it"


def test_get_test_function_benchmark():
    a = hg.BenchmarkAnalyzer(BENCH_SRC)
    tf = a.get_test_function(a.get_decorated_functions())
    assert tf.name == "test_main"


def test_get_test_function_toplevel_caller():
    src = "@perftest\ndef mykernel(a):\n    return a\ndef bench_runner():\n    return mykernel(1)\n"
    a = hg.BenchmarkAnalyzer(src)
    tf = a.get_test_function(a.get_decorated_functions())
    assert tf.name == "bench_runner"


def test_extract_tensor_creation():
    a = hg.BenchmarkAnalyzer(BENCH_SRC)
    dec = a.get_decorated_functions()
    tensors = a.extract_tensor_creation(dec["torch_ref"])
    assert tensors[0].var_name == "y"
    assert tensors[0].dtype_expr == "torch.bfloat16"


def test_extract_tensor_creation_syntax_error():
    a = hg.BenchmarkAnalyzer("x = 1")
    bad = hg.FuncInfo(name="f", params=[], source="def f(:\n  pass", decorator="", lineno=1)
    assert a.extract_tensor_creation(bad) == []


def test_extract_call_to():
    a = hg.BenchmarkAnalyzer(BENCH_SRC)
    dec = a.get_decorated_functions()
    call = a.extract_call_to(dec["test_main"], "torch_ref")
    assert call.args == ["x", "w", "1e-06"]


def test_extract_call_to_not_found():
    a = hg.BenchmarkAnalyzer(BENCH_SRC)
    dec = a.get_decorated_functions()
    assert a.extract_call_to(dec["test_main"], "nonexistent") is None


def test_call_func_name_variants():
    import ast

    a = hg.BenchmarkAnalyzer("x = 1")
    assert a._call_func_name(ast.parse("foo()").body[0].value) == "foo"
    assert a._call_func_name(ast.parse("a.b.c()").body[0].value) == "a.b.c"


def test_get_toplevel_statements():
    a = hg.BenchmarkAnalyzer(BENCH_SRC)
    tops = a.get_toplevel_statements()
    assert any("set_default_device" in t for t in tops)
    assert any("manual_seed" in t for t in tops)


# ---- config builder ----


def test_parse_shape_string():
    assert hg._parse_shape_string("(256, 128) bf16") == ((256, 128), "bf16")
    assert hg._parse_shape_string("(64)") == ((64,), "bfloat16")
    assert hg._parse_shape_string("garbage") == ((), "")
    assert hg._parse_shape_string("(a, b) bf16") == ((), "")


def test_dim_names():
    assert hg._dim_names(3) == ["M", "N", "K"]
    assert len(hg._dim_names(8)) == 8
    assert hg._dim_names(8)[0] == "D0"


def test_default_configs():
    all_c, unpack, cfg_str = hg._default_configs()
    assert "torch.bfloat16" in all_c
    assert unpack == "M, N, dtype = cfg"


def test_build_configs_default_when_empty():
    assert hg._build_configs({}) == hg._default_configs()


def test_build_configs_from_shapes():
    candidate = {
        "input_shapes": [
            {"call_num": 5, "shape": "(256, 128) bf16"},
            {"call_num": 2, "shape": "(512, 128) fp16"},
        ]
    }
    all_c, unpack, cfg_str = hg._build_configs(candidate)
    assert "torch.bfloat16" in all_c
    assert "M, N, dtype = cfg" == unpack
    # Scaling expands to >= 6 configs.
    assert all_c.count("(") >= 6


def test_build_configs_unparseable_shapes_fallback():
    candidate = {"input_shapes": [{"call_num": 1, "shape": "nope"}]}
    assert hg._build_configs(candidate) == hg._default_configs()


# ---- predicates ----


@pytest.mark.parametrize(
    "name,expected",
    [
        ("eps", True),
        ("epsilon", True),
        ("dropout", True),
        ("use_model_sensitive", True),
        ("x", False),
    ],
)
def test_is_scalar_param(name, expected):
    assert hg._is_scalar_param(name) is expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("weight", True),
        ("bias", True),
        ("gamma", True),
        ("up_weight", True),
        ("x", False),
    ],
)
def test_is_weight_param(name, expected):
    assert hg._is_weight_param(name) is expected


@pytest.mark.parametrize(
    "s,expected",
    [
        ("foo", True),
        ("x1", True),
        ("True", False),
        ("None", False),
        ("1e-6", False),
        ("torch.randn(3)", False),
    ],
)
def test_is_variable(s, expected):
    assert hg._is_variable(s) is expected


def test_match_call_args_to_params():
    call = hg.CallInfo(func_name="f", args=["x", "1e-6"], kwargs={"bias": "b", "dtype": "torch.bf16"})
    out = hg._match_call_args_to_params(call, ["a", "eps", "bias"])
    names = [p for p, _ in out]
    assert "a" in names and "eps" in names and "bias" in names


# ---- adapter generators ----


def test_generate_setup_inputs_no_target():
    a = hg.BenchmarkAnalyzer("x = 1")
    body = hg._generate_setup_inputs(a, None, "M, N, dtype = cfg", None, None)
    assert "torch.randn" in body
    assert 'return {"x": x}' in body


def test_generate_setup_inputs_with_kernel():
    a = hg.BenchmarkAnalyzer(BENCH_SRC)
    dec = a.get_decorated_functions()
    body = hg._generate_setup_inputs(
        a,
        dec["test_main"],
        "M, N, dtype = cfg",
        dec["torch_ref"],
        dec["triton_kernel"],
    )
    assert "weight" in body
    assert "eps" in body


def test_generate_run_kernel_passthrough():
    a = hg.BenchmarkAnalyzer("x = 1")
    body = hg._generate_run_kernel(a, None, None)
    assert "inputs.get" in body


def test_generate_run_kernel_with_func():
    a = hg.BenchmarkAnalyzer(BENCH_SRC)
    dec = a.get_decorated_functions()
    body = hg._generate_run_kernel(a, dec["test_main"], dec["triton_kernel"])
    assert "triton_kernel(" in body
    assert "while isinstance(result, tuple)" in body


def test_generate_run_ref_delegates_when_missing():
    a = hg.BenchmarkAnalyzer("x = 1")
    assert "run_kernel(inputs)" in hg._generate_run_ref(a, None, None, None)


def test_generate_run_func_body_without_call():
    a = hg.BenchmarkAnalyzer("x = 1")
    fi = hg.FuncInfo(name="k", params=["self", "a", "b"], source="def k(self,a,b):\n  return a", decorator="", lineno=1)
    body = hg._generate_run_func_body(a, None, fi)
    assert 'inputs.get("a")' in body


# ---- maybe_generate_harness ----


def test_maybe_generate_harness_missing_file(tmp_path):
    out = hg.maybe_generate_harness(
        benchmark_file=str(tmp_path / "nope.py"),
        candidate={},
        source_file="",
        out_dir=tmp_path,
    )
    assert out is None


def test_maybe_generate_harness_no_decorated(tmp_path, monkeypatch):
    _inject_fake_validate(monkeypatch, harness_ok=True, bench_ok=False)
    bench = tmp_path / "b.py"
    bench.write_text("import torch\nx = 1\n", encoding="utf-8")
    out = hg.maybe_generate_harness(
        benchmark_file=str(bench),
        candidate={},
        source_file="",
        out_dir=tmp_path,
    )
    assert out is None


def test_maybe_generate_harness_no_kernel_or_ref(tmp_path, monkeypatch):
    _inject_fake_validate(monkeypatch, harness_ok=True, bench_ok=False)
    bench = tmp_path / "b.py"
    bench.write_text("@benchmark\ndef test_x():\n    pass\n", encoding="utf-8")
    out = hg.maybe_generate_harness(
        benchmark_file=str(bench),
        candidate={},
        source_file="",
        out_dir=tmp_path,
    )
    assert out is None


def test_maybe_generate_harness_success(tmp_path, monkeypatch):
    _inject_fake_validate(monkeypatch, harness_ok=True, bench_ok=False)
    logs = []
    bench = tmp_path / "bench_rms.py"
    bench.write_text(BENCH_SRC, encoding="utf-8")
    out = hg.maybe_generate_harness(
        benchmark_file=str(bench),
        candidate={"input_shapes": [{"call_num": 1, "shape": "(256, 128) bf16"}]},
        source_file="",
        out_dir=tmp_path,
        log_fn=logs.append,
    )
    assert out is not None
    assert "--correctness" in out.test_command
    assert Path(out.harness_path).is_file()


def test_maybe_generate_harness_already_valid_skips(tmp_path, monkeypatch):
    # static_check says the input benchmark is already a valid harness -> skip.
    _inject_fake_validate(monkeypatch, harness_ok=True, bench_ok=True, force_file=True)
    bench = tmp_path / "bench_rms.py"
    bench.write_text(BENCH_SRC, encoding="utf-8")
    out = hg.maybe_generate_harness(
        benchmark_file=str(bench),
        candidate={},
        source_file="",
        out_dir=tmp_path,
    )
    assert out is None


def _fake_static_check(harness_ok, bench_ok):
    """Build a static_check stub keyed on whether the path is a harness file."""

    def static_check(path):
        # Key off the file name only; the tmp_path dir itself contains the
        # test name ("...harness_success") and must not be misread as harness.
        if Path(path).name.startswith("harness_"):
            return (harness_ok, [] if harness_ok else ["bad"])
        return (bench_ok, [] if bench_ok else ["needs-gen"])

    return static_check


def _inject_fake_validate(monkeypatch, *, harness_ok, bench_ok, force_file=False):
    """Patch validate_harness.static_check deterministically.

    maybe_generate_harness does ``from validate_harness import static_check``
    against the real module on disk; patch its attribute on the real (or a
    freshly injected) module object so both L1/L2 call sites see the stub.
    """
    stub = _fake_static_check(harness_ok, bench_ok)
    validator_dir = _TOOLS_DIR.parent / "skills" / "unittest"
    if (validator_dir / "validate_harness.py").is_file():
        if str(validator_dir) not in sys.path:
            sys.path.insert(0, str(validator_dir))
        import validate_harness as real_vh

        monkeypatch.setattr(real_vh, "static_check", stub)
    else:
        mod = types.ModuleType("validate_harness")
        mod.static_check = stub
        monkeypatch.setitem(sys.modules, "validate_harness", mod)
    if force_file:
        monkeypatch.setattr(hg.Path, "is_file", lambda self: True)


# ---- new type-inference predicates (P1) ----


@pytest.mark.parametrize(
    "name,expected",
    [
        ("block_tables", True),
        ("seq_lens", True),
        ("context_lens", True),
        ("kv_indptr", True),
        ("block_table_foo", True),
        ("query", False),
        ("scale", False),
    ],
)
def test_is_index_param(name, expected):
    assert hg._is_index_param(name) is expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("max_seq_len", True),
        ("num_kv_heads", True),
        ("num_queries_per_kv", True),
        ("head_size", True),
        ("block_size", True),
        ("query", False),
        ("scale", False),
    ],
)
def test_is_int_scalar_param(name, expected):
    assert hg._is_int_scalar_param(name) is expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("scale", True),
        ("softmax_scale", True),
        ("k_scale", True),
        ("k_scale_cache", False),
        ("query", False),
    ],
)
def test_is_float_scalar_param(name, expected):
    assert hg._is_float_scalar_param(name) is expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("dtype", True),
        ("kv_cache_dtype", True),
        ("q_dtype", True),
        ("query", False),
    ],
)
def test_is_dtype_param(name, expected):
    assert hg._is_dtype_param(name) is expected


# ---- attention-like harness: union keys + correct types (P1 + P2) ----

ATTN_BENCH_SRC = """\
import torch

@perftest
def torch_ref(query, k_cache, v_cache, block_tables, seq_lens, max_seq_len,
              kv_cache_dtype, num_kv_heads, scale, alibi_slopes,
              k_scale_cache, v_scale_cache, num_queries_per_kv, dtype):
    return query

@perftest
def triton_kernel(query, k_cache, v_cache, block_tables, seq_lens, max_seq_len,
                  kv_cache_dtype, num_kv_heads, scale, alibi_slopes,
                  k_scale, v_scale):
    return query

@benchmark
def test_main():
    triton_kernel(query, k_cache, v_cache, block_tables, seq_lens, max_seq_len,
                  kv_cache_dtype, num_kv_heads, scale, alibi_slopes, k_scale, v_scale)
    torch_ref(query, k_cache, v_cache, block_tables, seq_lens, max_seq_len,
              kv_cache_dtype, num_kv_heads, scale, alibi_slopes,
              k_scale_cache, v_scale_cache, num_queries_per_kv, dtype)
"""


def _setup_keys(body: str) -> set[str]:
    """Parse the dict keys returned by a generated setup_inputs body."""
    import ast

    tree = ast.parse("def setup_inputs(cfg):\n" + body)
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for k in node.value.keys:
                if isinstance(k, ast.Constant):
                    keys.add(k.value)
    return keys


def _referenced_keys(body: str) -> set[str]:
    """Parse inputs.get("X") / inputs["X"] keys referenced by a run_* body."""
    import ast

    tree = ast.parse("def run_x(inputs):\n" + body)
    refs: set[str] = set()
    for node in ast.walk(tree):
        # inputs.get("X")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "inputs"
                and node.args and isinstance(node.args[0], ast.Constant)):
            refs.add(node.args[0].value)
        # inputs["X"]
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == "inputs"
                and isinstance(node.slice, ast.Constant)):
            refs.add(node.slice.value)
    return refs


def test_setup_inputs_union_covers_ref_only_keys():
    """run_ref must never reference a key setup_inputs didn't create (P2)."""
    a = hg.BenchmarkAnalyzer(ATTN_BENCH_SRC)
    dec = a.get_decorated_functions()
    test_func = a.get_test_function(dec)
    ref_func, kernel_func = a.classify_functions(dec)

    setup_body = hg._generate_setup_inputs(
        a, test_func, "M, N, dtype = cfg", ref_func, kernel_func)
    ref_body = hg._generate_run_ref(a, test_func, ref_func, kernel_func)

    setup_keys = _setup_keys(setup_body)
    ref_refs = _referenced_keys(ref_body)

    # ref-only params are present in setup_inputs (union of kernel + ref).
    for k in ("k_scale_cache", "v_scale_cache", "num_queries_per_kv", "dtype"):
        assert k in setup_keys, f"{k} missing from setup_inputs (would KeyError)"
    # every key run_ref touches exists -> no smoke-test KeyError.
    assert ref_refs <= setup_keys, f"run_ref refs not in setup: {ref_refs - setup_keys}"


def test_setup_inputs_infers_correct_types():
    """Index/scalar/dtype args must not be built as 2D float tensors (P1)."""
    a = hg.BenchmarkAnalyzer(ATTN_BENCH_SRC)
    dec = a.get_decorated_functions()
    test_func = a.get_test_function(dec)
    ref_func, kernel_func = a.classify_functions(dec)

    body = hg._generate_setup_inputs(
        a, test_func, "M, N, dtype = cfg", ref_func, kernel_func)

    # index tensors are int, not randn float
    assert 'block_tables = torch.zeros(M, N, dtype=torch.int32' in body
    assert 'seq_lens = torch.zeros(M, N, dtype=torch.int32' in body
    # int scalars
    assert "max_seq_len = 1" in body
    assert "num_kv_heads = 1" in body
    assert "num_queries_per_kv = 1" in body
    # float scalar
    assert "scale = 1.0" in body
    # dtype carried from cfg
    assert "kv_cache_dtype = dtype" in body
    # block_tables/seq_lens must NOT be randn float
    assert "block_tables = torch.randn" not in body
    assert "seq_lens = torch.randn" not in body


def test_run_func_body_uses_get_not_subscript():
    """run_* bodies must use inputs.get to avoid KeyError (P2)."""
    a = hg.BenchmarkAnalyzer(ATTN_BENCH_SRC)
    dec = a.get_decorated_functions()
    test_func = a.get_test_function(dec)
    ref_func, kernel_func = a.classify_functions(dec)
    body = hg._generate_run_func_body(a, test_func, ref_func)
    assert 'inputs.get(' in body
    assert 'inputs["' not in body
