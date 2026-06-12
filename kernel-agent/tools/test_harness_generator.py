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
        "harness_generator_under_test", _TOOLS_DIR / "harness_generator.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec: dataclasses with PEP 563 annotations resolve their
    # module via sys.modules during class creation.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


hg = _load_module()


BENCH_SRC = '''\
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
'''


# ---- BenchmarkAnalyzer ----

def test_get_imports():
    a = hg.BenchmarkAnalyzer(BENCH_SRC)
    imports = a.get_imports()
    assert any("import torch" in i for i in imports)


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
    src = (
        "@perftest\ndef alpha(a):\n    return a\n"
        "@perftest\ndef beta(a):\n    return a\n"
    )
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
    src = (
        "@perftest\ndef mykernel(a):\n    return a\n"
        "def bench_runner():\n    return mykernel(1)\n"
    )
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

@pytest.mark.parametrize("name,expected", [
    ("eps", True), ("epsilon", True), ("dropout", True),
    ("use_model_sensitive", True), ("x", False),
])
def test_is_scalar_param(name, expected):
    assert hg._is_scalar_param(name) is expected


@pytest.mark.parametrize("name,expected", [
    ("weight", True), ("bias", True), ("gamma", True),
    ("up_weight", True), ("x", False),
])
def test_is_weight_param(name, expected):
    assert hg._is_weight_param(name) is expected


@pytest.mark.parametrize("s,expected", [
    ("foo", True), ("x1", True), ("True", False),
    ("None", False), ("1e-6", False), ("torch.randn(3)", False),
])
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
        a, dec["test_main"], "M, N, dtype = cfg", dec["torch_ref"], dec["triton_kernel"],
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
        benchmark_file=str(tmp_path / "nope.py"), candidate={}, source_file="",
        out_dir=tmp_path,
    )
    assert out is None


def test_maybe_generate_harness_no_decorated(tmp_path, monkeypatch):
    _inject_fake_validate(monkeypatch, harness_ok=True, bench_ok=False)
    bench = tmp_path / "b.py"
    bench.write_text("import torch\nx = 1\n", encoding="utf-8")
    out = hg.maybe_generate_harness(
        benchmark_file=str(bench), candidate={}, source_file="", out_dir=tmp_path,
    )
    assert out is None


def test_maybe_generate_harness_no_kernel_or_ref(tmp_path, monkeypatch):
    _inject_fake_validate(monkeypatch, harness_ok=True, bench_ok=False)
    bench = tmp_path / "b.py"
    bench.write_text("@benchmark\ndef test_x():\n    pass\n", encoding="utf-8")
    out = hg.maybe_generate_harness(
        benchmark_file=str(bench), candidate={}, source_file="", out_dir=tmp_path,
    )
    assert out is None


def test_maybe_generate_harness_success(tmp_path, monkeypatch):
    _inject_fake_validate(monkeypatch, harness_ok=True, bench_ok=False)
    logs = []
    bench = tmp_path / "bench_rms.py"
    bench.write_text(BENCH_SRC, encoding="utf-8")
    out = hg.maybe_generate_harness(
        benchmark_file=str(bench), candidate={"input_shapes": [{"call_num": 1, "shape": "(256, 128) bf16"}]},
        source_file="", out_dir=tmp_path, log_fn=logs.append,
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
        benchmark_file=str(bench), candidate={}, source_file="", out_dir=tmp_path,
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
