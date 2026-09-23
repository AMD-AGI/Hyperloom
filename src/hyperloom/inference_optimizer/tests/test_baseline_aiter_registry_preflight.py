"""A boot must not walk into a kernel the compiled registry never got.

PR #1457 taught the GEMM integrate lane to check serving-.so coverage before it boots,
but left the baseline executor able only to name ``aiter_jit_registry_mismatch`` after
the fact. PRELUDE's first measurement and every FRAMEWORK variant boot through the
baseline executor, so a CSV whose kernels the compiled module never registered kills
those rounds with no recovery.

Which CSVs those are is not a guess: ``aiter/jit/core.py::get_config_file`` either takes
the env var literally or, when it is unset, merges the model overlays on top of the
shipped default.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import _aiter_jit
from hyperloom.orchestrator.actions.executors import baseline as baseline_mod


@pytest.fixture(autouse=True)
def _isolate_serving_package(tmp_path, monkeypatch):
    package = tmp_path / "aiter"
    package.mkdir()
    (package / "__init__.py").write_text("raise AssertionError('AITER must not be imported')\n", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "aiter", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("AITER_JIT_DIR", "INFERENCE_OPTIMIZER_AITER_JIT_DIR", "VLLM_VENV_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_aiter_jit, "AITER_JIT_PROBE_PATHS", ())


def _configs(tmp_path: Path) -> Path:
    configs = tmp_path / "aiter" / "configs"
    (configs / "model_configs").mkdir(parents=True)
    return configs


def test_a_set_env_resolves_to_exactly_what_it_names(tmp_path):
    """Setting the variable turns overlay discovery off; aiter reads the list verbatim."""
    configs = _configs(tmp_path)
    (configs / "a8w8_tuned_gemm.csv").write_text("kernelName\n", encoding="utf-8")
    (configs / "model_configs" / "dsv3_a8w8_tuned_gemm.csv").write_text("kernelName\n", encoding="utf-8")

    assert _aiter_jit.csvs_aiter_will_load(configs, "a8w8_tuned_gemm", "/tuned/mine.csv") == [Path("/tuned/mine.csv")]
    # A ``:``-joined value is a list, not a path -- and Path("a.csv:b.csv").is_file() is
    # False, so treating it as one silently reports the whole set as covered.
    assert _aiter_jit.csvs_aiter_will_load(configs, "a8w8_tuned_gemm", "/a.csv:/b.csv") == [
        Path("/a.csv"),
        Path("/b.csv"),
    ]


def test_an_unset_env_pulls_in_the_model_overlays(tmp_path):
    """The shipped default is prepended and every matching overlay merges on top."""
    configs = _configs(tmp_path)
    shipped = configs / "a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
    shipped.write_text("kernelName\n", encoding="utf-8")
    overlay = configs / "model_configs" / "dsv3_a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
    overlay.write_text("kernelName\n", encoding="utf-8")
    untuned = configs / "model_configs" / "a8w8_blockscale_bpreshuffle_untuned_gemm.csv"
    untuned.write_text("kernelName\n", encoding="utf-8")

    resolved = _aiter_jit.csvs_aiter_will_load(configs, "a8w8_blockscale_bpreshuffle_tuned_gemm", "")

    assert resolved == [shipped, overlay]
    assert untuned not in resolved


def _aiter_tree(tmp_path: Path, *, csv_rows: str, so_contains: bytes, overlay: bool) -> Path:
    """A minimal aiter package: jit/ beside configs/, one bpreshuffle table."""
    jit = tmp_path / "aiter" / "jit"
    jit.mkdir(parents=True)
    (jit / "module_gemm_a8w8_blockscale_bpreshuffle.so").write_bytes(b"\x7fELF" + so_contains)
    configs = _configs(tmp_path)
    header = "M,N,K,kernelName,libtype\n"
    (configs / "a8w8_blockscale_bpreshuffle_tuned_gemm.csv").write_text(header, encoding="utf-8")
    if overlay:
        (configs / "model_configs" / "dsv3_a8w8_blockscale_bpreshuffle_tuned_gemm.csv").write_text(
            header + csv_rows, encoding="utf-8"
        )
    return jit


def test_an_overlay_an_unset_env_merges_is_checked(tmp_path, monkeypatch):
    """fmoe_ck tunes AITER_CONFIG_FMOE, then fails on a bpreshuffle kernel it never tuned.

    Nothing names that table, so a check keyed on what the round tuned never looks at it,
    and the round is lost with no recovery.
    """
    jit = _aiter_tree(
        tmp_path,
        csv_rows="16,512,2048,a8w8_blockscale_bpreshuffle_never_built,ck\n",
        so_contains=b"a8w8_blockscale_bpreshuffle_something_else",
        overlay=True,
    )
    monkeypatch.setenv("INFERENCE_OPTIMIZER_AITER_JIT_DIR", str(jit))

    outcome = _aiter_jit.prepare_serving_so_for_csvs(
        {"AITER_CONFIG_FMOE": "/tuned/fmoe.csv"},
        backup_dir=tmp_path / "backup",
    )

    assert outcome["action"] == "invalidate"
    assert [Path(p).name for p in outcome["removed"]] == ["module_gemm_a8w8_blockscale_bpreshuffle.so"]
    assert not (jit / "module_gemm_a8w8_blockscale_bpreshuffle.so").exists()
    record = outcome["jit_build"]
    assert record["status"] == "ok"
    assert record["module_names"] == ["module_gemm_a8w8_blockscale_bpreshuffle.so"]
    assert (Path(record["modules_backup_path"]) / record["module_names"][0]).read_bytes() == (
        b"\x7fELF" + b"a8w8_blockscale_bpreshuffle_something_else"
    )


@pytest.mark.parametrize("cache_location", ["runtime-env", "home", "wrapper-override"])
@pytest.mark.parametrize("overlay", [False, True], ids=["shipped", "overlay"])
def test_unset_csvs_respect_package_and_wrapper_boundaries(tmp_path, monkeypatch, cache_location, overlay):
    package_jit = _aiter_tree(
        tmp_path,
        csv_rows="16,512,2048,a8w8_blockscale_bpreshuffle_missing,ck\n",
        so_contains=b"a8w8_blockscale_bpreshuffle_old",
        overlay=overlay,
    )
    if not overlay:
        shipped = package_jit.parent / "configs" / "a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
        shipped.write_text("kernelName,libtype\na8w8_blockscale_bpreshuffle_missing,ck\n", encoding="utf-8")
    home = tmp_path / "home"
    jit = home / ".aiter" / "jit" if cache_location == "home" else tmp_path / "runtime" / "jit"
    (jit / "build").mkdir(parents=True)
    (jit / "build" / "stamp").write_bytes(b"runtime build")
    name = "module_gemm_a8w8_blockscale_bpreshuffle.so"
    (jit / name).write_bytes(b"a8w8_blockscale_bpreshuffle_old")
    (jit / "module_attention.so").write_bytes(b"unrelated")
    decoy_configs = jit.parent / "configs"
    decoy_configs.mkdir()
    (decoy_configs / "a8w8_blockscale_bpreshuffle_tuned_gemm.csv").write_text(
        "kernelName,libtype\na8w8_blockscale_bpreshuffle_old,ck\n", encoding="utf-8"
    )
    monkeypatch.delenv("AITER_JIT_DIR", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_AITER_JIT_DIR", raising=False)
    assert list(importlib.util.find_spec("aiter").submodule_search_locations) == [str(package_jit.parent)]
    if cache_location == "home":
        monkeypatch.setattr(Path, "home", lambda: home)
        monkeypatch.setattr(_aiter_jit.os, "access", lambda *_: False)
    else:
        env = "AITER_JIT_DIR" if cache_location == "runtime-env" else "INFERENCE_OPTIMIZER_AITER_JIT_DIR"
        monkeypatch.setenv(env, str(jit))

    outcome = _aiter_jit.prepare_serving_so_for_csvs({}, backup_dir=tmp_path / "backup")

    assert "aiter" not in sys.modules
    if cache_location == "wrapper-override":
        assert outcome == {"action": "skip", "jit_dir": str(jit)}
        assert (jit / name).read_bytes() == b"a8w8_blockscale_bpreshuffle_old"
        assert (jit / "build" / "stamp").read_bytes() == b"runtime build"
        assert (jit / "module_attention.so").read_bytes() == b"unrelated"
        assert (package_jit / name).read_bytes() == b"\x7fELF" + b"a8w8_blockscale_bpreshuffle_old"
        return
    assert outcome["action"] == "invalidate"
    assert outcome["jit_dir"] == str(jit)
    assert outcome["removed"] == [str(jit / name)]
    assert not (jit / name).exists()
    assert not (jit / "build").exists()
    assert (jit / "module_attention.so").read_bytes() == b"unrelated"
    assert (package_jit / name).read_bytes() == b"\x7fELF" + b"a8w8_blockscale_bpreshuffle_old"
    record = outcome["jit_build"]
    assert (Path(record["modules_backup_path"]) / name).read_bytes() == b"a8w8_blockscale_bpreshuffle_old"
    assert (Path(record["backup_path"]) / "stamp").read_bytes() == b"runtime build"


def test_a_consistent_install_costs_nothing(tmp_path, monkeypatch):
    """This runs before every boot, so the common case must not rebuild anything."""
    jit = _aiter_tree(
        tmp_path,
        csv_rows="16,512,2048,a8w8_blockscale_bpreshuffle_built,ck\n",
        so_contains=b"a8w8_blockscale_bpreshuffle_built",
        overlay=True,
    )
    monkeypatch.setenv("INFERENCE_OPTIMIZER_AITER_JIT_DIR", str(jit))

    outcome = _aiter_jit.prepare_serving_so_for_csvs({}, backup_dir=tmp_path / "backup")

    assert outcome["action"] == "skip"
    assert (jit / "module_gemm_a8w8_blockscale_bpreshuffle.so").exists()


def test_a_pinned_env_does_not_rebuild_for_a_table_it_turns_off(tmp_path, monkeypatch):
    """Setting the variable disables overlay discovery, so the overlay is out of scope."""
    jit = _aiter_tree(
        tmp_path,
        csv_rows="16,512,2048,a8w8_blockscale_bpreshuffle_never_built,ck\n",
        so_contains=b"a8w8_blockscale_bpreshuffle_something_else",
        overlay=True,
    )
    monkeypatch.chdir(tmp_path)
    pinned = Path("pinned.csv")
    pinned.write_text("M,N,K,kernelName,libtype\n16,512,2048,a8w8_blockscale_bpreshuffle_something_else,ck\n", "utf-8")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_AITER_JIT_DIR", str(jit))

    outcome = _aiter_jit.prepare_serving_so_for_csvs(
        {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": str(pinned)},
        backup_dir=tmp_path / "backup",
    )

    assert outcome["action"] == "skip"
    assert (jit / "module_gemm_a8w8_blockscale_bpreshuffle.so").exists()


def test_a_shipped_table_this_cannot_decode_does_not_skip_the_check(tmp_path, monkeypatch):
    """The tables ship with aiter; their encoding and field sizes are not ours to assume."""
    jit = _aiter_tree(tmp_path, csv_rows="", so_contains=b"built", overlay=False)
    bad = tmp_path / "aiter" / "configs" / "model_configs" / "dsv3_a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
    bad.write_bytes(b"\xff\xfeM,N,K,kernelName,libtype\n")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_AITER_JIT_DIR", str(jit))

    outcome = _aiter_jit.prepare_serving_so_for_csvs({}, backup_dir=tmp_path / "backup")

    assert outcome["action"] == "skip"


def test_a_tree_without_a_configs_dir_still_checks_what_the_round_named(tmp_path, monkeypatch):
    """Only the unset branch reads configs/; a pinned value is taken verbatim."""
    jit = tmp_path / "jit"
    jit.mkdir()
    (jit / "module_gemm_a8w8_blockscale_bpreshuffle.so").write_bytes(b"\x7fELFbuilt")
    monkeypatch.chdir(tmp_path)
    pinned = Path("pinned.csv")
    pinned.write_text("M,N,K,kernelName,libtype\n16,512,2048,a8w8_blockscale_bpreshuffle_missing,ck\n", "utf-8")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_AITER_JIT_DIR", str(jit))

    outcome = _aiter_jit.prepare_serving_so_for_csvs(
        {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": str(pinned)},
        backup_dir=tmp_path / "backup",
    )

    assert outcome["action"] == "invalidate"
    assert outcome["removed"] == [str(jit / "module_gemm_a8w8_blockscale_bpreshuffle.so")]
    assert not (jit / "module_gemm_a8w8_blockscale_bpreshuffle.so").exists()
    record = outcome["jit_build"]
    assert record["status"] == "ok"
    assert record["module_names"] == ["module_gemm_a8w8_blockscale_bpreshuffle.so"]
    assert (Path(record["modules_backup_path"]) / record["module_names"][0]).read_bytes() == b"\x7fELFbuilt"


@pytest.mark.asyncio
async def test_the_preflight_runs_even_when_the_round_names_no_csv(tmp_path, monkeypatch):
    """No env is exactly the case that needs the unset-branch check, not a reason to skip."""
    seen: list[tuple[dict, Path]] = []

    def _prepare(envs, backup_dir=None):
        seen.append((dict(envs), backup_dir))
        return {"action": "skip"}

    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._aiter_jit.prepare_serving_so_for_csvs",
        _prepare,
    )

    await baseline_mod._prepare_aiter_serving_so({"RUN_EVAL": "false"}, tmp_path)

    assert seen == [({}, tmp_path / "aiter_jit_backup")]


@pytest.mark.asyncio
async def test_only_the_csv_variables_travel(tmp_path, monkeypatch):
    seen: list[dict] = []

    def _prepare(envs, backup_dir=None):
        seen.append(dict(envs))
        return {"action": "skip"}

    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._aiter_jit.prepare_serving_so_for_csvs",
        _prepare,
    )

    await baseline_mod._prepare_aiter_serving_so(
        {
            "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": "/tuned/bpreshuffle.csv",
            "AITER_CONFIG_GEMM_BF16": "  ",
            "RUN_EVAL": "false",
        },
        tmp_path,
    )

    # A blank value is an unset variable, and the rest of the round's environment is not
    # the coverage check's business.
    assert seen == [{"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": "/tuned/bpreshuffle.csv"}]


@pytest.mark.asyncio
async def test_a_preflight_failure_does_not_abort_the_round(tmp_path, monkeypatch):
    """A jit directory the check cannot read is not a reason to lose the measurement."""

    def _prepare(envs, backup_dir=None):
        raise OSError("jit dir is read-only")

    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._aiter_jit.prepare_serving_so_for_csvs",
        _prepare,
    )

    await baseline_mod._prepare_aiter_serving_so({"AITER_CONFIG_GEMM_BF16": "/tuned/bf16.csv"}, tmp_path)


def test_the_preflight_runs_before_the_config_is_materialized():
    """Ordering is the whole point: after the boot it is a classifier, not a fix."""
    import ast

    source = Path(baseline_mod.__file__).read_text(encoding="utf-8")
    run_once = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_run_once"
    )
    called: list[str] = []
    for node in ast.walk(run_once):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        if name in ("_prepare_aiter_serving_so", "materialize_config_with_envs"):
            called.append((node.lineno, name))

    ordered = [name for _lineno, name in sorted(called)]
    assert ordered[:2] == ["_prepare_aiter_serving_so", "materialize_config_with_envs"]
    preflight = source.index("await _prepare_aiter_serving_so(base_extra_envs, output_dir)")
    materialize = source.index("config_path = materialize_config_with_envs(")
    assert preflight < materialize


def test_the_error_names_the_module_when_the_env_cannot(tmp_path):
    """A round's env does not always reach the module at fault.

    fmoe_ck tunes AITER_CONFIG_FMOE, which maps to no serving module, yet it boots
    against every CSV aiter merges -- so the missing kernel can belong to bpreshuffle.
    An env-keyed drop unlinks nothing there and the retry repeats the failure.
    """
    from hyperloom.orchestrator.actions.executors._aiter_jit import registry_mismatch_modules

    observed = (
        "RuntimeError: gemm_a8w8_blockscale_bpreshuffle kernel "
        "'a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1"
        "_16x16x1_1x32x1x8_8_2x1_intrawave_v1' is not present in the compiled registry. "
        "The tuned CSV references a kernel that was not built into aiter."
    )

    assert registry_mismatch_modules(observed) == ("module_gemm_a8w8_blockscale_bpreshuffle",)
    # A cktile kernel carries its libtype in its own name.
    assert registry_mismatch_modules("kernel 'a8w8_blockscale_cktile_x' is not present") == (
        "module_gemm_a8w8_blockscale_cktile",
    )
    # Nothing named, nothing to unlink — the env-keyed drop stays the only behaviour.
    assert registry_mismatch_modules("Capture cuda graph failed: HIP error") == ()


def test_named_modules_are_unlinked_on_top_of_the_env_s_own(tmp_path, monkeypatch):
    """The two sources add up; neither replaces the other."""
    from hyperloom.orchestrator.actions.executors import _aiter_jit

    jit = tmp_path / "jit"
    jit.mkdir()
    for stem in ("module_gemm_a8w8", "module_gemm_a8w8_blockscale_bpreshuffle", "module_attention"):
        (jit / f"{stem}.so").write_bytes(b"\x7fELF")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_AITER_JIT_DIR", str(jit))

    outcome = _aiter_jit.drop_serving_so_for_envs(
        {"AITER_CONFIG_GEMM_A8W8": "/tuned/a8w8.csv"},
        backup_dir=tmp_path / "backup",
        also_modules=("module_gemm_a8w8_blockscale_bpreshuffle",),
    )

    assert not (jit / "module_gemm_a8w8.so").exists()
    assert not (jit / "module_gemm_a8w8_blockscale_bpreshuffle.so").exists()
    assert outcome["action"] == "invalidate"
    assert (jit / "module_attention.so").read_bytes() == b"\x7fELF"
    record = outcome["jit_build"]
    assert record["status"] == "ok"
    assert record["module_scope"] == ["module_gemm_a8w8", "module_gemm_a8w8_blockscale_bpreshuffle"]
    assert record["module_names"] == ["module_gemm_a8w8.so", "module_gemm_a8w8_blockscale_bpreshuffle.so"]
    assert outcome["removed"] == [str(jit / name) for name in record["module_names"]]
    for name in record["module_names"]:
        assert (Path(record["modules_backup_path"]) / name).read_bytes() == b"\x7fELF"


def _fake_aiter_tree(root, *, csv_rows: str, so_contains: bytes) -> object:
    """A minimal aiter package: jit/ beside configs/, one shipped table."""
    jit = root / "aiter" / "jit"
    jit.mkdir(parents=True)
    (jit / "module_gemm_a8w8_blockscale_bpreshuffle.so").write_bytes(b"\x7fELF" + so_contains)
    shipped = root / "aiter" / "configs" / "model_configs"
    shipped.mkdir(parents=True)
    (shipped / "a8w8_blockscale_bpreshuffle_tuned_gemm_dsv3.csv").write_text(
        "M,N,K,kernelName,libtype\n" + csv_rows, encoding="utf-8"
    )
    return jit
