# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ``multi_node/scripts/kernel_node_ops.py``.

The Ray-free, pod-side kernel ops runner for the Dynamo backend. Stdlib-only,
imported via importlib. These guard the safety-critical behaviours: py_compile
auto-revert on a bad patch, the bench staging path-traversal guard, and the
status -> returncode contract the sandbox-side callers depend on.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load(unique_name: str):
    path = _repo_root() / "multi_node" / "scripts" / "kernel_node_ops.py"
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _last_json(capsys) -> dict:
    # Each op emits exactly one pretty-printed JSON document to stdout.
    return json.loads(capsys.readouterr().out.strip())


def test_safe_name_sanitizes_truncates_and_falls_back():
    k = _load("kno_safe")
    assert k._safe_name("a/b c:d") == "a_b_c_d"
    assert k._safe_name("ok.name-1_2") == "ok.name-1_2"
    assert k._safe_name("") == "patch"
    assert len(k._safe_name("x" * 200)) == 80


def test_apply_non_py_target_skips_compile(tmp_path, capsys):
    k = _load("kno_apply_nonpy")
    target = tmp_path / "kernel.cpp"
    target.write_text("// old", encoding="utf-8")
    ns = argparse.Namespace(
        target_path=str(target), backup_dir=str(tmp_path / "bak"),
        kernel_id="k1", patch_b64=_b64("// new content"),
    )
    rc = k._do_apply(ns)
    payload = _last_json(capsys)
    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["compile"]["status"] == "skipped"
    assert target.read_text(encoding="utf-8") == "// new content"


def test_apply_valid_py_compiles_ok(tmp_path, capsys):
    k = _load("kno_apply_py_ok")
    target = tmp_path / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")
    ns = argparse.Namespace(
        target_path=str(target), backup_dir=str(tmp_path / "bak"),
        kernel_id="k", patch_b64=_b64("y = 2\n"),
    )
    rc = k._do_apply(ns)
    payload = _last_json(capsys)
    assert rc == 0 and payload["status"] == "ok"
    assert payload["compile"]["status"] == "ok"
    assert target.read_text(encoding="utf-8") == "y = 2\n"


def test_apply_bad_py_auto_reverts(tmp_path, capsys):
    # SAFETY: a syntactically invalid .py patch must be auto-reverted so the pod
    # is never left with an unimportable kernel file.
    k = _load("kno_apply_py_bad")
    target = tmp_path / "mod.py"
    target.write_text("good = 1\n", encoding="utf-8")
    ns = argparse.Namespace(
        target_path=str(target), backup_dir=str(tmp_path / "bak"),
        kernel_id="k", patch_b64=_b64("def broken(:\n"),
    )
    rc = k._do_apply(ns)
    payload = _last_json(capsys)
    assert rc == 1
    assert payload["status"] == "failed"
    assert "auto-reverted" in payload["error"]
    # Original content restored.
    assert target.read_text(encoding="utf-8") == "good = 1\n"


def test_apply_missing_target_fails(tmp_path, capsys):
    k = _load("kno_apply_missing")
    ns = argparse.Namespace(
        target_path=str(tmp_path / "nope.py"),
        backup_dir=str(tmp_path / "bak"), kernel_id="k", patch_b64=_b64("x=1"),
    )
    rc = k._do_apply(ns)
    payload = _last_json(capsys)
    assert rc == 1 and payload["status"] == "failed"
    assert "does not exist" in payload["error"]


def test_apply_bad_base64_fails(tmp_path, capsys):
    k = _load("kno_apply_b64")
    target = tmp_path / "f.txt"
    target.write_text("orig", encoding="utf-8")
    ns = argparse.Namespace(
        target_path=str(target), backup_dir=str(tmp_path / "bak"),
        kernel_id="k", patch_b64="!!!not-base64!!!",
    )
    rc = k._do_apply(ns)
    payload = _last_json(capsys)
    assert rc == 1 and payload["status"] == "failed"
    assert "base64" in payload["error"]


def test_revert_missing_backup_is_noop(tmp_path, capsys):
    k = _load("kno_revert_noop")
    ns = argparse.Namespace(
        target_path=str(tmp_path / "t.py"),
        backup_path=str(tmp_path / "nope.bak"),
    )
    rc = k._do_revert(ns)
    payload = _last_json(capsys)
    assert rc == 0
    assert payload["status"] == "noop_missing_backup"


def test_revert_restores_from_backup(tmp_path, capsys):
    k = _load("kno_revert_ok")
    backup = tmp_path / "b.bak"
    backup.write_text("restored", encoding="utf-8")
    target = tmp_path / "t.py"
    target.write_text("current", encoding="utf-8")
    ns = argparse.Namespace(target_path=str(target), backup_path=str(backup))
    rc = k._do_revert(ns)
    payload = _last_json(capsys)
    assert rc == 0 and payload["status"] == "restored"
    assert target.read_text(encoding="utf-8") == "restored"


def test_bench_rejects_absolute_and_parent_staging_paths(tmp_path, capsys):
    # SECURITY: staged files must be workspace-relative with no '..' so a patch
    # payload cannot write outside the bench workspace.
    k = _load("kno_bench_traversal")
    ns = argparse.Namespace(
        workspace=str(tmp_path / "ws"), bench_command="true",
        files_b64_json=json.dumps({"../escape.txt": _b64("x")}),
        result_glob="*.json", timeout_sec=10,
    )
    rc = k._do_bench(ns)
    payload = _last_json(capsys)
    assert rc == 1 and payload["status"] == "failed"
    assert ".." in payload["error"] or "relative" in payload["error"]


def test_bench_stages_files_runs_and_collects_artifacts(tmp_path, capsys):
    k = _load("kno_bench_ok")
    ns = argparse.Namespace(
        workspace=str(tmp_path / "ws"),
        bench_command="cat kernel.txt > /dev/null; echo '{\"tput\": 42}' > result.json",
        files_b64_json=json.dumps({"kernel.txt": _b64("kernel body")}),
        result_glob="*.json", timeout_sec=30,
    )
    rc = k._do_bench(ns)
    payload = _last_json(capsys)
    assert rc == 0 and payload["status"] == "ok"
    assert payload["returncode"] == 0
    assert any(a["path"].endswith("result.json") for a in payload["artifacts"])
    art = next(a for a in payload["artifacts"] if a["path"].endswith("result.json"))
    assert art["content"] == {"tput": 42}  # JSON artifacts parsed inline
    # The staged kernel file landed inside the workspace.
    assert (tmp_path / "ws" / "kernel.txt").read_text(encoding="utf-8") == "kernel body"


def test_bench_invalid_files_json_fails(tmp_path, capsys):
    k = _load("kno_bench_badjson")
    ns = argparse.Namespace(
        workspace=str(tmp_path / "ws"), bench_command="true",
        files_b64_json="{not json", result_glob="*.json", timeout_sec=10,
    )
    rc = k._do_bench(ns)
    payload = _last_json(capsys)
    assert rc == 1 and payload["status"] == "failed"
    assert "JSON" in payload["error"]


def test_emit_status_to_returncode_contract():
    k = _load("kno_emit")
    # ok/restored/noop_missing_backup -> 0; everything else -> 1.
    for ok_status in ("ok", "restored", "noop_missing_backup"):
        assert k._emit({"status": ok_status}) == 0
    for bad_status in ("failed", "error", ""):
        assert k._emit({"status": bad_status}) == 1
