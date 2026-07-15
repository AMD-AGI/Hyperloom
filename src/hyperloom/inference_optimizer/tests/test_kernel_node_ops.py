# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ``multi_node/scripts/kernel_node_ops.py``.

The Ray-free, pod-side kernel ops runner for the Infera backend. Stdlib-only,
imported via importlib. These guard the safety-critical behaviours: py_compile
auto-revert on a bad patch, the bench staging path-traversal guard, and the
status -> returncode contract the sandbox-side callers depend on.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import types
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def patch_env(tmp_path, monkeypatch):
    fw = tmp_path / "fw"
    fw.mkdir()
    bak = tmp_path / "bak"
    bak.mkdir()
    monkeypatch.setenv("INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", f"{fw}/")
    monkeypatch.setenv("HYPERLOOM_MN_KERNEL_BACKUP_DIR", str(bak))
    return fw, bak


def _strip_pod_script_header(body: str) -> str:
    lines = body.splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    lines = [ln for ln in lines if ln.strip() != "from __future__ import annotations"]
    return "\n".join(lines).strip()


def _bundle_kernel_node_ops() -> str:
    root = _repo_root() / "multi_node" / "scripts"
    chunks = [_strip_pod_script_header((root / dep).read_text(encoding="utf-8")) for dep in ("patch_path_safety.py",)]
    main_body = _strip_pod_script_header((root / "kernel_node_ops.py").read_text(encoding="utf-8"))
    return "from __future__ import annotations\n\n" + "\n\n".join(chunks) + "\n\n" + main_body + "\n"


def _load(unique_name: str):
    mod = types.ModuleType(unique_name)
    mod.__dict__["__file__"] = str(_repo_root() / "multi_node" / "scripts" / "kernel_node_ops.py")
    exec(compile(_bundle_kernel_node_ops(), "kernel_node_ops_bundle.py", "exec"), mod.__dict__)
    sys.modules[unique_name] = mod
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


def test_apply_non_py_target_skips_compile(patch_env, capsys):
    fw, bak = patch_env
    k = _load("kno_apply_nonpy")
    target = fw / "kernel.cpp"
    target.write_text("// old", encoding="utf-8")
    ns = argparse.Namespace(
        target_path=str(target),
        backup_dir=str(bak),
        kernel_id="k1",
        patch_b64=_b64("// new content"),
    )
    rc = k._do_apply(ns)
    payload = _last_json(capsys)
    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["compile"]["status"] == "skipped"
    assert target.read_text(encoding="utf-8") == "// new content"


def test_apply_valid_py_compiles_ok(patch_env, capsys):
    fw, bak = patch_env
    k = _load("kno_apply_py_ok")
    target = fw / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")
    ns = argparse.Namespace(
        target_path=str(target),
        backup_dir=str(bak),
        kernel_id="k",
        patch_b64=_b64("y = 2\n"),
    )
    rc = k._do_apply(ns)
    payload = _last_json(capsys)
    assert rc == 0 and payload["status"] == "ok"
    assert payload["compile"]["status"] == "ok"
    assert target.read_text(encoding="utf-8") == "y = 2\n"


def test_apply_bad_py_auto_reverts(patch_env, capsys):
    # SAFETY: a syntactically invalid .py patch must be auto-reverted so the pod
    # is never left with an unimportable kernel file.
    fw, bak = patch_env
    k = _load("kno_apply_py_bad")
    target = fw / "mod.py"
    target.write_text("good = 1\n", encoding="utf-8")
    ns = argparse.Namespace(
        target_path=str(target),
        backup_dir=str(bak),
        kernel_id="k",
        patch_b64=_b64("def broken(:\n"),
    )
    rc = k._do_apply(ns)
    payload = _last_json(capsys)
    assert rc == 1
    assert payload["status"] == "failed"
    assert "auto-reverted" in payload["error"]
    # Original content restored.
    assert target.read_text(encoding="utf-8") == "good = 1\n"


def test_apply_missing_target_fails(patch_env, capsys):
    fw, bak = patch_env
    k = _load("kno_apply_missing")
    ns = argparse.Namespace(
        target_path=str(fw / "nope.py"),
        backup_dir=str(bak),
        kernel_id="k",
        patch_b64=_b64("x=1"),
    )
    rc = k._do_apply(ns)
    payload = _last_json(capsys)
    assert rc == 1 and payload["status"] == "failed"
    assert "does not exist" in payload["error"]


def test_apply_bad_base64_fails(patch_env, capsys):
    fw, bak = patch_env
    k = _load("kno_apply_b64")
    target = fw / "f.txt"
    target.write_text("orig", encoding="utf-8")
    ns = argparse.Namespace(
        target_path=str(target),
        backup_dir=str(bak),
        kernel_id="k",
        patch_b64="!!!not-base64!!!",
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


def test_revert_restores_from_backup(patch_env, capsys):
    fw, bak = patch_env
    k = _load("kno_revert_ok")
    backup = bak / "b.bak"
    backup.write_text("restored", encoding="utf-8")
    target = fw / "t.py"
    target.write_text("current", encoding="utf-8")
    ns = argparse.Namespace(target_path=str(target), backup_path=str(backup))
    rc = k._do_revert(ns)
    payload = _last_json(capsys)
    assert rc == 0 and payload["status"] == "restored"
    assert target.read_text(encoding="utf-8") == "restored"


def test_revert_rejects_backup_outside_root(patch_env, capsys):
    fw, bak = patch_env
    k = _load("kno_revert_bad_backup")
    target = fw / "t.py"
    target.write_text("current", encoding="utf-8")
    outside = bak.parent / "evil.bak"
    outside.write_text("pwned", encoding="utf-8")
    ns = argparse.Namespace(target_path=str(target), backup_path=str(outside))
    rc = k._do_revert(ns)
    payload = _last_json(capsys)
    assert rc == 1 and payload["status"] == "failed"
    assert "backup_path" in payload["error"]


def test_revert_rejects_target_outside_framework(patch_env, capsys):
    fw, bak = patch_env
    k = _load("kno_revert_bad_target")
    outside = fw.parent / "escape.py"
    outside.write_text("x", encoding="utf-8")
    backup = bak / "b.bak"
    backup.write_text("restored", encoding="utf-8")
    ns = argparse.Namespace(target_path=str(outside), backup_path=str(backup))
    rc = k._do_revert(ns)
    payload = _last_json(capsys)
    assert rc == 1 and payload["status"] == "failed"
    assert "target_path" in payload["error"]


def test_apply_rejects_backup_dir_outside_root(patch_env, capsys):
    fw, bak = patch_env
    k = _load("kno_apply_bad_bdir")
    target = fw / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")
    ns = argparse.Namespace(
        target_path=str(target),
        backup_dir=str(bak.parent / "escape"),
        kernel_id="k",
        patch_b64=_b64("y = 2\n"),
    )
    rc = k._do_apply(ns)
    payload = _last_json(capsys)
    assert rc == 1 and payload["status"] == "failed"
    assert "backup_dir" in payload["error"]


def test_bench_rejects_absolute_and_parent_staging_paths(tmp_path, capsys):
    # SECURITY: staged files must be workspace-relative with no '..' so a patch
    # payload cannot write outside the bench workspace.
    k = _load("kno_bench_traversal")
    ns = argparse.Namespace(
        workspace=str(tmp_path / "ws"),
        bench_command="true",
        files_b64_json=json.dumps({"../escape.txt": _b64("x")}),
        result_glob="*.json",
        timeout_sec=10,
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
        result_glob="*.json",
        timeout_sec=30,
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
        workspace=str(tmp_path / "ws"),
        bench_command="true",
        files_b64_json="{not json",
        result_glob="*.json",
        timeout_sec=10,
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
