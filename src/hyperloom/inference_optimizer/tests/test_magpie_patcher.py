# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the redundant ``--concurrent-requests`` eval-flag strip in
``_magpie_patcher.py`` (the flag-strip regex and per-script apply helper).

SGLang custom-tokenizer trust patching and end-to-end
``magpie_scripts_patch_status`` / ``ensure_eval_concurrency_compat`` /
``ensure_client_trust_compat`` coverage lives in
``test_magpie_patcher_unit.py`` and ``test_preflight_client_trust_compat.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from hyperloom.orchestrator.actions.executors import _magpie_patcher as mp
from hyperloom.orchestrator.actions.executors._magpie_patcher import (
    magpie_scripts_patch_status,
)

_EVAL_SCRIPT_WITH_FLAG = """\
#!/usr/bin/env bash
if [[ "$PHASE" != "server" && "${RUN_EVAL}" = "true" ]]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC || exit $?
    append_lm_eval_summary
fi
"""

_EVAL_SCRIPT_CLEAN = """\
#!/usr/bin/env bash
if [[ "$PHASE" != "server" && "${RUN_EVAL}" = "true" ]]; then
    run_eval --framework lm-eval --port "$PORT" || exit $?
    append_lm_eval_summary
fi
"""


def _write_bench_script(root: Path, name: str, src: str) -> Path:
    script_dir = root / "Magpie" / "scripts" / "benchmark"
    script_dir.mkdir(parents=True, exist_ok=True)
    script = script_dir / name
    script.write_text(src, encoding="utf-8")
    script.chmod(0o755)
    return script


class TestStripEvalConcurrencyFlag:
    def test_strips_bare_conc(self):
        out = mp._strip_eval_concurrency_flag(_EVAL_SCRIPT_WITH_FLAG)
        assert out is not None
        assert "--concurrent-requests" not in out
        assert 'run_eval --framework lm-eval --port "$PORT" || exit $?' in out

    def test_strips_quoted_and_braced(self):
        for frag in ('--concurrent-requests "$CONC"', "--concurrent-requests ${CONC}"):
            src = f'    run_eval --framework lm-eval --port "$PORT" {frag} || exit $?\n'
            out = mp._strip_eval_concurrency_flag(src)
            assert out is not None
            assert "--concurrent-requests" not in out
            assert 'run_eval --framework lm-eval --port "$PORT" || exit $?' in out

    def test_returns_none_without_marker(self):
        assert mp._strip_eval_concurrency_flag(_EVAL_SCRIPT_CLEAN) is None

    def test_returns_none_on_unrecognised_shape(self):
        # Marker present but value isn't $CONC -> regex miss -> None.
        weird = "    run_eval --framework lm-eval --concurrent-requests 64 || exit $?\n"
        assert mp._strip_eval_concurrency_flag(weird) is None


class TestApplyEvalFlagPatch:
    def test_strips_all_generic_scripts(self, tmp_path: Path):
        for name in ("sglang_mi300x.sh", "vllm_mi355x.sh"):
            _write_bench_script(tmp_path, name, _EVAL_SCRIPT_WITH_FLAG)
        scripts_dir = tmp_path / "Magpie" / "scripts" / "benchmark"
        assert mp._apply_eval_flag_patch_atomic(scripts_dir) is True
        for name in ("sglang_mi300x.sh", "vllm_mi355x.sh"):
            text = (scripts_dir / name).read_text(encoding="utf-8")
            assert "--concurrent-requests" not in text

    def test_idempotent(self, tmp_path: Path):
        _write_bench_script(tmp_path, "sglang_mi300x.sh", _EVAL_SCRIPT_WITH_FLAG)
        scripts_dir = tmp_path / "Magpie" / "scripts" / "benchmark"
        assert mp._apply_eval_flag_patch_atomic(scripts_dir) is True
        first = (scripts_dir / "sglang_mi300x.sh").read_text(encoding="utf-8")
        assert mp._apply_eval_flag_patch_atomic(scripts_dir) is True
        assert (scripts_dir / "sglang_mi300x.sh").read_text(encoding="utf-8") == first

    def test_clean_dir_is_noop_true(self, tmp_path: Path):
        _write_bench_script(tmp_path, "sglang_mi300x.sh", _EVAL_SCRIPT_CLEAN)
        scripts_dir = tmp_path / "Magpie" / "scripts" / "benchmark"
        pre = (scripts_dir / "sglang_mi300x.sh").read_text(encoding="utf-8")
        assert mp._apply_eval_flag_patch_atomic(scripts_dir) is True
        assert (scripts_dir / "sglang_mi300x.sh").read_text(encoding="utf-8") == pre

    def test_unrecognised_shape_returns_false(self, tmp_path: Path, caplog):
        _write_bench_script(
            tmp_path,
            "sglang_mi300x.sh",
            "    run_eval --framework lm-eval --concurrent-requests 64 || exit $?\n",
        )
        scripts_dir = tmp_path / "Magpie" / "scripts" / "benchmark"
        with caplog.at_level(logging.WARNING):
            assert mp._apply_eval_flag_patch_atomic(scripts_dir) is False
        assert any("could not be stripped" in r.getMessage() for r in caplog.records)


def test_status_strips_eval_flag_and_reports_ok(tmp_path: Path):
    script = _write_bench_script(tmp_path, "sglang_mi300x.sh", _EVAL_SCRIPT_WITH_FLAG)
    status = magpie_scripts_patch_status(tmp_path)
    assert status.eval_flag_ok is True
    assert "--concurrent-requests" not in script.read_text(encoding="utf-8")


def test_atomic_write_text_preserves_file_mode(tmp_path: Path):
    target = tmp_path / "bench.sh"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o755)
    pre_mode = target.stat().st_mode
    assert mp.atomic_write_text(target, "#!/bin/sh\npatched\n", log_prefix="_magpie_patcher")
    assert target.stat().st_mode == pre_mode


def test_status_eval_flag_false_on_unrecognised_shape(tmp_path: Path):
    _write_bench_script(
        tmp_path,
        "sglang_mi300x.sh",
        "    run_eval --framework lm-eval --concurrent-requests 64 || exit $?\n",
    )
    status = magpie_scripts_patch_status(tmp_path)
    assert status.eval_flag_ok is False
    assert status.ok is False
