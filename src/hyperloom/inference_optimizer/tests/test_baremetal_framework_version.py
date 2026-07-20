# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavior lock for version-aware SGLang/vLLM/AITER installs in
install_baremetal.sh, plus CLI version flags and the version-suffixed isolated
vLLM venv path.

Regression target: the installer used to skip framework installs whenever the
package was importable, ignoring the requested version, silently dropping
upgrades. These tests drive the real shell functions/flags with stubs to assert
the version-matching skip/reinstall decision and flag wiring.
"""

from __future__ import annotations

import subprocess

from pathlib import Path

from hyperloom.inference_optimizer import setup

_INSTALL_SH = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"


def _slice(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def _version_helpers() -> str:
    text = _INSTALL_SH.read_text(encoding="utf-8")
    return _slice(text, "normalize_version() {", "\ntorch_required_triton_version() {")


def _run(runner: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(runner)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _write_runner(tmp_path: Path, name: str, body: list[str]) -> Path:
    runner = tmp_path / name
    runner.write_text(
        "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", *body]) + "\n",
        encoding="utf-8",
    )
    return runner


# --- version helper unit tests -------------------------------------------------


def test_normalize_version_strips_v_prefix_and_suffixes(tmp_path: Path):
    runner = _write_runner(
        tmp_path,
        "norm.sh",
        [
            _version_helpers(),
            'echo "$(normalize_version v0.5.12)"',
            'echo "$(normalize_version 0.5.12)"',
            'echo "$(normalize_version 0.5.12.post1)"',
            'echo "$(normalize_version 0.5.12+git8514f05)"',
            'echo "$(normalize_version 7.2.0)"',
        ],
    )
    out = _run(runner).stdout.splitlines()
    assert out == ["0.5.12", "0.5.12", "0.5.12", "0.5.12", "7.2.0"]


def test_version_matches_is_loose_and_rejects_empty(tmp_path: Path):
    runner = _write_runner(
        tmp_path,
        "match.sh",
        [
            _version_helpers(),
            'version_matches v0.5.12 0.5.12 && echo v_vs_plain_MATCH || echo v_vs_plain_NO',
            'version_matches 0.5.12.post1 0.5.12 && echo suffix_MATCH || echo suffix_NO',
            'version_matches 0.5.12 0.5.13 && echo diff_MATCH || echo diff_NO',
            'version_matches "" 0.5.12 && echo empty_MATCH || echo empty_NO',
        ],
    )
    out = _run(runner).stdout.splitlines()
    assert out == ["v_vs_plain_MATCH", "suffix_MATCH", "diff_NO", "empty_NO"]


def test_sglang_target_version_empty_for_wheel_path(tmp_path: Path):
    # The 3.10 wheel path has no comparable sglang release target (7.2.0 is the
    # AMD ROCm PyPI index version, not the sglang package version), so it must
    # return empty; the source path returns SGLANG_REF.
    runner = _write_runner(
        tmp_path,
        "target.sh",
        [
            _version_helpers(),
            "SGLANG_ROCM_PYPI_VERSION=7.2.0",
            "SGLANG_REF=v0.5.12",
            'echo "wheel=[$(sglang_target_version 3.10)]"',
            'echo "source=[$(sglang_target_version 3.12)]"',
        ],
    )
    out = _run(runner).stdout.splitlines()
    assert out == ["wheel=[]", "source=[v0.5.12]"]


# --- sglang install decision (source path, python 3.12) ------------------------


def _sglang_framework_fn() -> str:
    text = _INSTALL_SH.read_text(encoding="utf-8")
    return _slice(
        text,
        "install_sglang_framework() {",
        "\n# Verify that the installed vLLM package resolves to a ROCm runtime.",
    )


def _sglang_runner_body(tmp_path: Path, installed_version: str, target_ref: str) -> list[str]:
    # A stub interpreter: `python -c "import ..."` always succeeds so the
    # end-of-function import guards pass without a real sglang.
    stub_py = tmp_path / "py-stub.sh"
    stub_py.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    stub_py.chmod(0o755)
    marker = tmp_path / "calls.txt"
    return [
        _version_helpers(),
        _sglang_framework_fn(),
        "CHECK_ONLY=0",
        "DRY_RUN=0",
        "SGLANG_ROCM_EXTRA=rocm720",
        "SGLANG_ROCM_PYPI_VERSION=7.2.0",
        f"SGLANG_REF={target_ref}",
        "AITER_REF=",
        "_AITER_REF_WAS_SET=",
        "AITER_REPO=https://example.invalid/aiter.git",
        "SGLANG_REPO=https://example.invalid/sglang.git",
        f"MARKER={marker}",
        "log() { :; }",
        "warn() { :; }",
        'die() { echo "DIE:$*" >&2; exit 42; }',
        f'resolve_python() {{ echo "{stub_py}"; }}',
        f'framework_deps_root() {{ echo "{tmp_path}/deps"; }}',
        "_py_has() { return 0; }",
        f'installed_dist_version() {{ echo "{installed_version}"; }}',
        'install_sglang_from_wheel() { echo wheel >> "$MARKER"; }',
        'install_sglang_from_source() { echo source >> "$MARKER"; }',
        'install_compatible_aiter() { :; }',
        "install_sglang_framework",
        'echo "MARKER_CONTENT:$(cat "$MARKER" 2>/dev/null || echo none)"',
    ]


def test_sglang_reinstalls_when_installed_version_differs(tmp_path: Path):
    body = _sglang_runner_body(tmp_path, installed_version="0.5.10", target_ref="v0.5.12")
    runner = _write_runner(tmp_path, "sglang-diff.sh", body)
    out = _run(runner).stdout
    assert "MARKER_CONTENT:source" in out


def test_sglang_skips_when_installed_version_matches(tmp_path: Path):
    body = _sglang_runner_body(tmp_path, installed_version="0.5.12", target_ref="v0.5.12")
    runner = _write_runner(tmp_path, "sglang-same.sh", body)
    out = _run(runner).stdout
    assert "MARKER_CONTENT:none" in out


# --- sglang wheel path (python 3.10): import-only, no version reinstall --------


def _sglang_wheel_runner_body(tmp_path: Path, sglang_importable: bool) -> list[str]:
    # Stub interpreter: heredoc (`python - <<PY`) reports py_mm=3.10 to force the
    # wheel path; any other invocation exits 0.
    stub_py = tmp_path / "py-wheel-stub.sh"
    stub_py.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "${1:-}" = "-" ]; then echo 3.10; exit 0; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    stub_py.chmod(0o755)
    marker = tmp_path / "wheel-calls.txt"
    # _py_has: sglang presence is parameterized; sgl_kernel/aiter present.
    py_has = (
        '_py_has() { case "$2" in sglang) return %d ;; *) return 0 ;; esac; }'
        % (0 if sglang_importable else 1)
    )
    return [
        _version_helpers(),
        _sglang_framework_fn(),
        "CHECK_ONLY=0",
        "DRY_RUN=0",
        "SGLANG_ROCM_EXTRA=rocm720",
        "SGLANG_ROCM_PYPI_VERSION=7.2.0",
        "SGLANG_REF=v0.5.12",
        "AITER_REF=",
        "_AITER_REF_WAS_SET=",
        "AITER_REPO=https://example.invalid/aiter.git",
        "SGLANG_REPO=https://example.invalid/sglang.git",
        f"MARKER={marker}",
        "log() { :; }",
        "warn() { :; }",
        'die() { echo "DIE:$*" >&2; exit 42; }',
        f'resolve_python() {{ echo "{stub_py}"; }}',
        f'framework_deps_root() {{ echo "{tmp_path}/deps"; }}',
        py_has,
        'installed_dist_version() { echo 0.5.12; }',
        'install_sglang_from_wheel() { echo wheel >> "$MARKER"; }',
        'install_sglang_from_source() { echo source >> "$MARKER"; }',
        'install_compatible_aiter() { :; }',
        "install_sglang_framework",
        'echo "MARKER_CONTENT:$(cat "$MARKER" 2>/dev/null || echo none)"',
    ]


def test_sglang_wheel_path_skips_when_importable(tmp_path: Path):
    # Regression: the 3.10 wheel path must NOT reinstall when sglang is already
    # importable (previously it compared 0.5.12 vs 7.2.0 and reinstalled always).
    body = _sglang_wheel_runner_body(tmp_path, sglang_importable=True)
    runner = _write_runner(tmp_path, "wheel-import.sh", body)
    out = _run(runner).stdout
    assert "MARKER_CONTENT:none" in out


def test_sglang_wheel_path_installs_when_missing(tmp_path: Path):
    body = _sglang_wheel_runner_body(tmp_path, sglang_importable=False)
    runner = _write_runner(tmp_path, "wheel-missing.sh", body)
    out = _run(runner).stdout
    assert "MARKER_CONTENT:wheel" in out


# --- vllm install decision (shared env, python 3.12) ---------------------------


def _vllm_framework_fn() -> str:
    text = _INSTALL_SH.read_text(encoding="utf-8")
    return _slice(
        text,
        "install_vllm_framework() {",
        "\n# Dispatch the optional bare-metal framework installer.",
    )


def _vllm_runner_body(tmp_path: Path, installed_version: str, target_version: str) -> list[str]:
    # Stub interpreter: heredoc (`python - <<PY`) prints 3.12 for py_mm; any
    # other invocation exits 0 so import/version guards pass.
    marker = tmp_path / "vllm-calls.txt"
    stub_py = tmp_path / "py-vllm-stub.sh"
    stub_py.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "${1:-}" = "-" ]; then echo 3.12; exit 0; fi\n'
        'for a in "$@"; do if [ "$a" = "install" ]; then echo pip >> "'
        + str(marker)
        + '"; fi; done\n'
        "exit 0\n",
        encoding="utf-8",
    )
    stub_py.chmod(0o755)
    return [
        _version_helpers(),
        _vllm_framework_fn(),
        "CHECK_ONLY=0",
        "DRY_RUN=0",
        "FRAMEWORK_ENV=shared",
        f"VLLM_VERSION={target_version}",
        "VLLM_ROCM_VARIANT=rocm722",
        "VLLM_ROCM_INDEX=https://example.invalid/wheels",
        "VLLM_VENV_ROOT=/tmp/does-not-exist-venv",
        f"MARKER={marker}",
        "log() { :; }",
        "warn() { :; }",
        'die() { echo "DIE:$*" >&2; exit 42; }',
        f'resolve_python() {{ echo "{stub_py}"; }}',
        "_py_has() { return 0; }",
        f'installed_dist_version() {{ echo "{installed_version}"; }}',
        "verify_vllm_rocm() { return 0; }",
        "write_rocm_torch_constraints() { : > \"$2\"; }",
        "link_vllm_into_shared_bin() { :; }",
        "install_vllm_framework",
        'echo "MARKER_CONTENT:$(cat "$MARKER" 2>/dev/null || echo none)"',
    ]


def test_vllm_reinstalls_when_installed_version_differs(tmp_path: Path):
    body = _vllm_runner_body(tmp_path, installed_version="0.21.0", target_version="0.22.0")
    runner = _write_runner(tmp_path, "vllm-diff.sh", body)
    out = _run(runner).stdout
    assert "MARKER_CONTENT:pip" in out


def test_vllm_skips_when_installed_version_matches(tmp_path: Path):
    body = _vllm_runner_body(tmp_path, installed_version="0.22.0", target_version="0.22.0")
    runner = _write_runner(tmp_path, "vllm-same.sh", body)
    out = _run(runner).stdout
    assert "MARKER_CONTENT:none" in out


# --- C: CLI version flags (end-to-end dry-run) ---------------------------------


def _dry_run(tmp_path: Path, args: list[str]) -> str:
    proc = subprocess.run(
        ["bash", str(_INSTALL_SH), "--dry-run", "--skip-base-check", *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={"REPO_ROOT": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )
    return proc.stdout


def test_vllm_version_flags_wire_through_and_recompute_index(tmp_path: Path):
    out = _dry_run(
        tmp_path,
        ["--install-framework", "vllm", "--vllm-version", "0.21.0", "--vllm-rocm-variant", "rocm720"],
    )
    assert "VLLM_VERSION=0.21.0" in out
    assert "VLLM_ROCM_VARIANT=rocm720" in out
    assert "VLLM_ROCM_INDEX=https://wheels.vllm.ai/rocm/0.21.0/rocm720" in out
    assert "vllm==0.21.0+rocm720" in out


def test_sglang_ref_flag_wires_through(tmp_path: Path):
    out = _dry_run(tmp_path, ["--install-framework", "sglang", "--sglang-ref", "v0.5.10"])
    assert "sglang.git@v0.5.10" in out


# --- D: version-suffixed default venv path -------------------------------------


def test_default_vllm_venv_path_gets_python_suffix(tmp_path: Path):
    out = _dry_run(tmp_path, ["--install-framework", "vllm"])
    assert "VLLM_VENV_ROOT=/opt/hyperloom/vllm-venv-py312" in out


def test_explicit_vllm_venv_root_is_not_suffixed(tmp_path: Path):
    out = _dry_run(
        tmp_path,
        ["--install-framework", "vllm", "--vllm-venv-root", "/custom/vllm"],
    )
    assert "VLLM_VENV_ROOT=/custom/vllm" in out
    assert "/custom/vllm-py312" not in out


# --- A: AITER version-aware skip -----------------------------------------------


def _aiter_runner_body(
    tmp_path: Path, installed_aiter: str, aiter_ref: str, ref_was_set: str
) -> list[str]:
    stub_py = tmp_path / "py-aiter-stub.sh"
    stub_py.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    stub_py.chmod(0o755)
    marker = tmp_path / "aiter-calls.txt"
    return [
        _version_helpers(),
        _sglang_framework_fn(),
        "CHECK_ONLY=0",
        "DRY_RUN=0",
        "SGLANG_ROCM_EXTRA=rocm720",
        "SGLANG_ROCM_PYPI_VERSION=7.2.0",
        "SGLANG_REF=v0.5.12",
        f"AITER_REF={aiter_ref}",
        f"_AITER_REF_WAS_SET={ref_was_set}",
        "AITER_REPO=https://example.invalid/aiter.git",
        "SGLANG_REPO=https://example.invalid/sglang.git",
        f"MARKER={marker}",
        "log() { :; }",
        "warn() { :; }",
        'die() { echo "DIE:$*" >&2; exit 42; }',
        f'resolve_python() {{ echo "{stub_py}"; }}',
        f'framework_deps_root() {{ echo "{tmp_path}/deps"; }}',
        "_py_has() { return 0; }",
        # sglang is up to date so only the AITER branch decides install.
        f'installed_dist_version() {{ if [ "$2" = aiter ]; then echo "{installed_aiter}"; else echo 0.5.12; fi; }}',
        'install_sglang_from_wheel() { :; }',
        'install_sglang_from_source() { :; }',
        f'install_compatible_aiter() {{ echo aiter >> "{marker}"; }}',
        "install_sglang_framework",
        'echo "MARKER_CONTENT:$(cat "$MARKER" 2>/dev/null || echo none)"',
    ]


def test_aiter_reinstalls_when_ref_pinned_and_version_differs(tmp_path: Path):
    body = _aiter_runner_body(tmp_path, installed_aiter="0.1.3", aiter_ref="v0.1.4", ref_was_set="x")
    runner = _write_runner(tmp_path, "aiter-diff.sh", body)
    out = _run(runner).stdout
    assert "MARKER_CONTENT:aiter" in out


def test_aiter_skips_when_ref_pinned_and_version_matches(tmp_path: Path):
    body = _aiter_runner_body(tmp_path, installed_aiter="0.1.4", aiter_ref="v0.1.4", ref_was_set="x")
    runner = _write_runner(tmp_path, "aiter-same.sh", body)
    out = _run(runner).stdout
    assert "MARKER_CONTENT:none" in out


def test_aiter_skips_when_ref_not_pinned_even_if_importable(tmp_path: Path):
    # Default auto-select: no explicit target, importable aiter must not trigger
    # a reinstall.
    body = _aiter_runner_body(tmp_path, installed_aiter="0.1.3", aiter_ref="", ref_was_set="")
    runner = _write_runner(tmp_path, "aiter-auto.sh", body)
    out = _run(runner).stdout
    assert "MARKER_CONTENT:none" in out
