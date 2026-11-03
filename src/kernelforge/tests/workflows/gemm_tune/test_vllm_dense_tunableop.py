# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys

from kernelforge.workflows.gemm_tune.tuners.vllm_dense_tunableop import (
    _candidate_pythonpath,
    _generate_candidate_sitecustomize,
    _generate_tunableop_script,
    count_tunableop_result_lines,
)


def _write_fake_torch(tmp_path, *, with_read_file: bool, read_file_raises: bool = False):
    torch_root = tmp_path / "fake_torch"
    cuda_dir = torch_root / "torch" / "cuda"
    cuda_dir.mkdir(parents=True)
    (torch_root / "torch" / "__init__.py").write_text("from . import cuda\n", encoding="utf-8")
    if with_read_file:
        read_body = (
            "        raise ValueError('corrupt tunableop csv')\n"
            if read_file_raises
            else "        open(os.environ['READ_MARKER'], 'w', encoding='utf-8').write(value)\n"
        )
        cuda_init = (
            "import os\n"
            "class _Tunable:\n"
            "    def enable(self, value): pass\n"
            "    def tuning_enable(self, value): pass\n"
            "    def record_untuned_enable(self, value): pass\n"
            "    def set_filename(self, value): self.filename = value\n"
            f"    def read_file(self, value):\n{read_body}"
            "tunable = _Tunable()\n"
        )
    else:
        cuda_init = (
            "class _Tunable:\n"
            "    def enable(self, value): pass\n"
            "    def tuning_enable(self, value): pass\n"
            "    def record_untuned_enable(self, value): pass\n"
            "    def set_filename(self, value): pass\n"
            "tunable = _Tunable()\n"
        )
    (cuda_dir / "__init__.py").write_text(cuda_init, encoding="utf-8")
    return torch_root


def _run_with_sitecustomize(tmp_path, *, torch_root, extra_env=None):
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    (site_dir / "sitecustomize.py").write_text(_generate_candidate_sitecustomize(), encoding="utf-8")

    candidate_file = tmp_path / "tunableop_results.csv"
    candidate_file.write_text("Validator,PT_VERSION,0\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "HL_TUNABLEOP_MODE": "candidate",
            "HL_TUNABLEOP_FILE": str(candidate_file),
            "HL_TUNABLEOP_VERBOSE": "1",
            "PYTHONPATH": os.pathsep.join([str(site_dir), str(torch_root)]),
        }
    )
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        [sys.executable, "-c", "print('candidate ran')"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_count_tunableop_result_lines_ignores_validators_and_garbage():
    text = "\n".join(
        [
            "Validator,PT_VERSION,0",
            "bad-line",
            "aten::mm,params,solution,1.23",
            "# comment",
            "",
        ]
    )
    assert count_tunableop_result_lines(text) == 1


def test_candidate_pythonpath_prepends_existing(monkeypatch, tmp_path):
    site_dir = tmp_path / "site"
    monkeypatch.setenv("PYTHONPATH", "/existing/path")
    assert _candidate_pythonpath(site_dir) == f"{site_dir}{os.pathsep}/existing/path"


def test_candidate_pythonpath_without_existing(monkeypatch, tmp_path):
    site_dir = tmp_path / "site"
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert _candidate_pythonpath(site_dir) == str(site_dir)


def test_candidate_sitecustomize_fails_closed_without_candidate_file_env(tmp_path):
    torch_root = _write_fake_torch(tmp_path, with_read_file=True)
    result = _run_with_sitecustomize(
        tmp_path,
        torch_root=torch_root,
        extra_env={"HL_TUNABLEOP_FILE": ""},
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "HL_TUNABLEOP_READ_FAILED" in combined
    assert "HL_TUNABLEOP_FILE or PYTORCH_TUNABLEOP_FILENAME" in combined
    assert "candidate ran" not in result.stdout


def test_candidate_sitecustomize_fails_closed_when_candidate_file_missing(tmp_path):
    torch_root = _write_fake_torch(tmp_path, with_read_file=True)
    missing = tmp_path / "missing_tunableop_results.csv"
    result = _run_with_sitecustomize(
        tmp_path,
        torch_root=torch_root,
        extra_env={"HL_TUNABLEOP_FILE": str(missing)},
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "HL_TUNABLEOP_READ_FAILED" in combined
    assert "TunableOp candidate file not found" in combined
    assert str(missing) in combined
    assert "candidate ran" not in result.stdout


def test_candidate_sitecustomize_fails_closed_without_read_file(tmp_path):
    torch_root = _write_fake_torch(tmp_path, with_read_file=False)
    result = _run_with_sitecustomize(tmp_path, torch_root=torch_root)

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "HL_TUNABLEOP_READ_FAILED" in combined
    assert "read_file unavailable" in combined
    assert "candidate ran" not in result.stdout


def test_candidate_sitecustomize_fails_closed_when_read_file_raises(tmp_path):
    torch_root = _write_fake_torch(tmp_path, with_read_file=True, read_file_raises=True)
    result = _run_with_sitecustomize(tmp_path, torch_root=torch_root)

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "HL_TUNABLEOP_READ_FAILED" in combined
    assert "corrupt tunableop csv" in combined
    assert "candidate ran" not in result.stdout


def test_candidate_sitecustomize_loads_file_when_read_file_works(tmp_path):
    torch_root = _write_fake_torch(tmp_path, with_read_file=True)
    marker = tmp_path / "read_marker"
    result = _run_with_sitecustomize(
        tmp_path,
        torch_root=torch_root,
        extra_env={"READ_MARKER": str(marker)},
    )

    assert result.returncode == 0, result.stderr
    assert "candidate ran" in result.stdout
    assert marker.read_text(encoding="utf-8") == str(tmp_path / "tunableop_results.csv")


def test_tunableop_script_failure_reports_status_and_traceback(tmp_path):
    torch_dir = tmp_path / "fake_torch" / "torch"
    torch_dir.mkdir(parents=True)
    (torch_dir / "__init__.py").write_text(
        "class _Tunable:\n"
        "    def enable(self, value): pass\n"
        "    def tuning_enable(self, value): pass\n"
        "    def record_untuned_enable(self, value): pass\n"
        "    def tune_gemm_in_file(self, path): raise RuntimeError('hipBLASLt unavailable')\n"
        "class cuda:\n"
        "    tunable = _Tunable()\n",
        encoding="utf-8",
    )
    untuned = tmp_path / "untuned.csv"
    untuned.write_text("", encoding="utf-8")
    script = _generate_tunableop_script(tmp_path, untuned, tmp_path / "tuned.csv", 10, "0")

    proc = subprocess.run(
        [sys.executable, str(script)],
        env={**os.environ, "PYTHONPATH": str(tmp_path / "fake_torch")},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 1
    assert '"status": "failed", "error": "hipBLASLt unavailable", "error_class": "RuntimeError"' in proc.stdout
    assert proc.stderr.startswith("Traceback (most recent call last):")
    assert proc.stderr.rstrip().endswith("RuntimeError: hipBLASLt unavailable")
