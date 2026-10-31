# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner installs the accuracy evaluator before the server boots, and the round finds it installed.

Run V5 of #156 (c4287): the warmup round of an accuracy test served its benchmark, then spent the silence window
installing lm-eval on an idle server whose output Magpie had not yet written anywhere, and the watchdog reaped it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from functools import partial
from pathlib import Path

import pytest
import yaml

from hyperloom.measure.manifest import ResultManifest
from hyperloom.measure.runner import eval_deps
from hyperloom.measure.runner import run as runner
from hyperloom.measure.runner.tests.test_run import _Backend, _go, _ok, _sheet, _status, env, key  # noqa: F401

pytestmark = pytest.mark.skipif(not (shutil.which("bash") and shutil.which("flock")), reason="needs bash and flock")

REF = "b315ef3b05176acc9732bb7fdec116abe1ecc476"
#: InferenceX's install, as far as the patch that makes it run once reads it: one function holding the pinned ref.
LIB = f"""#!/usr/bin/env bash
export PORT="${{PORT:-8888}}"

_install_lm_eval_deps() {{
    local lm_eval_ref="{REF}"
    echo install >> "$INSTALLS"
    eval "$INSTALL_EXTRA"
    touch "$INSTALLED"
}}
"""


class _Image:
    """A sandbox whose ``python3`` imports lm_eval once the install has run, under its own ``sys.prefix``."""

    def __init__(self, root: Path, *, lib: str = LIB, extra: str = ":") -> None:
        self.checkout = root / "ix"
        (self.checkout / "benchmarks").mkdir(parents=True, exist_ok=True)
        (self.checkout / "benchmarks" / "benchmark_lib.sh").write_text(lib)
        self.prefix = root / "prefix"
        self.prefix.mkdir()
        self.installs = root / "installs"
        self.installed = root / "installed"
        bin_dir = root / "bin"
        bin_dir.mkdir()
        python3 = bin_dir / "python3"
        python3.write_text(
            "#!/bin/sh\n"
            f'if [ "$2" = "import sys; print(sys.prefix)" ]; then echo {self.prefix}; exit 0; fi\n'
            f'if [ "$2" = "import lm_eval" ]; then [ -f {self.installed} ] && exit 0; echo "No module named lm_eval" >&2; exit 1; fi\n'
            "exit 0\n"
        )
        python3.chmod(0o755)
        self.environ = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "INSTALLS": str(self.installs),
            "INSTALLED": str(self.installed),
            "INSTALL_EXTRA": extra,
        }

    def install_count(self) -> int:
        return self.installs.read_text().count("install") if self.installs.exists() else 0

    def round_install(self) -> None:
        """What the accuracy round runs after its benchmark: the same library's install."""
        script = f'source "{self.checkout / "benchmarks" / "benchmark_lib.sh"}" && _install_lm_eval_deps'
        subprocess.run(["bash", "-c", script], check=True, env=self.environ)


def _accuracy_sheet(**procedure) -> dict:
    text = yaml.safe_dump({"benchmark": {"model": "/m", "framework": "vllm", "envs": {"TP": 1, "RUN_EVAL": "true"}}})
    sheet = _sheet(config=[{"yaml": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}])
    sheet["procedure"].update({"benchmark_backend": "magpie", "accuracy": True, **procedure})
    return sheet


@pytest.fixture
def image(tmp_path, env, monkeypatch):
    sandbox = _Image(tmp_path)
    env.update(sandbox.environ)
    monkeypatch.setattr(runner, "ensure_magpie", lambda commit, cache: tmp_path / "magpie")
    return sandbox


def _round_after(sandbox: _Image, seen: list[int]):
    """A backend that records how many installs had run when the test started, then installs as its round would."""

    def run(call):
        seen.append(sandbox.install_count())
        sandbox.round_install()
        return _ok(call)

    return _Backend(run)


def test_an_accuracy_test_installs_lm_eval_before_its_server_boots_and_its_round_does_not_again(
    tmp_path, env, image, monkeypatch
):
    seen: list[int] = []
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _round_after(image, seen))

    assert _go(tmp_path, env, _accuracy_sheet()) == 0, _status(tmp_path)

    assert seen == [1], "the install ran before the test, not in its round"
    assert image.install_count() == 1, "the round found lm-eval installed"
    assert (image.prefix / f".hyperloom-lm-eval-{REF}").is_file()
    manifest = ResultManifest.from_bytes((tmp_path / "out" / "manifest.json").read_bytes())
    assert eval_deps.EVAL_DEPS_LOG in {f.path for f in manifest.files}
    phases = json.loads((tmp_path / "out" / "manifest.json").read_text())["timing"]["phases"]
    assert "eval_deps" in [p["name"] for p in phases]


@pytest.mark.parametrize(
    "procedure", [{"accuracy": False}, {"benchmark_backend": "bypass"}], ids=["no accuracy", "not magpie"]
)
def test_a_test_that_does_not_evaluate_through_lm_eval_installs_nothing(tmp_path, env, image, monkeypatch, procedure):
    seen: list[int] = []
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(lambda call: (seen.append(1), _ok(call))[1]))

    assert _go(tmp_path, env, _accuracy_sheet(**procedure)) == 0, _status(tmp_path)

    assert seen == [1] and image.install_count() == 0
    assert not (tmp_path / "out" / "files" / eval_deps.EVAL_DEPS_LOG).exists()


def test_an_install_that_does_not_end_stops_the_runner_before_the_test_with_its_output(tmp_path, env, monkeypatch):
    sandbox = _Image(tmp_path, extra="echo fetching from the index; sleep 60")
    env.update(sandbox.environ)
    monkeypatch.setattr(runner, "ensure_magpie", lambda commit, cache: tmp_path / "magpie")
    monkeypatch.setattr(runner, "install_eval_deps", partial(eval_deps.install_eval_deps, timeout_sec=1.0))
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)
    started = time.monotonic()

    assert _go(tmp_path, env, _accuracy_sheet()) == 2

    assert time.monotonic() - started < 30, "the install's sleep was killed with its shell"
    assert backend.calls == []
    status = _status(tmp_path)
    assert status["error_class"] == "runner_start_failed"
    assert "did not end within 1 s" in status["message"] and "fetching from the index" in status["message"]


def test_lm_eval_that_does_not_import_after_the_install_is_refused_with_the_reason(tmp_path):
    sandbox = _Image(
        tmp_path, lib=LIB.replace('    touch "$INSTALLED"\n', '    echo "pip: no matching distribution"\n')
    )

    with pytest.raises(eval_deps.EvalDepsError) as raised:
        eval_deps.install_eval_deps(sandbox.checkout, environ=sandbox.environ, log_path=tmp_path / "log")

    assert "No module named lm_eval" in str(raised.value) and "pip: no matching distribution" in str(raised.value)


def test_a_checkout_whose_install_cannot_run_once_is_left_to_the_round_and_said_so(tmp_path):
    sandbox = _Image(tmp_path, lib="#!/usr/bin/env bash\nother() { :; }\n")

    note = eval_deps.install_eval_deps(sandbox.checkout, environ=sandbox.environ, log_path=tmp_path / "log")

    assert note is not None and "cannot be made to install it once" in note
    assert sandbox.install_count() == 0


def test_the_install_runs_in_the_round_s_environment_so_the_round_does_not_install_again(tmp_path, env, monkeypatch):
    """Run V6 of #156 (c4469): ``eval_deps`` succeeded, yet the warmup round sat idle on its server until the watchdog.

    The point's variables are the round's, not the runner's: here they name the user base pip installs into when the
    image's site-packages is not the account's, as ``PYTHONUSERBASE`` does. Installed in the runner's environment,
    lm-eval was where the round's ``python3`` does not look, and the round installed it again on an idle server.
    """
    prefix, runner_base, round_base = tmp_path / "prefix", tmp_path / "runner-base", tmp_path / "round-base"
    for directory in (prefix, runner_base, round_base):
        directory.mkdir()
    checkout = tmp_path / "ix"
    (checkout / "benchmarks").mkdir(parents=True)
    (checkout / "benchmarks" / "benchmark_lib.sh").write_text(
        LIB.replace('touch "$INSTALLED"', 'touch "${PYTHONUSERBASE:-$DEFAULT_BASE}/lm_eval"')
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").write_text(
        "#!/bin/sh\n"
        f'if [ "$2" = "import sys; print(sys.prefix)" ]; then echo {prefix}; exit 0; fi\n'
        'case "$2" in *"import lm_eval"*) [ -f "${PYTHONUSERBASE:-$DEFAULT_BASE}/lm_eval" ] && exit 0;'
        ' echo "No module named lm_eval" >&2; exit 1;; esac\n'
        "exit 0\n"
    )
    (bin_dir / "python3").chmod(0o755)
    installs = tmp_path / "installs"
    env.update(
        {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "INSTALLS": str(installs),
            "INSTALL_EXTRA": ":",
            "DEFAULT_BASE": str(runner_base),
        }
    )
    monkeypatch.setattr(runner, "ensure_inferencex", lambda commit, cache: checkout)
    monkeypatch.setattr(runner, "ensure_magpie", lambda commit, cache: tmp_path / "magpie")
    round_env = {**env, "PYTHONUSERBASE": str(round_base)}

    def run(call):
        script = f'source "{checkout / "benchmarks" / "benchmark_lib.sh"}" && _install_lm_eval_deps'
        subprocess.run(["bash", "-c", script], check=True, env=round_env)
        return _ok(call)

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(run))
    text = yaml.safe_dump(
        {
            "benchmark": {
                "model": "/m",
                "framework": "vllm",
                "envs": {"TP": 1, "RUN_EVAL": "true", "pythonuserbase": str(round_base)},
            }
        }
    )
    sheet = _sheet(config=[{"yaml": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}])
    sheet["procedure"].update({"benchmark_backend": "magpie", "accuracy": True})

    assert _go(tmp_path, env, sheet) == 0, _status(tmp_path)

    assert installs.read_text().count("install") == 1, "the round found lm-eval installed for its own python3"
    assert (round_base / "lm_eval").is_file() and not (runner_base / "lm_eval").exists()


def test_an_install_the_round_would_repeat_stops_the_runner_before_the_test(tmp_path):
    sandbox = _Image(tmp_path)
    # No marker can be written: the prefix lies below a regular file, and ``touch`` refuses the spare one below /tmp.
    (tmp_path / "file").write_text("")
    python3 = Path(sandbox.environ["PATH"].split(":")[0]) / "python3"
    python3.write_text(python3.read_text().replace(str(sandbox.prefix), str(tmp_path / "file" / "prefix")))
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "touch").write_text(
        '#!/bin/sh\ncase "$1" in /tmp/hyperloom-lm-eval-*) exit 1;; esac\nexec /usr/bin/touch "$@"\n'
    )
    (stubs / "touch").chmod(0o755)
    sandbox.environ["PATH"] = f"{stubs}:{sandbox.environ['PATH']}"

    with pytest.raises(eval_deps.EvalDepsError) as raised:
        eval_deps.install_eval_deps(sandbox.checkout, environ=sandbox.environ, log_path=tmp_path / "log")

    assert "would install it again" in str(raised.value)
    assert "no marker for lm-eval" in str(raised.value), "the install's own output says why"
