# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The start-up script as a file of the cache entry, and the one checked line a template step runs it with.

Each run goes through the real script against the rig's stand-in cache, as :mod:`test_bootstrap` does for the argv form.
"""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from hyperloom.measure.runner.bootstrap import BOOTSTRAP_SCRIPT, runner_start_shell_line
from hyperloom.measure.runner.cache_build import CacheConflict, ensure_runner_cache
from hyperloom.measure.runner.cache_layout import (
    BOOTSTRAP_NAME,
    MANIFEST_NAME,
    MANIFEST_PIN_ENV,
    check_entry,
    manifest_from_json,
    manifest_to_json,
)
from hyperloom.measure.runner.tests.bootstrap_rig import CacheRef, Rig, copy_cache, read_probe, run_script, tripwire

_SPEC = b'{"id": 11}'
_ARGS = ("--spec-wait-sec", "20")
#: What a tampered script leaves behind if it ever runs.
_RAN = "tampered-script-ran"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(cache: CacheRef) -> Path:
    return cache.root / cache.sha


def _pin(cache: CacheRef) -> dict[str, str]:
    return {MANIFEST_PIN_ENV: _sha256(_entry(cache) / MANIFEST_NAME)}


def _start(
    rig: Rig, work: Path, env: dict[str, str], cache: CacheRef | None = None
) -> subprocess.CompletedProcess[str]:
    """One run of the checked line, as a template step's ``sh -c``, with the network aimed at a tripwire."""
    (work / "in").mkdir(parents=True, exist_ok=True)
    (work / "in" / "spec.json").write_bytes(_SPEC)
    with tripwire() as wire:
        done = run_script(("/bin/sh", "-c", runner_start_shell_line(*_ARGS)), rig, work, {**wire.env, **env}, cache)
    assert wire.connections == 0, "the start tried to reach the network"
    return done


def _assert_refused(work: Path, done: subprocess.CompletedProcess[str]) -> None:
    """The step ended as a runner that could not start, by the verify step, and nothing of the script ran."""
    assert done.returncode == 2, done.stderr
    status = json.loads((work / "out" / "status.json").read_text())
    assert (status["state"], status["error_class"]) == ("failed", "runner_start_failed")
    assert status["message"].startswith("runner bootstrap, step verify, exit 1:")
    assert "bootstrap: cache" not in done.stderr, "the script's first step ran"
    assert not (work / _RAN).exists()
    assert not (work / "scratch").exists()


def _tamper(entry: Path, work: Path) -> None:
    """Append to the entry's script a line that marks that it ran; everything else of the entry is left as it was."""
    script = entry / BOOTSTRAP_NAME
    script.write_text(script.read_text().replace("set -eu\n", f"set -eu\ntouch {work / _RAN}\n", 1))


def test_a_build_writes_the_script_executable_and_its_hash_into_the_manifest(rig: Rig) -> None:
    entry = _entry(rig.cache)
    script = entry / BOOTSTRAP_NAME
    assert script.read_text(encoding="utf-8") == BOOTSTRAP_SCRIPT
    assert script.read_text().startswith("#!/bin/sh\n")
    assert stat.S_IMODE(script.stat().st_mode) == 0o755
    assert manifest_from_json((entry / MANIFEST_NAME).read_bytes()).bootstrap_sha256 == _sha256(script)


def test_the_checked_line_runs_the_entrys_script_and_the_runner_gets_its_arguments(rig: Rig, tmp_path: Path) -> None:
    done = _start(rig, tmp_path, _pin(rig.cache))
    assert done.returncode == 0, done.stderr
    probe = read_probe(tmp_path / "out")
    assert probe["argv"][-2:] == list(_ARGS)
    assert probe["revision"] == rig.cache.sha
    assert (tmp_path / "out" / "status.json").read_bytes() == _SPEC


def test_the_checked_line_refuses_a_tampered_script(rig: Rig, tmp_path: Path) -> None:
    cache = copy_cache(rig, tmp_path / "cache")
    _tamper(_entry(cache), tmp_path)
    work = tmp_path / "work"
    done = _start(rig, work, _pin(cache), cache)
    _assert_refused(work, done)
    assert f"{BOOTSTRAP_NAME}: FAILED" in done.stderr


def test_the_checked_line_refuses_a_manifest_that_is_not_the_pinned_one(rig: Rig, tmp_path: Path) -> None:
    done = _start(rig, tmp_path, {MANIFEST_PIN_ENV: "0" * 64})
    _assert_refused(tmp_path, done)
    assert f"{MANIFEST_NAME}: FAILED" in done.stderr


def test_the_checked_line_refuses_to_start_without_a_pin(rig: Rig, tmp_path: Path) -> None:
    done = _start(rig, tmp_path, {})
    _assert_refused(tmp_path, done)
    assert "sha256sum" not in done.stderr, "the line went on to a check it has nothing to check against"


def test_the_checked_line_refuses_an_entry_whose_manifest_records_no_script(rig: Rig, tmp_path: Path) -> None:
    """A manifest that records no script hash (one written before entries had a script) starts nothing."""
    cache = copy_cache(rig, tmp_path / "cache")
    manifest = _entry(cache) / MANIFEST_NAME
    manifest.write_text(manifest.read_text().replace('"bootstrap_sha256": ', '"no_script": ', 1))
    work = tmp_path / "work"
    _assert_refused(work, _start(rig, work, _pin(cache), cache))


def test_the_line_is_posix_sh_and_quotes_the_runner_arguments(tmp_path: Path) -> None:
    line = runner_start_shell_line("--spec-wait-sec", "it's $HOME")
    assert "\n" not in line
    subprocess.run(["/bin/sh", "-n", "-c", line], check=True)
    assert line.count("exec sh ") == 1 and """ --spec-wait-sec 'it'"'"'s $HOME'; fi;""" in line


def _rewrite_manifest(entry: Path, bootstrap_sha256: str) -> None:
    manifest = manifest_from_json((entry / MANIFEST_NAME).read_bytes())
    data = json.loads(manifest_to_json(manifest))
    data["bootstrap_sha256"] = bootstrap_sha256
    if not bootstrap_sha256:
        del data["bootstrap_sha256"]
    (entry / MANIFEST_NAME).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _older_builder(entry: Path) -> None:
    script = entry / BOOTSTRAP_NAME
    script.write_text(BOOTSTRAP_SCRIPT + "# an older builder's script\n")
    _rewrite_manifest(entry, _sha256(script))


def _before_the_script(entry: Path) -> None:
    (entry / BOOTSTRAP_NAME).unlink()
    _rewrite_manifest(entry, "")


@pytest.mark.parametrize("make", [_older_builder, _before_the_script], ids=["another-builder", "no-script"])
def test_an_entry_with_another_builders_script_or_none_gets_this_one_and_a_manifest_that_pins_it(
    rig: Rig, tmp_path: Path, make: Callable[[Path], None]
) -> None:
    cache = copy_cache(rig, tmp_path / "cache")
    entry = _entry(cache)
    make(entry)
    used = ensure_runner_cache(rig.repo, cache.sha, cache.root, rig.build_options())
    assert used.reused is True
    assert (entry / BOOTSTRAP_NAME).read_text() == BOOTSTRAP_SCRIPT
    assert stat.S_IMODE((entry / BOOTSTRAP_NAME).stat().st_mode) == 0o755
    assert used.manifest_sha256 == _sha256(entry / MANIFEST_NAME)
    assert manifest_from_json((entry / MANIFEST_NAME).read_bytes()).bootstrap_sha256 == _sha256(entry / BOOTSTRAP_NAME)
    assert check_entry(entry, cache.sha, used.manifest_sha256, verify_hash=True) == []


def test_an_entry_whose_script_is_not_the_one_its_manifest_pins_is_refused_and_left_as_it_was(
    rig: Rig, tmp_path: Path
) -> None:
    cache = copy_cache(rig, tmp_path / "cache")
    entry = _entry(cache)
    _tamper(entry, tmp_path)
    before = (_sha256(entry / BOOTSTRAP_NAME), _sha256(entry / MANIFEST_NAME))
    with pytest.raises(CacheConflict, match=f"{BOOTSTRAP_NAME} is missing or does not match the hash in the manifest"):
        ensure_runner_cache(rig.repo, cache.sha, cache.root, rig.build_options())
    assert (_sha256(entry / BOOTSTRAP_NAME), _sha256(entry / MANIFEST_NAME)) == before
