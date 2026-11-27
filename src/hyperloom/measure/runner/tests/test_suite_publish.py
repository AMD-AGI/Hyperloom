# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The cache build publishes the inference_suite staging with the commit's own inference_suite, and fails without it; a
local session publishes the one this interpreter imports."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import inference_suite
from inference_suite.staging import MANIFEST_FILE, StagedTool, StagingManifest
from hyperloom.measure.runner import cache_build, suite_publish
from hyperloom.measure.runner.cache_build import CacheEntry
from hyperloom.measure.runner.tools import ToolError
from hyperloom.measure.runner.suite_publish import (
    SuitePublishError,
    publish_inference_suite,
    publish_own_inference_suite,
)

_FAKE_MAIN = """
import json, os, sys
if os.environ.get("FAIL"):
    sys.exit("staging_missing: the interpreter build is pinned at abc")
stagings = sys.argv[sys.argv.index("--stagings") + 1]
print(os.path.join(stagings, "k"))
with open(os.path.join(os.path.dirname(stagings), "seen.json"), "w") as seen:
    json.dump({"argv": sys.argv[1:], "module": __file__}, seen)
"""


def _commit_source(tmp_path: Path) -> Path:
    """An entry's source tree whose own ``inference_suite`` records how it was run."""
    package = tmp_path / "entry" / "source" / "src" / "inference_suite"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(_FAKE_MAIN)
    return tmp_path / "entry" / "source"


def test_the_commits_own_inference_suite_publishes_into_the_runner_caches_inference_suite_directory(tmp_path):
    source = _commit_source(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()

    published = publish_inference_suite(source, cache)

    assert published == cache / "inference-suite" / "k"
    seen = json.loads((cache / "seen.json").read_text())
    assert seen["argv"] == ["publish", "--stagings", str(cache / "inference-suite")]
    assert Path(seen["module"]).is_relative_to(source / "src")
    assert not list(source.rglob("__pycache__")), "the entry's source is checked file by file on reuse"


def test_the_inference_suite_this_interpreter_imports_publishes_for_runners_started_from_this_installation(
    tmp_path, monkeypatch
):
    packages = _commit_source(tmp_path) / "src"
    monkeypatch.setattr(inference_suite, "__file__", str(packages / "inference_suite" / "__init__.py"))
    cache = tmp_path / "cache"
    cache.mkdir()

    assert publish_own_inference_suite(cache) == cache / "inference-suite" / "k"
    seen = json.loads((cache / "seen.json").read_text())
    assert seen["argv"] == ["publish", "--stagings", str(cache / "inference-suite")]
    assert Path(seen["module"]).is_relative_to(packages)


def test_a_failed_publish_is_an_error_naming_its_staging_missing_line(tmp_path, monkeypatch):
    source = _commit_source(tmp_path)
    monkeypatch.setenv("FAIL", "1")

    with pytest.raises(SuitePublishError, match="exited 1: staging_missing: the interpreter build is pinned at abc"):
        publish_inference_suite(source, tmp_path)


def _entry(monkeypatch, tmp_path, publish):
    entry = CacheEntry(tmp_path / "cache" / ("a" * 40), "b" * 64, reused=True)
    monkeypatch.setattr(cache_build, "ensure_runner_cache", lambda repo, sha, root, options: entry)
    monkeypatch.setattr(cache_build, "publish_inference_suite", publish)
    return ["--sha", "a" * 40, "--repo", str(tmp_path), "--out", str(tmp_path / "cache")]


def test_the_cache_build_command_publishes_the_entrys_staging_and_names_it(tmp_path, monkeypatch, capsys):
    calls = []

    def publish(source, root):
        calls.append((source, root))
        return root / "inference-suite" / "k"

    assert cache_build.main(_entry(monkeypatch, tmp_path, publish)) == 0

    assert calls == [(tmp_path / "cache" / ("a" * 40) / "source", tmp_path / "cache")]
    assert json.loads(capsys.readouterr().out)["inference_suite"] == str(tmp_path / "cache" / "inference-suite" / "k")


def test_the_cache_build_command_fails_when_the_staging_cannot_be_published(tmp_path, monkeypatch, capsys):
    def publish(source, root):
        raise SuitePublishError("publishing the inference_suite staging exited 2: staging_missing: offline")

    assert cache_build.main(_entry(monkeypatch, tmp_path, publish)) == 1
    assert "runner cache not built: publishing the inference_suite staging" in capsys.readouterr().err


_PIN = "c" * 40


def _published_staging(cache: Path, tools: dict[str, str]) -> None:
    """The staging the fake publish names, as a real publish leaves it: with its manifest."""
    staging = cache / "inference-suite" / "k"
    staging.mkdir(parents=True)
    manifest = StagingManifest({name: StagedTool(commit, {"f": "0" * 64}) for name, commit in tools.items()})
    (staging / MANIFEST_FILE).write_bytes(manifest.to_bytes())


def test_each_publish_fills_the_runner_caches_inferencex_store_with_the_stagings_commit(tmp_path, monkeypatch):
    source = _commit_source(tmp_path)
    cache = tmp_path / "cache"
    _published_staging(cache, {"inferencex": _PIN})
    filled = []
    monkeypatch.setattr(suite_publish, "fill_inferencex_store", lambda root, commit: filled.append((root, commit)))

    publish_inference_suite(source, cache)
    monkeypatch.setattr(inference_suite, "__file__", str(source / "src" / "inference_suite" / "__init__.py"))
    publish_own_inference_suite(cache)

    assert filled == [(cache, _PIN), (cache, _PIN)]


def test_a_store_that_cannot_be_filled_is_a_warning_and_the_publish_goes_on(tmp_path, monkeypatch, caplog):
    source = _commit_source(tmp_path)
    cache = tmp_path / "cache"
    _published_staging(cache, {"inferencex": _PIN})

    def unreachable(root, commit):
        raise ToolError(f"InferenceX {commit} was not fetched")

    monkeypatch.setattr(suite_publish, "fill_inferencex_store", unreachable)

    with caplog.at_level(logging.WARNING, logger=suite_publish.__name__):
        assert publish_inference_suite(source, cache) == cache / "inference-suite" / "k"

    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert "InferenceX store was not filled, so each test fetches InferenceX from INFERENCEX_REPO or upstream: " in (
        record.getMessage()
    )
    assert f"InferenceX {_PIN} was not fetched" in record.getMessage()


def test_a_staging_without_inferencex_fills_nothing(tmp_path, monkeypatch):
    source = _commit_source(tmp_path)
    cache = tmp_path / "cache"
    _published_staging(cache, {"other": _PIN})
    monkeypatch.setattr(suite_publish, "fill_inferencex_store", lambda root, commit: pytest.fail("filled"))

    assert publish_inference_suite(source, cache) == cache / "inference-suite" / "k"
