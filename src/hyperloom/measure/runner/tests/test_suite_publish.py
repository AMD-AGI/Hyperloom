# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The cache build publishes the inference_suite staging with the commit's own inference_suite, and fails without it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.measure.runner import cache_build
from hyperloom.measure.runner.cache_build import CacheEntry
from hyperloom.measure.runner.suite_publish import SuitePublishError, publish_inference_suite

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
