# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A test whose ``HF_DATASETS_CACHE`` is not writable loads its datasets from a copy of its own: ``datasets`` opens a
lock for writing on every load, so a read-only cache would fail every AgentX replay and every evaluation."""

from __future__ import annotations

import errno
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.measure.runner import datasets_cache
from hyperloom.measure.runner.datasets_cache import (
    DATASETS_CACHE_ENV,
    LOCAL_DATASETS_DIR,
    DatasetsNeeded,
    cache_dir_name,
    datasets_needed,
    is_writable,
    local_datasets_cache,
)
from inference_suite.client_env import CLIENT_SITE_PACKAGES
from inference_suite.staging import GSM8K_TASK

TRACE = "semianalysis_cc_traces_weka_062126"
TRACE_REPO = "semianalysisai/cc-traces-weka-062126"
CAPPED = f"{TRACE}_256k"
CAPPED_REPO = f"{TRACE_REPO}-256k"
GSM8K = "openai/gsm8k"


def _point(*, agentx: bool, envs: dict[str, str] | None = None) -> str:
    bench: dict[str, object] = {"framework": "sglang", "envs": envs or {}}
    if agentx:
        bench["agentx"] = {"enabled": True}
    return yaml.safe_dump({"benchmark": bench})


def _staged(root: Path, *, dataset_path: str | None = GSM8K) -> Path:
    """A staged InferenceX root with the client's loader registry and the GSM8K task."""
    registry = root / CLIENT_SITE_PACKAGES / "aiperf/plugin/plugins.yaml"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        yaml.safe_dump(
            {
                "public_dataset_loader": {
                    TRACE: {"class": "x:Y", "metadata": {"hf_dataset_name": TRACE_REPO, "hf_split": "train"}},
                    CAPPED: {"class": "x:Y", "metadata": {"hf_dataset_name": CAPPED_REPO}},
                    "sharegpt": {"class": "x:Z", "metadata": {}},
                }
            }
        ),
        encoding="utf-8",
    )
    task = root / GSM8K_TASK
    task.parent.mkdir(parents=True)
    task.write_text(f"task: gsm8k\ndataset_path: {dataset_path}\ndataset_name: main\n" if dataset_path else "", "utf-8")
    return root


def _prepared(cache: Path, repo: str, payload: bytes = b"arrow") -> Path:
    """``repo`` prepared below ``cache`` as ``datasets`` leaves it, with the locks it takes."""
    name = cache_dir_name(repo)
    version = cache / name / "default" / "0.0.0"
    data = version / "0123abcd"
    data.mkdir(parents=True)
    (data / "dataset_info.json").write_text('{"config_name": "default"}', encoding="utf-8")
    (data / "data-train.arrow").write_bytes(payload)
    (version / "0123abcd_builder.lock").touch()
    (version / "0123abcd.incomplete_info.lock").touch()
    root_lock = f"_{str(cache).replace('/', '_')}_{name}_default_0.0.0_0123abcd.lock"
    (cache / root_lock).touch()
    return cache / name


@pytest.fixture
def read_only(monkeypatch: pytest.MonkeyPatch):
    """The cache refuses every file made in it, as a read-only mount does (the tests may run as root, where a mode
    refuses nothing)."""

    def refuse(**_: object) -> tuple[int, str]:
        raise OSError(errno.EROFS, os.strerror(errno.EROFS))

    monkeypatch.setattr(datasets_cache, "tempfile", SimpleNamespace(mkstemp=refuse))


def _files(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() or path.is_symlink()}


# --- naming what the test loads ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("repo", "directory"),
    [
        (GSM8K, "openai___gsm8k"),
        (TRACE_REPO, "semianalysisai___cc-traces-weka-062126"),
        ("Acme/SquadV2", "Acme___squad_v2"),
    ],
)
def test_a_repository_is_prepared_in_the_directory_datasets_names_it(repo: str, directory: str) -> None:
    assert cache_dir_name(repo) == directory


def test_an_agentx_point_loads_its_trace_corpus_as_the_clients_registry_names_it(tmp_path: Path) -> None:
    staged = _staged(tmp_path)
    needed = datasets_needed(
        [_point(agentx=True, envs={"WEKA_LOADER_OVERRIDE": CAPPED}), _point(agentx=True)],
        base_client_env={"WEKA_LOADER_OVERRIDE": TRACE},
        accuracy=False,
        staged_root=staged,
    )
    assert needed == DatasetsNeeded((CAPPED_REPO, TRACE_REPO), ())


def test_a_test_with_accuracy_loads_the_dataset_its_staged_task_names(tmp_path: Path) -> None:
    point = _point(agentx=False, envs={"RUN_EVAL": "true"})
    staged = _staged(tmp_path / "a", dataset_path="acme/gsm8k-mirror")
    assert datasets_needed([point], base_client_env={}, accuracy=True, staged_root=staged).repos == (
        "acme/gsm8k-mirror",
    )
    unstated = _staged(tmp_path / "b", dataset_path=None)
    assert datasets_needed([point], base_client_env={}, accuracy=True, staged_root=unstated).repos == (GSM8K,)
    assert datasets_needed([point], base_client_env={}, accuracy=True, staged_root=None).repos == (GSM8K,)


def test_a_fixed_test_without_accuracy_loads_no_dataset(tmp_path: Path) -> None:
    point = _point(agentx=False, envs={"WEKA_LOADER_OVERRIDE": TRACE})
    needed = datasets_needed([point], base_client_env={}, accuracy=False, staged_root=_staged(tmp_path))
    assert needed == DatasetsNeeded()


def test_a_corpus_the_registry_does_not_name_is_unresolved(tmp_path: Path) -> None:
    points = [
        _point(agentx=True, envs={"WEKA_LOADER_OVERRIDE": "sharegpt"}),
        _point(agentx=True, envs={"WEKA_LOADER_OVERRIDE": "acme/own-traces"}),
    ]
    needed = datasets_needed(points, base_client_env={}, accuracy=False, staged_root=_staged(tmp_path))
    assert needed == DatasetsNeeded(("acme/own-traces",), ("sharegpt",))
    without_staging = datasets_needed(points[:1], base_client_env={}, accuracy=False, staged_root=None)
    assert without_staging == DatasetsNeeded((), ("sharegpt",))


# --- the writability probe --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("code", [errno.EROFS, errno.EACCES, errno.EPERM])
def test_a_refused_file_makes_the_cache_unwritable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    def refuse(**_: object) -> tuple[int, str]:
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(datasets_cache, "tempfile", SimpleNamespace(mkstemp=refuse))
    assert not is_writable(tmp_path)


def test_another_error_leaves_the_cache_as_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def full(**_: object) -> tuple[int, str]:
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(datasets_cache, "tempfile", SimpleNamespace(mkstemp=full))
    assert is_writable(tmp_path)


def test_a_writable_cache_is_probed_without_leaving_a_file(tmp_path: Path) -> None:
    assert is_writable(tmp_path)
    assert list(tmp_path.iterdir()) == []


# --- the test's own copy ----------------------------------------------------------------------------------------------


def test_an_unwritable_cache_is_copied_for_the_test_its_datasets_only_and_without_locks(
    tmp_path: Path, read_only: None
) -> None:
    cache, work = tmp_path / "shared" / "hf-datasets", tmp_path / "work"
    cache.mkdir(parents=True)
    trace = _prepared(cache, TRACE_REPO, b"t" * 4096)
    _prepared(cache, GSM8K, b"g" * 1024)
    _prepared(cache, "acme/unused", b"u" * 99)
    outside = tmp_path / "elsewhere.arrow"
    outside.write_bytes(b"outside")
    (trace / "escape.arrow").symlink_to(outside)
    (trace / "inside.arrow").symlink_to(trace / "default/0.0.0/0123abcd/data-train.arrow")
    environ = {DATASETS_CACHE_ENV: str(cache)}
    notes: list[str] = []
    asked = DatasetsNeeded((TRACE_REPO, GSM8K, "acme/not-cached"))

    with local_datasets_cache(environ, work, lambda: asked, notes) as copy:
        assert copy == work / LOCAL_DATASETS_DIR
        assert environ[DATASETS_CACHE_ENV] == str(copy)
        copied = _files(copy)
        assert sorted({path.split("/")[0] for path in copied}) == [
            "openai___gsm8k",
            "semianalysisai___cc-traces-weka-062126",
        ]
        assert not [path for path in copied if path.endswith(".lock")]
        assert "semianalysisai___cc-traces-weka-062126/escape.arrow" not in copied
        inside = copy / "semianalysisai___cc-traces-weka-062126/inside.arrow"
        assert not inside.is_symlink() and inside.read_bytes() == b"t" * 4096
        assert (copy / "openai___gsm8k/default/0.0.0/0123abcd/data-train.arrow").read_bytes() == b"g" * 1024

    assert environ == {DATASETS_CACHE_ENV: str(cache)}, "put back when the test ends"
    assert not (work / LOCAL_DATASETS_DIR).exists(), "the copy is the test's and goes with it"
    (note,) = notes
    assert f"{DATASETS_CACHE_ENV} {cache} is not writable" in note
    assert f"copied {TRACE_REPO}, {GSM8K} (" in note and "MiB, 5 files" in note
    assert "the cache holds no acme/not-cached" in note and "1 links leaving the cache" in note


def test_a_writable_cache_is_left_as_it_is(tmp_path: Path) -> None:
    environ = {DATASETS_CACHE_ENV: str(tmp_path)}
    _prepared(tmp_path, GSM8K)
    notes: list[str] = []

    def needed() -> DatasetsNeeded:
        raise AssertionError("a writable cache does not ask what the test loads")

    with local_datasets_cache(environ, tmp_path / "work", needed, notes) as copy:
        assert copy is None and environ == {DATASETS_CACHE_ENV: str(tmp_path)}
    assert notes == [] and not (tmp_path / "work").exists()


@pytest.mark.parametrize(
    ("environ", "sheet_env", "asked"),
    [
        ({}, {}, DatasetsNeeded((GSM8K,))),
        ({DATASETS_CACHE_ENV: "CACHE"}, {DATASETS_CACHE_ENV: "/sheet/own"}, DatasetsNeeded((GSM8K,))),
        ({DATASETS_CACHE_ENV: "CACHE"}, {}, DatasetsNeeded()),
    ],
)
def test_no_copy_without_a_cache_with_the_sheets_own_or_without_a_dataset(
    tmp_path: Path, read_only: None, environ: dict[str, str], sheet_env: dict[str, str], asked: DatasetsNeeded
) -> None:
    environ = {name: str(tmp_path) if value == "CACHE" else value for name, value in environ.items()}
    before = dict(environ)
    with local_datasets_cache(environ, tmp_path / "work", lambda: asked, [], sheet_env=sheet_env) as copy:
        assert copy is None and environ == before
    assert not (tmp_path / "work").exists()


def test_without_the_variable_the_default_below_hf_home_is_copied_and_the_variable_removed_after(
    tmp_path: Path, read_only: None
) -> None:
    home = tmp_path / "hf"
    _prepared(home / "datasets", GSM8K)
    environ = {"HF_HOME": str(home)}
    notes: list[str] = []

    with local_datasets_cache(environ, tmp_path / "work", lambda: DatasetsNeeded((GSM8K,)), notes) as copy:
        assert copy == tmp_path / "work" / LOCAL_DATASETS_DIR
        assert environ == {"HF_HOME": str(home), DATASETS_CACHE_ENV: str(copy)}
        assert (copy / "openai___gsm8k").is_dir()
    assert environ == {"HF_HOME": str(home)} and not (tmp_path / "work" / LOCAL_DATASETS_DIR).exists()
    assert f"{home / 'datasets'} is not writable" in notes[0]


@pytest.mark.parametrize("name", ["HF_HOME", "XDG_CACHE_HOME", DATASETS_CACHE_ENV])
def test_without_the_variable_a_sheet_naming_its_own_home_keeps_its_cache(
    tmp_path: Path, read_only: None, name: str
) -> None:
    _prepared(tmp_path / "hf" / "datasets", GSM8K)
    environ = {"HF_HOME": str(tmp_path / "hf")}
    with local_datasets_cache(
        environ, tmp_path / "work", lambda: DatasetsNeeded((GSM8K,)), [], sheet_env={name: "/sheet/own"}
    ) as copy:
        assert copy is None and environ == {"HF_HOME": str(tmp_path / "hf")}


@pytest.mark.parametrize("repo", ["/..", "a/..", "../x", "a/b/../c", "a b/c"])
def test_a_name_that_is_no_repository_id_copies_nothing_from_outside_the_cache(
    tmp_path: Path, read_only: None, repo: str
) -> None:
    cache = tmp_path / "cache"
    _prepared(cache, GSM8K)
    environ = {DATASETS_CACHE_ENV: str(cache)}
    notes: list[str] = []

    with local_datasets_cache(environ, tmp_path / "work", lambda: DatasetsNeeded((repo,)), notes) as copy:
        assert copy is not None and _files(copy) == set()
    assert f"the cache holds no {repo}" in notes[0]
    assert not (tmp_path / "work").exists() or not list((tmp_path / "work").iterdir())
    with pytest.raises(ValueError, match="not a Hugging Face repository id"):
        cache_dir_name(repo)


def test_a_corpus_that_cannot_be_named_copies_a_small_cache_whole(tmp_path: Path, read_only: None) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    for repo in (TRACE_REPO, "acme/other"):
        _prepared(cache, repo)
    environ = {DATASETS_CACHE_ENV: str(cache)}
    notes: list[str] = []

    with local_datasets_cache(environ, tmp_path / "work", lambda: DatasetsNeeded((), ("sharegpt",)), notes) as copy:
        assert copy is not None
        assert {path.split("/")[0] for path in _files(copy)} == {cache_dir_name(TRACE_REPO), "acme___other"}
        assert not [path for path in _files(copy) if path.endswith(".lock")]
    assert "copied the whole cache" in notes[0]


def test_a_large_cache_is_not_copied_whole_for_a_corpus_that_cannot_be_named(
    tmp_path: Path, read_only: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(datasets_cache, "WHOLE_CACHE_MAX_BYTES", 10)
    cache = tmp_path / "cache"
    cache.mkdir()
    _prepared(cache, GSM8K, b"g" * 64)
    _prepared(cache, "acme/other", b"o" * 64)
    environ = {DATASETS_CACHE_ENV: str(cache)}
    notes: list[str] = []

    with local_datasets_cache(
        environ, tmp_path / "work", lambda: DatasetsNeeded((GSM8K,), ("sharegpt",)), notes
    ) as copy:
        assert copy is not None
        assert {path.split("/")[0] for path in _files(copy)} == {"openai___gsm8k"}
    unnamed, copied = notes
    assert "sharegpt names no dataset repository" in unnamed and f"copied {GSM8K} (" in copied


def test_a_copy_that_fails_leaves_nothing_behind_and_the_environment_as_it_was(
    tmp_path: Path, read_only: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    _prepared(cache, GSM8K)
    environ = {DATASETS_CACHE_ENV: str(cache)}

    def full(source: str, target: str) -> None:
        Path(target).write_bytes(b"part")
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(datasets_cache.shutil, "copy2", full)
    with pytest.raises(OSError, match="No space"):
        with local_datasets_cache(environ, tmp_path / "work", lambda: DatasetsNeeded((GSM8K,)), []):
            pytest.fail("the test does not run on a partial copy")
    assert environ == {DATASETS_CACHE_ENV: str(cache)} and not (tmp_path / "work" / LOCAL_DATASETS_DIR).exists()


# --- an offline load from the copy ------------------------------------------------------------------------------------

_PREPARE = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    import datasets
    from datasets.arrow_writer import ArrowWriter
    root = Path(sys.argv[1]) / "acme___tiny" / "default" / "0.0.0" / "0123abcd"
    root.mkdir(parents=True)
    features = datasets.Features({"question": datasets.Value("string")})
    with ArrowWriter(features=features, path=str(root / "tiny-train.arrow")) as writer:
        for question in ("a", "b", "c"):
            writer.write({"question": question})
        rows, size = writer.finalize()
    split = datasets.SplitInfo(name="train", num_examples=rows, num_bytes=size, dataset_name="tiny")
    datasets.DatasetInfo(
        features=features, config_name="default", dataset_name="tiny", splits=datasets.SplitDict({"train": split})
    ).write_to_directory(str(root))
    """
)
_LOAD = "import datasets; print(','.join(datasets.load_dataset('acme/tiny', split='train')['question']))"


def _offline(cache: Path, home: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "HF_HOME": str(home / "hf"),
        DATASETS_CACHE_ENV: str(cache),
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
    }


def test_datasets_loads_the_copy_offline_and_takes_its_locks_there(tmp_path: Path, read_only: None) -> None:
    pytest.importorskip("datasets")
    cache = tmp_path / "shared"
    subprocess.run([sys.executable, "-c", _PREPARE, str(cache)], check=True, env=_offline(cache, tmp_path))
    before = _files(cache)
    environ = {DATASETS_CACHE_ENV: str(cache)}

    with local_datasets_cache(environ, tmp_path / "work", lambda: DatasetsNeeded(("acme/tiny",)), []) as copy:
        assert copy is not None
        loaded = subprocess.run(
            [sys.executable, "-c", _LOAD], env=_offline(copy, tmp_path), capture_output=True, text=True, check=True
        )
        assert loaded.stdout.strip() == "a,b,c"
        assert [path for path in _files(copy) if path.endswith(".lock")], "datasets took its lock in the copy"
    assert _files(cache) == before, "the shared cache was only read"
