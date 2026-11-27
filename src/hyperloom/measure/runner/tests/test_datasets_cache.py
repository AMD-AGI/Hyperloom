# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A test whose datasets cache is not writable loads its datasets through a link tree of its own: ``datasets`` opens a
lock for writing on every load, so a read-only cache would fail every AgentX replay and every evaluation."""

from __future__ import annotations

import errno
import os
import pwd
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.measure.failures import DISK_FULL, HF_CACHE_PERMISSION, FailureClass, classify_test_failure
from hyperloom.measure.runner import datasets_cache
from hyperloom.measure.runner.datasets_cache import (
    DATASETS_CACHE_ENV,
    LOCAL_DATASETS_DIR,
    DatasetsNeeded,
    _cache_dir_name,
    _effective_cache,
    _is_writable,
    datasets_needed,
    local_datasets_cache,
)
from hyperloom.measure.runner.service import RunnerFailure
from inference_suite.client_env import CLIENT_SITE_PACKAGES
from inference_suite.staging import GSM8K_TASK

TRACE = "semianalysis_cc_traces_weka_062126"
TRACE_REPO = "semianalysisai/cc-traces-weka-062126"
CAPPED = f"{TRACE}_256k"
CAPPED_REPO = f"{TRACE_REPO}-256k"
GSM8K = "openai/gsm8k"


def _point(*, agentx: bool, envs: dict[str, object] | None = None) -> str:
    bench: dict[str, object] = {"framework": "sglang", "envs": envs or {}}
    if agentx:
        bench["agentx"] = {"enabled": True}
    return yaml.safe_dump({"benchmark": bench})


def _staged(root: Path, *, task: str | None = f"task: gsm8k\ndataset_path: {GSM8K}\ndataset_name: main\n") -> Path:
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
    if task is not None:
        (root / GSM8K_TASK).parent.mkdir(parents=True)
        (root / GSM8K_TASK).write_text(task, "utf-8")
    return root


def _prepared(cache: Path, repo: str, payload: bytes = b"arrow") -> Path:
    """``repo`` prepared below ``cache`` as ``datasets`` leaves it, with the locks it takes."""
    name = _cache_dir_name(repo)
    assert name is not None
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


def _entries(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() or path.is_symlink()}


def _with_cache(tmp_path: Path, *repos: str) -> tuple[Path, dict[str, str]]:
    cache = tmp_path / "shared" / "hf-datasets"
    cache.mkdir(parents=True)
    for repo in repos:
        _prepared(cache, repo)
    return cache, {DATASETS_CACHE_ENV: str(cache)}


# --- naming what the test loads ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("repo", "directory"),
    [
        (GSM8K, "openai___gsm8k"),
        (TRACE_REPO, "semianalysisai___cc-traces-weka-062126"),
        ("Acme/SquadV2", "Acme___squad_v2"),
        ("/..", None),
        ("a/..", None),
        ("../x", None),
        ("a b/c", None),
    ],
)
def test_a_repository_is_prepared_in_the_directory_datasets_names_it(repo: str, directory: str | None) -> None:
    assert _cache_dir_name(repo) == directory


def test_an_agentx_point_loads_its_trace_corpus_as_the_clients_registry_names_it(tmp_path: Path) -> None:
    needed = datasets_needed(
        [_point(agentx=True, envs={"WEKA_LOADER_OVERRIDE": CAPPED}), _point(agentx=True)],
        base_client_env={"WEKA_LOADER_OVERRIDE": TRACE},
        accuracy=False,
        staged_root=_staged(tmp_path),
    )
    assert needed == DatasetsNeeded((CAPPED_REPO, TRACE_REPO), ())


def test_a_test_with_accuracy_loads_the_dataset_its_staged_task_names(tmp_path: Path) -> None:
    point = _point(agentx=False, envs={"RUN_EVAL": "true"})

    def named(task: str | None) -> tuple[str, ...]:
        staged = _staged(tmp_path / str(len(list(tmp_path.iterdir()))), task=task)
        return datasets_needed([point], base_client_env={}, accuracy=True, staged_root=staged).repos

    assert named("task: gsm8k\ndataset_path: 'acme/gsm8k-mirror'  # mirrored\n") == ("acme/gsm8k-mirror",)
    assert named("task: gsm8k\n") == ()
    assert named("dataset_path: [unclosed\n") == ()
    assert named(None) == ()
    assert datasets_needed([point], base_client_env={}, accuracy=True, staged_root=None) == DatasetsNeeded()


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
    assert needed == DatasetsNeeded((), ("sharegpt", "acme/own-traces"))
    without_staging = datasets_needed(points[:1], base_client_env={}, accuracy=True, staged_root=None)
    assert without_staging == DatasetsNeeded((), ("sharegpt",))


# --- the cache datasets uses ------------------------------------------------------------------------------------------

_OWN_HOME = pwd.getpwuid(os.getuid()).pw_dir
_ROOT_HOME = pwd.getpwnam("root").pw_dir
_CACHES = [
    ({"HOME": "/h"}, "/h/.cache/huggingface/datasets"),
    ({"HOME": "/h/"}, "/h/.cache/huggingface/datasets"),
    ({"HOME": ""}, "/.cache/huggingface/datasets"),
    ({}, f"{_OWN_HOME.rstrip('/')}/.cache/huggingface/datasets"),
    ({"HOME": "/h", "XDG_CACHE_HOME": "/x"}, "/x/huggingface/datasets"),
    ({"HOME": "/h", "XDG_CACHE_HOME": "~/x"}, "/h/x/huggingface/datasets"),
    ({"HOME": "/h", "XDG_CACHE_HOME": ""}, None),
    ({"HOME": "/h", "HF_HOME": "/hf"}, "/hf/datasets"),
    ({"HOME": "/h", "HF_HOME": "~/hf"}, "/h/hf/datasets"),
    ({"HOME": "/h", "HF_HOME": "~root/hf"}, f"{_ROOT_HOME}/hf/datasets"),
    ({"HOME": "/h", "HF_HOME": ""}, None),
    ({"HOME": "/h", "HF_HOME": "/hf", DATASETS_CACHE_ENV: "/d"}, "/d"),
    ({"HOME": "/h", DATASETS_CACHE_ENV: "~/d"}, None),
    ({"HOME": "/h", DATASETS_CACHE_ENV: ""}, None),
]


@pytest.mark.parametrize(("environ", "cache"), _CACHES)
def test_the_cache_is_the_one_datasets_resolves(environ: dict[str, str], cache: str | None) -> None:
    assert _effective_cache(environ) == (None if cache is None else Path(cache))


def test_the_cache_rule_is_datasets_own(tmp_path: Path) -> None:
    pytest.importorskip("datasets")
    names = ("HOME", "HF_HOME", "XDG_CACHE_HOME", DATASETS_CACHE_ENV)
    for environ, cache in _CACHES:
        env = {k: v for k, v in os.environ.items() if k not in names} | environ
        said = subprocess.run(
            [sys.executable, "-c", "import datasets.config as c; print(c.HF_DATASETS_CACHE)"],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert (said if os.path.isabs(said) else None) == cache, environ


# --- the writability probe --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("code", [errno.EROFS, errno.EACCES, errno.EPERM])
def test_a_refused_file_makes_the_cache_unwritable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    def refuse(**_: object) -> tuple[int, str]:
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(datasets_cache, "tempfile", SimpleNamespace(mkstemp=refuse))
    assert not _is_writable(tmp_path)


def test_another_error_leaves_the_cache_as_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def full(**_: object) -> tuple[int, str]:
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(datasets_cache, "tempfile", SimpleNamespace(mkstemp=full))
    assert _is_writable(tmp_path)


def test_a_writable_cache_is_probed_without_leaving_a_file(tmp_path: Path) -> None:
    assert _is_writable(tmp_path)
    assert list(tmp_path.iterdir()) == []


# --- the test's link tree ---------------------------------------------------------------------------------------------


def test_an_unwritable_cache_is_linked_for_the_test_its_datasets_only_in_real_directories_without_locks(
    tmp_path: Path, read_only: None
) -> None:
    cache, environ = _with_cache(tmp_path, GSM8K, "acme/unused")
    trace = _prepared(cache, TRACE_REPO, b"t" * 4096)
    outside = tmp_path / "elsewhere.arrow"
    outside.write_bytes(b"outside")
    (trace / "escape.arrow").symlink_to(outside)
    (trace / "inside.arrow").symlink_to(trace / "default/0.0.0/0123abcd/data-train.arrow")
    (trace / "linked-dir").symlink_to(trace / "default", target_is_directory=True)
    work = tmp_path / "work"
    notes: list[str] = []
    asked = DatasetsNeeded((TRACE_REPO, GSM8K, "acme/not-cached"))

    with local_datasets_cache(environ, work, lambda: asked, notes) as tree:
        assert tree == work / LOCAL_DATASETS_DIR
        assert environ[DATASETS_CACHE_ENV] == str(tree)
        linked = _entries(tree)
        assert {path.split("/")[0] for path in linked} == {"openai___gsm8k", "semianalysisai___cc-traces-weka-062126"}
        assert not [path for path in linked if path.endswith(".lock")]
        assert {"escape.arrow", "linked-dir"}.isdisjoint(path.split("/", 1)[1] for path in linked)
        for path in linked:
            assert (tree / path).is_symlink() and Path(os.readlink(tree / path)).is_relative_to(cache), path
            assert not (tree / path).parent.is_symlink(), "directories are the tree's own"
        inside = tree / "semianalysisai___cc-traces-weka-062126/inside.arrow"
        assert os.readlink(inside) == str(trace / "default/0.0.0/0123abcd/data-train.arrow")
        assert inside.read_bytes() == b"t" * 4096
        assert len(linked) == 5

    assert environ == {DATASETS_CACHE_ENV: str(cache)}, "put back when the test ends"
    assert not (work / LOCAL_DATASETS_DIR).exists(), "the tree is the test's and goes with it"
    assert (trace / "default/0.0.0/0123abcd/data-train.arrow").read_bytes() == b"t" * 4096, "the cache is untouched"
    (note,) = notes
    assert f"{DATASETS_CACHE_ENV} {cache} is not writable" in note
    assert f"linked {TRACE_REPO}, {GSM8K} (5 files)" in note
    assert "the cache holds no acme/not-cached" in note and "2 links in the cache leaving their dataset" in note


def test_a_dataset_directory_that_is_a_link_is_linked_from_the_directory_it_names(
    tmp_path: Path, read_only: None
) -> None:
    store = tmp_path / "store"
    real = _prepared(store, GSM8K)
    cache, environ = _with_cache(tmp_path)
    (cache / "openai___gsm8k").symlink_to(real, target_is_directory=True)

    with local_datasets_cache(environ, tmp_path / "work", lambda: DatasetsNeeded((GSM8K,)), []) as tree:
        assert tree is not None and not (tree / "openai___gsm8k").is_symlink()
        target = os.readlink(tree / "openai___gsm8k/default/0.0.0/0123abcd/data-train.arrow")
        assert target == str(real.resolve() / "default/0.0.0/0123abcd/data-train.arrow")


def test_a_writable_cache_is_left_as_it_is(tmp_path: Path) -> None:
    cache, environ = _with_cache(tmp_path, GSM8K)
    before = _entries(cache)
    notes: list[str] = []

    def needed() -> DatasetsNeeded:
        raise AssertionError("a writable cache does not ask what the test loads")

    with local_datasets_cache(environ, tmp_path / "work", needed, notes) as tree:
        assert tree is None and environ == {DATASETS_CACHE_ENV: str(cache)}
    assert notes == [] and not (tmp_path / "work").exists() and _entries(cache) == before


@pytest.mark.parametrize(
    ("environ", "sheet_env"),
    [
        ({DATASETS_CACHE_ENV: "MISSING"}, {}),
        ({DATASETS_CACHE_ENV: "CACHE"}, {DATASETS_CACHE_ENV: "/sheet/own"}),
        ({DATASETS_CACHE_ENV: "CACHE"}, {DATASETS_CACHE_ENV: ""}),
    ],
)
def test_no_tree_without_a_cache_or_with_the_sheets_own(
    tmp_path: Path, read_only: None, environ: dict[str, str], sheet_env: dict[str, str]
) -> None:
    cache, _ = _with_cache(tmp_path, GSM8K)
    environ = {name: {"CACHE": str(cache), "MISSING": str(tmp_path / "none")}[value] for name, value in environ.items()}
    before = dict(environ)
    with local_datasets_cache(
        environ, tmp_path / "work", lambda: DatasetsNeeded((GSM8K,)), [], sheet_env=sheet_env
    ) as tree:
        assert tree is None and environ == before
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize(
    ("asked", "said"),
    [
        (DatasetsNeeded(), None),
        (DatasetsNeeded((), ("sharegpt",)), "sharegpt names no dataset repository the staged client knows"),
        (DatasetsNeeded(("acme/not-cached",)), "holds none of acme/not-cached"),
    ],
)
def test_with_nothing_to_link_the_variable_is_left_as_given(
    tmp_path: Path, read_only: None, asked: DatasetsNeeded, said: str | None
) -> None:
    cache, environ = _with_cache(tmp_path, GSM8K)
    notes: list[str] = []
    with local_datasets_cache(environ, tmp_path / "work", lambda: asked, notes) as tree:
        assert tree is None and environ == {DATASETS_CACHE_ENV: str(cache)}
    assert not (tmp_path / "work" / LOCAL_DATASETS_DIR).exists()
    assert notes == ([] if said is None else [notes[0]]) and (said is None or said in notes[0])


def test_without_the_variable_the_default_below_hf_home_is_linked_and_the_variable_removed_after(
    tmp_path: Path, read_only: None
) -> None:
    home = tmp_path / "hf"
    _prepared(home / "datasets", GSM8K)
    environ = {"HF_HOME": str(home)}
    notes: list[str] = []

    with local_datasets_cache(environ, tmp_path / "work", lambda: DatasetsNeeded((GSM8K,)), notes) as tree:
        assert tree == tmp_path / "work" / LOCAL_DATASETS_DIR
        assert environ == {"HF_HOME": str(home), DATASETS_CACHE_ENV: str(tree)}
        assert (tree / "openai___gsm8k").is_dir()
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
    ) as tree:
        assert tree is None and environ == {"HF_HOME": str(tmp_path / "hf")}


# --- a tree that cannot be made ---------------------------------------------------------------------------------------


def _fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str, code: int) -> RunnerFailure:
    cache, environ = _with_cache(tmp_path, GSM8K)

    real = getattr(os, where)

    def refused(path: object, *rest: object, **named: object):
        if not Path(str(path)).is_relative_to(cache):  # the cleanup lists the tree with the same call
            return real(path, *rest, **named)
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(datasets_cache.os, where, (lambda *_: False) if where == "access" else refused)
    with pytest.raises(RunnerFailure) as failed:
        with local_datasets_cache(environ, tmp_path / "work", lambda: DatasetsNeeded((GSM8K,)), []):
            pytest.fail("the test does not run on a part of its tree")
    assert environ == {DATASETS_CACHE_ENV: str(cache)} and not (tmp_path / "work" / LOCAL_DATASETS_DIR).exists()
    return failed.value


@pytest.mark.parametrize(
    ("where", "code"),
    [
        ("scandir", errno.EACCES),  # a directory that cannot be listed
        ("access", errno.EACCES),  # a file that cannot be read
        ("scandir", errno.ESTALE),
        ("symlink", errno.EIO),
    ],
)
def test_a_cache_that_cannot_be_linked_ends_the_test_as_the_caches_refusal(
    tmp_path: Path, read_only: None, monkeypatch: pytest.MonkeyPatch, where: str, code: int
) -> None:
    failure = _fails(tmp_path, monkeypatch, where, code)
    assert failure.error_class == HF_CACHE_PERMISSION
    verdict = classify_test_failure(returncode=1, error_class=failure.error_class)
    assert verdict.failure_class is FailureClass.INFRA and verdict.failure_reason == HF_CACHE_PERMISSION


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EDQUOT])
def test_a_full_scratch_ends_the_test_as_disk_full(
    tmp_path: Path, read_only: None, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    assert _fails(tmp_path, monkeypatch, "symlink", code).error_class == DISK_FULL


# --- a load through the tree ------------------------------------------------------------------------------------------

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


def test_datasets_loads_through_the_tree_offline_and_takes_its_locks_there(tmp_path: Path, read_only: None) -> None:
    pytest.importorskip("datasets")
    cache = tmp_path / "shared"
    subprocess.run([sys.executable, "-c", _PREPARE, str(cache)], check=True, env=_offline(cache, tmp_path))
    before = {path: (cache / path).stat().st_mtime_ns for path in _entries(cache)}
    environ = {DATASETS_CACHE_ENV: str(cache)}

    with local_datasets_cache(environ, tmp_path / "work", lambda: DatasetsNeeded(("acme/tiny",)), []) as tree:
        assert tree is not None
        loaded = subprocess.run(
            [sys.executable, "-c", _LOAD], env=_offline(tree, tmp_path), capture_output=True, text=True, check=True
        )
        assert loaded.stdout.strip() == "a,b,c"
        locks = [path for path in _entries(tree) if path.endswith(".lock")]
        assert locks and not [path for path in locks if (tree / path).is_symlink()], "its locks are the tree's own"
    assert {path: (cache / path).stat().st_mtime_ns for path in _entries(cache)} == before, "the cache was only read"
