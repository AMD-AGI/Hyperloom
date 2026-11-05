# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner cache's layout code: the lock's parser, the tree hash, the manifest and the check the runner makes."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import BinaryIO

import pytest

from hyperloom.measure.runner import cache_layout
from hyperloom.measure.runner.cache_layout import (
    LOCAL_SOURCE_DIR,
    LOCAL_WHEELS_DIR,
    MANIFEST_NAME,
    REQUIREMENTS_FILE,
    SOURCE_ARCHIVE,
    CacheError,
    FileDigest,
    Manifest,
    check_cache,
    check_entry,
    file_sha256,
    manifest_from_json,
    manifest_to_json,
    parse_lock,
    requirements_text,
    stage_entry,
    tree_digest,
)
from hyperloom.measure.runner.tests.bootstrap_rig import Rig, copy_cache

_HASH = "a" * 64
_RUNNER_MAIN = "src/hyperloom/measure/runner/__main__.py"


def test_the_lock_parser_reads_pins_with_their_hashes_and_skips_comments() -> None:
    text = (
        "#\n# header\n#\n"
        f"Typing_Extensions==4.15.0 \\\n    --hash=sha256:{_HASH} \\\n    --hash=sha256:{'b' * 64}\n"
        "    # via\n    #   rich\n"
        f"\nPyYAML==6.0.3 \\\n    --hash=sha256:{'c' * 64}\n"
    )
    pins = parse_lock(text)
    assert [(pin.name, pin.version) for pin in pins] == [("typing-extensions", "4.15.0"), ("pyyaml", "6.0.3")]
    assert pins[0].hashes == {_HASH, "b" * 64} and pins[1].hashes == {"c" * 64}


@pytest.mark.parametrize(
    ("text", "line"),
    [
        (f"a==1 \\\n    --hash=sha256:{_HASH}\nb>=2\n", 3),
        (f"--index-url https://example.invalid/simple\na==1 \\\n    --hash=sha256:{_HASH}\n", 1),
        (f"a==1 \\\n    --hash=md5:{'a' * 32}\n", 2),
        (f"    --hash=sha256:{_HASH}\n", 1),
    ],
)
def test_the_lock_parser_refuses_what_is_not_a_pin_or_a_hash_and_names_the_line(text: str, line: int) -> None:
    with pytest.raises(CacheError, match=f"line {line} is not a pin or a hash"):
        parse_lock(text)


def _make_tree(root: Path) -> Path:
    (root / "pkg").mkdir(parents=True)
    (root / "a.txt").write_text("alpha")
    (root / "pkg" / "b.py").write_text("beta")
    (root / "pkg" / "run.sh").write_text("#!/bin/sh\n")
    return root


def test_the_tree_hash_is_the_same_for_a_copy_and_changes_with_anything_a_runner_could_notice(tmp_path: Path) -> None:
    base = tree_digest(_make_tree(tmp_path / "base"))
    assert base.files == 3 and base.size == len("alpha") + len("beta") + len("#!/bin/sh\n")
    copy = _make_tree(tmp_path / "copy")
    assert tree_digest(copy) == base
    (copy / "a.txt").write_text("alphA")
    assert tree_digest(copy).sha256 != base.sha256 and tree_digest(copy).size == base.size
    (copy / "a.txt").write_text("alpha")
    (copy / "pkg" / "run.sh").chmod(0o755)
    assert tree_digest(copy).sha256 != base.sha256
    (copy / "pkg" / "run.sh").chmod(0o644)
    assert tree_digest(copy) == base
    (copy / "pkg" / "new.py").write_text("")
    assert tree_digest(copy).files == 4 and tree_digest(copy).sha256 != base.sha256
    (copy / "pkg" / "new.py").unlink()
    (copy / "pkg" / "b.py").rename(copy / "pkg" / "c.py")
    assert tree_digest(copy).sha256 != base.sha256


def test_a_symbolic_link_counts_by_its_target_and_is_not_followed(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path / "tree")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "big.bin").write_bytes(b"x" * 1000)
    (tree / "link").symlink_to("../outside")
    digest = tree_digest(tree)
    assert digest.files == 4 and digest.size < 100
    (tree / "link").unlink()
    (tree / "link").symlink_to("../elsewhere")
    assert tree_digest(tree).sha256 != digest.sha256


def _manifest(rig: Rig) -> Manifest:
    return manifest_from_json((rig.source.parent / MANIFEST_NAME).read_text(encoding="utf-8"))


def test_the_manifest_round_trips_and_says_what_was_built(rig: Rig) -> None:
    manifest = _manifest(rig)
    assert manifest.sha == rig.cache.sha and manifest.source == tree_digest(rig.source)
    archive = rig.source.parent / SOURCE_ARCHIVE
    assert manifest.source_archive == FileDigest(archive.stat().st_size, file_sha256(archive))
    assert sorted(wheel.name for wheel in manifest.wheels) == ["already-pkg", "lock-pkg", "sdist-pkg", "setuptools"]
    assert manifest_from_json(manifest_to_json(manifest)) == manifest
    assert json.loads(manifest_to_json(manifest))["schema"] == 1


def test_a_manifest_written_before_entries_had_a_script_or_an_archive_still_reads(rig: Rig) -> None:
    data = json.loads(manifest_to_json(_manifest(rig)))
    del data["bootstrap_sha256"], data["source_archive"]
    older = manifest_from_json(json.dumps(data))
    assert (older.bootstrap_sha256, older.source_archive) == ("", None)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not json", "the manifest cannot be read"),
        ("[]", "the manifest cannot be read"),
        ('{"schema": 2}', "has schema 2"),
        ('{"schema": 1, "sha": "x"}', "the manifest cannot be read: KeyError"),
    ],
)
def test_a_manifest_that_cannot_be_read_is_refused(text: str, message: str) -> None:
    with pytest.raises(CacheError, match=message):
        manifest_from_json(text)


def test_a_manifest_that_is_not_text_is_refused_by_the_check(rig: Rig, tmp_path: Path) -> None:
    cache = copy_cache(rig, tmp_path / "cache")
    assert check_cache(cache.root, cache.sha) == []
    (cache.root / cache.sha / MANIFEST_NAME).write_bytes(b"\xff\xfe")
    (problem,) = check_cache(cache.root, cache.sha)
    assert problem.startswith("the manifest cannot be read")


def test_an_entry_whose_manifest_names_another_commit_is_refused(rig: Rig) -> None:
    other = "0" * 40
    assert check_entry(rig.source.parent, rig.cache.sha) == []
    assert check_entry(rig.source.parent, other) == [
        f"the manifest of {rig.source.parent} is for commit {rig.cache.sha}, not {other}"
    ]


def test_the_checker_prints_the_first_few_problems_and_counts_the_rest(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("hyperloom.measure.runner.cache_layout._checked", lambda *_: [f"problem {n}" for n in range(8)])
    assert cache_layout.main(["/cache", "a" * 40]) == 1
    assert capsys.readouterr().out.splitlines() == [*(f"problem {n}" for n in range(5)), "... and 3 more"]


def test_the_checker_reports_a_cache_it_cannot_read_as_a_problem(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unreadable(*_: object) -> list[str]:
        raise PermissionError("denied")

    monkeypatch.setattr("hyperloom.measure.runner.cache_layout._checked", unreadable)
    assert cache_layout.main(["/cache", "a" * 40]) == 1
    assert "could not be read: denied" in capsys.readouterr().out


def test_the_checker_runs_as_a_script_read_from_stdin_and_prints_what_is_wrong(rig: Rig, tmp_path: Path) -> None:
    cache = copy_cache(rig, tmp_path / "cache")
    for wheel in (cache.root / cache.sha / "wheels").iterdir():
        wheel.unlink()
    script = Path(cache_layout.__file__).read_text(encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-", str(cache.root), cache.sha], input=script, capture_output=True, text=True, check=False
    )
    assert done.returncode == 1
    assert "missing wheel already_pkg-1.0-py3-none-any.whl for already-pkg==1.0" in done.stdout
    assert "missing wheel lock_pkg-1.0-py3-none-any.whl for lock-pkg==1.0" in done.stdout


def _edit_manifest(entry: Path, edit: Callable[[dict[str, dict[str, str]]], None]) -> None:
    manifest = json.loads((entry / MANIFEST_NAME).read_text())
    edit({wheel["name"]: wheel for wheel in manifest["wheels"]})
    (entry / MANIFEST_NAME).write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    ("edit", "named"),
    [
        (lambda wheels: wheels["lock-pkg"].update(sha256="0" * 64), "lock_pkg-1.0-py3-none-any.whl for lock-pkg==1.0"),
        (
            lambda wheels: wheels["sdist-pkg"].update(sdist_sha256="1" * 64),
            "sdist_pkg-1.0-py3-none-any.whl for sdist-pkg",
        ),
        (lambda wheels: wheels["sdist-pkg"].update(sdist_sha256=""), "sdist_pkg-1.0-py3-none-any.whl for sdist-pkg"),
    ],
)
def test_a_wheel_that_is_not_or_was_not_built_from_a_file_the_lock_lists_is_refused(
    rig: Rig, tmp_path: Path, edit: Callable[[dict[str, dict[str, str]]], None], named: str
) -> None:
    cache = copy_cache(rig, tmp_path / "cache")
    _edit_manifest(cache.root / cache.sha, edit)
    problems = check_cache(cache.root, cache.sha)
    assert [problem for problem in problems if named in problem and "a file the lock lists for it" in problem]


def test_a_wheel_the_lock_does_not_pin_or_a_pin_without_a_wheel_is_refused(rig: Rig, tmp_path: Path) -> None:
    cache = copy_cache(rig, tmp_path / "cache")
    _edit_manifest(cache.root / cache.sha, lambda wheels: wheels["lock-pkg"].update(name="other-pkg"))
    assert check_cache(cache.root, cache.sha) == [
        "no wheel for lock-pkg==1.0 in the manifest",
        "the manifest lists other-pkg==1.0, which the lock does not pin",
    ]


def test_the_requirements_pin_each_manifest_wheel_to_its_recorded_hash(rig: Rig) -> None:
    lines = requirements_text(_manifest(rig))
    for wheel in _manifest(rig).wheels:
        assert f"{wheel.name}=={wheel.version} \\\n    --hash=sha256:{wheel.sha256}\n" in lines
    assert len(parse_lock(lines)) == 4


def _staged(scratch: Path) -> list[str]:
    return sorted(path.name for path in scratch.iterdir()) if scratch.is_dir() else []


def test_the_checker_stages_the_source_wheels_and_requirements_only_for_a_cache_that_checks_out(
    rig: Rig, tmp_path: Path
) -> None:
    cache = copy_cache(rig, tmp_path / "cache")
    entry, scratch = cache.root / cache.sha, tmp_path / "scratch"
    assert cache_layout.main([str(cache.root), cache.sha, str(scratch)]) == 0
    assert _staged(scratch) == sorted([LOCAL_SOURCE_DIR, LOCAL_WHEELS_DIR, REQUIREMENTS_FILE])
    # The source is the archive unpacked, file for file the tree the builder recorded; the wheels are byte copies.
    assert tree_digest(scratch / LOCAL_SOURCE_DIR) == _manifest(rig).source
    assert {path.name: path.read_bytes() for path in (scratch / LOCAL_WHEELS_DIR).iterdir()} == {
        path.name: path.read_bytes() for path in (entry / "wheels").iterdir()
    }
    assert (scratch / REQUIREMENTS_FILE).read_text() == requirements_text(_manifest(rig))
    # A second check of a cache that no longer checks out leaves none of what the first one staged.
    (entry / "wheels" / "lock_pkg-1.0-py3-none-any.whl").unlink()
    assert cache_layout.main([str(cache.root), cache.sha, str(scratch)]) == 1
    assert _staged(scratch) == []


def test_a_manifest_swapped_after_it_was_read_does_not_reach_the_requirements(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The requirements pip enforces come from the manifest the check read, not from a second read of the file."""
    cache = copy_cache(rig, tmp_path / "cache")
    entry = cache.root / cache.sha
    checked = _manifest(rig)
    swapped = dataclasses.replace(
        checked, wheels=tuple(dataclasses.replace(wheel, sha256="0" * 64) for wheel in checked.wheels)
    )
    copy = cache_layout._copy_hashed

    def copy_then_swap(path: Path, sink: BinaryIO) -> FileDigest:
        (entry / MANIFEST_NAME).write_text(manifest_to_json(swapped), encoding="utf-8")
        return copy(path, sink)

    monkeypatch.setattr(cache_layout, "_copy_hashed", copy_then_swap)
    out = tmp_path / "scratch" / REQUIREMENTS_FILE
    assert cache_layout.main([str(cache.root), cache.sha, str(out.parent)]) == 0
    assert manifest_from_json((entry / MANIFEST_NAME).read_text(encoding="utf-8")) == swapped
    assert out.read_text(encoding="utf-8") == requirements_text(checked)
    assert "0" * 64 not in out.read_text(encoding="utf-8")


Member = tuple[tarfile.TarInfo, bytes]


def _members(raw: bytes) -> list[Member]:
    """The members of the tar archive ``raw``, each with its content (empty for what is not a regular file)."""
    members: list[Member] = []
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        for info in tar:
            handle = tar.extractfile(info) if info.isreg() else None
            members.append((info, handle.read() if handle else b""))
    return members


def _tar(members: Iterable[Member]) -> bytes:
    """A tar archive of ``members``; a regular file's size is its content's."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for info, content in members:
            info.size = len(content) if info.isreg() else 0
            tar.addfile(info, io.BytesIO(content) if info.isreg() else None)
    return out.getvalue()


def test_the_source_is_unpacked_from_the_bytes_that_were_hashed_not_from_the_cache_file_again(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``source.tar`` rewritten in the cache right after it was hashed, with an archive that is valid but another one:
    the check passes on the pinned manifest and what it unpacked is the archive it hashed."""
    cache = copy_cache(rig, tmp_path / "cache")
    entry, scratch = cache.root / cache.sha, tmp_path / "scratch"
    archive = entry / SOURCE_ARCHIVE
    pin = hashlib.sha256((entry / MANIFEST_NAME).read_bytes()).hexdigest()
    other = _tar(
        (info, b"raise SystemExit('the swapped runner')\n" if info.name == _RUNNER_MAIN else content)
        for info, content in _members(archive.read_bytes())
    )
    assert other != archive.read_bytes() and _RUNNER_MAIN in {info.name for info, _ in _members(other)}
    hash_and_copy = cache_layout._copy_hashed

    def hash_then_swap(path: Path, sink: BinaryIO) -> FileDigest:
        found = hash_and_copy(path, sink)
        if path.name == SOURCE_ARCHIVE:
            archive.write_bytes(other)
        return found

    monkeypatch.setattr(cache_layout, "_copy_hashed", hash_then_swap)
    assert stage_entry(cache.root, cache.sha, scratch, pin) == []
    assert archive.read_bytes() == other, "the cache's archive was not rewritten"
    assert tree_digest(scratch / LOCAL_SOURCE_DIR) == _manifest(rig).source
    assert (scratch / LOCAL_SOURCE_DIR / _RUNNER_MAIN).read_bytes() == (rig.source / _RUNNER_MAIN).read_bytes()


def _file(name: str) -> Member:
    info = tarfile.TarInfo(name)
    info.mode = 0o644
    return info, b"escaped\n"


def _link(name: str, target: str) -> Member:
    info = tarfile.TarInfo(name)
    info.type, info.linkname = tarfile.SYMTYPE, target
    return info, b""


@pytest.mark.parametrize(
    ("escape", "refused"),
    [
        (lambda outside: [_file("../../escaped")], True),
        (lambda outside: [_link("out", "../.."), _file("out/escaped")], True),
        (lambda outside: [_file(str(outside / "escaped"))], False),
    ],
    ids=["parent", "link-out", "absolute"],
)
def test_an_archive_member_that_would_land_outside_the_staged_source_never_does(
    rig: Rig, tmp_path: Path, escape: Callable[[Path], list[Member]], refused: bool
) -> None:
    """An archive the manifest records (no pin) whose extra member climbs out of ``hl-source``, through ``..`` or a
    link, is refused and nothing is staged; an absolute member name is staged inside ``hl-source`` (the data filter
    strips its leading ``/``). In no case is anything written outside the scratch directory."""
    cache = copy_cache(rig, tmp_path / "cache")
    entry, scratch = cache.root / cache.sha, tmp_path / "scratch"
    raw = _tar([*_members((entry / SOURCE_ARCHIVE).read_bytes()), *escape(tmp_path)])
    (entry / SOURCE_ARCHIVE).write_bytes(raw)
    manifest = json.loads((entry / MANIFEST_NAME).read_text())
    manifest["source_archive"] = {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    (entry / MANIFEST_NAME).write_text(json.dumps(manifest))
    problems = stage_entry(cache.root, cache.sha, scratch)
    assert not (tmp_path / "escaped").exists(), "a member was written outside the scratch directory"
    if refused:
        assert len(problems) == 1 and "outside the destination" in problems[0], problems
        assert _staged(scratch) == []
    else:
        assert problems == []
        assert (scratch / LOCAL_SOURCE_DIR / str(tmp_path / "escaped").lstrip("/")).read_bytes() == b"escaped\n"
