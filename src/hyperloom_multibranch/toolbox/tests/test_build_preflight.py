# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The host checks of native builds: each item that is not ready is refused by name, and a ready host passes."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

from hyperloom.measure.jit.native.config import BuildStoreConfigError
from hyperloom.measure.runner.native.recipes import RECIPE_FACTORIES
from hyperloom_multibranch.toolbox import build_preflight as preflight
from hyperloom_multibranch.toolbox.build_preflight import (
    ImageUpstream,
    PreflightSite,
    build_preflight,
    changeable_by_others,
    image_upstreams,
    signing_key_problem,
    tracelens_problem,
    unreachable,
)

#: Every directory above a test's own is taken as root's with mode 0755: what this suite controls is below.
_OPEN_MODE = 0o40755
_PIN = "614c597bc0c66cabf6a15152037324acc7386090"
_PACKAGE = "FakeTraceLensW206"


def stat_open_above(base: Path):
    """``os.stat`` below ``base``; a root-owned 0755 directory above it (pytest's own tmp root is 0700)."""

    def stat_fn(path: Path) -> os.stat_result:
        if Path(path) == base or base in Path(path).parents:
            return os.stat(path)
        return os.stat_result((_OPEN_MODE, 0, 0, 1, 0, 0, 0, 0, 0, 0))

    return stat_fn


def git_repo(path: Path, files: dict[str, str] | None = None) -> str:
    """A git checkout at ``path`` with one commit of ``files``; its commit."""
    path.mkdir(parents=True, exist_ok=True)
    for name, text in (files or {"README": "sglang\n"}).items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text)
    git = ["git", "-c", "safe.directory=*", "-C", str(path)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    identity = ["-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, *identity, "commit", "-q", "-m", "c"], check=True)
    return subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def sglang_dist(image_root: Path, direct_url: dict, name: str = "sglang") -> None:
    """An installed ``name`` in ``image_root`` whose ``direct_url.json`` is ``direct_url``."""
    info = image_root / f"{name}-0.5.0.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.5.0\n")
    (info / "direct_url.json").write_text(json.dumps(direct_url))


# --- what user 65534 can reach --------------------------------------------------------------------------------------


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "file").write_text("x")
    for directory in (tmp_path, tmp_path / "a", tmp_path / "a" / "b"):
        directory.chmod(0o755)
    (tmp_path / "a" / "b" / "file").chmod(0o644)
    return tmp_path


def test_a_tree_open_to_others_is_reachable(tree: Path) -> None:
    reach = partial(unreachable, stat_fn=stat_open_above(tree))

    assert reach(tree / "a" / "b") is None
    assert reach(tree / "a" / "b" / "file") is None


def test_a_private_directory_on_the_way_is_named_with_its_mode_and_owner(tree: Path) -> None:
    (tree / "a").chmod(0o700)

    problem = unreachable(tree / "a" / "b" / "file", stat_fn=stat_open_above(tree))

    assert problem is not None and problem.startswith(f"{tree / 'a'} (mode 0700, owner uid {os.geteuid()}")
    assert problem.endswith("does not let user 65534 pass through")


def test_a_file_others_cannot_read_or_run_is_named(tree: Path) -> None:
    target = tree / "a" / "b" / "file"
    target.chmod(0o640)
    assert unreachable(target, stat_fn=stat_open_above(tree)).endswith("does not let user 65534 read it")

    target.chmod(0o644)
    assert unreachable(target, stat_fn=stat_open_above(tree)) is None
    assert unreachable(target, run=True, stat_fn=stat_open_above(tree)).endswith("does not let user 65534 run it")


def test_what_user_65534_owns_is_judged_by_the_owner_bits(tree: Path) -> None:
    def owned_by_nobody(path: Path) -> os.stat_result:
        found = stat_open_above(tree)(path)
        if Path(path) != tree / "a":
            return found
        fields = list(found)[:10]
        fields[0], fields[4] = found.st_mode & ~0o077, 65534
        return os.stat_result(fields)

    assert unreachable(tree / "a" / "b", stat_fn=owned_by_nobody) is None


def test_a_link_into_a_private_directory_names_the_directory_it_resolves_to(tree: Path) -> None:
    (tree / "private").mkdir(mode=0o700)
    (tree / "private" / "python").write_text("")
    (tree / "a" / "python").symlink_to(tree / "private" / "python")

    problem = unreachable(tree / "a" / "python", run=True, stat_fn=stat_open_above(tree))

    assert problem is not None and problem.startswith(f"{tree / 'private'} (mode 0700")


def test_a_missing_path_is_said_to_be_missing(tree: Path) -> None:
    assert (
        unreachable(tree / "a" / "nothing", stat_fn=stat_open_above(tree)) == f"{tree / 'a' / 'nothing'} does not exist"
    )


# --- the signing key ------------------------------------------------------------------------------------------------


def test_a_signing_key_is_refused_naming_its_mode_or_its_owner(tmp_path: Path) -> None:
    key = tmp_path / "signing.pem"
    key.write_text("k")
    key.chmod(0o600)
    assert signing_key_problem(key, os.geteuid()) is None

    key.chmod(0o640)
    assert "mode 0640" in signing_key_problem(key, os.geteuid())

    key.chmod(0o600)
    other = os.geteuid() + 1
    problem = signing_key_problem(key, other)
    assert f"this user (uid {other})" in problem and f"owner uid {os.geteuid()}" in problem


# --- the image's upstream commits ----------------------------------------------------------------------------------


def _upstreams(image_root: Path, checkout: Path) -> list[ImageUpstream]:
    return image_upstreams(image_root, {"sgl_kernel": checkout})


def test_the_image_commit_is_the_one_its_installed_upstream_names(tmp_path: Path) -> None:
    sglang_dist(
        tmp_path / "site", {"url": "https://github.com/sgl-project/sglang.git", "vcs_info": {"commit_id": _PIN}}
    )
    sglang_dist(tmp_path / "site", {"vcs_info": {"commit_id": "4" * 40}}, name="vllm")

    assert _upstreams(tmp_path / "site", tmp_path / "none") == [
        ImageUpstream("sgl_kernel", _PIN, None),
        ImageUpstream("vllm", "4" * 40, None),
    ]


def test_the_image_commit_of_an_editable_sglang_is_the_head_it_was_installed_from(tmp_path: Path) -> None:
    head = git_repo(tmp_path / "sgl-workspace" / "sglang", {"python/pyproject.toml": "[project]\n"})
    url = (tmp_path / "sgl-workspace" / "sglang" / "python").as_uri()
    sglang_dist(tmp_path / "site", {"url": url, "dir_info": {"editable": True}})

    assert _upstreams(tmp_path / "site", tmp_path / "none") == [
        ImageUpstream("sgl_kernel", head, tmp_path / "sgl-workspace" / "sglang")
    ]


def test_without_an_installed_sglang_the_image_commit_is_its_checkout_head_or_unknown(tmp_path: Path) -> None:
    head = git_repo(tmp_path / "sglang")
    (tmp_path / "site").mkdir()

    assert _upstreams(tmp_path / "site", tmp_path / "sglang") == [
        ImageUpstream("sgl_kernel", head, tmp_path / "sglang")
    ]
    assert _upstreams(tmp_path / "site", tmp_path / "none") == []


def test_every_native_component_says_where_its_upstream_commit_is_read() -> None:
    assert set(preflight._DISTRIBUTIONS) == set(RECIPE_FACTORIES)


# --- TraceLens ------------------------------------------------------------------------------------------------------


@pytest.fixture
def fake_tracelens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    site = tmp_path / "tl-site"
    (site / _PACKAGE).mkdir(parents=True)
    (site / _PACKAGE / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(site))
    return site


def _installed(site: Path, version: str) -> None:
    info = site / f"{_PACKAGE}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {_PACKAGE}\nVersion: {version}\n")


def test_an_installed_tracelens_is_compared_by_the_commit_its_version_names(fake_tracelens: Path) -> None:
    _installed(fake_tracelens, "0.1.0.dev20261024+g614c597")
    assert tracelens_problem(_PIN, package=_PACKAGE) is None

    problem = tracelens_problem("647cbf27" + "0" * 32, package=_PACKAGE)
    assert problem == f"{_PACKAGE} at {fake_tracelens} is commit 614c597, not the pinned 647cbf27{'0' * 32}"


def test_a_tracelens_checkout_is_compared_by_its_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    head = git_repo(tmp_path / "TraceLens", {f"{_PACKAGE}/__init__.py": ""})
    monkeypatch.syspath_prepend(str(tmp_path / "TraceLens"))

    assert tracelens_problem(head, package=_PACKAGE) is None
    assert f"is commit {head}, not the pinned {_PIN}" in tracelens_problem(_PIN, package=_PACKAGE)


def test_a_tracelens_that_names_no_commit_must_import_what_the_toolbox_uses(fake_tracelens: Path) -> None:
    (fake_tracelens / _PACKAGE / "config_utils.py").write_text("")

    assert tracelens_problem(_PIN, package=_PACKAGE, users=(f"{_PACKAGE}.config_utils",)) is None
    problem = tracelens_problem(_PIN, package=_PACKAGE, users=(f"{_PACKAGE}.Agent.Analysis.utils.config_utils",))
    assert problem.startswith(f"{_PACKAGE} at {fake_tracelens} names no commit and {_PACKAGE}.Agent")


def test_a_tracelens_that_is_not_importable_is_named() -> None:
    assert tracelens_problem(_PIN, package="NoSuchTraceLensW206") == (
        f"NoSuchTraceLensW206 is not importable; install the pinned commit {_PIN}"
    )


# --- the whole preflight --------------------------------------------------------------------------------------------


@pytest.fixture
def ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PreflightSite:
    """A host that is ready: root, setpriv, the image's checkout, everything open to 65534 (this interpreter and tree
    are outside the test's directory, taken as open), the pinned TraceLens."""
    ref = git_repo(tmp_path / "sources" / "sgl_kernel" / "staged")
    (tmp_path / "sources" / "sgl_kernel" / "staged").rename(tmp_path / "sources" / "sgl_kernel" / ref)
    sglang_dist(
        tmp_path / "image", {"url": "https://github.com/sgl-project/sglang.git", "vcs_info": {"commit_id": ref}}
    )
    (tmp_path / "store" / "builds").mkdir(parents=True)
    (tmp_path / "trust.json").write_text("{}")
    (tmp_path / "signing.pem").write_text("")
    (tmp_path / "signing.pem").chmod(0o600)
    tmp_path.chmod(0o755)
    for top in (tmp_path / "sources", tmp_path / "image", tmp_path / "store"):
        for directory, _, _ in os.walk(top):
            Path(directory).chmod(0o755)
    monkeypatch.setattr(preflight, "geteuid", lambda: 0)
    monkeypatch.setattr(preflight, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(preflight, "tracelens_problem", lambda: None)
    monkeypatch.setattr(preflight, "unreachable", partial(unreachable, stat_fn=stat_open_above(tmp_path)))
    return PreflightSite(
        tmp_path / "store",
        tmp_path / "trust.json",
        tmp_path / "signing.pem",
        tmp_path / "sources",
        tmp_path / "image",
        sys.executable,
    )


def _refusal(site: PreflightSite) -> str:
    with pytest.raises(BuildStoreConfigError) as refused:
        build_preflight(site)
    return str(refused.value)


def test_a_ready_host_passes(ready: PreflightSite) -> None:
    build_preflight(ready)


@pytest.mark.parametrize(
    ("break_it", "says"),
    [
        (lambda m: m.setattr(preflight, "geteuid", lambda: 1000), "runs as uid 1000, not root"),
        (lambda m: m.setattr(preflight, "which", lambda name: None), "setpriv is not on PATH"),
        (lambda m: m.setattr(preflight, "tracelens_problem", lambda: "TraceLens at /x is commit 647cbf2"), "647cbf2"),
        (
            lambda m: m.setattr(preflight, "image_upstreams", lambda root: []),
            "the image's upstream commits cannot be read",
        ),
    ],
    ids=["not-root", "no-setpriv", "tracelens", "no-image-commit"],
)
def test_each_host_item_that_is_not_ready_is_refused_by_name(
    ready: PreflightSite, monkeypatch: pytest.MonkeyPatch, break_it, says: str
) -> None:
    break_it(monkeypatch)

    problem = _refusal(ready)

    assert problem.startswith("native builds are configured (HYPERLOOM_BUILD_STORE) and this host is not ready")
    assert says in problem


def test_sources_without_the_image_commit_are_refused_naming_what_they_have(
    ready: PreflightSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    (present,) = os.listdir(ready.sources / "sgl_kernel")
    monkeypatch.setattr(preflight, "image_upstreams", lambda root: [ImageUpstream("sgl_kernel", "1" * 40, None)])

    problem = _refusal(ready)

    assert f"has no checkout of the image's sgl_kernel commit {'1' * 40}" in problem
    assert "holds no checkout of what the image installs" in problem
    assert f"(it has [{present!r}])" in problem


def test_a_checkout_named_for_the_image_commit_but_at_another_is_refused(
    ready: PreflightSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    (present,) = os.listdir(ready.sources / "sgl_kernel")
    (ready.sources / "sgl_kernel" / present).rename(ready.sources / "sgl_kernel" / ("2" * 40))
    monkeypatch.setattr(preflight, "image_upstreams", lambda root: [ImageUpstream("sgl_kernel", "2" * 40, None)])

    problem = _refusal(ready)

    assert f"is at {present}, not at the image's commit {'2' * 40}" in problem


def test_a_path_65534_cannot_reach_is_refused_naming_the_item_and_the_directory(
    ready: PreflightSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready.image_root.chmod(0o700)
    (present,) = os.listdir(ready.sources / "sgl_kernel")
    (ready.sources / "sgl_kernel").chmod(0o711)
    (ready.sources / "sgl_kernel" / present).chmod(0o750)

    problem = _refusal(ready)

    assert (
        f"user 65534 cannot reach HYPERLOOM_BUILD_IMAGE_ROOT {ready.image_root}: {ready.image_root} (mode 0700"
        in problem
    )
    assert f"cannot reach the sgl_kernel checkout {ready.sources / 'sgl_kernel' / present}: " in problem
    assert "(mode 0750" in problem


def test_an_image_interpreter_not_on_path_is_refused(ready: PreflightSite, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight, "which", lambda name: "/usr/bin/setpriv" if name == "setpriv" else None)

    assert "HYPERLOOM_BUILD_PYTHON 'python3.99' is not found on PATH" in _refusal(replace(ready, python="python3.99"))


def test_the_toolbox_interpreter_and_tree_are_among_what_65534_must_reach(ready: PreflightSite) -> None:
    checked = {
        what: path for what, path, _ in preflight._reach_items(ready, [ready.sources / "sgl_kernel" / ("3" * 40)])
    }

    assert checked["the toolbox interpreter"] == Path(sys.executable)
    assert checked["the toolbox source tree"].name == "child.py" and checked["the toolbox source tree"].is_file()
    assert checked["HYPERLOOM_BUILD_IMAGE_ROOT"] == ready.image_root
    assert checked["the sgl_kernel checkout"] == ready.sources / "sgl_kernel" / ("3" * 40)


# --- what no other user may change ----------------------------------------------------------------------------------


def _checkout(site: PreflightSite) -> Path:
    (ref,) = os.listdir(site.sources / "sgl_kernel")
    return site.sources / "sgl_kernel" / ref


@pytest.mark.parametrize(
    ("loosen", "item", "names"),
    [
        (lambda s: s.store / "builds", "HYPERLOOM_BUILD_STORE", lambda s: s.store / "builds"),
        (lambda s: _checkout(s) / "README", "the sgl_kernel checkout", lambda s: _checkout(s) / "README"),
        (lambda s: s.trust, "HYPERLOOM_BUILD_TRUST", lambda s: s.trust),
        (lambda s: s.sources, "the sgl_kernel checkout", lambda s: s.sources),
        (lambda s: s.key.parent, "HYPERLOOM_BUILD_SIGNING_KEY", lambda s: s.key.parent),
    ],
    ids=["store-subdir", "checkout-file", "trust", "sources-above-checkout", "key-directory"],
)
@pytest.mark.parametrize("mode_bit", [0o020, 0o002], ids=["group", "others"])
def test_what_another_user_could_write_is_refused_naming_the_item_and_the_path(
    ready: PreflightSite, loosen, item: str, names, mode_bit: int
) -> None:
    target = loosen(ready)
    target.chmod(stat.S_IMODE(os.stat(target).st_mode) | mode_bit)

    problem = _refusal(ready)

    assert f"users other than uid {os.geteuid()} could change {item} " in problem
    assert f"{names(ready)} (mode " in problem and "can be written by its group or others" in problem


def test_a_sticky_directory_above_is_no_refusal_and_a_link_below_is_not_followed(ready: PreflightSite) -> None:
    ready.key.parent.chmod(0o1777)
    outside = ready.key.parent / "open"
    outside.write_text("")
    outside.chmod(0o666)
    (_checkout(ready) / "link").symlink_to(outside)

    build_preflight(ready)


def test_an_item_of_another_user_is_refused(ready: PreflightSite) -> None:
    """As root, a directory in the store is given away; otherwise the service is said to be another user than the one
    that owns the test's directories."""
    as_root = os.geteuid() == 0
    if as_root:
        os.chown(ready.store / "builds", 4242, 4242)
        assert changeable_by_others(ready.store, 0) is None
    service = 0 if as_root else os.geteuid() + 4242

    problem = changeable_by_others(ready.store, service, tree=True, files=False)

    assert problem is not None and problem.endswith("is another user's")
    assert not as_root or problem.startswith(f"{ready.store / 'builds'} (mode")


@pytest.mark.skipif(os.geteuid() != 0, reason="giving a link to another user needs root")
def test_a_link_of_another_user_on_the_way_is_refused_though_it_points_at_the_service_s_checkout(
    ready: PreflightSite,
) -> None:
    """Another user could replace the link after the check and redirect what the build later reads."""
    shared = ready.key.parent / "shared"
    shared.mkdir(mode=0o1777)
    shared.chmod(0o1777)
    link = shared / "sources"
    link.symlink_to(ready.sources)
    assert changeable_by_others(link, 0, tree=True) is None

    os.lchown(link, 4242, 4242)

    problem = changeable_by_others(link, 0, tree=True)
    assert problem is not None and problem.startswith(f"{link} (mode") and problem.endswith("is another user's")
    assert "could change the sgl_kernel checkout" in _refusal(replace(ready, sources=link))


def test_a_component_the_sources_have_no_directory_for_is_not_required(
    ready: PreflightSite, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An sglang image that also installs vLLM builds sgl-kernel without a vLLM checkout; a vLLM image the other way."""
    (present,) = os.listdir(ready.sources / "sgl_kernel")
    image = [ImageUpstream("sgl_kernel", present, None), ImageUpstream("vllm", "5" * 40, None)]
    monkeypatch.setattr(preflight, "image_upstreams", lambda root: image)
    build_preflight(ready)

    (ready.sources / "vllm" / ("6" * 40)).mkdir(parents=True)
    assert f"has no checkout of the image's vllm commit {'5' * 40}" in _refusal(ready)
