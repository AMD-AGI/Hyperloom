# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``python -m hyperloom_multibranch.toolbox.build_setup``: a host's native builds set up from a fake image, idempotent,
never replacing a key, and saying which path user 65534 cannot reach."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from functools import partial
from pathlib import Path

import pytest

from hyperloom.measure.jit.native.config import trust_from_file
from hyperloom.measure.jit.native.trust import Signer
from hyperloom_multibranch.toolbox import build_preflight as preflight
from hyperloom_multibranch.toolbox.build_preflight import git_head, unreachable
from hyperloom_multibranch.toolbox.build_setup import BuildSetup, SetupError, main, set_up
from hyperloom_multibranch.toolbox.tests.test_build_preflight import git_repo, sglang_dist, stat_open_above

_KERNEL = "python/sglang/kernels/aot/csrc/elementwise/activation.cu"
_SIX = {
    "HYPERLOOM_BUILD_STORE",
    "HYPERLOOM_BUILD_TRUST",
    "HYPERLOOM_BUILD_SIGNING_KEY",
    "HYPERLOOM_BUILD_SOURCES",
    "HYPERLOOM_BUILD_IMAGE_ROOT",
    "HYPERLOOM_BUILD_GFX",
    "HYPERLOOM_BUILD_PYTHON",
}


@pytest.fixture
def image(tmp_path: Path) -> tuple[Path, str]:
    """A fake image: ``sglang`` installed editable from its checkout, and the image's interpreter beside it."""
    checkout = tmp_path / "sgl-workspace" / "sglang"
    ref = git_repo(checkout, {"python/pyproject.toml": "[project]\n", _KERNEL: "// kernel\n"})
    root = tmp_path / "opt" / "venv" / "lib" / "python3.10" / "site-packages"
    sglang_dist(root, {"url": (checkout / "python").as_uri(), "dir_info": {"editable": True}})
    (tmp_path / "opt" / "venv" / "bin").mkdir()
    (tmp_path / "opt" / "venv" / "bin" / "python3").write_text("")
    return root, ref


def _sourced(env_file: Path) -> dict[str, str]:
    """What the launcher's ``set -a; . build.env; set +a`` exports, read back from a real shell."""
    script = 'set -a; . "$1"; set +a; env -0'
    done = subprocess.run(["sh", "-c", script, "sh", str(env_file)], capture_output=True, check=True, env={})
    pairs = (entry.split("=", 1) for entry in done.stdout.decode().split("\0") if "=" in entry)
    return {name: value for name, value in pairs if name.startswith("HYPERLOOM_")}


def test_setup_from_an_image_makes_the_store_key_trust_checkout_and_environment(
    tmp_path: Path, image: tuple[Path, str]
) -> None:
    image_root, ref = image
    root = tmp_path / "build root"

    done = set_up(BuildSetup(root, image_root, "gfx950", "/opt/venv/bin/python3", builder_image="sha256:" + "a" * 64))

    assert [line.rsplit(": ", 1)[-1] for line in done[1:4]] == ["created", "created", "created"]
    key = root / "signing.pem"
    found = os.stat(key)
    assert (stat.S_IMODE(found.st_mode), found.st_uid) == (0o600, os.geteuid())
    trust_from_file(root / "trust.json").check(b"record", Signer.from_file(key).sign(b"record"))
    checkout = root / "sources" / "sgl_kernel" / ref
    assert git_head(checkout) == ref and (checkout / _KERNEL).read_text() == "// kernel\n"
    assert _sourced(root / "build.env") == {
        "HYPERLOOM_BUILD_STORE": str(root / "store"),
        "HYPERLOOM_BUILD_TRUST": str(root / "trust.json"),
        "HYPERLOOM_BUILD_SIGNING_KEY": str(key),
        "HYPERLOOM_BUILD_SOURCES": str(root / "sources"),
        "HYPERLOOM_BUILD_IMAGE_ROOT": str(image_root),
        "HYPERLOOM_BUILD_GFX": "gfx950",
        "HYPERLOOM_BUILD_PYTHON": "/opt/venv/bin/python3",
        "HYPERLOOM_BUILDER_IMAGE": "sha256:" + "a" * 64,
    }
    assert (root / "store").is_dir() and stat.S_IMODE(os.stat(root / "sources").st_mode) == 0o755


def test_setup_again_keeps_everything_and_never_replaces_the_key(tmp_path: Path, image: tuple[Path, str]) -> None:
    image_root, _ = image
    setup = BuildSetup(tmp_path / "root", image_root, "gfx942", "python3")
    set_up(setup)
    key, trust = (tmp_path / "root" / "signing.pem").read_bytes(), (tmp_path / "root" / "trust.json").read_bytes()

    again = set_up(setup)

    assert [line.rsplit(": ", 1)[-1] for line in again[1:4]] == ["kept", "kept", "kept"]
    assert (tmp_path / "root" / "signing.pem").read_bytes() == key
    assert (tmp_path / "root" / "trust.json").read_bytes() == trust
    assert set(_sourced(tmp_path / "root" / "build.env")) == _SIX


def test_an_existing_key_others_can_read_is_reported_and_left_as_it_is(tmp_path: Path, image: tuple[Path, str]) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "signing.pem").write_bytes(b"not mine to replace")
    (root / "signing.pem").chmod(0o644)

    with pytest.raises(SetupError, match=r"the existing signing key .* \(mode 0644, owner uid .*left as it is"):
        set_up(BuildSetup(root, image[0], "gfx950", "python3"))

    assert (root / "signing.pem").read_bytes() == b"not mine to replace"
    assert stat.S_IMODE(os.stat(root / "signing.pem").st_mode) == 0o644


def test_an_existing_trust_file_without_the_key_is_reported_and_left_as_it_is(
    tmp_path: Path, image: tuple[Path, str]
) -> None:
    setup = BuildSetup(tmp_path / "root", image[0], "gfx950", "python3")
    set_up(setup)
    trust = tmp_path / "root" / "trust.json"
    other = json.loads(trust.read_text())
    other["keys"][0]["public_key_pem"] = "-----BEGIN PUBLIC KEY-----\nother\n-----END PUBLIC KEY-----\n"
    trust.write_text(json.dumps(other))

    with pytest.raises(SetupError, match="does not list the public key of"):
        set_up(setup)

    assert json.loads(trust.read_text()) == other


def test_a_ref_and_repo_given_override_what_the_image_says(tmp_path: Path) -> None:
    ref = git_repo(tmp_path / "upstream", {_KERNEL: "// upstream\n"})
    (tmp_path / "bare-image").mkdir()
    setup = BuildSetup(
        tmp_path / "root", tmp_path / "bare-image", "gfx950", "python3", None, ref, str(tmp_path / "upstream")
    )

    set_up(setup)

    assert git_head(tmp_path / "root" / "sources" / "sgl_kernel" / ref) == ref


def test_an_image_that_names_no_sglang_commit_asks_for_one(tmp_path: Path) -> None:
    (tmp_path / "bare-image").mkdir()
    setup = BuildSetup(tmp_path / "root", tmp_path / "bare-image", "gfx950", "python3", sglang_repo=str(tmp_path / "x"))

    with pytest.raises(SetupError, match="pass --sglang-ref and --sglang-repo"):
        set_up(setup)


def test_a_checkout_at_another_commit_is_reported_and_left_as_it_is(tmp_path: Path, image: tuple[Path, str]) -> None:
    image_root, ref = image
    other = git_repo(tmp_path / "root" / "sources" / "sgl_kernel" / ref, {"other": "x"})

    with pytest.raises(SetupError, match=f"is at {other}, not at {ref}; it was left as it is"):
        set_up(BuildSetup(tmp_path / "root", image_root, "gfx950", "python3"))


def test_the_command_exits_1_naming_each_path_65534_cannot_reach(
    tmp_path: Path, image: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    image_root, _ = image
    image_root.parent.chmod(0o700)

    code = main([str(tmp_path / "root"), "--image-root", str(image_root), "--gfx", "gfx950"])

    out, err = capsys.readouterr()
    assert code == 1
    assert f"environment {tmp_path / 'root' / 'build.env'}" in out
    assert f"build_setup: user 65534 cannot reach HYPERLOOM_BUILD_IMAGE_ROOT {image_root}: " in err


def test_the_command_exits_0_when_65534_reaches_everything_and_reads_the_image_interpreter(
    tmp_path: Path, image: tuple[Path, str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    image_root, ref = image
    tmp_path.chmod(0o755)
    for top in (tmp_path / "opt", tmp_path / "sgl-workspace"):
        for directory, _, files in os.walk(top):
            Path(directory).chmod(0o755)
            for name in files:
                (Path(directory) / name).chmod(0o755)
    monkeypatch.setattr(preflight, "unreachable", partial(unreachable, stat_fn=stat_open_above(tmp_path)))

    code = main([str(tmp_path / "root"), "--image-root", str(image_root), "--gfx", "gfx950"])

    assert (code, capsys.readouterr().err) == (0, "")
    assert _sourced(tmp_path / "root" / "build.env")["HYPERLOOM_BUILD_PYTHON"] == str(tmp_path / "opt/venv/bin/python3")
    assert git_head(tmp_path / "root" / "sources" / "sgl_kernel" / ref) == ref


def test_each_upstream_the_image_installs_gets_its_checkout_where_the_image_keeps_one(
    tmp_path: Path, image: tuple[Path, str]
) -> None:
    image_root, ref = image
    vllm = git_repo(tmp_path / "app" / "vllm", {"csrc/ops.cu": "// ops\n"})
    sglang_dist(image_root, {"url": (tmp_path / "app" / "vllm").as_uri(), "dir_info": {"editable": True}}, name="vllm")

    done = set_up(BuildSetup(tmp_path / "root", image_root, "gfx950", "python3"))

    assert f"sgl_kernel checkout of {ref}: created" in done and f"vllm checkout of {vllm}: created" in done
    assert git_head(tmp_path / "root" / "sources" / "vllm" / vllm) == vllm


def test_an_upstream_the_image_keeps_no_checkout_of_is_said_to_be_not_made(
    tmp_path: Path, image: tuple[Path, str]
) -> None:
    image_root, ref = image
    sglang_dist(
        image_root,
        {"url": "https://github.com/vllm-project/vllm.git", "vcs_info": {"commit_id": "4" * 40}},
        name="vllm",
    )

    done = set_up(BuildSetup(tmp_path / "root", image_root, "gfx950", "python3"))

    assert f"sgl_kernel checkout of {ref}: created" in done
    assert f"vllm checkout of {'4' * 40}: not made: the image keeps no checkout to clone it from" in done
    assert not (tmp_path / "root" / "sources" / "vllm").exists()
