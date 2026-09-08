# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rtk installer refuses anything it did not pin, and is safe to re-run.

Nothing here touches the network: every test substitutes ``_download`` with a
local archive, which is also the point -- what is being checked is the handling
around the download (digest, archive shape, destination), not github.
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
from pathlib import Path

import pytest

from kernelforge import rtk_install


def _tarball(path: Path, entries: dict[str, bytes]) -> str:
    """Write a gzipped tar holding ``entries``; return its sha256."""
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def offline(monkeypatch, tmp_path):
    """Serve a caller-supplied archive in place of the real download.

    Returns a callable taking the archive's entries and returning its digest,
    which the test then pins (or deliberately does not) via ``_ASSETS``.
    """

    def _serve(entries: dict[str, bytes]) -> str:
        source = tmp_path / "served.tar.gz"
        digest = _tarball(source, entries)

        def _fake_download(url: str, target: Path) -> None:
            target.write_bytes(source.read_bytes())

        monkeypatch.setattr(rtk_install, "_download", _fake_download)
        return digest

    return _serve


def _pin(monkeypatch, digest: str) -> None:
    """Make every platform resolve to one asset with ``digest``."""
    monkeypatch.setattr(rtk_install, "_current_asset", lambda: rtk_install._Asset("served.tar.gz", digest))


def _never_present(monkeypatch) -> None:
    """No usable rtk on PATH, so the installer always does the work."""
    monkeypatch.setattr(rtk_install, "installed_version", lambda binary="rtk": "")


def test_a_pinned_archive_installs_and_is_executable(offline, monkeypatch, tmp_path):
    """The happy path: digest matches, the binary lands, and it can be run."""
    digest = offline({"rtk": b"#!/bin/sh\nexit 0\n"})
    _pin(monkeypatch, digest)
    _never_present(monkeypatch)

    path, outcome = rtk_install.install_rtk(tmp_path / "bin")

    assert outcome == "installed"
    assert path == tmp_path / "bin" / "rtk"
    assert path.read_bytes() == b"#!/bin/sh\nexit 0\n"
    assert path.stat().st_mode & 0o111, "installed binary must be executable"


def test_a_digest_that_does_not_match_is_refused(offline, monkeypatch, tmp_path):
    """The pin is the security boundary, so a mismatch installs nothing.

    This is the case the pin exists for: fetching the release's own
    ``checksums.txt`` would agree with whatever that release currently serves,
    and so would not catch it.
    """
    offline({"rtk": b"not the pinned build"})
    _pin(monkeypatch, "0" * 64)
    _never_present(monkeypatch)

    with pytest.raises(rtk_install.RtkInstallError, match="refusing to install"):
        rtk_install.install_rtk(tmp_path / "bin")
    assert not (tmp_path / "bin" / "rtk").exists()


def test_an_archive_without_the_binary_is_refused(offline, monkeypatch, tmp_path):
    """A well-hashed archive that does not carry ``rtk`` is still a failure."""
    digest = offline({"README": b"nothing useful here"})
    _pin(monkeypatch, digest)
    _never_present(monkeypatch)

    with pytest.raises(rtk_install.RtkInstallError, match="no 'rtk' binary"):
        rtk_install.install_rtk(tmp_path / "bin")


def test_the_archive_cannot_choose_where_anything_lands(offline, monkeypatch, tmp_path):
    """An entry named with a traversal is written to the destination, not through it.

    Members are matched on base name and copied out one at a time rather than
    with ``extractall``, so the path inside the archive never reaches the
    filesystem.
    """
    digest = offline({"../../escaped/rtk": b"payload"})
    _pin(monkeypatch, digest)
    _never_present(monkeypatch)

    path, _ = rtk_install.install_rtk(tmp_path / "bin")

    assert path == tmp_path / "bin" / "rtk"
    assert not (tmp_path / "escaped").exists()
    assert not (tmp_path.parent / "escaped").exists()


def test_the_pinned_version_already_on_path_downloads_nothing(monkeypatch, tmp_path):
    """Re-running the installer is a no-op, so install.sh can call it every time."""
    monkeypatch.setattr(rtk_install, "installed_version", lambda binary="rtk": rtk_install.RTK_VERSION)
    monkeypatch.setattr(rtk_install.shutil, "which", lambda name: str(tmp_path / "rtk"))

    def _refuse(url: str, target: Path) -> None:
        raise AssertionError("downloaded despite the pinned version already being present")

    monkeypatch.setattr(rtk_install, "_download", _refuse)

    path, outcome = rtk_install.install_rtk(tmp_path / "bin")

    assert outcome == "present"
    assert path == tmp_path / "rtk"


def test_force_downloads_over_a_matching_version(offline, monkeypatch, tmp_path):
    """``--force`` exists to replace a binary that reports the right version."""
    digest = offline({"rtk": b"fresh"})
    _pin(monkeypatch, digest)
    monkeypatch.setattr(rtk_install, "installed_version", lambda binary="rtk": rtk_install.RTK_VERSION)

    path, outcome = rtk_install.install_rtk(tmp_path / "bin", force=True)

    assert outcome == "installed"
    assert path.read_bytes() == b"fresh"


def test_a_destination_that_cannot_be_created_is_named_in_the_error(offline, monkeypatch, tmp_path):
    """install.sh turns the message into a warning, so it has to say what to fix."""
    digest = offline({"rtk": b"payload"})
    _pin(monkeypatch, digest)
    _never_present(monkeypatch)
    blocker = tmp_path / "bin"
    blocker.write_text("a file where the directory should go", encoding="utf-8")

    with pytest.raises(rtk_install.RtkInstallError, match="cannot create"):
        rtk_install.install_rtk(blocker / "nested")


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the write bit, so os.access always says yes")
def test_an_unwritable_destination_is_named_in_the_error(offline, monkeypatch, tmp_path):
    """A directory that exists but cannot be written to fails before downloading."""
    digest = offline({"rtk": b"payload"})
    _pin(monkeypatch, digest)
    _never_present(monkeypatch)
    target = tmp_path / "bin"
    target.mkdir()
    target.chmod(0o500)
    try:
        with pytest.raises(rtk_install.RtkInstallError, match="not writable"):
            rtk_install.install_rtk(target)
    finally:
        target.chmod(0o700)


def test_an_unsupported_platform_says_so_rather_than_guessing(monkeypatch):
    """No asset for this machine is a clear failure, not a wrong download."""
    monkeypatch.setattr(rtk_install.platform, "system", lambda: "Plan9")
    monkeypatch.setattr(rtk_install.platform, "machine", lambda: "mips")

    with pytest.raises(rtk_install.RtkInstallError, match="no build for plan9/mips"):
        rtk_install._current_asset()


def test_every_pinned_asset_carries_a_full_sha256() -> None:
    """A blank or truncated digest would silently accept any download."""
    for key, asset in rtk_install._ASSETS.items():
        assert len(asset.sha256) == 64, key
        assert set(asset.sha256) <= set("0123456789abcdef"), key


def test_the_default_destination_is_where_the_entry_point_lives() -> None:
    """Writing beside ``kernelforge`` is what makes the binary reachable on PATH."""
    assert rtk_install.default_destination() == Path(rtk_install.sysconfig.get_path("scripts"))


def test_the_quickstart_does_not_send_readers_to_the_wrong_package() -> None:
    """PyPI ``rtk`` is a Raspberry Pi GPIO library, not the token filter."""
    quickstart = Path(__file__).resolve().parents[3] / "docs" / "kernelforge" / "quickstart.md"
    if not quickstart.exists():  # pragma: no cover - docs are absent from a wheel install
        pytest.skip("docs tree not present in this layout")
    text = quickstart.read_text(encoding="utf-8")
    assert "kernelforge install-rtk" in text
    assert "`pip install rtk`" not in text.replace("Do **not** `pip install rtk`", "")
