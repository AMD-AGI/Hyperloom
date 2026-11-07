# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""start.sh of the hosting layer: what it refuses on an image, how it keeps the specialists from the application's
credentials (an account, else the layer's bubblewrap), and what it hands app_bootstrap.sh."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest

from hyperloom.common.env_safety import missing_capabilities
from hyperloom.common.id_map import unmapped_ids
from hyperloom.common.tests._id_map_helpers import HOST_MAP, ONE_ID_MAP

SH = shutil.which("sh") or "sh"
START = Path(__file__).resolve().parent.parent / "start.sh"
_CERT = "-----BEGIN CERTIFICATE-----\nQUJD\n-----END CERTIFICATE-----\n"


def _no_account_here() -> str | None:
    """Why start.sh, run from this process (as root, by the stubbed ``id``), would make no account here: it reads its
    own maps (/proc/self/uid_map, gid_map), which must map uid and gid 2000, and its capability bounding set
    (/proc/self/status), which must hold CAP_KILL, CAP_SETGID and CAP_SETUID; ``None`` when it would make one."""
    unmapped = unmapped_ids((2000,), (2000,))
    if unmapped is not None:
        return unmapped
    status = Path("/proc/self/status").read_text(encoding="ascii").splitlines()
    bounding = next((int(line.split()[1], 16) for line in status if line.startswith("CapBnd:")), None)
    missing = [] if bounding is None else missing_capabilities(bounding)
    return f"the bounding set lacks {', '.join(missing)}" if missing else None


_NO_ACCOUNT_HERE = _no_account_here()
needs_account_here = pytest.mark.skipif(
    _NO_ACCOUNT_HERE is not None, reason=f"start.sh makes no account on this host: {_NO_ACCOUNT_HERE}"
)
#: The functions of start.sh that judge the user namespace maps, and the die their refusals go through.
_MAP_FUNCTIONS = ("die", "decimal", "in_id_map", "id_mapped")
#: The tail of the line start.sh says where it makes no account and the layer's bubblewrap confines the specialists.
_NO_ACCOUNT = "; no account is made, and the specialists run under the layer's bubblewrap ({layer}/bin/bwrap)"
#: The tail of its refusal where the layer has no bubblewrap either.
_NEITHER = "; and the layer has no bubblewrap ({layer}/bin/bwrap) to confine them instead, so the specialists could not"


def _tool(path: Path, name: str, body: str = "exit 0") -> None:
    path.mkdir(exist_ok=True)
    (path / name).write_text(f"#!/bin/sh\n{body}\n")
    (path / name).chmod(0o755)


@pytest.fixture
def layer(tmp_path) -> Path:
    """A layer directory, as the layer image ships it (with ``bin/bwrap``), whose bootstrap records what it was started
    with. The bwrap here is a stub that records a call: start.sh must never run it."""
    root = tmp_path / "layer"
    root.mkdir()
    shutil.copy(START, root / "start.sh")
    _tool(
        root,
        "app_bootstrap.sh",
        'echo bootstrap >>"$CALLS"; echo "PATH=$PATH" >"$CALLS.boot"; '
        'env | grep -E "^(SSL_CERT_FILE|NODE_EXTRA_CA_CERTS|HYPERLOOM_SPECIALIST_USER|HYPERLOOM_HOST_LAYER)=" '
        '>>"$CALLS.boot" || true',
    )
    _tool(root / "bin", "claude", "echo 1.0.0")
    _tool(root / "bin", "bwrap", 'echo "bwrap $*" >>"$CALLS"')
    return root


@pytest.fixture
def image(tmp_path) -> Path:
    """The tools of an image as stubs: root, git, setpriv, bash, mount and the account tools, each recording its call.
    The real mount must never run in a test: it would change the propagation of the host's root mount."""
    tools = tmp_path / "tools"
    for name in ("git", "setpriv"):
        _tool(tools, name)
    _tool(tools, "mount", 'echo "mount $*" >>"$CALLS"')
    _tool(tools, "bash", 'exec /bin/bash "$@"')
    _tool(tools, "id", "echo 0")
    _tool(tools, "getent", 'echo "getent $*" >>"$CALLS"; exit 2')
    for name in ("groupadd", "useradd"):
        _tool(tools, name, f'echo "{name} $*" >>"$CALLS"')
    return tools


def _bare_path(tmp_path, image, *names) -> str:
    """A PATH of the image's stubs and the named real tools only, so a tool the image lacks cannot be found."""
    essentials = tmp_path / "essentials"
    essentials.mkdir(exist_ok=True)
    for name in names:
        found = shutil.which(name)
        if found and not (essentials / name).exists():
            (essentials / name).symlink_to(found)
    return f"{image}:{essentials}"


def _boot(tmp_path) -> list[str]:
    """What the bootstrap was started with: ``PATH=...`` first, then the variables it records that were set."""
    return (tmp_path / "calls.boot").read_text().splitlines()


def _run(tmp_path, layer, image, *sh_options, prefix=(), **env):
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    state.chmod(0o700)  # private to root, as shared storage may be
    full = {
        "PATH": f"{image}:/usr/bin:/bin",
        "CALLS": str(tmp_path / "calls"),
        "HYPERLOOM_STATE_DIR": str(state),
        "TMPDIR": str(tmp_path / "local-tmp"),
        **env,
    }
    Path(full["TMPDIR"]).mkdir(exist_ok=True)
    done = subprocess.run(
        [*prefix, SH, *sh_options, str(layer / "start.sh")], capture_output=True, text=True, env=full, timeout=60
    )
    calls = tmp_path / "calls"
    return done, (calls.read_text().splitlines() if calls.exists() else [])


#: The real tools a run of start.sh needs past its own checks, for a PATH that holds nothing else (:func:`_bare_path`).
_ESSENTIALS = ("cat", "cut", "dirname", "env", "grep", "mkdir", "mv", "chmod")


@pytest.mark.parametrize("missing", ["git", "bash"])
def test_an_image_without_git_or_bash_is_refused_by_name(tmp_path, layer, image, missing):
    (image / missing).unlink()
    # Keep the real tool out of reach: only the stub dir and a dir with the bare essentials are searched.
    done, calls = _run(tmp_path, layer, image, PATH=_bare_path(tmp_path, image, *_ESSENTIALS))
    assert done.returncode == 1 and f"has no {missing}" in done.stderr, done.stderr
    assert not calls


def test_an_image_without_setpriv_makes_no_account_and_leaves_the_specialists_to_the_layers_bubblewrap(
    tmp_path, layer, image
):
    """setpriv starts a specialist as its account, so without it there is none: the bubblewrap tier, on any host."""
    (image / "setpriv").unlink()
    done, calls = _run(
        tmp_path,
        layer,
        image,
        PATH=_bare_path(tmp_path, image, *_ESSENTIALS),
        HYPERLOOM_SPECIALIST_USER="hl-specialist",
    )
    assert done.returncode == 0, done.stderr
    said = "the image has no setpriv (util-linux), which starts the specialists as their account"
    assert done.stderr.splitlines() == [f"hosted layer: {said}" + _NO_ACCOUNT.format(layer=layer)]
    assert calls == ["mount --make-rprivate /", "bootstrap"]  # no account looked up or made, bwrap never run
    boot = _boot(tmp_path)
    assert f"HYPERLOOM_HOST_LAYER={layer}" in boot
    assert not [line for line in boot if line.startswith("HYPERLOOM_SPECIALIST_USER=")]


def test_without_the_layers_bubblewrap_an_image_without_setpriv_is_refused_naming_both(tmp_path, layer, image):
    (image / "setpriv").unlink()
    (layer / "bin" / "bwrap").unlink()
    done, calls = _run(tmp_path, layer, image, PATH=_bare_path(tmp_path, image, *_ESSENTIALS))
    assert done.returncode == 1 and not calls, done.stderr
    assert done.stderr.startswith("hosted layer: the image has no setpriv (util-linux), "), done.stderr
    assert _NEITHER.format(layer=layer) in done.stderr


def test_not_root_makes_no_account_and_leaves_the_specialists_to_the_layers_bubblewrap(tmp_path, layer, image):
    _tool(image, "id", "echo 1000")
    done, calls = _run(tmp_path, layer, image)
    assert done.returncode == 0, done.stderr
    said = "this start runs as uid 1000, not root, so it cannot make the specialist account"
    assert done.stderr.splitlines()[0] == f"hosted layer: {said}" + _NO_ACCOUNT.format(layer=layer)
    assert not [c for c in calls if c.startswith(("getent", "groupadd", "useradd"))]
    assert calls[-1] == "bootstrap"


def test_not_root_without_the_layers_bubblewrap_is_refused(tmp_path, layer, image):
    _tool(image, "id", "echo 1000")
    (layer / "bin" / "bwrap").unlink()
    done, calls = _run(tmp_path, layer, image)
    assert done.returncode == 1 and not calls, done.stderr
    assert "this start runs as uid 1000, not root" in done.stderr and _NEITHER.format(layer=layer) in done.stderr


@needs_account_here
def test_a_uid_taken_by_another_name_is_refused(tmp_path, layer, image):
    _tool(
        image,
        "getent",
        'case "$1" in passwd) [ "$2" = 2000 ] && echo "someone:x:2000:2000::/h:/bin/sh" && exit 0;; esac; exit 2',
    )
    done, _ = _run(tmp_path, layer, image)
    assert done.returncode == 1 and "uid 2000 is taken by the account someone" in done.stderr


@needs_account_here
def test_a_name_with_another_uid_is_refused(tmp_path, layer, image):
    _tool(
        image,
        "getent",
        'case "$1 $2" in "passwd hl-specialist") echo "hl-specialist:x:999:999::/h:/bin/sh"; exit 0;; esac; exit 2',
    )
    done, _ = _run(tmp_path, layer, image)
    assert done.returncode == 1 and "exists with a uid other than 2000" in done.stderr


# ---- the user namespace maps: the functions of start.sh that judge them, run without the rest of the script ----


def _functions(*names: str) -> str:
    """These functions' definitions as start.sh has them, each from ``name() {`` to the ``}`` that closes it at the
    start of a line, so a test can source them without running the script."""
    lines = START.read_text().splitlines()
    defined: list[str] = []
    for name in names:
        first = lines.index(f"{name}() {{")
        defined.extend(lines[first : lines.index("}", first) + 1])
    return "\n".join(defined) + "\n"


def _sourced(
    tmp_path: Path, call: str, *args: str, functions: tuple[str, ...] = _MAP_FUNCTIONS
) -> subprocess.CompletedProcess[str]:
    """Run ``call`` (``args`` are its ``$1``...) under start.sh's ``set -eu``, with only ``functions`` sourced (its map
    functions by default)."""
    defined = tmp_path / "functions.sh"
    defined.write_text(_functions(*functions))
    script = f'set -eu\n. "{defined}"\nno_account=""\n{call}\n'
    return subprocess.run([SH, "-c", script, "sh", *args], capture_output=True, text=True, timeout=60)


def _maps(tmp_path: Path, uid_map: str | None, gid_map: str | None) -> tuple[Path, Path]:
    """A uid map and a gid map with these contents (``None``: no such file); returns their paths."""
    paths = (tmp_path / "uid_map", tmp_path / "gid_map")
    for path, text in zip(paths, (uid_map, gid_map), strict=True):
        if text is not None:
            path.write_text(text)
    return paths


@pytest.mark.parametrize(
    ("text", "taken"),
    [
        ("0", True),
        ("2000", True),
        ("4294967295", True),
        ("", False),
        ("-1", False),
        ("+1", False),
        ("01", False),
        ("1a", False),
        ("1 2", False),
    ],
)
def test_decimal_takes_only_a_number_as_the_kernel_writes_it(tmp_path, text, taken):
    done = _sourced(tmp_path, 'if decimal "$1"; then echo yes; else echo no; fi', text)
    assert done.stdout == ("yes\n" if taken else "no\n"), done.stderr


@pytest.mark.parametrize(
    ("text", "answer"),
    [
        (HOST_MAP, "0|0 0 4294967295|"),
        (ONE_ID_MAP, "1|0 1234 1|"),
        ("0 1234 1\n1 100000 1999\n2000 300000 1\n", "0|0 1234 1; 1 100000 1999; 2000 300000 1|"),
        ("0 0 1\n1 100000 1999\n", "1|0 0 1; 1 100000 1999|"),
        ("1 100000 2000\n", "0|1 100000 2000|"),
        ("2001 0 5\n", "1|2001 0 5|"),
        ("1990 0 11", "0|1990 0 11|"),
        ("", "1||"),
        (None, "2||it cannot be read"),
        ("0 1234\n", "2||the line '0 1234' is not 'inside outside count'"),
        ("0 -1 5\n", "2||the line '0 -1 5' is not 'inside outside count'"),
        ("0 01 5\n", "2||the line '0 01 5' is not 'inside outside count'"),
        ("0 0 x\n", "2||the line '0 0 x' is not 'inside outside count'"),
        ("0 0 4294967295 7\n", "2||the line '0 0 4294967295 7' is not 'inside outside count'"),
        ("1990 0 11\n\n", "2|1990 0 11|the line '' is not 'inside outside count'"),
    ],
    ids=[
        "host",
        "one-id",
        "ranges-covering",
        "ranges-ending-at-1999",
        "range-ending-at-2000",
        "range-starting-at-2001",
        "no-final-newline",
        "empty",
        "missing",
        "two-fields",
        "signed",
        "leading-zero",
        "not-a-number",
        "four-fields",
        "blank-line",
    ],
)
def test_in_id_map_says_whether_2000_is_mapped_or_why_the_map_cannot_be_read(tmp_path, text, answer):
    """0: mapped, 1: not, 2: the map cannot be read; then the map in short, and why it cannot be read."""
    path = tmp_path / "map"
    if text is not None:
        path.write_text(text)
    done = _sourced(tmp_path, 'rc=0; in_id_map 2000 "$1" || rc=$?; echo "$rc|$map_text|$map_error"', str(path))
    assert (done.returncode, done.stdout) == (0, answer + "\n"), done.stderr


_MAPPED = 'if id_mapped 2000 "$1" "$2"; then echo mapped; else echo "not: $no_account"; fi'


@pytest.mark.parametrize(
    ("uid_map", "gid_map", "kind", "quoted"),
    [
        (ONE_ID_MAP, ONE_ID_MAP, "uid", "0 1234 1"),
        (HOST_MAP, ONE_ID_MAP, "gid", "0 1234 1"),
        ("0 0 1\n1 100000 1999\n", HOST_MAP, "uid", "0 0 1; 1 100000 1999"),
        ("", HOST_MAP, "uid", "empty"),
    ],
    ids=["one-id", "gid-only", "ranges-ending-below-2000", "empty"],
)
def test_an_id_this_user_namespace_does_not_map_is_not_mapped_naming_the_map(tmp_path, uid_map, gid_map, kind, quoted):
    paths = _maps(tmp_path, uid_map, gid_map)
    done = _sourced(tmp_path, _MAPPED, *map(str, paths))
    assert (done.returncode, done.stderr) == (0, ""), done.stderr
    path = paths[0] if kind == "uid" else paths[1]
    assert done.stdout.startswith(
        f"not: {kind} 2000 is not mapped in this container's user namespace ({path}: '{quoted}'): "
    ), done.stdout
    assert "EINVAL" in done.stdout and "fakeroot" in done.stdout


def test_maps_of_several_ranges_that_cover_2000_pass(tmp_path):
    paths = _maps(tmp_path, "0 1234 1\n1 100000 1999\n2000 300000 1\n", "0 1234 1\n1000 5000 1001\n")
    done = _sourced(tmp_path, _MAPPED, *map(str, paths))
    assert (done.returncode, done.stdout, done.stderr) == (0, "mapped\n", "")


@pytest.mark.parametrize(
    ("uid_map", "gid_map", "kind", "says"),
    [
        (None, HOST_MAP, "uid", "it cannot be read"),
        (HOST_MAP, None, "gid", "it cannot be read"),
        ("0 1234\n", HOST_MAP, "uid", "the line '0 1234' is not 'inside outside count'"),
        ("0 -1 5\n", HOST_MAP, "uid", "the line '0 -1 5' is not 'inside outside count'"),
        ("0 0 4294967295 7\n", HOST_MAP, "uid", "the line '0 0 4294967295 7' is not 'inside outside count'"),
    ],
    ids=["uid-map-missing", "gid-map-missing", "two-fields", "signed", "four-fields"],
)
def test_a_map_that_cannot_be_read_is_refused_naming_the_file(tmp_path, uid_map, gid_map, kind, says):
    paths = _maps(tmp_path, uid_map, gid_map)
    done = _sourced(tmp_path, _MAPPED, *map(str, paths))
    assert (done.returncode, done.stdout) == (1, ""), done.stderr
    path = paths[0] if kind == "uid" else paths[1]
    assert done.stderr.startswith(f"hosted layer: cannot read {path} as a user namespace map ({says})"), done.stderr


#: A process status file as the kernel writes it, with its capability bounding set left to fill in.
_STATUS = (
    "Name:\tsh\nUid:\t0\t0\t0\t0\nCapInh:\t0000000000000000\nCapPrm:\t000001ffffffffff\n"
    "CapEff:\t{effective}\nCapBnd:\t{bounding}\nCapAmb:\t0000000000000000\nNoNewPrivs:\t0\n"
)
_HELD = 'if launch_capabilities "$1"; then echo held; else echo "not: $no_account"; fi'


def _capable(tmp_path: Path, text: str | None) -> subprocess.CompletedProcess[str]:
    """``launch_capabilities`` of start.sh on a status file with this text (``None``: no such file)."""
    status = tmp_path / "status"
    if text is not None:
        status.write_text(text)
    return _sourced(tmp_path, _HELD, str(status), functions=("launch_capabilities",))


@pytest.mark.parametrize(
    ("bounding", "missing"),
    [
        ("000001ffffffffff", None),
        ("00000000000000e0", None),
        ("0000000000000000", "CAP_KILL, CAP_SETGID, CAP_SETUID"),
        ("000001ffffffffc0", "CAP_KILL"),
        ("000001ffffffffa0", "CAP_SETGID"),
        ("000001ffffffff60", "CAP_SETUID"),
        ("000001ffffffff5f", "CAP_KILL, CAP_SETUID"),
    ],
    ids=["all", "only-the-three", "none", "no-kill", "no-setgid", "no-setuid", "a-higher-digit-is-not-read"],
)
def test_the_launch_capabilities_are_read_from_the_bounding_set_and_each_missing_one_is_named(
    tmp_path, bounding, missing
):
    """The bounding set, not the effective set, of the shell: root's effective set once it execs the application."""
    done = _capable(tmp_path, _STATUS.format(effective="000001ffffffffff", bounding=bounding))
    said = f"not: root here lacks {missing} ({tmp_path / 'status'}: CapBnd {bounding}), so it cannot start a specialist"
    assert (done.returncode, done.stderr) == (0, ""), done.stderr
    assert done.stdout == "held\n" if missing is None else done.stdout.startswith(said), done.stdout


def test_an_effective_set_without_them_does_not_count_when_the_bounding_set_holds_them(tmp_path):
    done = _capable(tmp_path, _STATUS.format(effective="0000000000000000", bounding="000001ffffffffff"))
    assert (done.returncode, done.stdout, done.stderr) == (0, "held\n", "")


@pytest.mark.parametrize(
    "text",
    [None, "Name:\tsh\nCapEff:\t0000000000000000\n", "CapBnd:\tzz\n", "CapBnd:\t0\n", "CapBnd:\n"],
    ids=["missing", "no-bounding-line", "not-hexadecimal", "one-digit", "empty"],
)
def test_a_status_that_does_not_say_counts_as_holding_them(tmp_path, text):
    """As root has them unless they were dropped; the application checks its own before it starts an account's."""
    assert _capable(tmp_path, text).stdout == "held\n"


def test_a_status_without_a_final_newline_is_still_read(tmp_path):
    done = _capable(tmp_path, "Name:\tsh\nCapBnd:\t0000000000000000")
    assert done.stdout.startswith("not: root here lacks CAP_KILL, CAP_SETGID, CAP_SETUID"), done.stdout


def _one_id_namespace() -> list[str] | None:
    """The command prefix of a new user namespace that maps one id (root, onto this process's own), or None here."""
    unshare = shutil.which("unshare")
    if unshare is None:
        return None
    prefix = [unshare, "--user", "--map-root-user", "--"]
    probe = subprocess.run([*prefix, "cat", "/proc/self/uid_map"], capture_output=True, text=True, check=False)
    return prefix if probe.returncode == 0 and len(probe.stdout.split()) == 3 else None


def _in_one_id_namespace(tmp_path: Path, layer: Path, image: Path) -> subprocess.CompletedProcess[str]:
    """start.sh, traced (``sh -x``), as the root of a real user namespace that maps only that root, as on a node
    whose containers get one id; the manifest's HYPERLOOM_SPECIALIST_USER is set."""
    env = {
        "PATH": f"{image}:/usr/bin:/bin",
        "CALLS": str(tmp_path / "calls"),
        "TMPDIR": str(tmp_path),
        "HYPERLOOM_SPECIALIST_USER": "hl-specialist",
    }
    return subprocess.run(
        [*_one_id_namespace(), SH, "-x", str(layer / "start.sh")], capture_output=True, text=True, env=env, timeout=60
    )


_ONE_ID_SAID = "hosted layer: uid 2000 is not mapped in this container's user namespace (/proc/self/uid_map: '0 "
needs_a_user_namespace = pytest.mark.skipif(
    _one_id_namespace() is None, reason="this host cannot make a user namespace (unshare --user)"
)


@needs_a_user_namespace
@pytest.mark.parametrize("existing", [False, True], ids=["new-account", "existing-account"])
def test_in_a_real_one_id_user_namespace_no_account_is_made_and_the_root_mount_is_made_private_for_bubblewrap(
    tmp_path, layer, image, existing
):
    """The kernel's own map says uid 2000 cannot be switched to: the start goes on without an account, the
    application's isolation is left to the layer's bubblewrap, and the root mount is made private after that
    decision, right before the bootstrap (bwrap cannot bind from a root mount marked unbindable)."""
    if existing:
        _tool(image, "getent", 'echo "getent $*" >>"$CALLS"; echo "hl-specialist:x:2000:2000::/h:/bin/sh"')
    done = _in_one_id_namespace(tmp_path, layer, image)
    assert done.returncode == 0, done.stderr
    lines = done.stderr.splitlines()
    said = [line for line in lines if line.startswith("hosted layer: ")]
    assert len(said) == 1 and said[0].startswith(_ONE_ID_SAID), done.stderr
    assert "EINVAL" in said[0] and said[0].endswith(_NO_ACCOUNT.format(layer=layer)), said
    trace = [line.lstrip("+ ") for line in lines if line.startswith("+")]
    private = "mount --make-rprivate /"
    assert private in trace, done.stderr
    assert trace.index("unset HYPERLOOM_SPECIALIST_USER") < trace.index(private), done.stderr
    assert trace[-1] == f"exec {layer}/app_bootstrap.sh", done.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    assert calls == [private, "bootstrap"], "no account may be looked up or made, and bwrap is the app's to run"
    boot = _boot(tmp_path)
    assert f"HYPERLOOM_HOST_LAYER={layer}" in boot
    assert not [line for line in boot if line.startswith("HYPERLOOM_SPECIALIST_USER=")]


@needs_a_user_namespace
def test_in_a_real_one_id_user_namespace_without_the_layers_bubblewrap_the_start_is_refused(tmp_path, layer, image):
    (layer / "bin" / "bwrap").unlink()
    done = _in_one_id_namespace(tmp_path, layer, image)
    assert done.returncode == 1, done.stderr
    said = [line for line in done.stderr.splitlines() if line.startswith("hosted layer: ")]
    assert len(said) == 1 and said[0].startswith(_ONE_ID_SAID), done.stderr
    assert "EINVAL" in said[0] and "fakeroot" in said[0] and _NEITHER.format(layer=layer) in said[0]
    assert not (tmp_path / "calls").exists(), "neither the account nor its group may be looked up or made"


def _without_launch_capabilities() -> list[str] | None:
    """The command prefix that starts a command with CAP_KILL, CAP_SETGID and CAP_SETUID dropped from its bounding
    set, as a container runtime drops them; None where this process cannot (it takes root and setpriv)."""
    setpriv = shutil.which("setpriv")
    if setpriv is None or os.geteuid() != 0:
        return None
    prefix = [setpriv, "--bounding-set=-kill,-setgid,-setuid", "--"]
    probe = subprocess.run([*prefix, "cat", "/proc/self/status"], capture_output=True, text=True, check=False)
    bounding = [line.split()[1] for line in probe.stdout.splitlines() if line.startswith("CapBnd:")]
    if probe.returncode != 0 or not bounding:
        return None
    return prefix if missing_capabilities(int(bounding[0], 16)) == ["CAP_KILL", "CAP_SETGID", "CAP_SETUID"] else None


@needs_account_here
@pytest.mark.skipif(_without_launch_capabilities() is None, reason="needs root and setpriv to drop capabilities")
def test_root_without_the_capabilities_to_switch_makes_no_account_and_leaves_the_specialists_to_bubblewrap(
    tmp_path, layer, image
):
    """On maps that hold uid and gid 2000, as a root whose runtime dropped CAP_KILL, CAP_SETGID and CAP_SETUID."""
    done, calls = _run(tmp_path, layer, image, prefix=_without_launch_capabilities())
    assert done.returncode == 0, done.stderr
    said = done.stderr.splitlines()
    assert len(said) == 1 and said[0].startswith("hosted layer: root here lacks CAP_KILL, CAP_SETGID, CAP_SETUID "), (
        said
    )
    assert said[0].endswith(_NO_ACCOUNT.format(layer=layer)), said
    assert calls == ["mount --make-rprivate /", "bootstrap"]


@needs_account_here
def test_the_script_checks_its_own_kernel_maps_and_capabilities_before_it_looks_up_the_account(tmp_path, layer, image):
    """The order the real-namespace test above proves, on a host that cannot make a user namespace (which skips that
    test): the script's own trace (``sh -x``) on this host's maps shows ``id_mapped`` and ``launch_capabilities``,
    whose answers are tested above, called on /proc/self's maps and status, and reading both maps, before the first
    account lookup."""
    done, calls = _run(tmp_path, layer, image, "-x")
    assert done.returncode == 0, done.stderr
    trace = [line.lstrip("+ ") for line in done.stderr.splitlines() if line.startswith("+")]
    lookups = [i for i, line in enumerate(trace) if line.startswith("getent ")]
    assert lookups, done.stderr
    before = trace[: lookups[0]]
    for check in ("id_mapped 2000 /proc/self/uid_map /proc/self/gid_map", "launch_capabilities /proc/self/status"):
        assert check in before, done.stderr
    mapped = trace.index("id_mapped 2000 /proc/self/uid_map /proc/self/gid_map")
    for kind in ("uid", "gid"):
        assert f"in_id_map 2000 /proc/self/{kind}_map" in trace[mapped : lookups[0]], done.stderr
    assert calls and calls[0].startswith("getent "), calls


def test_a_layer_without_the_cli_is_refused(tmp_path, layer, image):
    (layer / "bin" / "claude").unlink()
    done, calls = _run(tmp_path, layer, image)
    assert done.returncode == 1 and "no claude CLI" in done.stderr
    assert not [c for c in calls if c.startswith("mount ")], "the root mount is touched only after every check"


@needs_account_here
def test_the_account_is_created_and_the_bootstrap_gets_the_cli_and_the_ca(tmp_path, layer, image):
    (layer / "ca.pem").write_text(_CERT)
    done, calls = _run(tmp_path, layer, image)
    assert done.returncode == 0, done.stderr
    assert "groupadd --gid 2000 hl-specialist" in calls
    assert any(
        c.startswith("useradd ") and "--uid 2000" in c and "--gid 2000" in c and c.endswith("hl-specialist")
        for c in calls
    )
    boot = _boot(tmp_path)
    assert boot[0].startswith(f"PATH={layer}/bin:")
    bundle = tmp_path / "local-tmp" / "hyperloom-host" / "ca-bundle.pem"
    assert f"SSL_CERT_FILE={bundle}" in boot and f"NODE_EXTRA_CA_CERTS={bundle}" in boot
    assert bundle.read_text().endswith(_CERT)
    assert stat.S_IMODE(bundle.stat().st_mode) == 0o644 and stat.S_IMODE(bundle.parent.stat().st_mode) == 0o755
    assert not list((tmp_path / "state").iterdir())  # nothing written to the (shared) state dir


@needs_account_here
def test_the_specialist_can_read_the_bundle_and_the_state_dir_stays_empty(tmp_path, layer, image):
    if os.geteuid() != 0 or shutil.which("setpriv") is None:
        pytest.skip("needs root and setpriv to act as uid 2000")
    (layer / "ca.pem").write_text(_CERT)
    local = Path(tempfile.mkdtemp())  # a path uid 2000 can traverse, unlike pytest's private tmp dirs
    try:
        local.chmod(0o755)
        done, _ = _run(tmp_path, layer, image, TMPDIR=str(local))
        assert done.returncode == 0, done.stderr
        bundle = local / "hyperloom-host" / "ca-bundle.pem"
        read = subprocess.run(
            ["setpriv", "--reuid=2000", "--regid=2000", "--clear-groups", "cat", str(bundle)],
            capture_output=True,
            text=True,
        )
        assert read.returncode == 0 and read.stdout.endswith(_CERT), read.stderr
        assert not list((tmp_path / "state").iterdir())
    finally:
        shutil.rmtree(local, ignore_errors=True)


@needs_account_here
def test_an_existing_account_is_left_alone_and_no_ca_means_no_bundle(tmp_path, layer, image):
    _tool(image, "getent", 'echo "hl-specialist:x:2000:2000::/h:/bin/sh"')
    done, calls = _run(tmp_path, layer, image)
    assert done.returncode == 0, done.stderr
    assert not [c for c in calls if c.startswith(("useradd", "groupadd"))]
    assert not (tmp_path / "local-tmp" / "hyperloom-host").exists()


_ROOT_STAYS = "so the root mount stays as it is; wrap each bwrap call in 'unshare --mount'"


@needs_account_here
def test_the_root_mount_is_made_private_once_right_before_the_bootstrap(tmp_path, layer, image):
    done, calls = _run(tmp_path, layer, image)
    assert done.returncode == 0 and done.stderr == "", done.stderr
    assert [c for c in calls if c.startswith("mount ")] == ["mount --make-rprivate /"]
    assert calls[-2:] == ["mount --make-rprivate /", "bootstrap"]


@needs_account_here
@pytest.mark.parametrize(
    ("stub", "said"),
    [
        ('echo "mount: /: permission denied." >&2; exit 32', "mount: /: permission denied."),
        ("exit 1", "no message"),
    ],
    ids=["with-a-message", "silent"],
)
def test_a_root_mount_that_cannot_be_made_private_is_logged_and_the_bootstrap_still_runs(
    tmp_path, layer, image, stub, said
):
    _tool(image, "mount", f'echo "mount $*" >>"$CALLS"; {stub}')
    done, calls = _run(tmp_path, layer, image)
    assert done.returncode == 0, done.stderr
    assert done.stderr.splitlines() == [f"hosted layer: mount --make-rprivate / failed ({said}), {_ROOT_STAYS}"]
    assert calls[-2:] == ["mount --make-rprivate /", "bootstrap"]


@needs_account_here
def test_an_image_without_mount_is_logged_and_the_bootstrap_still_runs(tmp_path, layer, image):
    (image / "mount").unlink()
    bare = _bare_path(tmp_path, image, *_ESSENTIALS)
    done, calls = _run(tmp_path, layer, image, PATH=bare)
    assert done.returncode == 0, done.stderr
    assert done.stderr.splitlines() == [f"hosted layer: the image has no mount (util-linux), {_ROOT_STAYS}"]
    assert calls[-1] == "bootstrap"


def test_a_layer_built_under_a_private_umask_is_readable_and_runnable_by_every_account(tmp_path):
    """build_layer.sh in a copy of the repo layout, with the installer stubbed, under umask 027."""
    root = tmp_path / "repo"
    for rel in (
        "scripts/hosted_app_image/build_layer.sh",
        "src/hyperloom/hosted/app_bootstrap.sh",
        "src/hyperloom/hosted/start.sh",
    ):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(START.parent.parent.parent.parent / rel, root / rel)
    for cmd in (
        ["init", "-q"],
        ["add", "-A"],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c"],
    ):
        subprocess.run(["git", "-C", str(root), *cmd], check=True, capture_output=True)
    stub = tmp_path / "stub"
    _tool(
        stub,
        "curl",
        'echo \'mkdir -p "$HOME/.local/bin" && printf "#!/bin/sh\\necho 2.0.0 stub\\n" >"$HOME/.local/bin/claude" && chmod 700 "$HOME/.local/bin/claude"\'',
    )
    ca = tmp_path / "ca.crt"
    ca.write_text(_CERT)
    out = tmp_path / "layers"
    done = subprocess.run(
        [
            "sh",
            "-c",
            f'umask 027; exec bash "{root}/scripts/hosted_app_image/build_layer.sh" --out "{out}" --ca-file "{ca}"',
        ],
        capture_output=True,
        text=True,
        env={"PATH": f"{stub}:/usr/bin:/bin"},
    )
    assert done.returncode == 0, done.stderr
    layer = next(out.iterdir())
    assert layer.name.endswith("-claude2.0.0")
    for path in [layer, *layer.rglob("*")]:
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir():
            assert mode == 0o755, path
        else:
            assert mode & 0o444 == 0o444, path  # a+r
            if path.name != "ca.pem":
                assert mode & 0o111 == 0o111, path  # the scripts and the CLI: a+x
