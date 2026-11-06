# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""start.sh of the hosting layer: what it refuses on an image, and what it hands app_bootstrap.sh."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest

from hyperloom.common.id_map import unmapped_ids
from hyperloom.common.tests._id_map_helpers import HOST_MAP, ONE_ID_MAP

SH = shutil.which("sh") or "sh"
START = Path(__file__).resolve().parent.parent / "start.sh"
_CERT = "-----BEGIN CERTIFICATE-----\nQUJD\n-----END CERTIFICATE-----\n"
#: start.sh reads this process's own maps (/proc/self/uid_map, gid_map) and goes past them only if they map uid and gid
#: 2000; what is not mapped here, if anything.
_NOT_MAPPED_HERE = unmapped_ids((2000,), (2000,))
needs_2000_mapped = pytest.mark.skipif(
    _NOT_MAPPED_HERE is not None, reason=f"start.sh refuses on this host's own maps: {_NOT_MAPPED_HERE}"
)
#: The functions of start.sh that judge the user namespace maps, and the die their refusals go through.
_MAP_FUNCTIONS = ("die", "decimal", "in_id_map", "require_mapped")


def _tool(path: Path, name: str, body: str = "exit 0") -> None:
    path.mkdir(exist_ok=True)
    (path / name).write_text(f"#!/bin/sh\n{body}\n")
    (path / name).chmod(0o755)


@pytest.fixture
def layer(tmp_path) -> Path:
    """A layer directory whose bootstrap records what it was started with."""
    root = tmp_path / "layer"
    root.mkdir()
    shutil.copy(START, root / "start.sh")
    _tool(
        root,
        "app_bootstrap.sh",
        'echo "PATH=$PATH" >"$CALLS.boot"; env | grep -E "^(SSL_CERT_FILE|NODE_EXTRA_CA_CERTS)=" >>"$CALLS.boot" || true',
    )
    _tool(root / "bin", "claude", "echo 1.0.0")
    return root


@pytest.fixture
def image(tmp_path) -> Path:
    """The tools of an image as stubs: root, git, setpriv, bash and the account tools, each recording its call."""
    tools = tmp_path / "tools"
    for name in ("git", "setpriv"):
        _tool(tools, name)
    _tool(tools, "bash", 'exec /bin/bash "$@"')
    _tool(tools, "id", "echo 0")
    _tool(tools, "getent", 'echo "getent $*" >>"$CALLS"; exit 2')
    for name in ("groupadd", "useradd"):
        _tool(tools, name, f'echo "{name} $*" >>"$CALLS"')
    return tools


def _run(tmp_path, layer, image, **env):
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
    done = subprocess.run([SH, str(layer / "start.sh")], capture_output=True, text=True, env=full, timeout=60)
    calls = tmp_path / "calls"
    return done, (calls.read_text().splitlines() if calls.exists() else [])


@pytest.mark.parametrize("missing", ["git", "setpriv"])
def test_an_image_without_git_or_setpriv_is_refused_by_name(tmp_path, layer, image, missing):
    (image / missing).unlink()
    # Keep the real tool out of reach: only the stub dir and a dir with the bare essentials are searched.
    essentials = tmp_path / "essentials"
    essentials.mkdir()
    for name in ("cat", "cut", "dirname", "mv", "chmod", "ls"):
        found = shutil.which(name)
        if found:
            (essentials / name).symlink_to(found)
    done, _ = _run(tmp_path, layer, image, PATH=f"{image}:{essentials}")
    assert done.returncode == 1 and f"has no {missing}" in done.stderr, done.stderr


def test_not_root_is_refused(tmp_path, layer, image):
    _tool(image, "id", "echo 1000")
    done, calls = _run(tmp_path, layer, image)
    assert done.returncode == 1 and "must run as root" in done.stderr and not calls


@needs_2000_mapped
def test_a_uid_taken_by_another_name_is_refused(tmp_path, layer, image):
    _tool(
        image,
        "getent",
        'case "$1" in passwd) [ "$2" = 2000 ] && echo "someone:x:2000:2000::/h:/bin/sh" && exit 0;; esac; exit 2',
    )
    done, _ = _run(tmp_path, layer, image)
    assert done.returncode == 1 and "uid 2000 is taken by the account someone" in done.stderr


@needs_2000_mapped
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


def _sourced(tmp_path: Path, call: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run ``call`` (``args`` are its ``$1``...) under start.sh's ``set -eu``, with only its map functions sourced."""
    functions = tmp_path / "map-functions.sh"
    functions.write_text(_functions(*_MAP_FUNCTIONS))
    script = f'set -eu\n. "{functions}"\n{call}\n'
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
def test_an_id_this_user_namespace_does_not_map_is_refused_naming_the_map(tmp_path, uid_map, gid_map, kind, quoted):
    paths = _maps(tmp_path, uid_map, gid_map)
    done = _sourced(tmp_path, 'require_mapped 2000 "$1" "$2"; echo mapped', *map(str, paths))
    assert (done.returncode, done.stdout) == (1, ""), done.stderr
    path = paths[0] if kind == "uid" else paths[1]
    assert done.stderr.startswith(
        f"hosted layer: {kind} 2000 is not mapped in this container's user namespace ({path}: '{quoted}'): "
    ), done.stderr
    assert "EINVAL" in done.stderr and "fakeroot" in done.stderr


def test_maps_of_several_ranges_that_cover_2000_pass(tmp_path):
    paths = _maps(tmp_path, "0 1234 1\n1 100000 1999\n2000 300000 1\n", "0 1234 1\n1000 5000 1001\n")
    done = _sourced(tmp_path, 'require_mapped 2000 "$1" "$2"; echo mapped', *map(str, paths))
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
    done = _sourced(tmp_path, 'require_mapped 2000 "$1" "$2"; echo mapped', *map(str, paths))
    assert (done.returncode, done.stdout) == (1, ""), done.stderr
    path = paths[0] if kind == "uid" else paths[1]
    assert done.stderr.startswith(f"hosted layer: cannot read {path} as a user namespace map ({says})"), done.stderr


def _one_id_namespace() -> list[str] | None:
    """The command prefix of a new user namespace that maps one id (root, onto this process's own), or None here."""
    unshare = shutil.which("unshare")
    if unshare is None:
        return None
    prefix = [unshare, "--user", "--map-root-user", "--"]
    probe = subprocess.run([*prefix, "cat", "/proc/self/uid_map"], capture_output=True, text=True, check=False)
    return prefix if probe.returncode == 0 and len(probe.stdout.split()) == 3 else None


@pytest.mark.skipif(_one_id_namespace() is None, reason="this host cannot make a user namespace (unshare --user)")
@pytest.mark.parametrize("existing", [False, True], ids=["new-account", "existing-account"])
def test_in_a_real_one_id_user_namespace_the_kernels_map_is_read_and_the_account_refused(
    tmp_path, layer, image, existing
):
    if existing:
        _tool(image, "getent", 'echo "getent $*" >>"$CALLS"; echo "hl-specialist:x:2000:2000::/h:/bin/sh"')
    env = {"PATH": f"{image}:/usr/bin:/bin", "CALLS": str(tmp_path / "calls"), "TMPDIR": str(tmp_path)}
    done = subprocess.run(
        [*_one_id_namespace(), SH, str(layer / "start.sh")], capture_output=True, text=True, env=env, timeout=60
    )
    assert done.returncode == 1, done.stderr
    assert "uid 2000 is not mapped in this container's user namespace (/proc/self/uid_map: '0 " in done.stderr
    assert "EINVAL" in done.stderr and "fakeroot" in done.stderr
    assert not (tmp_path / "calls").exists(), "neither the account nor its group may be looked up or made"


@needs_2000_mapped
def test_a_layer_without_the_cli_is_refused(tmp_path, layer, image):
    (layer / "bin" / "claude").unlink()
    done, _ = _run(tmp_path, layer, image)
    assert done.returncode == 1 and "no claude CLI" in done.stderr


@needs_2000_mapped
def test_the_account_is_created_and_the_bootstrap_gets_the_cli_and_the_ca(tmp_path, layer, image):
    (layer / "ca.pem").write_text(_CERT)
    done, calls = _run(tmp_path, layer, image)
    assert done.returncode == 0, done.stderr
    assert "groupadd --gid 2000 hl-specialist" in calls
    assert any(
        c.startswith("useradd ") and "--uid 2000" in c and "--gid 2000" in c and c.endswith("hl-specialist")
        for c in calls
    )
    boot = (tmp_path / "calls.boot").read_text().splitlines()
    assert boot[0].startswith(f"PATH={layer}/bin:")
    bundle = tmp_path / "local-tmp" / "hyperloom-host" / "ca-bundle.pem"
    assert f"SSL_CERT_FILE={bundle}" in boot and f"NODE_EXTRA_CA_CERTS={bundle}" in boot
    assert bundle.read_text().endswith(_CERT)
    assert stat.S_IMODE(bundle.stat().st_mode) == 0o644 and stat.S_IMODE(bundle.parent.stat().st_mode) == 0o755
    assert not list((tmp_path / "state").iterdir())  # nothing written to the (shared) state dir


@needs_2000_mapped
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


@needs_2000_mapped
def test_an_existing_account_is_left_alone_and_no_ca_means_no_bundle(tmp_path, layer, image):
    _tool(image, "getent", 'echo "hl-specialist:x:2000:2000::/h:/bin/sh"')
    done, calls = _run(tmp_path, layer, image)
    assert done.returncode == 0, done.stderr
    assert not [c for c in calls if c.startswith(("useradd", "groupadd"))]
    assert not (tmp_path / "local-tmp" / "hyperloom-host").exists()


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
