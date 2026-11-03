"""A KEEP's editable packages, kept beside its landed.patch: what the gate measured.

landed.patch carries the workspace only (save_landed_patch), so an edit a KEEP made to
another editable package -- AITER, under site-packages -- lived only in its container.
The model-m launchers rebuild those edits from the agent's own landed_aiter.py, and nothing
checked that it could: Model M day-0 b0's iterations 51 and 52 were AITER-only KEEPs that
landed_aiter.py never received, so the chunk roll of 2026-10-22 02:20 rebuilt a tree
without them, and the round-end A/B measured round 1's tree as round 2's (runner
model-m-day0/plans/00_ARBOR_FROM_DAY0.md §4, 04:18). Operator: build the guard.

On every KEEP the driver copies each editable package's .py files that differ from the
container's image (changed or added) to <session>/landed_packages/<name>/, and lists them
with their digests in <session>/landed_packages.json. A chunk's launcher, after
landed_aiter.py, puts back every listed file the container does not hold as captured, and
says which: what the last KEEP measured is what the next one is graded against.

    python -m arbor.execute.packages capture --container C --config CFG --session S
    python -m arbor.execute.packages restore --container C --session S

capture  the container's package files that differ from its image -> the session
restore  the listed files, wherever the container differs from them; exit 1 if it cannot
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from arbor.execute.workspace import HOST, run_in

MANIFEST = "landed_packages.json"
FILES = "landed_packages"
PY_SUMS = "find . -name '*.py' -not -path '*/__pycache__/*' -print0 | sort -z | xargs -0 -r sha256sum"


def _sums(text: str) -> dict[str, str]:
    """sha256sum's output as {path: digest}."""
    out = {}
    for line in text.splitlines():
        digest, _, path = line.partition("  ")
        if path:
            out[path] = digest
    return out


def _sh(container: str, cmd: str) -> str:
    p = run_in(container, ["bash", "-c", cmd], capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise RuntimeError(f"`{cmd[:120]}` failed: {(p.stderr or p.stdout).strip()[-300:]}")
    return p.stdout


def editable(config: dict) -> list[dict]:
    """The config's editable packages other than the workspace (which landed.patch carries)."""
    ws = str(config.get("workspace_root") or "").rstrip("/")
    return [p for p in config.get("editable_packages") or []
            if p.get("name") and p.get("path") and str(p["path"]).rstrip("/") != ws]


def image_sums(container: str):
    """A reader of the container's image: its .py digests under a path, from a throwaway container."""
    image = subprocess.run(["docker", "inspect", "--format", "{{.Image}}", container],
                           capture_output=True, text=True, timeout=60, check=True).stdout.strip()

    def read(path: str) -> dict[str, str]:
        p = subprocess.run(["docker", "run", "--rm", "--entrypoint", "bash", image, "-c",
                            f"cd {shlex.quote(path)} && {PY_SUMS}"],
                           capture_output=True, text=True, timeout=900)
        if p.returncode != 0:
            raise RuntimeError(f"could not list the image's {path}: {p.stderr.strip()[-300:]}")
        return _sums(p.stdout)
    read.image = image
    return read


def capture(container: str, config: dict, session: str | Path, pristine=None) -> dict:
    """Copy each editable package's files that differ from the image to the session.

    ``pristine(path) -> {file: digest}`` reads the image (default: image_sums). Returns
    the manifest written, or {} when there is nothing to capture. Raises on failure; the
    driver records that and keeps its verdict, as for landed.patch.
    """
    pkgs = editable(config)
    if not pkgs or (container == HOST and pristine is None):
        return {}
    pristine = pristine or image_sums(container)
    session = Path(session)
    tmp = session / f".{FILES}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    manifest: dict = {}
    for p in pkgs:
        name, path = p["name"], str(p["path"])
        img, cur = pristine(path), _sums(_sh(container, f"cd {shlex.quote(path)} && {PY_SUMS}"))
        files = sorted(f for f in cur if img.get(f) != cur[f])
        if files:
            out = tmp / name
            out.mkdir(parents=True)
            data = run_in(container, ["bash", "-c", f"cd {shlex.quote(path)} && tar -cf - "
                                      + " ".join(shlex.quote(f) for f in files)],
                          capture_output=True, timeout=600)
            if data.returncode != 0:
                raise RuntimeError(f"{name}: could not read {len(files)} file(s) from the container")
            subprocess.run(["tar", "-C", str(out), "-xf", "-"], input=data.stdout, check=True, timeout=600)
        manifest[name] = {"path": path, "image": getattr(pristine, "image", None),
                          "files": {f: cur[f] for f in files}, "added": [f for f in files if f not in img]}
    shutil.rmtree(session / FILES, ignore_errors=True)
    if tmp.exists():
        os.replace(tmp, session / FILES)
    part = session / f".{MANIFEST}.tmp"
    part.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    os.replace(part, session / MANIFEST)
    return manifest


def restore(container: str, session: str | Path) -> dict[str, list[str]]:
    """Put back each listed file the container does not hold as captured: {package: files}."""
    session = Path(session)
    manifest = json.loads((session / MANIFEST).read_text())
    put: dict[str, list[str]] = {}
    for name, m in manifest.items():
        files, path = m.get("files") or {}, str(m["path"])
        if not files:
            continue
        listed = " ".join(shlex.quote(f) for f in files)

        def differ() -> list[str]:
            got = _sums(_sh(container, f"cd {shlex.quote(path)} && sha256sum -- {listed} 2>/dev/null; true"))
            return [f for f in files if got.get(f) != files[f]]
        bad = differ()
        if not bad:
            continue
        data = subprocess.run(["tar", "-C", str(session / FILES / name), "-cf", "-", *bad],
                              capture_output=True, check=True, timeout=600).stdout
        tar = ["tar", "-C", path, "-xf", "-"]
        p = subprocess.run(tar if container == HOST else ["docker", "exec", "-i", container, *tar],
                           input=data, capture_output=True, timeout=600)
        if p.returncode != 0:
            raise RuntimeError(f"{name}: could not write {len(bad)} file(s): {p.stderr.decode()[-300:]}")
        # A stale bytecode cache must not outlive its source.
        pyc = " ".join(f"{shlex.quote(os.path.dirname(f) or '.')}/__pycache__/"
                       f"{shlex.quote(os.path.basename(f)[:-3])}.*.pyc" for f in bad)
        _sh(container, f"cd {shlex.quote(path)} && rm -f {pyc}")
        still = differ()
        if still:
            raise RuntimeError(f"{name}: {len(still)} file(s) still differ after the restore: {still[:5]}")
        put[name] = bad
    return put


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m arbor.execute.packages", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="the container's package files that differ from its image -> the session")
    c.add_argument("--container", required=True)
    c.add_argument("--config", required=True)
    c.add_argument("--session", required=True)
    r = sub.add_parser("restore", help="put back the files the last KEEP measured")
    r.add_argument("--container", required=True)
    r.add_argument("--session", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "capture":
            m = capture(a.container, json.loads(Path(a.config).read_text()), a.session)
            for name, v in m.items():
                print(f"packages: {name}: {len(v['files'])} file(s) differ from the image, captured")
            return 0
        manifest = json.loads((Path(a.session) / MANIFEST).read_text())
        put = restore(a.container, a.session)
        for name, v in manifest.items():
            n = len(v.get("files") or {})
            if put.get(name):
                print(f"packages: WARNING: {name}: {len(put[name])} of {n} file(s) were not as the last KEEP "
                      f"measured, restored: {', '.join(put[name][:8])} (landed_aiter.py does not rebuild them)")
            else:
                print(f"packages: {name}: {n} file(s) as the last KEEP measured")
        return 0
    except Exception as exc:  # noqa: BLE001 - the launcher decides
        print(f"packages: {a.cmd} failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
