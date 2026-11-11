"""The editable trees outside the workspace (AITER under site-packages), committed and reverted with each iteration.

The workspace is a git repository the harness commits, reverts and patches every iteration. The other
``editable_packages`` were touched only at a KEEP (arbor/execute/packages.py copies their changed ``.py`` files). So a
change confined to one of them was booked "REVERT: no changes" and stayed in place, and one beside a workspace change was
measured with it but outlived the REVERT, credited to whatever came next. A tuned MoE table an agent wrote into AITER's
``configs/model_configs/``, which AITER scans at load, did both.

Each of those trees is a git repository from the campaign's start (``ensure``), with a baseline commit of what the tree
held then, bytecode and JIT outputs kept out (``EXCLUDES``), and the baseline recorded in ``<session>/trees.json``. Every
iteration then:

- **sees** a change in any tree, untracked files included (``changed``);
- **commits** each changed tree with the workspace's message (``commit``), recorded in ``iter_N/trees.json``;
- **shows** those commits to the critic beside the workspace's diff (``diff``);
- **reverts** them with the workspace (``revert``, by the iteration's commit subject).

A KEEP's capture takes every file a tree changed since its baseline, not only ``.py`` (packages.py).

Under a host (component mode) nothing here runs: the trees are the host's.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from arbor.execute.workspace import GIT_EXCLUDE_PATTERNS, HOST, has_changes, run_in

FILE = "trees.json"           # <session>/: each tree's baseline, for the container it was taken in
ITER_FILE = "trees.json"      # iter_N/: the trees this iteration committed, {path: sha}
EXCLUDES = (*GIT_EXCLUDE_PATTERNS, "*.so", "*.o", "jit/build/", "jit/flydsl_cache/")
_IDENTITY = ("-c", "user.name=arbor", "-c", "user.email=arbor@localhost")


def editable(config: dict) -> list[dict]:
    """The config's editable packages other than the workspace (which landed.patch carries). Here rather than in
    arbor/execute/packages.py, which imports this module, so the two do not import each other."""
    ws = str(config.get("workspace_root") or "").rstrip("/")
    return [p for p in config.get("editable_packages") or []
            if p.get("name") and p.get("path") and str(p["path"]).rstrip("/") != ws]


def paths(config: dict) -> list[str]:
    """The editable packages other than the workspace, which this module tracks: none under a host."""
    from arbor import component
    if component.is_component(config):
        return []
    return [str(p["path"]).rstrip("/") for p in editable(config)]


def _git(container: str, tree: str, *args: str, check: bool = True, timeout: int = 900) -> str:
    r = run_in(container, ["git", *_IDENTITY, "-C", tree, *args], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])} in {tree}: {(r.stderr or r.stdout).strip()[-300:]}")
    return r.stdout.strip()


def _container_id(container: str) -> str:
    if container == HOST:
        return HOST
    try:
        import subprocess
        r = subprocess.run(["docker", "inspect", "--format", "{{.Id}}", container],
                           capture_output=True, text=True, timeout=60)
        return r.stdout.strip() or container
    except (OSError, ValueError):
        return container


def ensure(container: str, config: dict, session_dir: str | Path) -> dict[str, str]:
    """Make each tree a git repository with a baseline commit, at every start; ``{path: baseline}``.

    A tree that is not a repository is initialized; one that is (a clone, a launcher's) is kept, its uncommitted state
    committed as the baseline. A baseline taken in this container is kept on a resume; a new container gets new ones,
    which hold what a launcher restored of earlier KEEPs. Recorded in ``<session>/trees.json``.
    """
    trees = paths(config)
    if not container or not trees:
        return {}
    session_dir = Path(session_dir)
    record_path = session_dir / FILE
    try:
        record = json.loads(record_path.read_text())
    except (OSError, ValueError):
        record = {}
    cid = _container_id(container)
    known = record.get("trees", {}) if record.get("container") == cid else {}
    baselines: dict[str, str] = {}
    for tree in trees:
        if run_in(container, ["test", "-d", tree], capture_output=True).returncode != 0:
            continue
        if _git(container, tree, "rev-parse", "--git-dir", check=False) == "":
            _git(container, tree, "init", "-q")
        _excludes(container, tree)
        head = _git(container, tree, "rev-parse", "--verify", "-q", "HEAD", check=False)
        if not head or has_changes(container, tree):
            _git(container, tree, "add", "-A")
            _git(container, tree, "commit", "-q", "--allow-empty", "-m", f"arbor: baseline of {Path(tree).name}")
        baselines[tree] = known.get(tree) or _git(container, tree, "rev-parse", "HEAD")
    record_path.write_text(json.dumps({"container": cid, "trees": baselines}, indent=1) + "\n")
    return baselines


def _excludes(container: str, tree: str) -> None:
    """Bytecode and JIT outputs stay out of every commit (``.git/info/exclude``, beside the tree's own ignores)."""
    script = (f"cd {shlex.quote(tree)} && f=$(git rev-parse --git-path info/exclude) && mkdir -p \"$(dirname \"$f\")\" "
              f"&& touch \"$f\" && for p in {' '.join(shlex.quote(p) for p in EXCLUDES)}; do "
              f"grep -qxF \"$p\" \"$f\" || printf '%s\\n' \"$p\" >> \"$f\"; done")
    run_in(container, ["bash", "-c", script], capture_output=True)


def baselines(session_dir: str | Path) -> dict[str, str]:
    """``{path: baseline}`` from ``<session>/trees.json``, or {}."""
    try:
        return json.loads((Path(session_dir) / FILE).read_text()).get("trees", {})
    except (OSError, ValueError):
        return {}


def changed(container: str, config: dict) -> list[str]:
    """The trees with uncommitted changes, untracked files included (bytecode and JIT outputs excluded)."""
    out = []
    for tree in paths(config):
        try:
            if has_changes(container, tree):
                out.append(tree)
        except RuntimeError:
            continue          # not a repository (not ensured): nothing to commit there
    return out


def commit(container: str, trees: list[str], message: str, iter_dir: str | Path | None = None) -> dict[str, str]:
    """Commit each tree with the iteration's message; ``{path: sha}``, also written to ``iter_N/trees.json``."""
    done: dict[str, str] = {}
    for tree in trees:
        _git(container, tree, "add", "-A")
        _git(container, tree, "commit", "-q", "-m", message)
        done[tree] = _git(container, tree, "rev-parse", "HEAD")
    if iter_dir is not None and done:
        (Path(iter_dir) / ITER_FILE).write_text(json.dumps(done, indent=1) + "\n")
    return done


def committed(iter_dir: str | Path) -> dict[str, str]:
    """The trees an iteration committed, from ``iter_N/trees.json``."""
    try:
        return json.loads((Path(iter_dir) / ITER_FILE).read_text())
    except (OSError, ValueError):
        return {}


def diff(container: str, iter_dir: str | Path) -> str:
    """The iteration's commits in the trees, as one diff whose paths start with each tree's name."""
    parts = []
    for tree, sha in committed(iter_dir).items():
        name = Path(tree).name
        try:
            parts.append(_git(container, tree, "diff", f"--src-prefix=a/{name}/", f"--dst-prefix=b/{name}/",
                              f"{sha}~1", sha))
        except RuntimeError:
            parts.append(f"(could not read the diff of {tree})")
    return "\n".join(p for p in parts if p)


def revert(container: str, trees: list[str], subject: str) -> list[str]:
    """Revert, in each tree, a HEAD commit whose subject is the iteration's (``iter-N: ...``); the trees reverted.
    Tracked files a server run touched since are restored first. Untracked files are left: the commit took the turn's,
    and what appears after it is a server's own output, not the change."""
    out = []
    for tree in trees:
        try:
            _git(container, tree, "checkout", "--", ".", check=False)
            if subject and _git(container, tree, "log", "-1", "--format=%s", check=False) == subject:
                _git(container, tree, "revert", "--no-edit", "HEAD")
                out.append(tree)
        except RuntimeError as exc:
            print(f"[arbor] revert failed in {tree}: {exc}")
    return out
