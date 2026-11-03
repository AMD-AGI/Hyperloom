"""The share: where collectors put campaign data for everyone to read.

Layout under the share root (default /shared/arbor, on a Primus-SaFE pod)::

    projects/<project>/<user>/<campaign>/realtime/          status.json, meta.json
    projects/<project>/<user>/<campaign>/knowledge/<type>/  summary, outcomes, findings,
                                                            profiles, dossier, artifacts

The pod is reached through the Primus-SaFE ssh gateway, which takes ssh
commands, scp and sftp but not rsync and not piped stdin (tested 2026-10-06).
So an upload is one sftp batch: make the directories, put each file under a
temporary name, rename it into place. sftp's rename replaces the target in one
step, so a reader never sees half a file. The gateway also returns exit status
0 for every command and delivers its stderr on stdout (tested 2026-10-07), so
``check`` reads markers the command prints, never its exit status; sftp's own
status is reliable.

The share root and its projects/ directory are created once, by a person; the
collectors only write below projects/.

Settings, each with an environment fallback:
    ARBOR_SHARE_HOST      ssh destination, an alias from ~/.ssh/config (default arbor-share)
    ARBOR_SHARE_ROOT      the share root on that host (default /shared/arbor)
    ARBOR_SHARE_SSH_OPTS  extra options for both ssh and sftp; use -F and -o forms
                          (-o Port=2222), since ssh and sftp spell -p differently
"""

from __future__ import annotations

import os
import posixpath
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_HOST = "arbor-share"
DEFAULT_ROOT = "/shared/arbor"
BASE_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]


def _env_opts() -> list[str]:
    return shlex.split(os.environ.get("ARBOR_SHARE_SSH_OPTS", ""))


@dataclass
class Share:
    host: str = field(default_factory=lambda: os.environ.get("ARBOR_SHARE_HOST") or DEFAULT_HOST)
    root: str = field(default_factory=lambda: os.environ.get("ARBOR_SHARE_ROOT") or DEFAULT_ROOT)
    opts: list[str] = field(default_factory=_env_opts)
    timeout: float = 180.0

    @property
    def projects(self) -> str:
        return posixpath.join(self.root, "projects")

    def campaign_dir(self, project: str, user: str, campaign: str) -> str:
        return posixpath.join(self.projects, project, user, campaign)

    # -- transport ---------------------------------------------------------

    def _run(self, cmd: list[str]) -> tuple[int, str, str]:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout,
                               stdin=subprocess.DEVNULL)
            return p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            return 124, "", f"timed out after {self.timeout:.0f} s"
        except OSError as exc:
            return 127, "", str(exc)

    def ssh(self, command: str) -> tuple[int, str, str]:
        return self._run(["ssh", *BASE_OPTS, *self.opts, self.host, command])

    def sftp(self, batch: list[str]) -> tuple[int, str, str]:
        with tempfile.NamedTemporaryFile("w", suffix=".sftp", delete=False) as fh:
            fh.write("\n".join(batch) + "\n")
            path = fh.name
        try:
            return self._run(["sftp", "-b", path, *BASE_OPTS, *self.opts, self.host])
        finally:
            os.unlink(path)

    # -- operations --------------------------------------------------------

    def check(self, user: str) -> dict:
        """Can this host reach the share and upload to it? Run before collecting.

        Each step's result, and for the first that fails, what to do about it.
        """
        out: dict = {"host": self.host, "root": self.root, "ok": False, "steps": []}

        def step(name: str, ok: bool, detail: str, fix: str = "") -> bool:
            out["steps"].append({"step": name, "ok": ok, "detail": detail})
            if not ok:
                out["fix"] = fix
            return ok

        projects = shlex.quote(self.projects)
        rc, so, se = self.ssh(f"echo __ARBOR_SHARE__; test -d {projects} && test -w {projects} "
                              f"&& echo WRITABLE")
        reached = rc in (0, 1) and "__ARBOR_SHARE__" in so
        if not step("ssh", reached,
                    f"reached {self.host}" if reached else (se.strip() or so.strip() or f"exit {rc}")[-300:],
                    f"ssh to {self.host} failed. Add the pod to ~/.ssh/config as {self.host} "
                    f"(the collector skills show the entry) with the key you registered "
                    f"on Primus-SaFE, and check `ssh {self.host} true`."):
            return out
        if not step("share root", "WRITABLE" in so,
                    f"{self.projects} {'is writable' if 'WRITABLE' in so else 'is missing or not writable'}",
                    f"{self.projects} must exist and be writable. A person creates it once: "
                    f"ssh {self.host} 'mkdir -p {self.projects} && chgrp 10008 {self.root} "
                    f"{self.projects} && chmod 1775 {self.root} {self.projects}' (the mode and "
                    f"group of /shared itself)"):
            return out

        probe_local = Path(tempfile.mkstemp(prefix="arbor-share-probe-")[1])
        probe_local.write_text(f"probe from {user}\n")
        remote = posixpath.join(self.projects, f".probe-{user}-{os.getpid()}")
        try:
            ok, err = self.push([(probe_local, remote)])
        finally:
            probe_local.unlink(missing_ok=True)
        self.ssh(f"rm -f {shlex.quote(remote)} {shlex.quote(remote + '.tmp')}")
        if not step("upload", ok, err or "a probe file uploaded and removed",
                    "sftp through the gateway failed; see detail. The collectors keep "
                    "writing locally until it works."):
            return out
        out["ok"] = True
        return out

    def push(self, files: list[tuple[Path, str]]) -> tuple[bool, str]:
        """Upload ``(local file, remote path)`` pairs, each replaced in one step.

        Remote paths must be below the share's projects/ directory; the
        directories between are created.
        """
        if not files:
            return True, ""
        dirs: list[str] = []
        lines: list[str] = []
        for local, remote in files:
            if not posixpath.normpath(remote).startswith(self.projects + "/"):
                return False, f"refusing a path outside {self.projects}: {remote}"
            for text in (str(local), remote):
                if '"' in text or "\n" in text:
                    return False, f"unsupported character in path: {text!r}"
            d = posixpath.dirname(remote)
            chain = []
            while d != self.projects and d not in dirs and d.startswith(self.projects):
                chain.append(d)
                d = posixpath.dirname(d)
            for d in reversed(chain):
                dirs.append(d)
                lines.append(f'-mkdir "{d}"')
        for local, remote in files:
            tmp = posixpath.join(posixpath.dirname(remote), "." + posixpath.basename(remote) + ".tmp")
            lines += [f'put "{local}" "{tmp}"', f'rename "{tmp}" "{remote}"']
        rc, so, se = self.sftp(lines)
        if rc != 0:
            return False, (se.strip() or so.strip() or f"sftp exit {rc}")[-500:]
        return True, ""
