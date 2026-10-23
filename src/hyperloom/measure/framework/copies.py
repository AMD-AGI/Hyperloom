# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every test reads private copies of the framework's tuned tables, so no test can change what another reads.

Needs no privileges and no mounts: aiter takes each table's path from an ``AITER_CONFIG_*`` variable, so the test gets a
private directory holding the session's snapshot of the tables, every one of those variables the test does not set
names the private copy (what aiter reads when it is unset, merged tables included), and the tables a test names itself
are copied into the same directory and named by their copy. The merge aiter does for a variable that lists several
tables writes the de-duplicated rows back into its sources, which are now the test's own files.

A value that lists ``DEFAULT_TABLE_PLACEHOLDER`` beside a file of tuned rows (a rows-only kernel table) is the image's
default table for that variable plus those rows; ``framework.rows`` takes the rows' shapes out of the default's copy
first, since aiter rewrites its sources and raises on a shape both carry, and the result says how many it took out.

What no variable redirects is checked instead: the shared tables are hashed against the snapshot before and after the
test, anything that differs is put back, and the result says so (``framework_base.json``, covered by the signature).
aiter writes the merged view of a variable to ``/tmp/aiter_configs`` whatever the environment says, and local sandboxes
share ``/tmp``; ``framework.merge`` gives the test's server a directory of its own for it instead, through a start-up
hook on its ``PYTHONPATH``, and the result says whether the hook did its work.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fnmatch
import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hyperloom.measure.farm import prepend_pythonpath, server_environment
from hyperloom.measure.framework.base import (
    FRAMEWORK_BASE_FILE,
    SESSION_SHA_ENV,
    SNAPSHOT_ENV,
    FrameworkBaseError,
    Snapshot,
    file_sha256,
    files_digest,
    load_snapshot,
    restore,
    shared_view,
    tree_files,
)
from hyperloom.measure.framework.merge import MergeView
from hyperloom.measure.framework.rows import DEFAULT_TABLE_PLACEHOLDER, RowsOnlyRefused, lay_rows_over
from hyperloom.measure.framework.tables import AITER_VARIABLE_PREFIX
from hyperloom.measure.socket import TestCall

__all__ = ["PrivateFramework", "private_framework"]

log = logging.getLogger(__name__)

_SCHEMA = "hyperloom.framework-base/v1"
_CONFIGS = "configs"
_SANDBOX_ROOT_ENV = "HYPERLOOM_SANDBOX_ROOT"
_UTF8 = "utf-8"
_TABLE_DIGEST_CHARS = 16


@dataclass(frozen=True)
class PrivateFramework:
    """The call to run, and the variables it now names private tables for (unsetting one of them no longer applies)."""

    call: TestCall
    redirected: frozenset[str] = frozenset()


class _Redirect:
    """Copies of the tables the test reads, in a directory of the test's own, and the variable values that name them."""

    def __init__(self, snapshot: Snapshot, private: Path) -> None:
        self._snapshot = snapshot
        self._configs = private / _CONFIGS
        self._tables = private / "tables"
        self._points = private / "points"
        self.copied: dict[str, list[dict[str, str]]] = {}
        self.rows_only: dict[str, dict[str, Any]] = {}
        self.hook: Path | None = None

    def value(self, name: str, value: str) -> str:
        """``value`` with each table it lists replaced by a private copy; a path that is no file stays as it was."""
        parts = value.split(os.pathsep)
        if DEFAULT_TABLE_PLACEHOLDER in parts:
            return self._rows_over_default(name, [part for part in parts if part != DEFAULT_TABLE_PLACEHOLDER])
        return os.pathsep.join(self._copy(name, part) for part in parts)

    def _copy(self, name: str, part: str) -> str:
        """The private copy of the table ``part`` names; every point that names the same table gets the same path.

        The copy lives at ``tables/<variable>/<digest>/<file name>``, named by what the table holds: Magpie's reuse
        check compares the server's environment word for word, so a path that counted the tables copied so far (one
        per point of a sweep arm) would make every round after the one that booted the server look like another server.
        Tables with the same file name and other content stay apart, and the server's write-back lands in the one copy.
        """
        source = Path(part)
        if not part or not source.is_file():
            return part
        return self._place(name, part, file_sha256(source), lambda target: shutil.copyfile(source, target))

    def _place(self, name: str, source: str, digest: str, write: Callable[[Path], object]) -> str:
        target = self._tables / name / digest[:_TABLE_DIGEST_CHARS] / Path(source).name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            write(target)
        table = {"source": source, "sha256": digest, "copy": str(target)}
        listed = self.copied.setdefault(name, [])
        if table not in listed:
            listed.append(table)
        return str(target)

    def _rows_over_default(self, name: str, rows: list[str]) -> str:
        """The tables aiter reads for ``name`` when it is unset, minus the shapes ``rows`` tunes, then ``rows``.

        A default table that keeps all its rows is the private copy ``default`` names; one that loses some is written
        beside the rows' copy, named by what it now holds.
        """
        variable = self._snapshot.variables.get(name)
        defaults = [Path(part) for part in (self.default(name) or "").split(os.pathsep) if Path(part).is_file()]
        if not defaults:
            raise RowsOnlyRefused(
                f"{name} has no default table in this image to lay the rows over: the install does not declare it, "
                "or ships no table for it. The rows alone would be read as the whole table"
            )
        copies = [self._copy(name, part) for part in rows]
        over = lay_rows_over(defaults, [Path(copy) for copy in copies], self._configs, variable and variable.tuned)
        self.rows_only[name] = over.document()
        listed = [
            self._place_kept(name, path, over.kept[path]) if path in over.kept else str(path) for path in defaults
        ]
        return os.pathsep.join([*listed, *copies])

    def _place_kept(self, name: str, private: Path, data: bytes) -> str:
        """The default table ``private`` is a copy of, as it is read once its overridden rows are out."""
        with contextlib.suppress(ValueError):
            private = self._snapshot.source / private.relative_to(self._configs)
        return self._place(
            name, str(private), hashlib.sha256(data).hexdigest(), lambda target: target.write_bytes(data)
        )

    def default(self, name: str) -> str | None:
        """What aiter reads for ``name`` when it is unset, as private files: its default table and the per-model ones.

        ``None`` for a variable the snapshot does not know, or one with neither a default table nor a per-model one.
        """
        variable = self._snapshot.variables.get(name)
        if variable is None:
            return None
        pattern = f"*{variable.tuned}*.csv"
        models = [
            str(self._configs / "model_configs" / model)
            for model in self._snapshot.model_order
            if fnmatch.fnmatchcase(model, pattern) and "untuned" not in model
        ]
        default = self._configs / variable.default
        if not default.is_file():
            # aiter skips a listed table that does not exist, and merges what is left only when more than one is listed.
            return os.pathsep.join([str(self._snapshot.source / variable.default), *models]) if models else None
        return os.pathsep.join([str(default), *models])

    def process_env(self, call: TestCall, unset: frozenset[str]) -> dict[str, str]:
        """The value of every ``AITER_CONFIG_*`` variable the test's process sees, with private tables."""
        named = {name for name in (*os.environ, *call.env_overrides) if name.startswith(AITER_VARIABLE_PREFIX)}
        values: dict[str, str] = {}
        for name in sorted(named | set(self._snapshot.variables)):
            inherited = call.env_overrides[name] if name in call.env_overrides else os.environ.get(name, "")
            given = "" if name in unset else str(inherited).strip()
            value = self.value(name, given) if given else self.default(name)
            if value is not None:
                values[name] = value
        return values

    def point(self, path: Path, index: int, unset: frozenset[str]) -> Path:
        """``path``, or a copy of it whose own ``AITER_CONFIG_*`` values name private tables."""
        config = yaml.safe_load(Path(path).read_text(encoding=_UTF8)) or {}
        envs = server_environment(config)
        named = [name for name in envs if str(name).startswith(AITER_VARIABLE_PREFIX)]
        hook = self.hook if "PYTHONPATH" in envs else None
        if not named and hook is None:
            return path
        for name in named:
            given = "" if name in unset else str(envs[name]).strip()
            if given:
                envs[name] = self.value(str(name), given)
            else:
                del envs[name]
        if hook is not None:
            prepend_pythonpath(envs, hook)
        self._points.mkdir(parents=True, exist_ok=True)
        target = self._points / f"point_{index}.yaml"
        target.write_text(yaml.safe_dump(config, sort_keys=False), encoding=_UTF8)
        return target

    def provenance(self) -> dict[str, list[dict[str, str]]]:
        """The point's own tables by variable: where each came from and its hash."""
        return {
            name: [{"source": table["source"], "sha256": table["sha256"]} for table in tables]
            for name, tables in self.copied.items()
        }

    def rewritten(self, private: Path) -> list[str]:
        """The private files that differ from what they were copied from: aiter's write-back, kept in the test."""
        configs = tree_files(self._configs)
        names = {
            name for name in (*configs, *self._snapshot.files) if configs.get(name) != self._snapshot.files.get(name)
        }
        changed = {f"{_CONFIGS}/{name}" for name in names}
        for listed in self.copied.values():
            for table in listed:
                copy = Path(table["copy"])
                if not copy.is_file() or file_sha256(copy) != table["sha256"]:
                    changed.add(str(copy.relative_to(private)))
        return sorted(changed)


def refuse_default_table(call: TestCall, why: str) -> None:
    """Refuse a call that names the image's default table when there are no private tables to resolve it from.

    Left in the environment the name would be a path aiter skips, and the test would read the rows alone.

    Raises:
        RowsOnlyRefused: When an environment of the call lists ``DEFAULT_TABLE_PLACEHOLDER``.
    """
    environments = [dict(call.env_overrides)]
    environments += [
        server_environment(yaml.safe_load(Path(point).read_text(encoding=_UTF8)) or {}) for point in call.points
    ]
    for envs in environments:
        named = sorted(str(name) for name, value in envs.items() if DEFAULT_TABLE_PLACEHOLDER in str(value))
        if named:
            raise RowsOnlyRefused(
                f"{named} lists the image's default table next to tuned rows, which needs the test's private "
                f"copies of the framework tables, and the test has none ({why})"
            )


def _private_base() -> str | None:
    """The sandbox's own ``work`` directory when the process runs in one, so the copies go with the sandbox."""
    root = os.environ.get(_SANDBOX_ROOT_ENV)
    work = Path(root) / "work" if root else None
    return str(work) if work is not None and work.is_dir() else None


class _TestBase:
    """What one test began on and left behind, and the document that says so."""

    def __init__(self, directory: Path, session_sha256: str | None) -> None:
        self._directory = directory
        self._session = session_sha256
        self._snapshot: Snapshot | None = None
        self._private: Path | None = None
        self._redirect: _Redirect | None = None
        self._merge: MergeView | None = None
        self.document: dict[str, Any] = {
            "schema": _SCHEMA,
            "isolation": "none",
            "reason": None,
            "source": None,
            "session_sha256": session_sha256,
            "private_sha256": None,
            "shared_sha256_start": None,
            "shared_sha256_end": None,
            "matches_session": None,
            "framework_base_changed": None,
            "restored": {"start": [], "end": []},
            "restore_failed": [],
            "redirected": [],
            "point_tables": {},
            "rows_only": {},
            "private_rewritten": [],
            "merge_isolation": "shared",
            "merge_reason": "no private directory",
            "merge_chain_failure": None,
            "merge_dir": None,
            "merge_patched_processes": 0,
            "merge_files": {},
        }

    def begin(self, call: TestCall) -> PrivateFramework:
        """Check the shared tables, make the private copies and name them; never refuses the test."""
        try:
            self._snapshot = load_snapshot(self._directory)
        except FrameworkBaseError as exc:
            return self._shared(call, str(exc))
        snapshot = self._snapshot
        self._session = self._session or snapshot.sha256
        self.document.update(source=str(snapshot.source), session_sha256=self._session)
        shared = self._watch_shared(snapshot, "start")
        try:
            return self._isolate(snapshot, call)
        except OSError as exc:
            self._drop_private()
            self.document["matches_session"] = None if shared is None else shared == self._session
            return self._shared(call, f"the private copies cannot be made: {type(exc).__name__}: {exc}")

    def _shared(self, call: TestCall, reason: str) -> PrivateFramework:
        refuse_default_table(call, reason)
        self.document.update(isolation="none", reason=reason)
        return PrivateFramework(call)

    def _isolate(self, snapshot: Snapshot, call: TestCall) -> PrivateFramework:
        private = self._private = Path(tempfile.mkdtemp(prefix="hl-framework-", dir=_private_base()))
        shutil.copytree(snapshot.copies, private / _CONFIGS)
        private_sha = files_digest(tree_files(private / _CONFIGS))
        redirect = self._redirect = _Redirect(snapshot, private)
        unset = frozenset(call.unset_envs) | frozenset(
            call.session_facts.reference_unset_envs if call.session_facts else ()
        )
        values = redirect.process_env(call, unset)
        hook_env = self._install_merge_hook(redirect, private)
        points = tuple(redirect.point(point, index, unset) for index, point in enumerate(call.points))
        self.document.update(
            isolation="private_copies",
            private_sha256=private_sha,
            matches_session=private_sha == self._session,
            redirected=sorted(values),
            point_tables=redirect.provenance(),
            rows_only=redirect.rows_only,
        )
        isolated = dataclasses.replace(call, points=points, env_overrides={**call.env_overrides, **values, **hook_env})
        return PrivateFramework(isolated, frozenset(values))

    def _install_merge_hook(self, redirect: _Redirect, private: Path) -> dict[str, str]:
        """The variables that give the test's server its own merge directory; none, and the reason, when it cannot."""
        try:
            self._merge = MergeView.make(private)
        except OSError as exc:
            self.document["merge_reason"] = f"the hook cannot be installed: {type(exc).__name__}: {exc}"
            return {}
        redirect.hook = self._merge.hook
        return self._merge.environment(os.environ.get("PYTHONPATH", ""))

    def _watch_shared(self, snapshot: Snapshot, when: str) -> str | None:
        """Hash the shared tables against the snapshot and put back what differs; the digest, ``None`` if unreadable."""
        try:
            digest, changed = shared_view(snapshot)
        except OSError as exc:
            log.warning("the shared framework tables cannot be read: %s", exc)
            return None
        self.document[f"shared_sha256_{when}"] = digest
        if changed:
            restored, failed = restore(snapshot, changed)
            self.document["restored"][when] = restored
            self.document["restore_failed"] = sorted({*self.document["restore_failed"], *failed})
        if when == "end":
            self.document["framework_base_changed"] = bool(changed)
        return digest

    def _drop_private(self) -> None:
        if self._private is not None:
            shutil.rmtree(self._private, ignore_errors=True)
        self._private = self._redirect = self._merge = None

    def finish(self, output_dir: Path) -> None:
        """Hash what the test left, put the shared tables back if it changed them, and write the document."""
        try:
            if self._snapshot is not None:
                self._watch_shared(self._snapshot, "end")
            if self._redirect is not None and self._private is not None:
                self.document["private_rewritten"] = self._redirect.rewritten(self._private)
            if self._merge is not None:
                self.document.update(self._merge.report())
        except OSError as exc:
            log.warning("the framework base cannot be measured after the test: %s", exc)
        finally:
            self._drop_private()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / FRAMEWORK_BASE_FILE).write_text(json.dumps(self.document, sort_keys=True), encoding=_UTF8)
        except OSError as exc:
            log.warning("framework_base.json cannot be written: %s", exc)


@contextlib.contextmanager
def private_framework(call: TestCall) -> Iterator[PrivateFramework]:
    """Run ``call`` on private copies of the session's tuned tables, as this process's environment names them.

    A process without the snapshot variables (a hosted sandbox is its own copy of the image, and a machine without the
    framework has nothing to share) runs the call as it is and leaves no ``framework_base.json``. With them, the test
    runs even when the copies cannot be made; its result then says it ran on the shared tables.
    """
    directory = os.environ.get(SNAPSHOT_ENV)
    if not directory:
        refuse_default_table(call, f"{SNAPSHOT_ENV} is not set")
        yield PrivateFramework(call)
        return
    base = _TestBase(Path(directory), os.environ.get(SESSION_SHA_ENV))
    try:
        yield base.begin(call)
    finally:
        base.finish(Path(call.output_dir))
