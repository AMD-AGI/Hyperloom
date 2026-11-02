# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Collect forge-loop runs into the workspace's Experience KB.

The workspace has one Experience KB, which an optimize run writes its framework Experiences to and a forge-loop run
its kernel ones, each under its own schema. The run takes it as an optimize run does (``use_workspace_kb``): the KB
settings of the workspace's ``.env`` that its environment does not set, and, when neither names a KB, the loopback
URL and token setup writes into that ``.env``. The KB is the one ``HYPERLOOM_KB_URL`` then names, with its token in
``HYPERLOOM_KB_TOKEN`` or in the file ``HYPERLOOM_KB_TOKEN_FILE`` names, which the knowledge base's scripts also search
during a ``--wiki-guided`` campaign's knowledge base round. A run is collected once as it ends (``collect_run``), and
``kernelforge export-runs --collect`` collects any run, such as one stopped before its end. Collection is
``kernelforge export-runs`` followed by ``hyperloom-kb-collect --mapping kernelforge-v1``. Collecting a run again
writes nothing new: an Experience already there unchanged reports ``unchanged``. With ``HYPERLOOM_KB_AUTO_PUSH`` on,
the KB is then asked to push what it holds to its global KB, as a Hyperloom session asks. A KB that cannot be reached,
refuses a record, or cannot push is reported; it never fails the run.

A loopback URL names the workspace's local service, which the launch brings to serving (``prepare_collection``), as
an optimize launch does: from ``$USER_DATA_PATH/experience-kb`` unless ``HYPERLOOM_KB_HOME`` names another home. It is
left serving after the run, and one already serving with other settings is left as it is, since another run may be
using it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from hyperloom_common.env import EnvValueError, env_bool
from hyperloom_common.workspace import ENV_USER_DATA_PATH, parse_env_assignments, workspace_dotenv_target
from hyperloom_kb import workspace as kb_workspace
from hyperloom_kb import (
    ConfigurationError,
    LocalServiceError,
    RemoteClient,
    RemoteClientError,
    RemoteConfig,
    ensure_local_service,
    experience_kb_from_env,
    is_loopback,
)
from hyperloom_kb.collect import CollectReport, MappingError, SourceDocumentError, collect, load_mapping
from hyperloom_kb.workspace import TOKEN_ENV, TOKEN_FILE_ENV, URL_ENV, configured_token

from kernelforge.workflows.forge_loop.campaign.experiments_layout import ExperimentsLayout
from kernelforge.workflows.forge_loop.export.document import normalize, write_document

log = logging.getLogger(__name__)

MAPPING = "kernelforge-v1"
DOCUMENT, RECEIPT, SPOOL = "runs.json", "collect_receipt.json", "spool"
AUTO_PUSH_ENV = "HYPERLOOM_KB_AUTO_PUSH"
# What one KB request may take. A collection makes one per candidate, so a KB that stops answering costs a run's end
# that much per candidate before the writes are spooled for the next collection.
REQUEST_TIMEOUT_SECONDS = 30.0
#: What stops one collection. The run is left as it was, so collecting it again starts over.
COLLECTION_ERRORS = (ConfigurationError, MappingError, RemoteClientError, SourceDocumentError, OSError)
# The settings of the workspace's .env a run takes for its Experience KB when its environment does not set them: the
# KB's, the global KB's, the local service's planner, and the data root its home is under.
_KB_SETTING_PREFIXES = ("HYPERLOOM_KB_", "HYPERLOOM_GLOBAL_KB_", "LOCAL_KB_PLANNER_")


def _take_kb_settings(env_file: Path, environ: MutableMapping[str, str]) -> None:
    if not env_file.is_file():
        return
    for key, value in parse_env_assignments(env_file.read_text(encoding="utf-8", errors="replace")).items():
        named = key == ENV_USER_DATA_PATH or key.startswith(_KB_SETTING_PREFIXES)
        if named and value and not environ.get(key, "").strip():
            environ[key] = value


def use_workspace_kb(environ: MutableMapping[str, str] = os.environ) -> str:
    """Name in ``environ`` the workspace's Experience KB, as an optimize run takes it: the KB settings of the
    workspace's ``.env`` the environment does not set, and, when neither names a KB, a loopback URL and a generated
    token written into that ``.env`` as setup writes them. The launch line when it wrote them or could not, else
    empty."""

    env_file = workspace_dotenv_target(environ)
    try:
        _take_kb_settings(env_file, environ)
        if environ.get(URL_ENV, "").strip():
            return ""
        kb_workspace.init_env(env_file)
        _take_kb_settings(env_file, environ)
    except (LocalServiceError, OSError, ValueError) as error:
        return f"  Experience KB: none, the workspace's settings could not be written to {env_file}: {error}"
    return f"  Experience KB: the workspace's local KB, whose URL and token were written to {env_file} as setup does"


def collection_environment(environ: Mapping[str, str] = os.environ) -> dict[str, str] | None:
    """The environment ``hyperloom_kb`` reads its KB from, with the token from its file when only a file names it;
    None without a KB URL."""

    if not environ.get(URL_ENV, "").strip():
        return None
    return {**environ, TOKEN_ENV: configured_token(environ)}


def collection_target(environ: Mapping[str, str] = os.environ) -> str:
    """The launch line saying where the run is collected as it ends, or why it cannot be; empty without a KB URL."""

    env = collection_environment(environ)
    if env is None:
        return ""
    if not env[TOKEN_ENV]:
        missing = f"neither {TOKEN_ENV} nor {TOKEN_FILE_ENV} gives a token"
        return f"  Experience KB collection: off, {URL_ENV} is set but {missing}"
    try:
        schema_ref = load_mapping(MAPPING).declaration.schema_ref
    except MappingError as error:
        return f"  Experience KB collection: off, the {MAPPING} mapping does not load: {error}"
    return f"  Experience KB collection: {env[URL_ENV]} ({schema_ref}) as the run ends"


def start_local_kb(environ: Mapping[str, str] = os.environ) -> str:
    """Bring the local service a loopback KB URL names to serving, and say what serves it; empty for a KB elsewhere or
    none. A service that cannot serve is reported, and the run's writes are spooled until one does."""

    env = collection_environment(environ)
    if env is None or not env[TOKEN_ENV] or not is_loopback(env[URL_ENV]):
        return ""
    home = kb_workspace.service_home(env)
    try:
        config = RemoteConfig.from_env(env, timeout_seconds=REQUEST_TIMEOUT_SECONDS)
        service = ensure_local_service(config, home, env=kb_workspace.service_env(env), restart=False)
    except (LocalServiceError, RemoteClientError, OSError) as error:
        return f"  Experience KB service: not serving ({error}); this run's Experiences wait in its spool until one is"
    started = "started" if service.process is not None else "serving"
    stale = f"; left as it is: {service.stale}" if service.stale else ""
    return f"  Experience KB service: {started} at {env[URL_ENV]} from {home}{stale}"


def prepare_collection(environ: Mapping[str, str] = os.environ, *, note: str = "") -> list[str]:
    """Start the local KB the run collects into when it is one, and the launch lines saying so, after ``note``, what
    ``use_workspace_kb`` said."""

    return [line for line in (note, start_local_kb(environ), collection_target(environ)) if line]


def collect_document(document: Path, work: Path, environ: Mapping[str, str] = os.environ) -> CollectReport | None:
    """Collect an exported ``document`` through the ``kernelforge-v1`` mapping, its receipt and the writes the KB has
    not taken yet kept in ``work``; None without a KB URL."""

    env = collection_environment(environ)
    if env is None:
        return None
    mapping = load_mapping(MAPPING)
    target = experience_kb_from_env(
        env, spool_root=work / SPOOL, declaration=mapping.declaration, timeout_seconds=REQUEST_TIMEOUT_SECONDS
    )
    return collect(mapping, document, kb=target, receipt=work / RECEIPT)


def describe(report: CollectReport, receipt: Path) -> str:
    """One line on what a collection wrote."""

    counts = report.to_dict()["counts"]
    statuses = ", ".join(f"{count} {status}" for status, count in counts["by_status"].items()) or "nothing written"
    return (
        f"collected {counts['collected']} of {counts['units']} candidates under {report.schema_ref} ({statuses}); "
        f"{counts['skipped']} skipped, {counts['errors']} failed, as {receipt} says"
    )


def push_to_global(environ: Mapping[str, str] = os.environ) -> str:
    """Ask the KB to push what it holds to its global KB when ``HYPERLOOM_KB_AUTO_PUSH`` is on, and say what became of
    it; empty when it was not asked."""

    env = collection_environment(environ)
    try:
        asked = env is not None and env_bool(AUTO_PUSH_ENV, env=env)
    except EnvValueError as error:
        return f"push skipped: {error}"
    if not asked:
        return ""
    try:
        report = RemoteClient(RemoteConfig.from_env(env, timeout_seconds=REQUEST_TIMEOUT_SECONDS)).push()
    except (RemoteClientError, OSError) as error:
        return f"push failed, and the next push sends these Experiences: {error}"
    rejected = report.get("rejected") if isinstance(report.get("rejected"), list) else []
    return (
        f"pushed to the global KB: {report.get('created', 0)} created, {report.get('unchanged', 0)} unchanged, "
        f"{len(rejected)} rejected"
    )


def collect_run(experiments_dir: Path, environ: Mapping[str, str] = os.environ) -> None:
    """Collect the run kept in ``experiments_dir`` as it ends, and push it when asked; reports and never raises."""

    if collection_environment(environ) is None:
        return
    work = ExperimentsLayout(Path(experiments_dir)).experience_collection
    try:
        write_document(normalize((Path(experiments_dir),)), work / DOCUMENT)
        report = collect_document(work / DOCUMENT, work, environ)
    except COLLECTION_ERRORS as error:
        log.warning("Experience KB collection of %s failed", experiments_dir, exc_info=True)
        print(
            f"  [experience-kb] not collected ({type(error).__name__}: {error}); collect it with "
            f"`kernelforge export-runs {experiments_dir} -o {work / DOCUMENT} --collect`",
            flush=True,
        )
        return
    if report is not None:
        print(f"  [experience-kb] {describe(report, work / RECEIPT)}", flush=True)
    if pushed := push_to_global(environ):
        print(f"  [experience-kb] {pushed}", flush=True)
