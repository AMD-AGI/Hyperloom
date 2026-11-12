# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve a PD launch once per session, from the first source that serves the query.

The sources, in order: the operator's per-role args (or an earlier record, replayed exactly); an InferenceX main
master multi-node disaggregated entry; an InferenceX main recipe no master entry names, found by its directory; a
row of the PD default table. A query none of them serves is refused, naming what each passed over. InferenceX that
cannot be read is passed over for the table, unless the session pinned an InferenceX commit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from inference_suite.argv import canonical_flag_name, is_flag
from inference_suite.contracts.base_launch import Workload
from inference_suite.frameworks import Framework
from inference_suite.multinode.inferencex import (
    NVIDIA_MASTER,
    Candidate,
    Candidates,
    RoleFit,
    lookup_candidates,
    master_candidates,
)
from inference_suite.multinode.query import DEFAULT_FRONTEND, PdQuery, RoleAsk
from inference_suite.multinode.record import (
    Frontend,
    PdLaunch,
    PdSource,
    PdSourceKind,
    Role,
    RoleLaunch,
)
from inference_suite.multinode.render import RenderedRole, recipe_role_args, render_role
from inference_suite.multinode.table import TABLE_COMMIT, TABLE_REPO, RoleShape, TableError, TableQuery, compose
from inference_suite.recipe.inferencex import MASTER, RUNNERS, srtctl_cli_args
from inference_suite.recipe.model import RecipeSourceError
from inference_suite.recipe.sources import Checkout, SourceFetchError

#: The KV transfer backend of a launch no source names one for, by framework.
DEFAULT_TRANSFER_BACKEND = {Framework.SGLANG: "mori", Framework.ATOM: "mooncake"}
_TRUST_REMOTE_CODE = "--trust-remote-code"
#: The router flags the PD launcher sets, in canonical spelling.
ROUTER_OWNED = frozenset({"host", "port", "pd-disaggregation", "prefill", "decode", "worker-urls"})


class PdResolveError(ValueError):
    """No source serves the query, or the pinned InferenceX commit cannot be read."""


@dataclass(frozen=True)
class OperatorArgs:
    """The operator's server args: for both roles, then for each role alone, and the router's."""

    shared: tuple[str, ...] = ()
    prefill: tuple[str, ...] = ()
    decode: tuple[str, ...] = ()
    router: tuple[str, ...] = ()

    @property
    def given(self) -> bool:
        return bool(self.shared or self.prefill or self.decode)

    def role(self, role: Role) -> tuple[str, ...]:
        return (*self.shared, *(self.prefill if role is Role.PREFILL else self.decode))


@dataclass(frozen=True)
class ResolveInputs:
    """What a resolution reads besides the query: the operator's args, an earlier launch to replay, how to check
    InferenceX out (``None`` reads none) and whether its commit is pinned."""

    operator: OperatorArgs = field(default_factory=OperatorArgs)
    replay: PdLaunch | None = None
    inferencex: Callable[[], Checkout] | None = None
    pinned: bool = False


# --- roles --------------------------------------------------------------------------------------------------------


def _asked_fit(ask: RoleAsk, gpus_per_node: int) -> RoleFit:
    """The topology a role takes from the operator, else one worker of the node's GPUs per node."""
    tp = ask.tp or gpus_per_node
    workers = max(1, len(ask.hosts) * gpus_per_node // max(tp, gpus_per_node))
    return RoleFit(workers, tp, ask.ep or 1, bool(ask.dp_attention), tp)


def _role_launch(ask: RoleAsk, fit: RoleFit, rendered: RenderedRole) -> RoleLaunch:
    return RoleLaunch(
        hosts=ask.hosts,
        workers=fit.workers,
        tp=fit.tp,
        ep=fit.ep,
        dp_attention=fit.dp_attention,
        args=rendered.args,
        env=rendered.env,
        stripped=rendered.stripped,
    )


def _asks(query: PdQuery) -> dict[Role, RoleAsk]:
    return {Role.PREFILL: query.prefill, Role.DECODE: query.decode}


def _router_args(args: Sequence[str]) -> tuple[str, ...]:
    """``args`` without the router flags the launcher sets, each with its values."""
    kept: list[str] = []
    dropping = False
    for arg in args:
        if is_flag(arg):
            dropping = canonical_flag_name(arg.partition("=")[0]) in ROUTER_OWNED
        if not dropping:
            kept.append(arg)
    return tuple(kept)


@dataclass(frozen=True)
class _Bound:
    """A source's launch before assembly: each role's rendered args and topology, the router's args, the KV
    transfer backend the source names and the source itself."""

    rendered: Mapping[Role, RenderedRole]
    fits: Mapping[Role, RoleFit]
    router_args: tuple[str, ...]
    transfer_backend: str
    source: PdSource


def _assemble(query: PdQuery, bound: _Bound) -> PdLaunch:
    """The launch ``bound`` makes on the query's hosts; the query's transfer backend first, then the source's."""
    asks = _asks(query)
    named = (bound.rendered[role].transfer_backend for role in Role)
    fallback = DEFAULT_TRANSFER_BACKEND[query.framework]
    backend = query.transfer_backend or bound.transfer_backend or next((b for b in named if b), fallback)
    roles = {role: _role_launch(asks[role], bound.fits[role], bound.rendered[role]) for role in Role}
    return PdLaunch(
        source=bound.source,
        framework=query.framework,
        model=query.model_path,
        served_model_name=query.model_id,
        context_length=context_length(query),
        transfer_backend=backend,
        prefill=roles[Role.PREFILL],
        decode=roles[Role.DECODE],
        frontend=Frontend(query.served_frontend, query.router_host, _router_args(bound.router_args)),
    )


# --- sources ------------------------------------------------------------------------------------------------------


def _operator(query: PdQuery, operator: OperatorArgs) -> PdLaunch:
    fits = {role: _asked_fit(ask, query.gpus_per_node) for role, ask in _asks(query).items()}
    rendered = {role: render_role(query.framework, operator.role(role), {}) for role in Role}
    source = PdSource(PdSourceKind.OPERATOR)
    return _assemble(query, _Bound(rendered, fits, operator.router, "", source))


def context_length(query: PdQuery) -> int | None:
    """The context the engines serve: for a fixed workload the least power of two that holds ``isl + osl``, for an
    AgentX one the engine's own (``None``)."""
    if query.isl is None or query.osl is None:
        return None
    return 1 << (query.isl + query.osl - 1).bit_length()


def _replay(query: PdQuery, launch: PdLaunch) -> PdLaunch:
    """``launch`` on the query's hosts, refused when it is for another framework, model or host count."""
    if launch.framework is not query.framework:
        raise PdResolveError(f"the replayed launch runs {launch.framework.value}, not {query.framework.value}")
    if launch.model != query.model_path:
        raise PdResolveError(f"the replayed launch serves {launch.model}, not {query.model_path}")
    for role, ask in _asks(query).items():
        if len(launch.role(role).hosts) != len(ask.hosts):
            raise PdResolveError(
                f"the replayed launch runs {role.value} on {len(launch.role(role).hosts)} hosts, not {len(ask.hosts)}"
            )
    return launch.with_hosts(query.prefill.hosts, query.decode.hosts, query.router_host)


def _checkout_files(checkout: Checkout, candidate: Candidate) -> dict[str, str]:
    paths = [MASTER, candidate.path]
    if candidate.kind is PdSourceKind.INFERENCEX_MASTER:
        paths.append(RUNNERS)
    elif (checkout.root / NVIDIA_MASTER).is_file():
        paths.append(NVIDIA_MASTER)
    bundle = checkout.bundle(sorted(set(paths)))
    return {path: bundle.sha256(path) for path in bundle.files}


def _candidate(query: PdQuery, checkout: Checkout, found: Sequence[Candidate]) -> PdLaunch:
    """The first of ``found`` at the nearest concurrency, with the others named as alternatives."""
    target = query.target_concurrency
    ordered = sorted(found, key=lambda candidate: (candidate.conc or target) - target)
    chosen, others = ordered[0], ordered[1:]
    roles = chosen.recipe["roles"]
    rendered = {
        role: render_role(
            query.framework, recipe_role_args(query.framework, roles[role.value]), roles[role.value].get("env") or {}
        )
        for role in Role
    }
    frontend = chosen.recipe.get("frontend") or {}
    source = PdSource(
        kind=chosen.kind,
        repo=checkout.repo,
        commit=checkout.commit,
        path=chosen.path,
        entry=chosen.entry,
        locator=chosen.locator,
        files=_checkout_files(checkout, chosen),
        alternatives=tuple(other.entry or other.name for other in others),
        warnings=chosen.warnings,
    )
    router = srtctl_cli_args("sglang", dict(frontend.get("args") or {}))
    return _assemble(query, _Bound(rendered, chosen.fits, router, chosen.transfer_backend, source))


def _table_shape(fit: RoleFit) -> RoleShape:
    return RoleShape(fit.tp, ep=fit.ep > 1, dp=fit.dp_attention)


def _table(query: PdQuery, warnings: Sequence[str]) -> PdLaunch:
    """The PD default table's row for the model, composed for the query's topology."""
    fits = {role: _asked_fit(ask, query.gpus_per_node) for role, ask in _asks(query).items()}
    hit = compose(
        query.framework,
        TableQuery(
            model_name=query.model_name,
            prefill=_table_shape(fits[Role.PREFILL]),
            decode=_table_shape(fits[Role.DECODE]),
            max_concurrency=query.target_concurrency,
            agentic=query.workload is Workload.AGENTX,
        ),
    )
    composed = {Role.PREFILL: hit.prefill, Role.DECODE: hit.decode}
    rendered = {
        role: render_role(query.framework, (_TRUST_REMOTE_CODE, *composed[role].args), composed[role].env)
        for role in Role
    }
    source = PdSource(
        kind=PdSourceKind.PD_TABLE,
        repo=TABLE_REPO,
        commit=TABLE_COMMIT,
        path=hit.table.name,
        entry=hit.key,
        files={hit.table.name: hit.table.sha256},
        warnings=(
            *warnings,
            f"no InferenceX recipe serves {query.model_id}; the PD default table's {hit.key} row runs",
        ),
    )
    router = hit.router_args if query.served_frontend is DEFAULT_FRONTEND[query.framework] else ()
    return _assemble(query, _Bound(rendered, fits, router, "", source))


def _inferencex(query: PdQuery, inputs: ResolveInputs, misses: list[str], warnings: list[str]) -> PdLaunch | None:
    """The launch InferenceX main serves the query, its master entries before its lookup, else ``None`` with every
    candidate passed over appended to ``misses`` and an unreadable InferenceX to ``warnings``."""
    if inputs.inferencex is None:
        return None
    try:
        checkout = inputs.inferencex()
        for source in (master_candidates, lookup_candidates):
            candidates: Candidates = source(query, checkout)
            misses.extend(candidates.misses)
            if candidates.found:
                return _candidate(query, checkout, candidates.found)
    except (SourceFetchError, RecipeSourceError, OSError) as error:
        if inputs.pinned:
            raise PdResolveError(f"the pinned InferenceX commit cannot be read: {error}") from error
        warnings.append(f"InferenceX cannot be read: {error}")
    return None


def resolve(query: PdQuery, inputs: ResolveInputs) -> PdLaunch:
    """The PD launch of the first source that serves ``query``.

    Raises:
        PdResolveError: no source serves the query, a replayed launch does not fit it, or the pinned InferenceX
            commit cannot be read.
    """
    if inputs.replay is not None:
        return _replay(query, inputs.replay)
    if inputs.operator.given:
        return _operator(query, inputs.operator)
    misses: list[str] = []
    warnings: list[str] = []
    launch = _inferencex(query, inputs, misses, warnings)
    if launch is not None:
        return launch
    try:
        return _table(query, warnings)
    except TableError as error:
        misses.append(f"PD default table: {error}")
    passed = "\n  ".join((*warnings, *misses))
    raise PdResolveError(
        f"no source serves {query.model_id} with {query.framework.value} on {query.board}; give each role's server "
        f"args, or pass a recorded launch. Passed over:\n  {passed}"
    )


__all__ = [
    "DEFAULT_TRANSFER_BACKEND",
    "ROUTER_OWNED",
    "OperatorArgs",
    "PdResolveError",
    "ResolveInputs",
    "context_length",
    "resolve",
]
