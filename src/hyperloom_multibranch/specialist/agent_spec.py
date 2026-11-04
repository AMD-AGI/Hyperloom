# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What MB asks of an autonomous branch's agent, which the onboarding (:mod:`agent_onboarding`) and Arbor's adapter
(:mod:`arbor_adapter`) both derive the agent's config from."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from hyperloom.common.workload_defaults import DEFAULT_EP
from hyperloom_multibranch.contracts.blackboard.records import OpShares

#: What the root's shares are, as the profile file names their source: its op shares, the blackboard's.
ROOT_PROFILE_SOURCE = "PRELUDE root profile"


@dataclass(frozen=True)
class AgentSpec:
    """What MB asks of the agent, which its config must be derived from: the workload, the metric it is judged on, the
    server arguments and environment of the commit its branch starts on, ``workspace``, the framework checkout it edits,
    ``cards``, the GPUs it may use, ``seconds``, how long it may run (0: until MB stops it), ``port``, the first of
    the ports its servers may listen on, ``client``, the command the gate measures a server on ``port`` with (empty when
    the session has none to give), ``baseline``, what the gate measured of the metric on the session's root (``None``
    before it did), ``knowledge``, the notes on the integration head the agent reads each iteration, and, for an agent
    run as a component, ``deadline``, the clock time it ends by, ``root``, the commit of ``workspace`` its patches
    are made against, and ``root_shares``, the op shares of the session root's profile (``None`` when it has none),
    which the profile file says came from ``root_shares_source``. Arbor (:mod:`arbor_adapter`) is also given ``ep``,
    the expert-parallel size the workload is served at, and may be given ``plan``, the file of the branch's share of
    the session's directions, ``pack``, the name of the pack it runs (``None`` for the adapter's default),
    ``launch``, the file of the launch the recipe's adds to (``None`` when the recipe's is the whole launch),
    ``dossier``, the dossier the session's dossier pass wrote, for its checkout (``None`` without one), and
    ``review``, how its review gate is handed over (:data:`~hyperloom_multibranch.specialist.arbor_adapter.REVIEW_HOST`
    when MB reviews before its branches open, :data:`~hyperloom_multibranch.specialist.arbor_adapter.REVIEW_OFF` when
    nobody does, ``""`` to leave the config without either); the onboarding model is told none of them."""

    model: str
    framework: str
    tp: int
    precision: str
    gpu_type: str
    workload: Mapping[str, object]
    metric: str
    server_args: str
    envs: Mapping[str, str]
    workspace: str
    cards: tuple[int, ...]
    seconds: float
    port: int
    client: str = ""
    baseline: float | None = None
    knowledge: str = ""
    deadline: float = 0.0
    root: str = ""
    root_shares: OpShares | None = None
    plan: str | None = None
    ep: int = DEFAULT_EP
    pack: str | None = None
    launch: str | None = None
    dossier: str | None = None
    root_shares_source: str = ROOT_PROFILE_SOURCE
    review: str = ""

    def to_document(self) -> dict[str, object]:
        return {
            "model": self.model,
            "framework": self.framework,
            "tp": self.tp,
            "precision": self.precision,
            "gpu_type": self.gpu_type,
            "workload": dict(self.workload),
            "metric": self.metric,
            "server_args": self.server_args,
            "envs": dict(self.envs),
            "workspace": self.workspace,
            "cards": list(self.cards),
            "seconds": self.seconds,
            "port": self.port,
            "client": self.client,
            "baseline": self.baseline,
            "knowledge": self.knowledge,
            "deadline": self.deadline,
            "root": self.root,
        }
