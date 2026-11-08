# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one reader of the session's frozen base launch for the orchestrator.

Every round's server starts from the base: its argv sits below the round's own args (last wins) and its environment
below the round's envs. The base comes from ``base_launch.json`` and must be the one ``SharedState`` recorded by digest;
a session that froze no base (a scriptable workload) starts from nothing. A round runs the client this install
stages, which is keyed by the InferenceX pin, so a base whose client names another commit is refused rather than
measured with one client and signed with the other.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from hyperloom.common.env_safety import is_allowed_external_env_key
from hyperloom.inference_optimizer.session.base_launch import base_launch_path, read_base_launch, write_base_launch
from hyperloom.inference_optimizer.session.paths import session_dir
from inference_suite import INFERENCEX_CLIENT_COMMIT
from inference_suite.contracts import BaseLaunch

from .shared_state import SharedState


class BaseLaunchMismatch(RuntimeError):
    """``base_launch.json`` is not the base the session recorded, or its client is off the InferenceX pin."""


@dataclass(frozen=True)
class ServerBase:
    """The server half of the base, its golden layer last: argv as one string and the environment a launch may set.

    Redacted values and keys a launch may not set from outside stay in ``base_launch.json`` and never reach a server.
    """

    args: str = ""
    env: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))


def _server_base(base: BaseLaunch) -> ServerBase:
    golden = base.golden_al
    args = (*base.args, *golden.args) if golden is not None else base.args
    layers = {**base.env, **golden.env} if golden is not None else base.env
    env = {
        name: value for name, value in layers.items() if isinstance(value, str) and is_allowed_external_env_key(name)
    }
    return ServerBase(args=" ".join(args), env=MappingProxyType(env))


@dataclass(frozen=True)
class FrozenBase:
    """The session's base launch and the file it is frozen in, which a suite config names by path and digest."""

    path: Path
    launch: BaseLaunch


def frozen_session_base() -> FrozenBase | None:
    """The current session's frozen base, ``None`` when it froze none.

    Raises:
        BaseLaunchMismatch: The file and the digest the session recorded disagree, or the base's client is off the
            InferenceX pin.
    """
    return frozen_base(session_dir())


def frozen_base(session: Path) -> FrozenBase | None:
    """The frozen base of the session at ``session``, ``None`` when it froze none.

    Raises:
        BaseLaunchMismatch: The file and the digest the session recorded disagree, or the base's client is off the
            InferenceX pin.
    """
    recorded = SharedState.load_or_init(session).base_launch_digest
    base = read_base_launch(session)
    found = base.digest() if base is not None else ""
    if found != recorded:
        raise BaseLaunchMismatch(
            f"{base_launch_path(session)} has digest {found or '(no file)'} but the session recorded "
            f"{recorded or '(none)'}; the session's base launch changed after it was frozen"
        )
    if base is not None and base.client is not None and base.client.commit != INFERENCEX_CLIENT_COMMIT:
        raise BaseLaunchMismatch(
            f"{base_launch_path(session)} was frozen with the InferenceX client at {base.client.commit}, but this "
            f"install pins {INFERENCEX_CLIENT_COMMIT}; start a new session to measure with the pinned client"
        )
    return None if base is None else FrozenBase(base_launch_path(session), base)


def freeze_base(session: Path, base: BaseLaunch) -> FrozenBase:
    """Freeze ``base`` as the base of the session at ``session``: its ``base_launch.json`` and the digest its state
    records, so :func:`frozen_base` reads it back."""
    path = write_base_launch(session, base)
    state = SharedState.load_or_init(session)
    state.base_launch_digest = base.digest()
    state.save(session)
    return FrozenBase(path, base)


def session_server_base() -> ServerBase:
    """The current session's server base; raises :class:`BaseLaunchMismatch` when the file and the state disagree."""
    frozen = frozen_session_base()
    return ServerBase() if frozen is None else _server_base(frozen.launch)


__all__ = [
    "BaseLaunchMismatch",
    "FrozenBase",
    "ServerBase",
    "freeze_base",
    "frozen_base",
    "frozen_session_base",
    "session_server_base",
]
