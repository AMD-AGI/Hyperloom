# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keep credentials out of a frozen base launch, which is written to the session and carried in every sheet.

Only the base's own env has a redacted form: the runner fills a :class:`~inference_suite.contracts.Redacted` value from
its host's variable of the same name. A credential anywhere else (an arg, an entrypoint word, the client's or the golden
layer's env) has no name a host could hold it under, so a base carrying one is refused rather than frozen.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping, Sequence

from hyperloom.common.env_safety import credential_arg_indices, is_credential_env_name, redact_secret_values
from inference_suite.contracts import BaseLaunch, Redacted


def redact_credential_env(base: BaseLaunch) -> BaseLaunch:
    """``base`` with the value of every credential-named env replaced by its sha256."""
    env = {
        name: Redacted(hashlib.sha256(value.encode()).hexdigest())
        if isinstance(value, str) and is_credential_env_name(name)
        else value
        for name, value in base.env.items()
    }
    return base if env == base.env else dataclasses.replace(base, env=env)


def _clear_env(what: str, env: Mapping[str, object]) -> list[str]:
    return [
        f"{what} {name}"
        for name, value in env.items()
        if isinstance(value, str) and (is_credential_env_name(name) or redact_secret_values(value) != value)
    ]


def _clear_words(what: str, words: Sequence[str]) -> list[str]:
    return [f"{what} {index}" for index in credential_arg_indices(words)]


def credentials_in_the_clear(base: BaseLaunch) -> list[str]:
    """Where ``base`` holds a credential that is not redacted, by place and env name or word index, never by value."""
    found = _clear_words("entrypoint word", base.entrypoint) + _clear_words("arg", base.args)
    found += _clear_env("env", base.env)
    found += _clear_words("client arg", base.client.args) + _clear_env("client env", base.client.env)
    if base.golden_al is not None:
        found += _clear_words("golden arg", base.golden_al.args) + _clear_env("golden env", base.golden_al.env)
    return found


__all__ = ["credentials_in_the_clear", "redact_credential_env"]
