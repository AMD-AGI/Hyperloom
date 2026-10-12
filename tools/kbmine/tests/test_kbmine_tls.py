# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The CA bundle has to reach both services, not just Pulse.

Both hosts are signed by an internal CA that a stock machine does not carry.
For a while ``--ca-bundle`` was wired into ``PulseClient`` only, so every KB
request fell back on the ambient trust store: it worked on a developer box
exporting ``SSL_CERT_FILE`` and failed with ``CERTIFICATE_VERIFY_FAILED``
anywhere else. Nothing in the report distinguishes the two, which is what made
it survive. These tests pin the wiring rather than the behaviour, because a
real handshake needs the internal CA and a token.
"""

from __future__ import annotations

import inspect
import re

import pytest

from kbmine.kb_store_client import KBStoreClient
from kbmine.pulse import PulseClient

_URLOPEN = re.compile(r"urlopen\(")


def test_both_clients_accept_a_ca_bundle() -> None:
    for client in (KBStoreClient, PulseClient):
        assert "ca_bundle" in inspect.signature(client.__init__).parameters, client.__name__


def test_a_bundle_produces_a_context_and_its_absence_does_not() -> None:
    plain = KBStoreClient("https://kb.invalid", "t")
    assert plain._ctx is None, "no bundle must leave stdlib defaults alone"

    # Any readable PEM would do; certifi's bundle is one, when certifi is installed.
    certifi = pytest.importorskip("certifi")

    scoped = KBStoreClient("https://kb.invalid", "t", ca_bundle=certifi.where())
    assert scoped._ctx is not None


def test_no_kb_request_bypasses_the_context() -> None:
    """A bare ``urlopen`` is the exact shape of the original bug."""
    source = inspect.getsource(KBStoreClient)
    calls = [line.strip() for line in source.splitlines() if _URLOPEN.search(line)]
    assert calls, "expected to find the request sites"
    for call in calls:
        assert "context=self._ctx" in call, f"urlopen without the CA context: {call}"


def test_the_cli_hands_its_bundle_to_the_kb_client() -> None:
    source = inspect.getsource(__import__("kbmine.cli", fromlist=["cli"]))
    assert "KBStoreClient(store_url, token, ca_bundle=args.ca_bundle)" in source
    assert "PulseClient(args.pulse_url, token, ca_bundle=args.ca_bundle)" in source
