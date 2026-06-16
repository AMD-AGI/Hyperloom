# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared shims for the ci/ unit-test package.

The unit-test CI image does not install ``requests``. Sibling suites
(``inference_optimizer``) therefore drop a minimal ``requests`` stub into
``sys.modules`` exposing only ``HTTPError`` / ``exceptions`` / ``utils``.
The ci modules under test reference ``requests.get`` / ``requests.post`` /
``requests.Session``, so augment whichever ``requests`` module is live (real
or stub) *in place* before the ci tests import their targets: tests can then
monkeypatch those attributes and client construction does not explode.
No-op when the real ``requests`` package is installed.
"""

from __future__ import annotations

import sys
import types


def _ensure_requests_attrs() -> None:
    mod = sys.modules.get("requests")
    if mod is None:
        try:
            import requests as mod  # type: ignore[no-redef]
        except ModuleNotFoundError:
            mod = types.ModuleType("requests")
            sys.modules["requests"] = mod

    if not hasattr(mod, "HTTPError"):
        mod.HTTPError = type("HTTPError", (Exception,), {})
    if not hasattr(mod, "exceptions"):
        mod.exceptions = types.SimpleNamespace(RequestException=Exception)
    if not hasattr(mod, "utils"):
        mod.utils = types.SimpleNamespace(quote=lambda path, safe="": path)

    def _no_network(*_a, **_k):
        raise RuntimeError("requests stub: network disabled in unit tests")

    for _name in ("get", "post", "put", "delete", "patch", "head", "request"):
        if not hasattr(mod, _name):
            setattr(mod, _name, _no_network)

    if not hasattr(mod, "Session"):
        class _Session:
            def __init__(self) -> None:
                self.headers: dict = {}
                self.verify = True

            def request(self, *_a, **_k):
                raise RuntimeError(
                    "requests stub: network disabled in unit tests")

            def get(self, *a, **k):
                return self.request("GET", *a, **k)

            def post(self, *a, **k):
                return self.request("POST", *a, **k)

            def close(self) -> None:
                pass

        mod.Session = _Session


_ensure_requests_attrs()
