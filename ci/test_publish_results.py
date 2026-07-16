# Copyright Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import publish_results


def test_publish_warns_but_allows_http_bearer(monkeypatch, capsys):
    """HTTP results services remain supported, but token use is visible."""
    calls = []

    class _Resp:
        status_code = 200
        text = "{}"

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"ok": True}

    def _post(endpoint, *, headers, json, timeout):
        calls.append((endpoint, headers, json, timeout))
        return _Resp()

    monkeypatch.setattr(publish_results, "requests", SimpleNamespace(post=_post, RequestException=Exception))

    out = publish_results.publish([{"run": {}}], "http://results.local", "tok", timeout=3, max_retries=1)

    assert out == {"ok": True}
    assert calls[0][0] == "http://results.local/api/import"
    assert calls[0][1]["Authorization"] == "Bearer tok"
    assert "WARNING" in capsys.readouterr().err
