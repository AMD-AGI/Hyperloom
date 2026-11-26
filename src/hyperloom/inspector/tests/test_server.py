# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import http.client
import json
import threading
from urllib.parse import quote

import pytest

from hyperloom.inspector import server as server_mod
from hyperloom.inspector.price_sheet import SHEET
from hyperloom.inspector.server import InspectorServer, ServeOptions
from hyperloom.inspector.tests.conftest import FAKE_KEY, T0
from hyperloom.inspector.token_terms import TERMS
from hyperloom.inspector.watcher import Watcher, WatchOptions


def _start(session_dir, replay=False):
    """A server on *session_dir*, its tokens priced by the built-in price sheet, as the Inspector does."""
    watcher = Watcher(session_dir, WatchOptions(replay=replay, poll_sec=0.05, sample_metrics=False))
    watcher.poll()
    httpd = InspectorServer(("127.0.0.1", 0), watcher)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return httpd


@pytest.fixture(autouse=True)
def modern_root(tmp_path, monkeypatch):
    """Where the modern build lives for this test; empty unless a test writes one, whatever the dev tree holds."""
    root = tmp_path / "modern"
    monkeypatch.setattr(server_mod, "_modern_root", lambda: root)
    return root


def _build_modern(root):
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text(
        '<!doctype html><title>modern</title><script type="module" src="./assets/index-Ab_1.js">'
    )
    (root / "assets" / "index-Ab_1.js").write_text("console.log(1)")
    (root / "assets" / "logo-x9.svg").write_text("<svg/>")
    (root / "assets" / "index-Ab_1.js.map").write_text("{}")
    (root / "secret.js").write_text("no")


def _stop(httpd):
    httpd.closing = True
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def server(session_dir):
    httpd = _start(session_dir)
    yield httpd
    _stop(httpd)


@pytest.fixture
def replay_server(session_dir):
    httpd = _start(session_dir, replay=True)
    yield httpd
    _stop(httpd)


def _get(httpd, path, method="GET", host=None):
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
    conn.request(method, path, headers={"Host": host or f"127.0.0.1:{httpd.server_address[1]}"})
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, body, resp


def test_snapshot_is_served_redacted(server):
    status, body, resp = _get(server, "/api/snapshot")
    assert status == 200
    assert resp.getheader("X-Content-Type-Options") == "nosniff"
    snap = json.loads(body)
    assert {t["task_id"] for t in snap["tasks"] if t["health"] == "crashed"} == {"t-scout", "t-recon"}
    assert FAKE_KEY.encode() not in body


@pytest.mark.parametrize("route", ["/", "/index.html"])
def test_the_build_is_served_at_root(server, modern_root, route):
    _build_modern(modern_root)
    status, body, resp = _get(server, route)
    assert status == 200
    assert resp.getheader("Content-Type").startswith("text/html")
    assert b"<title>modern</title>" in body
    # Scripts run only from files (the CSP has no 'unsafe-inline' for scripts).
    assert "script-src 'self';" in resp.getheader("Content-Security-Policy")


def test_without_a_build_root_says_how_to_build_it(server):
    status, body, resp = _get(server, "/")
    assert status == 503
    assert resp.getheader("Content-Type").startswith("text/plain")
    assert b"npm run build" in body
    assert b'"In a pod"' in body
    # An install pip built without Node has no frontend/ to build in, and neither has an sdist built that way: only
    # a source checkout with Node, or the release, brings the UI back.
    assert b"install it again from a source checkout" in body
    assert b"sdist" in body
    assert _get(server, "/assets/index-Ab_1.js")[0] == 404


def test_build_assets_are_served_immutable(server, modern_root):
    _build_modern(modern_root)
    status, body, resp = _get(server, "/assets/index-Ab_1.js")
    assert (status, body) == (200, b"console.log(1)")
    assert resp.getheader("Content-Type").startswith("text/javascript")
    assert "immutable" in resp.getheader("Cache-Control")
    assert resp.getheader("X-Content-Type-Options") == "nosniff"
    status, _, resp = _get(server, "/assets/logo-x9.svg")
    assert (status, resp.getheader("Content-Type")) == (200, "image/svg+xml")


@pytest.mark.parametrize(
    "route",
    [
        "/assets/missing.js",
        "/assets/index-Ab_1.js.map",
        "/assets/../secret.js",
        "/assets/%2e%2e/secret.js",
        "/assets/sub/index-Ab_1.js",
        "/assets/.hidden.js",
        "/assets/index.html",
        "/secret.js",
        "/modern/index.html",
        "/static/modern/assets/index-Ab_1.js",
        "/classic",
        "/app.js",
        "/static/app.js",
        "/index.py",
        "/../server.py",
    ],
)
def test_only_files_of_the_build_are_served(server, modern_root, route):
    _build_modern(modern_root)
    assert _get(server, route)[0] == 404


def test_favicon_is_empty_not_missing(server):
    status, body, _ = _get(server, "/favicon.ico")
    assert status == 204
    assert body == b""


@pytest.mark.parametrize("path", ["../../etc/passwd", ".env", "/etc/passwd", "runs/../.env"])
def test_artifact_path_jail(server, path):
    status, body, _ = _get(server, "/api/artifact?path=" + quote(path))
    assert status in (403, 404)
    assert FAKE_KEY.encode() not in body


_ARCHIVED = "archived secret"


@pytest.fixture
def archived(session_dir):
    """The application-private ``archive`` with a file in it, and a link to that file from a served directory."""
    (session_dir / "archive").mkdir()
    (session_dir / "archive" / "package.tar").write_text(_ARCHIVED, encoding="utf-8")
    (session_dir / "runs" / "package.tar").symlink_to(session_dir / "archive" / "package.tar")
    return session_dir


@pytest.mark.parametrize(
    "query",
    [
        "/api/artifact?path=archive/package.tar",
        "/api/artifact?path=runs/../archive/package.tar",
        "/api/artifact?path=runs/package.tar",
        "/api/diff?a=archive/package.tar&b=runs/specialist/t-scout/process.log",
        "/api/diff?a=runs/specialist/t-scout/process.log&b=runs/package.tar",
    ],
)
def test_the_private_archive_is_refused(archived, query):
    httpd = _start(archived)
    try:
        status, body, _ = _get(httpd, query)
        listed = json.loads(_get(httpd, "/api/artifacts")[1])["artifacts"]
    finally:
        _stop(httpd)
    assert status == 403 and _ARCHIVED.encode() not in body
    assert not [a["path"] for a in listed if "package.tar" in a["path"]]


def test_foreign_host_header_is_refused(server):
    status, _, _ = _get(server, "/api/snapshot", host="evil.example:8765")
    assert status == 403


def test_log_artifact_is_redacted_and_followable(server):
    status, body, _ = _get(server, "/api/artifact?path=runs/specialist/t-scout/process.log")
    assert status == 200
    rec = json.loads(body)
    assert "model refused" in rec["text"] and FAKE_KEY not in rec["text"]
    status, body, _ = _get(server, f"/api/artifact?path=runs/specialist/t-scout/process.log&offset={rec['end']}")
    assert json.loads(body)["text"] == ""


def test_trace_is_metadata_only(server):
    rec = json.loads(_get(server, "/api/artifact?path=runs/roofline/t-roof/rank0.pt.trace.json.gz")[1])
    assert rec["mode"] == "meta" and "text" not in rec


def test_config_diff(server):
    a = "runs/baseline/t-base/baseline_config.with_envs.yaml"
    b = "runs/roofline/t-roof/baseline_config.with_envs.yaml"
    rec = json.loads(_get(server, f"/api/diff?a={quote(a)}&b={quote(b)}")[1])
    assert ["replace", 3, "  A: '1'", 3, "  A: '2'"] in rec["rows"]


def test_artifact_index_and_conversation(server):
    paths = {a["path"] for a in json.loads(_get(server, "/api/artifacts")[1])["artifacts"]}
    assert "state.json" in paths and ".env" not in paths
    conv = json.loads(_get(server, "/api/conversation?call_id=c1")[1])
    assert FAKE_KEY not in conv["prompt"]
    assert _get(server, "/api/conversation?call_id=nope")[0] == 404
    assert json.loads(_get(server, "/api/conversation?task_id=spec1&turn=1")[1])["response"] == "spec"
    assert _get(server, "/api/conversation?call_id=")[0] == 404


def test_tokens_are_served_priced_by_the_built_in_sheet(server):
    tokens = json.loads(_get(server, "/api/snapshot")[1])["tokens"]
    assert tokens["totals"]["total"] == 2650 and tokens["totals"]["cost_usd"] == 0.0162
    assert tokens["pricing"]["source"] == SHEET.source
    for path in ("/api/tokens", "/api/tokens?download=1"):
        status, body, _ = _get(server, path)
        doc = json.loads(body)
        assert status == 200 and doc["pricing"]["source"] == SHEET.source, path
        assert doc["totals"]["calls"] == 3 and doc["totals"]["cost_usd"] == 0.0162, path


def test_token_timeline_is_served_and_downloadable(server, replay_server):
    status, body, _ = _get(server, "/api/tokens")
    assert status == 200
    doc = json.loads(body)
    assert doc["schema"] == "hyperloom.inspector.token_timeline/v1"
    assert [e["call_id"] for e in doc["events"]] == ["c1", "c2", "c3"]
    assert doc["terms"] == TERMS
    snapshot = json.loads(_get(server, "/api/snapshot")[1])
    assert snapshot["tokens"]["totals"]["total"] == 2650
    assert snapshot["tokens"]["terms"] == TERMS  # not mangled by redaction
    status, body, resp = _get(server, "/api/tokens?download=1")
    assert resp.getheader("Content-Disposition") == 'attachment; filename="20260925T105719Z-test.tokens.json"'
    assert json.loads(body)["totals"]["calls"] == 3
    _get(replay_server, f"/api/replay?t={T0 + 620}&paused=1", method="POST")
    assert [e["call_id"] for e in json.loads(_get(replay_server, "/api/tokens")[1])["events"]] == ["c1"]


def test_replay_control_only_in_replay(server, replay_server):
    assert _get(server, "/api/replay?t=1", method="POST")[0] == 409
    status, body, _ = _get(replay_server, f"/api/replay?t={T0 + 305}&paused=1", method="POST")
    assert status == 200 and json.loads(body)["now"] == T0 + 305
    snap = json.loads(_get(replay_server, "/api/snapshot")[1])
    assert snap["summary"]["phase"] == "PRELUDE"


def test_sse_sends_hello_then_items_then_reset(replay_server):
    _get(replay_server, f"/api/replay?t={T0 + 305}&paused=1", method="POST")
    conn = http.client.HTTPConnection("127.0.0.1", replay_server.server_address[1], timeout=5)
    conn.request("GET", "/api/stream", headers={"Host": "localhost"})
    resp = conn.getresponse()
    assert resp.getheader("Content-Type") == "text/event-stream"

    def next_event():
        name = None
        while True:
            line = resp.fp.readline().decode().rstrip("\n")
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                return name, json.loads(line[6:])

    assert next_event()[0] == "hello"
    _get(replay_server, f"/api/replay?t={T0 + 700}", method="POST")
    seen = {}
    while "items" not in seen:
        name, data = next_event()
        seen[name] = data
    assert [c["call_id"] for c in seen["items"]["items"]["llm_call"]] == ["c1", "c2"]
    _get(replay_server, f"/api/replay?t={T0 + 10}", method="POST")
    while (event := next_event())[0] != "reset":
        pass
    conn.close()


def test_serve_never_puts_its_files_in_the_session(session_dir, monkeypatch):
    monkeypatch.setattr(InspectorServer, "serve_forever", lambda self, poll_interval=0.5: None)
    before = sorted(p for p in session_dir.rglob("*"))
    inside = session_dir / "scratch"
    server_mod.serve(session_dir, ServeOptions(port=0, history_dir=inside, token_dir=inside))
    assert sorted(p for p in session_dir.rglob("*")) == before


def test_serve_prices_with_the_built_in_sheet(session_dir, tmp_path, monkeypatch, capsys):
    served = []
    monkeypatch.setattr(InspectorServer, "serve_forever", lambda self, poll_interval=0.5: served.append(self))
    state = tmp_path / "state"
    server_mod.serve(session_dir, ServeOptions(port=0, token_dir=state))
    tokens = served[-1].watcher.snapshot()["tokens"]
    assert tokens["totals"]["calls"] == 3
    assert tokens["totals"]["cost_usd"] == 0.0162 and tokens["pricing"]["source"] == SHEET.source
    assert [p.suffixes[-2:] for p in state.iterdir()] == [[".tokens", ".json"]]
    assert "Prices: built-in price sheet\n" in capsys.readouterr().out
