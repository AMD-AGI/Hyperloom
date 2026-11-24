# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The HTTP contract between the KB Store client and the service, against a stand-in ``urlopen``."""

from __future__ import annotations

import hashlib
import json
import urllib.error
from io import BytesIO
from pathlib import Path

import pytest

from hyperloom_common import kb_store
from hyperloom_common.kb_store import KBStoreClient, KBStoreError, Upload

BASE = "https://kb.example"
CID = "inference:m:h:f"
SID = "session-1"
SESSION = f"{BASE}/v1/kb/{CID}/sessions/{SID}"
SCOPE = {"kernel_optimizer": "forge", "tp": 8}


class _Reply(BytesIO):
    status = 200


class _Service:
    """Records every request and answers from a ``(method, url)`` table."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.replies: dict[tuple[str, str], object] = {}

    def reply(self, method: str, url: str, answer: object) -> None:
        self.replies[(method, url)] = answer

    def __call__(self, request, timeout=None):
        if isinstance(request, str):
            method, url, data, headers = "GET", request, None, {}
        else:
            method, url, data, headers = (
                request.get_method(),
                request.full_url,
                request.data,
                dict(request.header_items()),
            )
        self.requests.append({"method": method, "url": url, "data": data, "headers": headers})
        answer = self.replies.get((method, url), b"")
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, (dict, list)):
            answer = json.dumps(answer).encode()
        return _Reply(answer)

    def calls(self) -> list[tuple]:
        return [
            (row["method"], row["url"], None if row["data"] is None else json.loads(row["data"]))
            for row in self.requests
        ]


@pytest.fixture
def service(monkeypatch) -> _Service:
    stand_in = _Service()
    monkeypatch.setattr(kb_store.urllib.request, "urlopen", stand_in)
    return stand_in


@pytest.fixture
def client() -> KBStoreClient:
    return KBStoreClient(BASE + "/", "token")


def _http_error(url: str, code: int, body: bytes = b"nope") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "status", {}, BytesIO(body))  # type: ignore[arg-type]


def _entry(path: str, content: bytes, **overrides) -> dict:
    entry = {
        "path": path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
        "download_url": f"https://objects.example/{path}",
    }
    entry.update(overrides)
    return entry


# --- routes and bodies --------------------------------------------------------


def test_every_control_call_sends_its_method_route_and_body(service: _Service, client: KBStoreClient) -> None:
    client.search_identities(
        scheme="inference", match={"framework_name": "sglang"}, hardware_in=["mi300x"], offset=10, limit=5
    )
    client.search_identities(scheme="kernel")
    client.get_rollup(CID, scope=SCOPE)
    client.get_rollup(CID)
    client.get_top_sessions(CID, metric="speedup", limit=4, offset=8)
    client.get_hyperloom_recipe_view(CID, scope=SCOPE)
    client.set_champion(CID, SID, metric="optimized_throughput", value=10.0, scope=SCOPE)
    client.set_champion(CID, SID, metric="speedup", value=1.5)
    client.get_session(CID, SID)
    client.replace_knowledge(CID, SID, {"value": {}}, scope=SCOPE, objective_schema="agentx-v1")
    client.replace_knowledge(CID, SID, {"value": {}})
    client.merge_knowledge(CID, SID, {"measured_speedup": 2.0})
    client.list_session_files(CID, SID)

    assert service.calls() == [
        (
            "POST",
            f"{BASE}/v1/kb/search",
            {
                "scheme": "inference",
                "match": {"framework_name": "sglang"},
                "offset": 10,
                "limit": 5,
                "hardware_in": ["mi300x"],
            },
        ),
        ("POST", f"{BASE}/v1/kb/search", {"scheme": "kernel", "match": {}, "offset": 0, "limit": 50}),
        ("GET", f"{BASE}/v1/kb/{CID}/sessions?kernel_optimizer=forge&tp=8", None),
        ("GET", f"{BASE}/v1/kb/{CID}/sessions", None),
        ("GET", f"{BASE}/v1/kb/{CID}/sessions/top?metric=speedup&limit=4&offset=8", None),
        ("GET", f"{BASE}/v1/kb/{CID}/views/hyperloom-recipe?kernel_optimizer=forge&tp=8", None),
        (
            "POST",
            f"{BASE}/v1/kb/{CID}/champion",
            {"session_id": SID, "metric": "optimized_throughput", "value": 10.0, "scope": SCOPE},
        ),
        ("POST", f"{BASE}/v1/kb/{CID}/champion", {"session_id": SID, "metric": "speedup", "value": 1.5}),
        ("GET", SESSION, None),
        (
            "POST",
            f"{BASE}/v1/kb/{CID}",
            {
                "knowledge": {"value": {}},
                "mode": "replace",
                "session_id": SID,
                "scope": SCOPE,
                "objective_schema": "agentx-v1",
            },
        ),
        ("POST", f"{BASE}/v1/kb/{CID}", {"knowledge": {"value": {}}, "mode": "replace", "session_id": SID}),
        ("POST", f"{BASE}/v1/kb/{CID}", {"knowledge": {"measured_speedup": 2.0}, "mode": "merge", "session_id": SID}),
        ("GET", f"{SESSION}/files", None),
    ]


def test_requests_carry_the_bearer_token_and_json_headers(service: _Service, client: KBStoreClient) -> None:
    client.merge_knowledge(CID, SID, {})
    headers = service.requests[0]["headers"]
    assert headers["Authorization"] == "Bearer token"
    assert headers["Accept"] == "application/json"
    assert headers["Content-type"] == "application/json"


def test_a_client_without_a_token_sends_no_authorization(service: _Service) -> None:
    KBStoreClient(BASE, "").get_session(CID, SID)
    assert "Authorization" not in service.requests[0]["headers"]


def test_path_segments_are_quoted_but_the_canonical_id_keeps_its_colons(
    service: _Service, client: KBStoreClient
) -> None:
    client.get_session("kernel:a b/c", "s 1")
    assert service.requests[0]["url"] == f"{BASE}/v1/kb/kernel:a%20b%2Fc/sessions/s%201"


def test_json_answers_are_returned_and_empty_ones_become_empty_values(service: _Service, client: KBStoreClient) -> None:
    service.reply("GET", SESSION, {"knowledge": {"a": 1}})
    assert client.get_session(CID, SID) == {"knowledge": {"a": 1}}
    assert client.search_identities(scheme="kernel") == {}
    assert client.list_session_files(CID, SID) == {}
    assert client.get_top_sessions(CID) == {}


def test_a_base_url_is_required() -> None:
    with pytest.raises(KBStoreError, match="base_url is required"):
        KBStoreClient("", "token")


# --- failures -------------------------------------------------------------------


@pytest.mark.parametrize(
    "read",
    [
        lambda client: client.get_session(CID, SID),
        lambda client: client.get_rollup(CID),
        lambda client: client.get_hyperloom_recipe_view(CID),
    ],
    ids=["session", "rollup", "view"],
)
def test_not_found_reads_answer_none(service: _Service, client: KBStoreClient, read) -> None:
    for url in (SESSION, f"{BASE}/v1/kb/{CID}/sessions", f"{BASE}/v1/kb/{CID}/views/hyperloom-recipe"):
        service.reply("GET", url, _http_error(url, 404))
    assert read(client) is None


def test_an_http_error_names_the_route_status_and_body_and_keeps_its_cause(
    service: _Service, client: KBStoreClient
) -> None:
    service.reply("POST", f"{BASE}/v1/kb/{CID}/champion", _http_error("", 409, b"write_conflict"))
    with pytest.raises(KBStoreError) as raised:
        client.set_champion(CID, SID, metric="m", value=1.0)
    assert str(raised.value) == f"POST /v1/kb/{CID}/champion -> HTTP 409: write_conflict"
    assert isinstance(raised.value.__cause__, urllib.error.HTTPError)
    assert raised.value.__cause__.code == 409


def test_a_server_error_on_a_none_answering_read_still_raises(service: _Service, client: KBStoreClient) -> None:
    service.reply("GET", SESSION, _http_error("", 503))
    with pytest.raises(KBStoreError, match="HTTP 503"):
        client.get_session(CID, SID)


def test_a_transport_failure_keeps_the_os_error_as_its_cause(service: _Service, client: KBStoreClient) -> None:
    service.reply("GET", SESSION, urllib.error.URLError(ConnectionRefusedError("refused")))
    with pytest.raises(KBStoreError, match="transport error") as raised:
        client.get_session(CID, SID)
    assert isinstance(raised.value.__cause__, OSError)


def test_a_non_json_answer_is_refused(service: _Service, client: KBStoreClient) -> None:
    service.reply("GET", SESSION, b"<html>")
    with pytest.raises(KBStoreError, match="response was not JSON"):
        client.get_session(CID, SID)


# --- upload ---------------------------------------------------------------------


def _upload_service(service: _Service, *, skip: tuple[str, ...] = ()) -> None:
    service.reply(
        "POST",
        f"{SESSION}/files:grant",
        {
            "uploads": [
                {"path": "a/one.txt", "upload_url": "https://objects.example/put-one", "skip": "a/one.txt" in skip},
                {"path": "two.bin", "upload_url": "https://objects.example/put-two", "skip": "two.bin" in skip},
            ]
        },
    )
    service.reply(
        "POST",
        f"{SESSION}/files:commit",
        {"artifacts": {"files": [{"path": "a/one.txt", "uri": "kb://one"}, {"path": "two.bin", "uri": "kb://two"}]}},
    )


def test_put_files_grants_uploads_with_the_digest_header_then_commits(
    service: _Service, client: KBStoreClient, tmp_path: Path
) -> None:
    _upload_service(service, skip=("two.bin",))
    (tmp_path / "one.txt").write_bytes(b"one")
    (tmp_path / "two.bin").write_bytes(b"two")
    refs = client.put_files(
        CID,
        SID,
        [
            Upload("a/one.txt", tmp_path / "one.txt", "rewrite", {"schema": "v1"}),
            Upload("two.bin", tmp_path / "two.bin"),
        ],
    )

    assert refs == {"a/one.txt": "kb://one", "two.bin": "kb://two"}
    entries = [
        {
            "path": "a/one.txt",
            "sha256": hashlib.sha256(b"one").hexdigest(),
            "size": 3,
            "kind": "rewrite",
            "meta": {"schema": "v1"},
        },
        {"path": "two.bin", "sha256": hashlib.sha256(b"two").hexdigest(), "size": 3, "kind": "other", "meta": {}},
    ]
    grant, put, commit = service.requests
    assert (grant["method"], grant["url"], json.loads(grant["data"])) == (
        "POST",
        f"{SESSION}/files:grant",
        {"files": entries},
    )
    assert (put["method"], put["url"], put["data"]) == ("PUT", "https://objects.example/put-one", b"one")
    assert put["headers"]["X-amz-meta-sha256"] == entries[0]["sha256"]
    assert put["headers"]["Content-type"] == "application/octet-stream"
    assert (commit["url"], json.loads(commit["data"])) == (
        f"{SESSION}/files:commit",
        {"files": entries, "verify": True},
    )


def test_put_dir_uploads_every_file_at_its_relative_path(
    service: _Service, client: KBStoreClient, tmp_path: Path
) -> None:
    _upload_service(service)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.txt").write_bytes(b"one")
    (tmp_path / "two.bin").write_bytes(b"two")
    assert client.put_dir(CID, SID, tmp_path) == {"a/one.txt": "kb://one", "two.bin": "kb://two"}
    granted = json.loads(service.requests[0]["data"])["files"]
    assert [(entry["path"], entry["kind"], entry["meta"]) for entry in granted] == [
        ("a/one.txt", "other", {}),
        ("two.bin", "other", {}),
    ]
    assert {row["url"] for row in service.requests[1:3]} == {
        "https://objects.example/put-one",
        "https://objects.example/put-two",
    }


def test_put_dir_refuses_a_path_that_is_not_a_directory(client: KBStoreClient, tmp_path: Path) -> None:
    with pytest.raises(KBStoreError, match="not a directory"):
        client.put_dir(CID, SID, tmp_path / "missing")


def test_nothing_to_upload_sends_nothing(service: _Service, client: KBStoreClient) -> None:
    assert client.put_files(CID, SID, []) == {}
    assert service.requests == []


@pytest.mark.parametrize(
    ("path", "match"),
    [
        ("", "empty"),
        ("/abs", "relative"),
        ("C:/x", "relative"),
        ("a\\b", "forward slashes"),
        ("a/../b", "traversal"),
        ("a//b", "traversal"),
        ("a\0b", "NUL"),
    ],
)
def test_put_files_refuses_an_unsafe_path_before_any_request(
    service: _Service, client: KBStoreClient, tmp_path: Path, path: str, match: str
) -> None:
    (tmp_path / "f").write_bytes(b"x")
    with pytest.raises(KBStoreError, match=match):
        client.put_files(CID, SID, [Upload(path, tmp_path / "f")])
    assert service.requests == []


def test_put_files_refuses_a_path_named_twice_or_a_missing_source(
    service: _Service, client: KBStoreClient, tmp_path: Path
) -> None:
    (tmp_path / "f").write_bytes(b"x")
    with pytest.raises(KBStoreError, match="duplicate artifact path"):
        client.put_files(CID, SID, [Upload("f", tmp_path / "f"), Upload("f", tmp_path / "f")])
    with pytest.raises(KBStoreError, match="not a file"):
        client.put_files(CID, SID, [Upload("g", tmp_path / "g")])
    assert service.requests == []


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        (lambda: _http_error("", 403, b"expired"), "HTTP 403: expired"),
        (lambda: urllib.error.URLError("down"), "upload of .* failed"),
    ],
    ids=["http", "transport"],
)
def test_a_failed_upload_raises_before_commit(
    service: _Service, client: KBStoreClient, tmp_path: Path, failure, match: str
) -> None:
    _upload_service(service, skip=("two.bin",))
    service.reply("PUT", "https://objects.example/put-one", failure())
    (tmp_path / "one.txt").write_bytes(b"one")
    with pytest.raises(KBStoreError, match=match) as raised:
        client.put_files(CID, SID, [Upload("a/one.txt", tmp_path / "one.txt")])
    assert raised.value.__cause__ is not None
    assert not any(row["url"].endswith("files:commit") for row in service.requests)


def test_an_upload_answered_with_an_unexpected_status_is_refused(
    monkeypatch, service: _Service, client: KBStoreClient, tmp_path: Path
) -> None:
    _upload_service(service, skip=("two.bin",))
    monkeypatch.setattr(_Reply, "status", 302)
    (tmp_path / "one.txt").write_bytes(b"one")
    with pytest.raises(KBStoreError, match="returned HTTP 302"):
        client.put_files(CID, SID, [Upload("a/one.txt", tmp_path / "one.txt")])


# --- download -------------------------------------------------------------------


def _serve(service: _Service, files: dict[str, bytes], **overrides) -> dict:
    listing = {"files": [_entry(path, content, **overrides.get(path, {})) for path, content in files.items()]}
    service.reply("GET", f"{SESSION}/files", listing)
    for path, content in files.items():
        service.reply("GET", f"https://objects.example/{path}", content)
    return listing


def _tree(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_download_writes_exactly_the_listed_files(service: _Service, client: KBStoreClient, tmp_path: Path) -> None:
    _serve(service, {"a/one.txt": b"one", "two.bin": b"two"})
    paths = client.download_session(CID, SID, tmp_path / "dest")
    assert sorted(paths) == [tmp_path / "dest" / "files" / "a" / "one.txt", tmp_path / "dest" / "files" / "two.bin"]
    assert _tree(tmp_path / "dest") == {"files/a/one.txt": b"one", "files/two.bin": b"two"}
    assert [row["url"] for row in service.requests][0] == f"{SESSION}/files"


def test_a_listing_the_caller_passes_is_used_without_listing_again(
    service: _Service, client: KBStoreClient, tmp_path: Path
) -> None:
    listing = _serve(service, {"one.txt": b"one"})
    service.reply("GET", f"{SESSION}/files", {"files": []})
    client.download_session(CID, SID, tmp_path / "dest", listing=listing)
    assert [row["url"] for row in service.requests] == ["https://objects.example/one.txt"]
    assert _tree(tmp_path / "dest") == {"files/one.txt": b"one"}


def test_an_empty_listing_makes_the_destination_and_fetches_nothing(
    service: _Service, client: KBStoreClient, tmp_path: Path
) -> None:
    assert client.download_session(CID, SID, tmp_path / "dest", listing={"files": None}) == []
    assert (tmp_path / "dest").is_dir()
    assert service.requests == []


@pytest.mark.parametrize(
    ("listing", "match"),
    [
        ([], "manifest must be an object"),
        ({"files": {}}, "files must be a list"),
        ({"files": ["x"]}, "entry 0 must be an object"),
        ({"files": [_entry("../x", b"x")]}, "traversal"),
        ({"files": [_entry("x", b"x"), _entry("x", b"x")]}, "duplicate artifact path"),
        ({"files": [_entry("x", b"x", sha256="ABC")]}, "invalid sha256"),
        ({"files": [_entry("x", b"x", sha256="A" * 64)]}, "invalid sha256"),
        ({"files": [_entry("x", b"x", size=-1)]}, "invalid size"),
        ({"files": [_entry("x", b"x", size=True)]}, "invalid size"),
        ({"files": [_entry("x", b"x", size="1")]}, "invalid size"),
        ({"files": [_entry("x", b"x", download_url="")]}, "no download_url"),
    ],
)
def test_a_malformed_manifest_is_refused_before_the_destination_is_touched(
    service: _Service, client: KBStoreClient, tmp_path: Path, listing, match: str
) -> None:
    with pytest.raises(KBStoreError, match=match):
        client.download_session(CID, SID, tmp_path / "dest", listing=listing)
    assert not (tmp_path / "dest").exists()
    assert service.requests == []


@pytest.mark.parametrize(
    ("override", "match"),
    [({"sha256": "0" * 64}, "sha256 mismatch"), ({"size": 4}, "size mismatch")],
    ids=["sha256", "size"],
)
def test_a_download_that_does_not_match_its_manifest_is_refused_and_leaves_no_file(
    service: _Service, client: KBStoreClient, tmp_path: Path, override: dict, match: str
) -> None:
    _serve(service, {"one.txt": b"one"}, **{"one.txt": override})
    with pytest.raises(KBStoreError, match=match):
        client.download_session(CID, SID, tmp_path / "dest")
    assert _tree(tmp_path / "dest") == {}


def test_a_failed_fetch_keeps_its_cause_and_leaves_no_partial_file(
    service: _Service, client: KBStoreClient, tmp_path: Path
) -> None:
    _serve(service, {"one.txt": b"one"})
    service.reply("GET", "https://objects.example/one.txt", urllib.error.URLError("reset"))
    with pytest.raises(KBStoreError, match="download of 'one.txt' failed") as raised:
        client.download_session(CID, SID, tmp_path / "dest")
    assert isinstance(raised.value.__cause__, OSError)
    assert _tree(tmp_path / "dest") == {}


def test_download_refuses_a_symlinked_destination_or_files_directory(
    service: _Service, client: KBStoreClient, tmp_path: Path
) -> None:
    listing = _serve(service, {"one.txt": b"one"})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "link").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(KBStoreError, match="download destination may not be a symlink"):
        client.download_session(CID, SID, tmp_path / "link", listing=listing)

    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "files").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(KBStoreError, match="files directory may not be a symlink"):
        client.download_session(CID, SID, dest, listing=listing)
    assert list(elsewhere.iterdir()) == []


def test_download_refuses_a_destination_that_is_a_file(
    service: _Service, client: KBStoreClient, tmp_path: Path
) -> None:
    (tmp_path / "dest").write_bytes(b"x")
    with pytest.raises(KBStoreError, match="download destination is not a directory"):
        client.download_session(CID, SID, tmp_path / "dest", listing={"files": []})


def test_download_refuses_to_write_through_a_symlinked_parent_directory(
    service: _Service, client: KBStoreClient, tmp_path: Path
) -> None:
    listing = _serve(service, {"sub/one.txt": b"one"})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "dest" / "files").mkdir(parents=True)
    (tmp_path / "dest" / "files" / "sub").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(KBStoreError, match="escapes files directory|may not be a symlink"):
        client.download_session(CID, SID, tmp_path / "dest", listing=listing)
    assert list(elsewhere.iterdir()) == []
    assert service.requests == []
