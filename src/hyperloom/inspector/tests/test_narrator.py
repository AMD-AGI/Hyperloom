# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import http.client
import json
import threading

import pytest

from hyperloom_common.llm import llm_config
from hyperloom_common.llm.llm_headers import parse_custom_headers
from hyperloom.inspector import digest
from hyperloom.inspector.narrator import (
    ClientInfo,
    Narrator,
    _AnthropicClient,
    build_client,
    load_llm_env,
)
from hyperloom.inspector.server import InspectorServer
from hyperloom.inspector.tests.conftest import FAKE_KEY, T0
from hyperloom.inspector.watcher import Watcher, WatchOptions


class FakeClient:
    def __init__(self, reply="All quiet.\n- explore is benchmarking", error=None, gate=None):
        self.reply, self.error, self.gate = reply, error, gate
        self.prompts: list[tuple[str, str]] = []

    def complete(self, *, system, user, operation):
        self.prompts.append((operation, user))
        if self.gate is not None:
            self.gate.wait(5)
        if self.error is not None:
            raise self.error
        return self.reply


class Tick:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _watcher(session_dir, replay=False):
    w = Watcher(session_dir, WatchOptions(replay=replay, poll_sec=0.05, sample_metrics=False))
    w.poll()
    return w


def _narrator(watcher, client=None, tick=None):
    client = FakeClient() if client is None else client
    return Narrator(watcher, client, ClientInfo("fake", "fake-model"), monotonic=tick or Tick())


# ----- digest ------------------------------------------------------------------------------------------------


def test_digest_names_the_phase_tasks_and_problems(session_dir):
    text = digest.build(_watcher(session_dir).snapshot())
    for section in ("## Run", "## State", "## Tasks", "## Recent bus traffic", "## LLM use"):
        assert section in text
    assert "crashed but recorded as succeeded" in text
    assert "Running:" in text
    assert FAKE_KEY not in text
    # state.json has no current action (SharedState dropped the field), so the prompt does not offer an empty one.
    assert "Phase " in text and "current action" not in text


def test_digest_is_capped_and_redacts_earlier_entries(session_dir):
    view = _watcher(session_dir).snapshot()
    earlier = [{"kind": "narration", "clock": T0, "text": f"key {FAKE_KEY}"}]
    text = digest.build(view, earlier)
    assert "## Earlier narration and questions" in text and FAKE_KEY not in text
    short = digest.build(view, earlier, max_chars=400)
    assert len(short) <= 400 and short.endswith("[digest truncated]")


# ----- narrator ----------------------------------------------------------------------------------------------


def test_unchanged_session_reuses_the_narration_and_force_respects_the_gap(session_dir):
    tick, client = Tick(), FakeClient()
    n = _narrator(_watcher(session_dir), client, tick)
    status, first = n.narrate()
    assert status == 200 and not first["reused"] and first["entry"]["kind"] == "narration"
    status, again = n.narrate()
    assert status == 200 and again["reused"] and again["entry"]["id"] == first["entry"]["id"]
    status, refused = n.narrate(force=True)
    assert status == 429 and "minimum gap" in refused["error"] and refused["entry"]["id"] == first["entry"]["id"]
    tick.t += 30
    status, forced = n.narrate(force=True)
    assert status == 200 and forced["entry"]["id"] != first["entry"]["id"]
    assert len(client.prompts) == 2


def test_focus_is_a_new_narration_and_reaches_the_prompt(session_dir):
    tick, client = Tick(), FakeClient()
    n = _narrator(_watcher(session_dir), client, tick)
    n.narrate()
    tick.t += 30
    status, data = n.narrate(focus="  memory   pressure ")
    assert status == 200 and not data["reused"] and data["entry"]["focus"] == "memory pressure"
    assert "Pay particular attention to: memory pressure" in client.prompts[-1][1]


def test_answers_see_earlier_entries_and_are_kept(session_dir):
    client = FakeClient()
    n = _narrator(_watcher(session_dir), client)
    n.narrate()
    client.reply = "The explore task."
    status, data = n.ask("What is running?")
    assert status == 200 and data["entry"]["kind"] == "answer" and data["entry"]["question"] == "What is running?"
    op, prompt = client.prompts[-1]
    assert op == "ask" and "## Question\nWhat is running?" in prompt and "explore is benchmarking" in prompt
    assert [e["kind"] for e in n.status()["entries"]] == ["narration", "answer"]
    assert n.ask("   ")[0] == 400


def test_replay_never_feeds_or_shows_entries_past_the_clock(session_dir):
    w = _watcher(session_dir, replay=True)
    tick, client = Tick(), FakeClient(reply="LATE narration")
    n = _narrator(w, client, tick)
    w.clock.seek(w.clock.end, paused=True)
    assert n.narrate()[0] == 200
    w.clock.seek(w.clock.start + 60, paused=True)
    assert n.status()["entries"] == []
    tick.t += 30
    client.reply = "early narration"
    assert n.narrate()[0] == 200
    assert "LATE narration" not in client.prompts[-1][1]
    assert [e["text"] for e in n.status()["entries"]] == ["early narration"]


def test_one_call_at_a_time(session_dir):
    gate = threading.Event()
    n = _narrator(_watcher(session_dir), FakeClient(gate=gate))
    worker = threading.Thread(target=n.narrate)
    worker.start()
    for _ in range(100):
        if n.status()["busy"]:
            break
        threading.Event().wait(0.01)
    status, data = n.ask("anything?")
    gate.set()
    worker.join(5)
    assert status == 429 and "already writing" in data["error"]
    assert not n.status()["busy"]


def test_client_errors_and_replies_are_redacted(session_dir):
    n = _narrator(_watcher(session_dir), FakeClient(error=RuntimeError(f"401 for key {FAKE_KEY}")))
    status, data = n.narrate()
    assert status == 502 and FAKE_KEY not in data["error"] and "RuntimeError" in data["error"]
    assert not n.status()["busy"]
    n = _narrator(_watcher(session_dir), FakeClient(reply=f"the key is {FAKE_KEY}"))
    assert FAKE_KEY not in n.narrate()[1]["entry"]["text"]


def test_without_a_client_the_narrator_says_why(session_dir):
    n = Narrator(_watcher(session_dir), None, ClientInfo(reason="no LLM credentials"))
    assert n.status()["available"] is False and n.status()["reason"] == "no LLM credentials"
    status, data = n.narrate()
    assert status == 503 and "no LLM credentials" in data["error"]
    assert n.ask("why?")[0] == 503


# ----- configuration -----------------------------------------------------------------------------------------


def test_backend_none_and_no_credentials_give_a_reason():
    client, info = build_client({"HYPERLOOM_INSPECTOR_LLM_BACKEND": "none"})
    assert client is None and "turned off" in info.reason
    client, info = build_client({})
    assert client is None and "no LLM credentials" in info.reason
    client, info = build_client({"HYPERLOOM_INSPECTOR_LLM_BACKEND": "bogus", "ANTHROPIC_API_KEY": "x"})
    assert client is None and "unknown" in info.reason


def test_anthropic_client_sends_one_request(monkeypatch):
    seen = {}

    def fake_completion(*, request, env=None, component="", operation=""):
        seen.update(request=request, component=component, operation=operation)
        return llm_config.AnthropicMessageResult(text="ok", stop_reason="end_turn", usage=None)

    monkeypatch.setattr(llm_config, "anthropic_completion", fake_completion)
    assert _AnthropicClient("claude-x", 50).complete(system="sys", user="hi", operation="explain") == "ok"
    request = seen["request"]
    assert (request.model, request.system, request.max_tokens) == ("claude-x", "sys", 50)
    assert request.messages == [{"role": "user", "content": "hi"}]
    assert (seen["component"], seen["operation"]) == ("inspector", "explain")


def test_llm_env_file_loads_only_missing_llm_settings(tmp_path):
    env_file = tmp_path / "llm.env"
    env_file.write_text(
        "# comment\n"
        f"export ANTHROPIC_API_KEY='{FAKE_KEY}'\n"
        "ANTHROPIC_BASE_URL=https://gateway.example\n"
        'HYPERLOOM_INSPECTOR_LLM_MODEL="claude-opus-5"\n'
        "CLAUDE_MODEL=from-file\n"
        "HF_TOKEN=hf_nope\n"
        "PATH=/evil\n"
        "OPENAI_API_KEY=\n"
    )
    environ = {"CLAUDE_MODEL": "from-env"}
    names = load_llm_env(env_file, environ)
    assert sorted(names) == ["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "HYPERLOOM_INSPECTOR_LLM_MODEL"]
    assert environ["ANTHROPIC_API_KEY"] == FAKE_KEY and environ["CLAUDE_MODEL"] == "from-env"
    assert "HF_TOKEN" not in environ and "PATH" not in environ and "OPENAI_API_KEY" not in environ


def test_llm_env_file_follows_references_in_custom_headers(tmp_path):
    env_file = tmp_path / "llm.env"
    env_file.write_text(
        'ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: ${GATEWAY_SUB_KEY}"\n'
        "GATEWAY_SUB_KEY=sub-123\n"
        "UNRELATED_SECRET=nope\n"
    )
    environ: dict[str, str] = {}
    assert load_llm_env(env_file, environ) == ["ANTHROPIC_CUSTOM_HEADERS", "GATEWAY_SUB_KEY"]
    assert environ["GATEWAY_SUB_KEY"] == "sub-123" and "UNRELATED_SECRET" not in environ


# ----- HTTP --------------------------------------------------------------------------------------------------


@pytest.fixture
def served(session_dir):
    started = []

    def start(narrator_factory):
        w = _watcher(session_dir)
        httpd = InspectorServer(("127.0.0.1", 0), w, narrator_factory(w))
        threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        started.append(httpd)
        return httpd

    yield start
    for httpd in started:
        httpd.closing = True
        httpd.shutdown()
        httpd.server_close()


def _request(httpd, method, path, body=None, ctype="application/json", host=None):
    port = httpd.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Host": host or f"127.0.0.1:{port}"}
    if body is not None:
        headers["Content-Type"] = ctype
    conn.request(method, path, body=body, headers=headers)
    resp = conn.getresponse()
    data = json.loads(resp.read() or b"{}")
    conn.close()
    return resp.status, data


def test_endpoints_narrate_and_answer(served):
    httpd = served(lambda w: _narrator(w))
    status, data = _request(httpd, "GET", "/api/narrator")
    assert status == 200 and data["available"] is True and data["entries"] == []
    status, data = _request(httpd, "POST", "/api/narrate", b"{}")
    assert status == 200 and data["entry"]["kind"] == "narration"
    status, data = _request(httpd, "POST", "/api/ask", json.dumps({"question": "what now?"}).encode())
    assert status == 200 and data["entry"]["question"] == "what now?"
    status, data = _request(httpd, "GET", "/api/narrator")
    assert [e["kind"] for e in data["entries"]] == ["narration", "answer"]


@pytest.mark.parametrize(
    ("body", "ctype", "code"),
    [
        (b"question=x", "application/x-www-form-urlencoded", 415),
        (b"{}", "text/plain", 415),
        (b"not json", "application/json", 400),
        (b"[1]", "application/json", 400),
        (json.dumps({"question": "x" * 5000}).encode(), "application/json", 400),
    ],
)
def test_posts_need_a_small_json_body(served, body, ctype, code):
    httpd = served(lambda w: _narrator(w))
    assert _request(httpd, "POST", "/api/ask", body, ctype)[0] == code


def test_foreign_host_and_missing_narrator(served):
    httpd = served(lambda w: None)
    assert _request(httpd, "POST", "/api/narrate", b"{}", host="evil.example")[0] == 403
    status, data = _request(httpd, "GET", "/api/narrator")
    assert status == 200 and data["available"] is False and "--narrate" in data["reason"]
    assert _request(httpd, "POST", "/api/narrate", b"{}")[0] == 503


def test_llm_env_file_reads_multi_line_quoted_values_and_comments(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ANTHROPIC_CUSTOM_HEADERS='Ocp-Apim-Subscription-Key: abc\n'\n"
        'ANTHROPIC_BASE_URL="https://gw.example/anthropic"  # the gateway\n'
        "CLAUDE_MODEL=claude-opus-5-5 # default\n",
        encoding="utf-8",
    )
    environ: dict[str, str] = {}
    load_llm_env(env_file, environ)
    assert parse_custom_headers(environ["ANTHROPIC_CUSTOM_HEADERS"], env=environ) == {
        "Ocp-Apim-Subscription-Key": "abc"
    }
    assert environ["ANTHROPIC_BASE_URL"] == "https://gw.example/anthropic"
    assert environ["CLAUDE_MODEL"] == "claude-opus-5-5"


def test_an_answer_cut_at_the_token_limit_says_so():
    from hyperloom.inspector.narrator import _with_cut_note

    marked = _with_cut_note("- one\n- Against the run:\n")
    assert marked.startswith("- one\n- Against the run:\n[answer cut off at the token limit")
    assert "HYPERLOOM_INSPECTOR_LLM_MAX_TOKENS" in marked


def test_the_clients_note_only_a_reply_cut_at_the_token_limit(monkeypatch):
    from hyperloom.inspector.narrator import _OpenAIClient

    reply = "- one\n- Against the run:\n"
    note = "[answer cut off at the token limit"
    for stop in ("end_turn", "max_tokens"):
        monkeypatch.setattr(
            llm_config, "anthropic_completion", lambda **_: llm_config.AnthropicMessageResult(reply, stop, None)
        )
        text = _AnthropicClient("claude-x", 50).complete(system="sys", user="hi", operation="ask")
        assert text == reply if stop == "end_turn" else text.startswith(reply + note)
    for stop in ("stop", "length"):
        monkeypatch.setattr(
            llm_config, "chat_completion", lambda *_, **__: llm_config.ChatCompletionResult(reply, stop, None)
        )
        text = _OpenAIClient(object(), "gpt-x", 50).complete(system="sys", user="hi", operation="ask")
        assert text == reply if stop == "stop" else text.startswith(reply + note)
