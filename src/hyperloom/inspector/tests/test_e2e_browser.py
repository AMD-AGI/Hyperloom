# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The page in a headless browser against the synthetic session: graph, drawer, viewer, diff, conversations.

Every test fails on page errors, console errors, unexpected HTTP errors and Content Security Policy violations. All
are skipped when ``frontend/inspector`` was not built into ``static/modern/``.

Optional: skipped unless the ``playwright`` package is installed and a Chromium launches. Point
``HYPERLOOM_E2E_CHROMIUM`` at a Chromium binary to skip ``playwright install``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from hyperloom.inspector import server as server_mod
from hyperloom.inspector.tests.conftest import FAKE_KEY
from hyperloom.inspector.tests.test_server import _start

sync_api = pytest.importorskip("playwright.sync_api")
expect = sync_api.expect

pytestmark = [
    pytest.mark.inspector_e2e,
    pytest.mark.skipif(
        server_mod._modern_page() is None, reason="the UI is not built (npm run build in frontend/inspector)"
    ),
]

# Records CSP violations before any page script runs; the console reports them too, but not as reliably.
_CSP_PROBE = """
window.__cspViolations = [];
document.addEventListener("securitypolicyviolation", (e) =>
  window.__cspViolations.push(`${e.violatedDirective} ${e.blockedURI}`));
"""


@dataclass(frozen=True)
class Ui:
    """Where the page keeps the parts the tests drive."""

    node: str
    edge: str
    filter: str
    dr_title: str
    dr_close: str
    drawer: str
    tab: str  # format with the tab name
    art_q: str
    art_link: str
    art_box: str
    art_diff: str
    art_view: str
    diff_table: str
    llm_rows: str
    conversation: str


UI = Ui(
    node="[data-testid=graph-node]",
    edge="[data-testid=graph-edge]",
    filter="[data-testid=graph-filter]",
    dr_title="[data-testid=dr-title]",
    dr_close="[data-testid=dr-close]",
    drawer="[data-testid=drawer]",
    tab="[data-testid=tab-{}]",
    art_q="[data-testid=art-q]",
    art_link="[data-testid=art-file] button.link",
    art_box="[data-testid=art-file] input[type=checkbox]",
    art_diff="[data-testid=art-diff]",
    art_view="[data-testid=art-view]",
    diff_table="[data-testid=diff]",
    llm_rows="[data-testid=llm-calls] tbody tr",
    conversation="[data-testid=conversation]",
)


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as pw:
        try:
            chromium = pw.chromium.launch(
                executable_path=os.environ.get("HYPERLOOM_E2E_CHROMIUM") or None, args=["--no-sandbox"]
            )
        except sync_api.Error as exc:
            pytest.skip(f"no Chromium for playwright: {str(exc).splitlines()[0]}")
        yield chromium
        chromium.close()


@pytest.fixture
def page(browser, session_dir):
    httpd = _start(session_dir)
    page = browser.new_page(viewport={"width": 1600, "height": 1200})
    page.add_init_script(_CSP_PROBE)
    problems: list[str] = []
    page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))

    # Calls without a recorded conversation (c2, c3) look it up and get a 404 by design; the browser logs those too.
    def expected(url: str) -> bool:
        return "/api/conversation?" in url

    def on_console(m):
        if m.type == "error" and not expected(m.location.get("url", "")):
            problems.append(f"console: {m.text}")

    def on_response(r):
        if r.status >= 400 and not expected(r.url):
            problems.append(f"HTTP {r.status} {r.url}")

    page.on("console", on_console)
    page.on("response", on_response)
    response = page.goto(f"http://127.0.0.1:{httpd.server_address[1]}/")
    assert "script-src 'self'" in response.headers["content-security-policy"]
    page.wait_for_selector(UI.node, timeout=10_000)
    yield page
    problems += [f"CSP: {v}" for v in page.evaluate("window.__cspViolations")]
    page.close()
    httpd.closing = True
    httpd.shutdown()
    httpd.server_close()
    assert problems == []


def test_graph_and_node_drawer(page):
    assert page.locator(UI.node).count() >= 5
    assert page.locator(UI.edge).count() >= 1
    page.locator(UI.node).first.click()
    expect(page.locator(UI.dr_title)).not_to_be_empty()
    expect(page.locator(UI.filter)).to_contain_text("filter")
    page.click(UI.dr_close)
    page.click(UI.filter)
    expect(page.locator(UI.filter, has_text="filter")).to_have_count(0)


def test_artifact_viewer_diff_and_redaction(page):
    page.click(UI.tab.format("artifacts"))
    page.fill(UI.art_q, "server.log")
    page.locator(UI.art_link).first.click()
    expect(page.locator(UI.art_view)).to_contain_text("INFO")
    page.fill(UI.art_q, "process.log")
    page.locator(UI.art_link).first.click()
    expect(page.locator(UI.art_view)).to_contain_text("model refused")
    assert FAKE_KEY not in page.text_content(UI.art_view)
    page.fill(UI.art_q, "yaml")
    boxes = page.locator(UI.art_box)
    expect(boxes).to_have_count(2)
    boxes.nth(0).check()
    boxes.nth(1).check()
    page.click(UI.art_diff)
    expect(page.locator(UI.diff_table)).to_contain_text("'2'")


def test_llm_call_conversation_lookup(page):
    page.click(UI.tab.format("llm"))
    rows = page.locator(UI.llm_rows)
    assert rows.count() >= 2
    shown = []
    for i in range(rows.count()):
        rows.nth(i).click()
        expect(page.locator(UI.conversation)).not_to_contain_text("loading")
        page.wait_for_timeout(300)
        shown.append(page.text_content(UI.drawer))
    assert any("use key" in t for t in shown)
    assert not any(FAKE_KEY in t for t in shown)
