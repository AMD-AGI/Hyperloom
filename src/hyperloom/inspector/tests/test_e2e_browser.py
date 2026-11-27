# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The page in a headless browser against the synthetic session: graph, drawer, viewer, diff, conversations, tokens.

Every test fails on page errors, console errors, unexpected HTTP errors and Content Security Policy violations. All
are skipped when ``frontend/inspector`` was not built into ``static/modern/``.

Optional: skipped unless the ``playwright`` package is installed and a Chromium launches. Point
``HYPERLOOM_E2E_CHROMIUM`` at a Chromium binary to skip ``playwright install``.
"""

from __future__ import annotations

import json
import os
import re
import socket
from dataclasses import dataclass

import pytest

from hyperloom.inspector import server as server_mod
from hyperloom.inspector.tests.conftest import FAKE_KEY, T0, iso
from hyperloom.inspector.tests.test_server import _start
from hyperloom.inspector.tests.test_tokens import TURNS, _add_ledger, _log
from hyperloom.inspector.token_terms import TERMS

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
    badges: str
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
    badges="[data-testid=badges]",
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


@dataclass(frozen=True)
class TokensUi:
    """The Tokens tab's selectors."""

    chart: str
    agents: str
    models: str
    download: str
    rate: str
    cost: str
    cost_chart: str
    pricing: str
    unpriced: str
    prices: str
    cache_hit: str
    total: str
    now: str
    legend: str


TOKENS = TokensUi(
    chart="[data-testid=token-chart]",
    agents="[data-testid=token-agents]",
    models="[data-testid=token-models]",
    download="[data-testid=token-download]",
    rate="[data-testid=token-rate]",
    cost="[data-testid=token-cost]",
    cost_chart="[data-testid=cost-chart]",
    pricing="[data-testid=token-pricing]",
    unpriced="[data-testid=token-unpriced]",
    prices="[data-testid=token-prices]",
    cache_hit="[data-testid=token-cache-hit]",
    total="[data-testid=token-total]",
    now="line.now",
    legend="[data-testid=token-legend] button.chip",
)

# The fixture ledger runs claude-opus-5-5, $4 / $20 per 1M tokens on the built-in price sheet: its three calls cost
# $0.0162.


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
def extra_calls():
    return []


@pytest.fixture
def running():
    return False


@pytest.fixture
def specialist_log():
    return None


@pytest.fixture
def page(browser, session_dir, extra_calls, running, specialist_log):
    if extra_calls:
        _add_ledger(session_dir, extra_calls)
    if specialist_log:
        _log(session_dir, specialist_log)
    if running:
        # This process on this host stands in for the session's optimizer, so the Inspector sees it alive.
        lock = {"hostname": socket.gethostname(), "pid": os.getpid()}
        (session_dir / "runtime" / "optimizer.lock").write_text(json.dumps(lock), encoding="utf-8")
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
    expect(page.locator(UI.badges)).to_contain_text("crashed but recorded as succeeded")
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


def test_token_tab_chart_and_download(page):
    page.click(UI.tab.format("tokens"))
    chart = page.locator(TOKENS.chart)
    expect(chart.locator("path")).not_to_have_count(0)
    expect(page.locator(TOKENS.agents)).to_contain_text("orchestration")
    expect(page.locator(TOKENS.agents)).to_contain_text("critic")
    assert page.locator(TOKENS.download).get_attribute("href") == "api/tokens?download=1"
    expect(page.locator(TOKENS.download)).to_have_text("Download receipt in JSON")
    expect(page.locator(TOKENS.cost)).to_contain_text("0.016")
    expect(page.locator(TOKENS.cost)).to_contain_text("llm_prices.json")
    expect(page.locator(TOKENS.cost_chart).locator("path")).not_to_have_count(0)
    expect(page.locator(TOKENS.pricing)).to_contain_text(
        re.compile(r"Price source\s*hyperloom_common/llm/llm_prices\.json, the built-in price sheet")
    )
    expect(page.locator(TOKENS.prices)).to_contain_text("claude-opus-5-5")  # the price table's "Priced as"
    # Under each price per 1M tokens, the tokens this run used at it and what they cost: 2,300 input at $4.
    for text in ("$4.00 / million-tokens", "Used in this run", "2,300 tokens", "Cost in this run", "$0.0092"):
        expect(page.locator("body")).to_contain_text(text)
    # The server's names on the cards; what each means is a hover hint (below).
    expect(page.locator(TOKENS.cache_hit)).to_contain_text("Input cache hit rate")
    page.click(TOKENS.rate)
    expect(page.locator(TOKENS.chart).locator("rect[fill^='#']")).not_to_have_count(0)
    expect(page.locator(TOKENS.cost_chart).locator("rect[fill^='#']")).not_to_have_count(0)


def test_token_names_show_their_definition_on_hover(page):
    page.click(UI.tab.format("tokens"))
    hint = page.locator("#hint-tip")
    expect(hint).to_be_hidden()
    # The moment the pointer is on a name: its definition, under the exact figure on a card.
    page.hover(TOKENS.cache_hit)
    expect(hint).to_have_text(TERMS["cache_hit"]["definition"], timeout=1000)
    assert page.get_attribute(TOKENS.cache_hit, "title") is None  # no native tooltip on top of it
    page.hover(TOKENS.total)
    expect(hint.locator("b")).to_contain_text(" tokens")
    expect(hint).to_contain_text(TERMS["total"]["definition"])
    # At its start: the "failed" count nested in it has a hint of its own, the innermost one shown.
    calls = page.locator(TOKENS.total).locator("[data-hint]").first
    calls.hover(position={"x": 2, "y": 2})
    expect(hint).to_have_text(TERMS["calls"]["definition"])
    calls.locator("[data-hint]").hover()
    expect(hint).to_have_text(TERMS["errors"]["definition"])
    # At the right edge, a short hint and then a long one (as a live figure changes it): the long one keeps its full
    # width and stays on screen.
    page.evaluate(
        """() => { const e = document.createElement('span');
                   Object.assign(e, {id: 'edge', textContent: 'e'});
                   e.style.cssText = 'position:fixed;right:0;top:300px';
                   e.dataset.hint = 'x';
                   document.body.append(e); }"""
    )
    page.hover("#edge")
    expect(hint).to_have_text("x")
    long = TERMS["cache_hit"]["definition"]
    page.evaluate("(text) => { document.getElementById('edge').dataset.hint = text; }", long)
    expect(hint).to_have_text(long)
    box = hint.bounding_box()
    assert box["width"] > 250 and box["x"] >= 0 and box["x"] + box["width"] <= 1600, box
    # Below the fold: the hover scrolls the heading in first, and the hint stays with it.
    page.set_viewport_size({"width": 1600, "height": 500})
    # A cost category: what it is, how it arises and how it is billed, a paragraph each.
    billed = TERMS["cache_read"]["definition"].split("\n")
    page.locator(TOKENS.agents).locator("th", has_text=TERMS["cache_read"]["label"]).hover()
    expect(hint.locator("div")).to_have_text(billed)
    page.set_viewport_size({"width": 1600, "height": 1200})
    # On its legend chip, then what a click does.
    page.locator(TOKENS.legend, has_text=TERMS["cache_read"]["label"]).hover()
    expect(hint.locator("div")).to_have_text([*billed, "Click to show or hide."])
    page.hover(TOKENS.rate)
    expect(hint).to_have_text(TERMS["per_minute"]["definition"])
    # A click re-renders the tab under the still pointer: the hint stays.
    page.click(TOKENS.rate)
    page.wait_for_timeout(100)
    expect(hint).to_have_text(TERMS["per_minute"]["definition"])
    page.hover(UI.tab.format("tokens"))
    expect(hint).to_be_hidden()
    page.focus(TOKENS.rate)
    expect(hint).to_have_text(TERMS["per_minute"]["definition"])
    page.keyboard.press("Escape")
    expect(hint).to_be_hidden()
    # The definitions are hover hints only, with no section of their own.
    expect(page.locator("body")).not_to_contain_text("Definitions")


_TOTAL_ENDS = """svg => {
  const edges = [...svg.querySelectorAll('path')]
    .filter((p) => p.classList.contains('edge') || p.getAttribute('fill') === 'none');
  const now = svg.querySelector('line.now');
  return {now: now && +now.getAttribute('x1'), right: svg.viewBox.baseVal.width - 8,
          ends: edges.map((p) => +p.getAttribute('d').split('L').pop().split(',')[0])};
}"""


@pytest.mark.parametrize("running", [True])
def test_while_the_run_is_running_token_charts_carry_the_moving_now_line(page):
    page.click(UI.tab.format("tokens"))
    for chart in (TOKENS.chart, TOKENS.cost_chart):
        line = page.locator(chart).locator(TOKENS.now)
        expect(line).to_contain_text("now ")
        first = line.text_content()
        # It moves with the session clock, as the timeline's does.
        expect(line).not_to_have_text(first, timeout=5000)
        # The fixture's last call was days ago: its running total runs on flat to the line.
        ends = page.locator(chart).evaluate(_TOTAL_ENDS)
        assert ends["ends"] and all(abs(x - ends["now"]) < 2 for x in ends["ends"]), ends


@pytest.mark.parametrize("specialist_log", [TURNS])
def test_a_specialist_that_ended_without_reporting_its_output_is_told_so_not_pending(page):
    # The fixture's t-perf succeeded, but its log has turns and no result line, so its output never came.
    page.click(UI.tab.format("tokens"))
    expect(page.locator("[data-testid=token-output-unreported]")).to_have_text("Output not reported: 3 turns")
    expect(page.locator("[data-testid=token-output-pending]")).to_have_count(0)
    expect(page.locator("body")).to_contain_text(
        "1 specialist ended without reporting its output (cancelled, say): their turns come from the live log, and "
        "their output is unknown and left out of the totals and the cost."
    )
    page.hover("[data-testid=token-output-unreported]")
    expect(page.locator("#hint-tip")).to_have_text(TERMS["output_unreported"]["definition"])


def test_once_the_run_has_gone_quiet_token_charts_end_at_its_last_activity(page):
    page.click(UI.tab.format("tokens"))
    for chart in (TOKENS.chart, TOKENS.cost_chart):
        expect(page.locator(chart).locator("path")).not_to_have_count(0)
        expect(page.locator(chart).locator(TOKENS.now)).to_have_count(0)
        ends = page.locator(chart).evaluate(_TOTAL_ENDS)
        assert ends["ends"] and all(x >= ends["right"] - 2 for x in ends["ends"]), ends


def test_the_price_sheet_says_what_its_rates_come_from(page):
    page.click(UI.tab.format("tokens"))
    expect(page.locator(TOKENS.pricing)).to_contain_text(re.compile(r"Built from\s*The LLM gateway's rates"))


# One call on a model the price sheet does not list, one that names no model.
UNPRICED = [
    {"component": "critic", "model": "unlisted-model", "input_tokens": 70, "ts": iso(T0 + 640)},
    {"component": "critic", "input_tokens": 30, "ts": iso(T0 + 650)},
]


@pytest.mark.parametrize("extra_calls", [UNPRICED])
def test_calls_without_a_price_are_flagged_with_why_not_costed_at_zero(page):
    page.click(UI.tab.format("tokens"))
    no_model, no_price = r"1 call(\(s\))? with no model recorded", r"1 call(\(s\))? with no price for the model"
    cost = page.locator(TOKENS.cost)
    expect(cost).to_contain_text("0.016")
    expect(cost).to_contain_text(re.compile(f"Unpriced: {no_model}, {no_price}"))
    models = page.locator(TOKENS.models)
    expect(models.locator("tr", has_text="no model recorded")).to_contain_text(re.compile(f"– · unpriced: {no_model}"))
    expect(models.locator("tr", has_text="unlisted-model")).to_contain_text(re.compile(f"– · unpriced: {no_price}"))
    critic = page.locator(TOKENS.agents).locator("tr", has_text="critic")
    expect(critic).to_contain_text(re.compile(f"· unpriced: {no_model}, {no_price}"))
    why = page.locator(TOKENS.unpriced)
    expect(why).to_have_class(re.compile(r"\bh-crashed\b"))
    expect(why).to_contain_text(re.compile(r"2 of \d+ LLM call(s|\(s\)), left out of the estimated cost\."))
    expect(why).to_contain_text(
        re.compile(
            r"1 LLM call(\(s\))? with no model recorded, as neither the call log nor the specialist's log names its "
            r"model\.\s*1 LLM call(\(s\))? with no price for the model: unlisted-model \(1 call(\(s\))?\), as the price "
            r"sheet lists no price for it\."
        )
    )
    # Right after the price source, ahead of how a cost is worked out.
    text = page.locator(TOKENS.pricing).text_content()
    assert text.index("Price source") < text.index("left out of the estimated") < text.index("Cost in this run ="), text
    cost.locator("[data-hint]", has_text="with no model recorded").hover()
    expect(page.locator("#hint-tip")).to_have_text(TERMS["no_model"]["definition"])


def test_the_tokens_tab_prices_with_the_built_in_sheet(page):
    for tab in ("tasks", "artifacts", "bus", "llm", "tokens", "journal"):
        expect(page.locator(UI.tab.format(tab))).to_be_visible()
    page.click(UI.tab.format("tokens"))
    expect(page.locator(TOKENS.agents)).to_contain_text(TERMS["cost_usd"]["label"])
    expect(page.locator(TOKENS.pricing)).to_contain_text("built-in price sheet")
    page.locator(TOKENS.agents).locator("th", has_text=TERMS["cache_read"]["label"]).hover()
    expect(page.locator("#hint-tip").locator("div")).to_have_text(TERMS["cache_read"]["definition"].split("\n"))


def test_folding_the_uplift_panel_leaves_its_summary(page):
    strips = page.locator("[data-testid=uplift-svg]")
    expect(strips).to_be_visible()
    page.click("[data-testid=uplift-toggle]")
    expect(page.locator("[data-testid=uplift-peek]")).to_be_visible()
    expect(strips).to_have_count(0)
