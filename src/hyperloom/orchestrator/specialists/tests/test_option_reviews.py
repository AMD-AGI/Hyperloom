# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A research specialist of a session that keeps its server's option list is shown the list, asked for a verdict on
every option the base launch leaves unset, and has what it returns held against the list."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.common.tests.handed_names import knob_words_named, pull_requests_named, server_options_named
from hyperloom.inference_optimizer.native_builds import OptionVerdict
from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR
from hyperloom.measure.runner.native.server_options import build_inventory, write_inventory
from hyperloom.orchestrator.prompts.specialist_prompt_builder import build_specialist_prompts
from hyperloom.orchestrator.specialists import option_reviews, runner as sr
from hyperloom.orchestrator.specialists.domains import get_domain
from hyperloom.orchestrator.specialists.model_dirs import session_reads
from hyperloom.orchestrator.specialists.profile import MODE_PATCH, MODE_RESEARCH
from hyperloom.orchestrator.specialists.prompt_inputs import SpecialistPromptInputs

_HELP = """\
-c,    --ctx-size N                     size of the prompt context (default: 0)
-t,    --threads N                      number of CPU threads to use during generation
--jinja, --no-jinja                     whether to use the chat template engine
--keep N                                number of tokens to keep from the initial prompt
--mlock                                 force system to keep model in RAM
"""
_UNSET = ("--threads", "--no-jinja", "--keep", "--mlock")


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A session directory that keeps the option list of a server whose base launch sets ``-c``."""
    root = tmp_path / "session"
    root.mkdir()
    executable = tmp_path / "llama-server"
    executable.write_bytes(b"binary")
    write_inventory(build_inventory("llama-server", executable, _HELP, "-c 4096"), root)
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, str(root))
    return root


@pytest.fixture
def bare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A session directory that keeps no option list."""
    root = tmp_path / "bare"
    root.mkdir()
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, str(root))
    return root


def _prompt(framework: str = "llamacpp", mode: str = MODE_RESEARCH) -> str:
    inputs = SpecialistPromptInputs(domain=get_domain("serving_specialist"), framework=framework, mode=mode)
    return "\n".join(build_specialist_prompts(inputs))


def _entries(*options: str, verdict: str = OptionVerdict.NO_EFFECT_ON_STEP.value) -> list[dict[str, str]]:
    return [{"option": option, "verdict": verdict, "note": "read its help line"} for option in options]


def _finalize(root: Path | None, workspace: Path, payload: dict, *, mode=MODE_RESEARCH):
    """The result ``_finalize`` builds for ``payload`` and the done file it writes but for its time, for a specialist
    of ``mode``."""
    runner = sr.SpecialistRunner(subprocess_config=sr.SpecialistSubprocessConfig(), session_dir=root)
    prep = sr._PreparedRun(
        domain=SimpleNamespace(key="serving_specialist"),
        gap="gap-1",
        workspace=workspace,
        profile=sr.SpecialistProfile(mode=mode),
    )
    ctx = SimpleNamespace(task=SimpleNamespace(task_id="t-review", params={"framework": "llamacpp"}), extra={})
    result = runner._finalize(ctx=ctx, prep=prep, sub_result=sr.SpecialistSubprocessResult(done_payload=payload))
    written = json.loads((workspace / "specialist_done.json").read_text(encoding="utf-8"))
    return result, {name: value for name, value in written.items() if name != "ts"}


def _review_notes(notes: list[str]) -> list[str]:
    return [note for note in notes if note.startswith("options_review")]


def test_the_request_names_the_file_and_no_option_of_the_server(session: Path) -> None:
    request = "\n".join(
        option_reviews.review_request(
            SpecialistPromptInputs(domain=get_domain("serving_specialist"), framework="llamacpp", mode=MODE_RESEARCH)
        )
    )

    assert f"``{session / 'server_options.json'}``" in request and f"each of the **{len(_UNSET)}** options" in request
    assert server_options_named(request) == {"--help"}
    assert not knob_words_named(request) and not pull_requests_named(request)


def test_no_review_is_asked_of_a_patch_specialist_or_where_the_session_keeps_no_option_list(
    session: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "options_review" in _prompt()
    assert "options_review" not in _prompt(mode=MODE_PATCH)
    assert "options_review" not in _prompt(framework="sglang")

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv(ENV_CURRENT_SESSION_DIR, str(empty))
    assert "options_review" not in _prompt()


@pytest.mark.parametrize(
    ("review", "notes"),
    [
        (_entries("-t", "--jinja", "--keep", "--mlock"), ["options_review_complete:4"]),
        (_entries("--mlock") + _entries("--keep", verdict="looks fine"), ["options_review_incomplete:3/4"]),
        (None, ["options_review_incomplete:4/4"]),
        (_entries(*_UNSET, "--no-such-option", "-c"), ["options_review_complete:4", "options_review_unknown:1"]),
    ],
    ids=["every-option-by-any-spelling", "missed-options", "no-review", "unlisted-spelling"],
)
def test_the_review_a_research_specialist_returns_is_noted_against_the_option_list(
    session: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture, review: list | None, notes: list[str]
) -> None:
    payload = {"proposal_set": [{"name": "v1"}], "summary": "s"}
    payload.update({} if review is None else {"options_review": review})

    result, written = _finalize(session, tmp_path, payload)

    assert _review_notes(result.notes) == notes
    assert set(notes) <= set(written["_specialist_notes"]) and written.get("options_review") == review
    assert result.status == "succeeded" and written["proposal_set"][0]["name"] == "v1", "the proposals are admitted"
    assert bool(caplog.records) == (notes != ["options_review_complete:4"]), "a review that falls short is said"


@pytest.mark.parametrize(("mode", "kept"), [(MODE_RESEARCH, False), (MODE_PATCH, True)])
def test_nothing_is_checked_without_an_option_list_or_of_a_patch_specialist(
    session: Path, bare: Path, tmp_path: Path, mode: str, kept: bool
) -> None:
    payload = {"proposal_set": [{"name": "v1"}], "summary": "s", "options_review": _entries("--mlock")}

    result, written = _finalize(session if kept else bare, tmp_path, dict(payload), mode=mode)
    unchecked, plain = _finalize(None, tmp_path, dict(payload), mode=mode)

    assert _review_notes(result.notes) == []
    assert (result.notes, written) == (unchecked.notes, plain), "as a runner with no session admits it"


def test_a_specialist_reads_the_option_list_its_session_keeps_beside_its_models(session: Path, tmp_path: Path) -> None:
    model = tmp_path / "models" / "m"
    model.mkdir(parents=True)
    task = {"framework": "llamacpp", "model_path": str(model)}

    assert session_reads(session, task) == (model, session / "server_options.json")
    assert session_reads(session, {**task, "framework": "sglang"}) == (model,)
    assert session_reads(None, task) == (model,)
