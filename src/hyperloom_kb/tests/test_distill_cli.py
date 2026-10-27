# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``hyperloom-kb distill`` takes an author from exploring Experiences to a Distill and keeps a directory of pages a
Curated Wiki searches in step with the KB; a draft the KB would refuse, and input it cannot read, exit 1."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from hyperloom_kb import FieldDeclaration, FieldKind, FieldRole
from hyperloom_kb.cli import main
from hyperloom_kb.distill import FormDeclaration
from hyperloom_kb.tests.distills import IDS, RULE, SCHEMA, TOKEN, make_draft, make_experience, serving

CONTRADICTING = make_experience(7, "run-c", "qwen3", "triton", "keep", 103.0)
_EMPTY_EDITS = {"add": [], "remove": [], "annotations": {}, "representatives": [], "notes": {}}


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Any:
    """Run ``hyperloom-kb`` against a client's service: ``(exit status, parsed stdout or stderr text)``."""

    def run(client: Any, *argv: str) -> tuple[int, Any]:
        monkeypatch.setenv("HYPERLOOM_KB_URL", client.config.base_url)
        monkeypatch.setenv("HYPERLOOM_KB_TOKEN", TOKEN)
        status = main(list(argv))
        out, err = capsys.readouterr()
        return status, json.loads(out) if out else err

    return run


def _write(path: Path, value: Any) -> str:
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path)


def test_an_author_explores_writes_and_maintains_a_distill_from_the_command_line(tmp_path: Path, cli: Any) -> None:
    form = _write(tmp_path / "rule.json", RULE.to_dict())
    draft = _write(tmp_path / "draft.json", {key: v for key, v in make_draft().items() if key != "schema_ref"})
    with serving(tmp_path / "kb") as (_, client):
        _, schema = cli(client, "schema", "show")
        _, experience = cli(client, "experience", "get", IDS[0])
        _, explored = cli(
            client, "distill", "explore", "--scope", '{"identity.model": ["qwen3"]}', "--group-by", "change.attention"
        )
        _, added = cli(client, "distill", "add-form", form)
        _, forms = cli(client, "distill", "forms")
        previewed = cli(client, "distill", "preview", draft)
        created = cli(client, "distill", "create", draft)
        entry_id = created[1]["entry_id"]
        client.write(CONTRADICTING)
        _, listed = cli(client, "distill", "list")
        _, changes = cli(client, "distill", "changes")
        _, exported = cli(client, "distill", "draft", entry_id)
        updated = cli(client, "distill", "update", entry_id, _write(tmp_path / "exported.json", exported))
        _, refreshed = cli(client, "distill", "list")
        _, first = cli(
            client, "distill", "get", entry_id, "--version", "1", "--follow-evidence", "--limit", "2",
            "--representatives", "2",
        )  # fmt: skip
        _, deleted = cli(client, "distill", "delete", entry_id, "--reason", "superseded")
        _, hidden = cli(client, "distill", "list")
        _, restored = cli(client, "distill", "restore-entry", entry_id)

    assert schema["paths"][0] == "objective" and experience["experience"]["id"] == IDS[0]
    assert explored["count"] == 6
    assert [(cell["values"]["change.attention"], cell["count"]) for cell in explored["cells"]] == [
        ("triton", 4),
        ("aiter", 2),
    ]
    assert (added["status"], [item["form_ref"] for item in forms["forms"]]) == ("created", [RULE.form_ref])
    # A draft that names no schema takes the service's default one.
    assert (previewed[0], previewed[1]["valid"], previewed[1]["population"]) == (0, True, 4)
    assert (created[0], created[1]["version"]) == (0, 1)
    assert listed["items"][0]["evidence_status"] == "contradicted"
    assert [item["entry_id"] for item in changes["items"]] == [entry_id]
    # Updating with the exported draft takes the new contradicting member in, which settles the change.
    assert exported == {**make_draft(), "relations": {"entries": [], "external": []}, **_EMPTY_EDITS}
    assert (updated[0], updated[1]["version"]) == (0, 2)
    assert refreshed["items"][0]["evidence_status"] == "consistent"
    assert (first["entry"]["version"], first["drift"]["evidence_status"]) == (1, "contradicted")
    assert (len(first["evidence"]["items"]), first["representatives"]) == (2, [IDS[2], IDS[4]])
    assert deleted == {"experience_id": entry_id, "status": "excluded"} and hidden["items"] == []
    assert restored["status"] == "included"


def test_a_draft_the_kb_would_refuse_and_input_it_cannot_read_exit_1(
    tmp_path: Path, cli: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    refused = _write(tmp_path / "refused.json", make_draft(claim={"effect": "hurts"}))
    with serving(tmp_path / "kb") as (_, client):
        client.register_distill_form(SCHEMA.schema_ref, RULE.to_dict())
        previewed = cli(client, "distill", "preview", refused)
        create_status, create_error = cli(client, "distill", "create", refused)
        missing_status, missing_error = cli(client, "distill", "create", str(tmp_path / "absent.json"))
        scope_status, scope_error = cli(client, "distill", "explore", "--scope", "{qwen3")
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(make_draft())))
        piped = cli(client, "distill", "create", "-")

    assert (previewed[0], previewed[1]["valid"]) == (1, False)
    assert "required field 'statement'" in previewed[1]["errors"][0]
    assert create_status == 1 and "Experience KB distill create failed" in create_error and "HTTP 400" in create_error
    assert missing_status == 1 and "cannot read the draft" in missing_error
    assert scope_status == 1 and "--scope is not JSON" in scope_error
    assert (piped[0], piped[1]["status"]) == (0, "created")


def _frontmatter(page: str) -> tuple[dict[str, Any], str]:
    head, body = page.removeprefix("---\n").split("\n---\n", 1)
    return {key: json.loads(value) for key, value in (line.split(": ", 1) for line in head.splitlines())}, body


def test_sync_mirrors_the_visible_distills_into_one_page_each_and_leaves_other_files_alone(
    tmp_path: Path, cli: Any
) -> None:
    pages = tmp_path / "wiki" / "distilled"
    with serving(tmp_path / "kb") as (_, client):
        client.register_distill_form(SCHEMA.schema_ref, RULE.to_dict())
        kept = str(client.create_distill(make_draft())["entry_id"])
        dropped = str(client.create_distill(make_draft(tags=["copy"]))["entry_id"])
        _, first = cli(client, "distill", "sync", "--dir", str(pages))
        folder = Path(first["schemas"][SCHEMA.schema_ref]["directory"])
        (folder / "README.md").write_text("Not the KB's.", encoding="utf-8")
        (folder / f"{kept}.md").write_text("Edited by hand.", encoding="utf-8")
        client.exclude(dropped, reason="a duplicate")
        client.write(CONTRADICTING)
        _, second = cli(client, "distill", "sync", "--dir", str(pages))
        _, third = cli(client, "distill", "sync", "--dir", str(pages))
    fields, body = _frontmatter((folder / f"{kept}.md").read_text(encoding="utf-8"))

    def counts(report: dict[str, Any]) -> tuple[int, ...]:
        done = report["schemas"][SCHEMA.schema_ref]
        return tuple(done[key] for key in ("created", "updated", "unchanged", "removed"))

    assert (counts(first), counts(second), counts(third)) == ((2, 0, 0, 0), (0, 1, 0, 1), (0, 0, 1, 0))
    assert sorted(path.name for path in folder.iterdir()) == ["README.md", f"{kept}.md"]
    assert (folder / "README.md").read_text(encoding="utf-8") == "Not the KB's."
    assert [(item["entry_id"], item["evidence_status"]) for item in second["needs_attention"]] == [
        (kept, "contradicted")
    ]
    assert (fields["id"], fields["version"], fields["form"], fields["tags"]) == (kept, 1, "rule", ["attention"])
    assert fields["scope"] == {"change.attention": ["triton"], "identity.model": ["qwen3"]}
    assert (fields["evidence_status"], fields["needs_attention"]) == ("contradicted", True)
    assert fields["evidence"] == {
        "snapshot": {"supports": 3, "contradicts": 1, "irrelevant": 0, "runs": 2},
        "current": {"supports": 3, "contradicts": 2, "irrelevant": 0, "undetermined": 0, "runs": 3},
    }
    assert fields["representatives"] == [IDS[2], IDS[4], IDS[1]]
    assert body.startswith("\n# Triton attention alone does not raise qwen3 throughput.\n")
    assert "**effect:** hurts" in body
    assert "Evidence at this sync (contradicted): 3 supporting, 2 contradicting, 0 undetermined, from 3 runs." in body
    # Ids stay in the frontmatter, out of the text a search matches.
    assert all(record_id not in body for record_id in (kept, *IDS, CONTRADICTING.id))


def test_a_page_writes_numbers_and_booleans_as_json_does(tmp_path: Path, cli: Any) -> None:
    measured = FormDeclaration(
        "measured",
        "A measured effect.",
        (
            FieldDeclaration("statement", "What holds.", FieldKind.TEXT, required=True, role=FieldRole.SUMMARY),
            FieldDeclaration("speedup", "Median speedup.", FieldKind.NUMBER),
            FieldDeclaration("reproduced", "Whether a rerun agreed.", FieldKind.BOOLEAN),
        ),
    )
    claim = {"statement": "Aiter attention speeds qwen3 up.", "speedup": 1.11, "reproduced": True}
    with serving(tmp_path / "kb") as (_, client):
        client.register_distill_form(SCHEMA.schema_ref, measured.to_dict())
        scope = {"identity.model": ["qwen3"], "change.attention": ["aiter"]}
        draft = make_draft(form_ref=measured.form_ref, claim=claim, scope=scope)
        entry_id = str(client.create_distill(draft)["entry_id"])
        _, synced = cli(client, "distill", "sync", "--dir", str(tmp_path / "pages"))
    page = Path(synced["schemas"][SCHEMA.schema_ref]["directory"]) / f"{entry_id}.md"

    _, body = _frontmatter(page.read_text(encoding="utf-8"))
    assert "**speedup:** 1.11" in body and "**reproduced:** true" in body
