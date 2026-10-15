# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The schema, Experiences, forms, and drafts the Distill tests share, and a served KB that holds them."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from hyperloom_kb import (
    Experience,
    ExperienceDeclaration,
    ExperienceHTTPService,
    ExperienceStatus,
    FieldDeclaration,
    FieldKind,
    FieldRole,
    HTTPServiceConfig,
    ObjectiveDeclaration,
    Provenance,
    RemoteClient,
    RemoteConfig,
    create_http_server,
    derive_experience_id,
)
from hyperloom_kb.distill import FormDeclaration
from hyperloom_kb.tests.conftest import fresh_database

TOKEN = "distill-token"
NOW = datetime(2026, 10, 13, tzinfo=timezone.utc)
SCHEMA = ExperienceDeclaration(
    objectives=(ObjectiveDeclaration("throughput@v1", "Maximize throughput."),),
    identity=(FieldDeclaration("model", "Model."), FieldDeclaration("tp", "Tensor parallel.", FieldKind.NUMBER)),
    baseline=(FieldDeclaration("throughput", "Baseline.", FieldKind.NUMBER, role=FieldRole.MEASUREMENT),),
    change=(
        FieldDeclaration("attention", "Attention backend.", group=True),
        FieldDeclaration("summary", "What changed.", role=FieldRole.SUMMARY),
    ),
    outcome=(
        FieldDeclaration("decision", "Decision.", role=FieldRole.DECISION, values=("keep", "revert")),
        FieldDeclaration("throughput", "Throughput.", FieldKind.NUMBER, role=FieldRole.MEASUREMENT),
    ),
    reflection=(FieldDeclaration("text", "Reflection.", kind=FieldKind.TEXT),),
)
OTHER = ExperienceDeclaration(
    objectives=(ObjectiveDeclaration("latency@v1", "Minimize latency."),),
    identity=(FieldDeclaration("model", "Model."),),
)
RULE = FormDeclaration(
    "rule",
    "What a change does under the scope.",
    (
        FieldDeclaration("statement", "What holds.", FieldKind.TEXT, required=True, role=FieldRole.SUMMARY),
        FieldDeclaration("effect", "Its direction.", values=("helps", "hurts", "neutral")),
    ),
)
PLOTTED = FormDeclaration(
    "plotted",
    "A rule with the plot it was read from.",
    (
        FieldDeclaration("statement", "What holds.", FieldKind.TEXT, required=True, role=FieldRole.SUMMARY),
        FieldDeclaration("plot", "The plot.", FieldKind.FILE, required=True),
    ),
)
#: (seq, run, model, attention, decision, outcome throughput); every baseline is 100.
CORPUS = (
    (0, "run-a", "qwen3", "triton", "revert", 95.0),
    (1, "run-a", "qwen3", "triton", "revert", 97.0),
    (2, "run-a", "qwen3", "triton", "keep", 104.0),
    (3, "run-a", "qwen3", "aiter", "keep", 112.0),
    (4, "run-b", "qwen3", "triton", "revert", 96.0),
    (5, "run-b", "qwen3", "aiter", "keep", 110.0),
    (6, "run-b", "llama", "triton", "keep", 120.0),
)


def make_experience(seq: int, run: str, model: str, attention: str, decision: str, throughput: float) -> Experience:
    at = NOW + timedelta(minutes=seq)
    return Experience(
        id=derive_experience_id("distill-test", run, seq),
        run_id=run,
        seq=seq,
        created_at=at,
        completed_at=at,
        identity={"model": model, "tp": 2},
        objective="throughput@v1",
        baseline={"throughput": 100.0},
        provenance=Provenance("distill-test", "1"),
        schema_ref=SCHEMA.schema_ref,
        status=ExperienceStatus.COMPLETE,
        rationale={"reasoning": f"Try {attention}."},
        change={"attention": attention, "summary": f"Switch attention to {attention}."},
        outcome={"decision": decision, "throughput": throughput},
        reflection={"text": "Measured."},
    )


EXPERIENCES = tuple(make_experience(*row) for row in CORPUS)
IDS = tuple(experience.id for experience in EXPERIENCES)


def make_draft(**changes: Any) -> dict[str, Any]:
    """Triton attention on qwen3 selects Experiences 0, 1, 2, and 4; the check says each revert supports the claim."""

    return {
        "schema_ref": SCHEMA.schema_ref,
        "form_ref": RULE.form_ref,
        "scope": {"identity.model": ["qwen3"], "change.attention": ["triton"]},
        "claim": {"statement": "Triton attention alone does not raise qwen3 throughput.", "effect": "hurts"},
        "check": {"outcome.decision": ["revert"]},
        "tags": ["attention"],
        "author": "zili",
        **changes,
    }


@contextmanager
def serving(
    home: Path, *, global_url: str | None = None, seeded: bool = True
) -> Iterator[tuple[ExperienceHTTPService, RemoteClient]]:
    """A KB at ``home`` syncing with ``global_url``, holding ``EXPERIENCES`` unless not ``seeded``."""

    global_kb = None if global_url is None else RemoteClient(RemoteConfig(global_url, TOKEN))
    app = ExperienceHTTPService(
        HTTPServiceConfig(home, TOKEN), SCHEMA, None, database=fresh_database(), global_kb=global_kb
    )
    server = create_http_server(app, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        client = RemoteClient(
            RemoteConfig(f"http://127.0.0.1:{server.server_address[1]}", TOKEN, spool_root=home.parent / "spool")
        )
        if seeded:
            for experience in EXPERIENCES:
                client.write(experience)
        yield app, client
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
