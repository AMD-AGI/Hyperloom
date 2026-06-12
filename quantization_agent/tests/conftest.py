"""Shared fixtures for quantization_agent tests.

Most tests build a synthetic workspace on tmp_path and feed it to the
classifier / retry loop directly. Two helpers are exposed here:

* ``build_workspace`` — turn a small dict of artifact stubs into a
  workspace dir, no Quark / SDK needed.
* ``FakeSDK`` — a callable matching the injection seam in
  ``driver.runner.run_one_attempt`` (``sdk_query_factory`` + ``sdk_options_cls``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# workspace builder
# ─────────────────────────────────────────────────────────────────────────────

# Validation-report snippets keyed by short tag. Tests pick the tag they need
# rather than copy-paste markdown.
_VALIDATION_REPORTS: dict[str, str] = {
    "all_ok": (
        "## Validation Report — quark-torch-result-validator\n\n"
        "**Step 4 — fuzzy tensor names**: ok\n\n"
        "**Step 1 — auxiliary files**: ok\n\n"
        "**Step 3 — config.json**: ok\n\n"
        "**Step 2 — MD5 spot-check**: ok\n"
    ),
    "md5_fail": (
        "**Step 4 — fuzzy tensor names**: ok\n"
        "**Step 1 — auxiliary files**: ok\n"
        "**Step 3 — config.json**: ok\n"
        "**Step 2 — MD5 spot-check**: FAIL\n"
    ),
    "config_fail": (
        "**Step 4 — fuzzy tensor names**: ok\n"
        "**Step 1 — auxiliary files**: ok\n"
        "**Step 3 — config.json**: FAIL\n"
        "**Step 2 — MD5 spot-check**: ok\n"
    ),
    "fuzzy_fail": (
        "**Step 4 — fuzzy tensor names**: FAIL\n"
        "**Step 1 — auxiliary files**: ok\n"
        "**Step 3 — config.json**: ok\n"
        "**Step 2 — MD5 spot-check**: ok\n"
    ),
    "aux_fail": (
        "**Step 4 — fuzzy tensor names**: ok\n"
        "**Step 1 — auxiliary files**: FAIL\n"
        "**Step 3 — config.json**: ok\n"
        "**Step 2 — MD5 spot-check**: ok\n"
    ),
    "must_validate_skipped": (
        "**Step 4 — fuzzy tensor names**: ok\n"
        "**Step 1 — auxiliary files**: ok\n"
        "**Step 3 — config.json**: skipped\n"
        "**Step 2 — MD5 spot-check**: skipped\n"
    ),
}


def _write_manifest(workspace: Path, quantized_dir: Path) -> None:
    """Write a minimal run_manifest.yaml. PyYAML required (skip otherwise)."""

    pytest.importorskip("yaml")
    import yaml

    payload = {
        "version": "1",
        "outputs": {"quantized_model_dir": str(quantized_dir)},
    }
    (workspace / "run_manifest.yaml").write_text(
        yaml.safe_dump(payload), encoding="utf-8"
    )


def _make_quantized_dir(
    workspace: Path,
    *,
    include_config: bool = True,
    include_weights: bool = True,
    include_tokenizer: bool = True,
) -> Path:
    qdir = workspace / "quantized-model"
    qdir.mkdir(parents=True, exist_ok=True)
    if include_config:
        (qdir / "config.json").write_text("{}", encoding="utf-8")
    if include_weights:
        (qdir / "model.safetensors").write_bytes(b"\x00" * 16)
    if include_tokenizer:
        (qdir / "tokenizer.json").write_text("{}", encoding="utf-8")
    return qdir


@dataclass
class WorkspaceBuilder:
    """Fluent builder mirroring the artifact fields SKILL.md writes.

    Defaults to a fully-successful run (manifest + quantized dir + all-ok
    validation_report + eval_report with 0% gap). Tests subtract or override
    pieces to simulate failure modes.
    """

    workspace: Path
    include_manifest: bool = True
    include_quantized_dir: bool = True
    include_config: bool = True
    include_weights: bool = True
    include_tokenizer: bool = True
    include_validation_report: bool = True
    validation_tag: str = "all_ok"
    include_eval_report: bool = True
    eval_report: dict[str, Any] | None = None
    eval_skipped_reason: str | None = None
    last_phase: str | None = None
    blocked_md: str | None = None
    fix_hypotheses: tuple[int, ...] = ()
    requantize_attempts: int | None = None
    eval_gap_threshold: float | None = None
    model_analysis: bool = True
    quant_plan: bool = True
    session_context: bool = True

    def build(self) -> Path:
        ws = self.workspace
        ws.mkdir(parents=True, exist_ok=True)

        if self.model_analysis:
            (ws / "model_analysis.json").write_text("{}", encoding="utf-8")
        if self.quant_plan:
            (ws / "quant_plan.json").write_text("{}", encoding="utf-8")
        if self.session_context:
            (ws / "session_context.json").write_text("{}", encoding="utf-8")

        if self.include_quantized_dir:
            qdir = _make_quantized_dir(
                ws,
                include_config=self.include_config,
                include_weights=self.include_weights,
                include_tokenizer=self.include_tokenizer,
            )
        else:
            qdir = ws / "quantized-model"  # path referenced by manifest but absent

        if self.include_manifest:
            _write_manifest(ws, qdir)

        if self.include_validation_report:
            (ws / "validation_report.md").write_text(
                _VALIDATION_REPORTS[self.validation_tag], encoding="utf-8"
            )

        if self.include_eval_report:
            payload = self.eval_report or {
                "metric_name": "gsm8k",
                "dataset": "gsm8k",
                "backend": "vllm",
                "source_score": 0.50,
                "quantized_score": 0.50,
                "relative_gap": 0.0,
            }
            (ws / "eval_report.json").write_text(json.dumps(payload), encoding="utf-8")
            (ws / "source_eval.md").write_text("source", encoding="utf-8")
            (ws / "quantized_eval.md").write_text("quantized", encoding="utf-8")

        if self.eval_skipped_reason is not None:
            (ws / "eval_skipped.txt").write_text(self.eval_skipped_reason, encoding="utf-8")

        if self.last_phase is not None:
            (ws / "last_phase.txt").write_text(self.last_phase, encoding="utf-8")

        if self.blocked_md is not None:
            (ws / "blocked.md").write_text(self.blocked_md, encoding="utf-8")

        for n in self.fix_hypotheses:
            (ws / f"fix_hypothesis_attempt_{n}.md").write_text(
                f"# Fix hypothesis for attempt {n}\n", encoding="utf-8"
            )

        if self.requantize_attempts is not None:
            (ws / "requantize_attempts.txt").write_text(
                str(self.requantize_attempts), encoding="utf-8"
            )

        if self.eval_gap_threshold is not None:
            (ws / "eval_gap_threshold.txt").write_text(
                str(self.eval_gap_threshold), encoding="utf-8"
            )

        return ws


@pytest.fixture
def build_workspace(tmp_path: Path) -> Callable[..., Path]:
    """Return a factory: ``build_workspace(**overrides) → Path``."""

    def _factory(**overrides: Any) -> Path:
        ws = overrides.pop("workspace", tmp_path)
        builder = WorkspaceBuilder(workspace=ws, **overrides)
        return builder.build()

    return _factory


# ─────────────────────────────────────────────────────────────────────────────
# fake SDK
# ─────────────────────────────────────────────────────────────────────────────

class FakeMessage:
    def __init__(self, text: str):
        self.content = [type("Block", (), {"text": text})()]


@dataclass
class FakeOptions:
    """Captures kwargs passed to ``sdk_options_cls``. Plays the role of
    ``claude_agent_sdk.ClaudeAgentOptions`` in tests without touching network.
    """

    kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs: Any):
        # Accept any kwarg set without validation.
        self.kwargs = dict(kwargs)


@dataclass
class FakeSDK:
    """Stub for ``sdk_query_factory`` — records prompts and replays scripted
    responses. Pass to ``run_one_attempt(sdk_query_factory=..., sdk_options_cls=...)``.

    ``side_effect`` (when set) is raised on the next call. ``scripted_chunks``
    yields each string as a separate message.
    """

    scripted_chunks: list[str] = field(default_factory=lambda: ["fake-sdk: ok"])
    side_effect: Exception | None = None
    received_prompts: list[str] = field(default_factory=list)
    received_options: list[FakeOptions] = field(default_factory=list)

    def __call__(self, *, prompt: str, options: FakeOptions) -> AsyncIterator[Any]:
        self.received_prompts.append(prompt)
        self.received_options.append(options)
        if self.side_effect is not None:
            err = self.side_effect

            async def _raise() -> AsyncIterator[Any]:
                raise err
                yield  # pragma: no cover

            return _raise()

        chunks = list(self.scripted_chunks)

        async def _gen() -> AsyncIterator[Any]:
            for c in chunks:
                yield FakeMessage(c)

        return _gen()


@pytest.fixture
def fake_sdk() -> FakeSDK:
    return FakeSDK()


@pytest.fixture
def fake_options_cls() -> type[FakeOptions]:
    return FakeOptions
