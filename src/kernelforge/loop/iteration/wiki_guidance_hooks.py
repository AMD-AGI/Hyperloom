# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A ``--wiki-guided`` campaign's knowledge base round as the iteration loop calls it: opening the round's state,
heading the Implementer's history with its note, reading what the batch session reported, and crediting the
measurement. Every hook is a no-op without the flag."""

from __future__ import annotations

from kernelforge.knowledge.wiki_tool import KnowledgeWikiTool
from kernelforge.loop.iteration.iteration_types import IterationResult
from kernelforge.loop.wiki_guided_rounds import GuidanceSinks, WikiGuidedRounds


class WikiGuidanceMixin:
    """The iteration loop's side of the knowledge base round."""

    def _open_wiki_rounds(self, resume: bool) -> WikiGuidedRounds:
        sinks = GuidanceSinks(lambda event: self.state_store.append_event(event), self._persistence_failed)
        knowledge_wiki = KnowledgeWikiTool.from_config(self.config)
        return WikiGuidedRounds(self.ic.workspace_dir, knowledge_wiki=knowledge_wiki, resume=resume, sinks=sinks)

    def _persistence_failed(self, what: str) -> None:
        self.persistence_degraded = True
        self.persistence_errors.append(what)
        self.persistence_errors = self.persistence_errors[-10:]

    def _with_guidance_note(self, history: str) -> str:
        """The Implementer's history headed by the guidance phase note, when the campaign is guided."""
        note = self.wiki_rounds.implementer_note() if self.wiki_rounds is not None else ""
        return f"{note}\n\n{history}" if note else history

    def _note_batch_report(self, iteration: int, session_sink: dict) -> None:
        if self.wiki_rounds is not None:
            final_text = str(session_sink.get("final_text") or "")
            self.wiki_rounds.note_session_report(iteration, final_text)

    def _record_wiki_outcome(self, result: IterationResult, decision_label: str) -> None:
        if self.wiki_rounds is not None:
            self.wiki_rounds.record(result.iteration, decision_label, detail=result.validation_summary)
