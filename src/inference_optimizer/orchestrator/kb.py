"""KnowledgeBase wrapper — DESIGN §6.

L4 long-term memory: cross-run lessons keyed by ``(user_id, model_family)``.
Underlying storage stays as JSONL files for v0.7 (entries.jsonl /
insights.jsonl) so external Sage / Critic processes can grep.

STATUS (v0.7):
    Pure-Python implementation. Cold-start gating, ingest, recall (via
    ``kb_query.py`` shell-out), persona read/append, conflict scan, and
    cross-run synthesis are all wired up. Embeddings are intentionally
    omitted — token-overlap / BM25 is enough until the hardware arm
    delivers real workload diversity.

References:
    - DESIGN §6.2 KB schema + cold-start
    - DESIGN §6.3 prompt-time injection
    - DESIGN §10.5.7 update_persona intent
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


__all__ = [
    "Conflict",
    "KBEntry",
    "KnowledgeBase",
    "KBError",
]


class KBError(RuntimeError):
    """Raised on irrecoverable filesystem / parse problems."""


@dataclass
class Conflict:
    """Two KB entries claim opposite outcomes for the same context."""

    entry_a: dict[str, Any]
    entry_b: dict[str, Any]
    reason: str


@dataclass
class KBEntry:
    """Strongly-typed wrapper around a single ``entries.jsonl`` row.

    Always serialise via :meth:`to_dict` to keep the on-disk schema
    stable.
    """

    category: str
    user_id: str
    model: str
    model_family: str
    action: str
    lesson: str
    tags: list[str]
    gain: float
    status: str   # keep / revert / fail / observation
    ts: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "user_id": self.user_id,
            "model": self.model,
            "model_family": self.model_family,
            "action": self.action,
            "lesson": self.lesson,
            "tags": list(self.tags),
            "gain": float(self.gain),
            "status": self.status,
            "ts": float(self.ts),
        }


# ---------------------------------------------------------------------------
def _model_family(model: str) -> str:
    """Coarse family bucket for cold-start gating.

    Strips trailing version / size suffixes so that
    ``llama-3-8b-instruct`` and ``llama-3.1-70b-base`` both map to
    ``llama``. Patterns are deliberately broad to maximise warm-start
    hits.
    """
    if not model:
        return "unknown"
    needle = str(model).lower()
    table: list[tuple[str, str]] = [
        ("deepseek", "deepseek"),
        ("kimi", "kimi"),
        ("mixtral", "mixtral"),
        ("qwen", "qwen"),
        ("llama", "llama"),
        ("mistral", "mistral"),
        ("phi", "phi"),
        ("gpt-oss", "gpt-oss"),
    ]
    for token, family in table:
        if token in needle:
            return family
    # last-resort: take everything before the first digit / dash
    m = re.match(r"^([a-z][a-z_]*)", needle.split("/")[-1])
    return m.group(1) if m else "unknown"


# ---------------------------------------------------------------------------
class KnowledgeBase:
    """Wrapper around ``kb/entries.jsonl`` + ``kb/insights.jsonl``."""

    PARTITION_KEY: str = "user_id"

    def __init__(
        self,
        session_dir: Path,
        user_id: str = "default",
        *,
        kb_dir: Path | None = None,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.user_id = user_id
        # Allow tests / multi-user setups to point at a shared KB outside
        # the session directory (DESIGN §6.2 partition by user_id).
        self.kb_dir = Path(kb_dir) if kb_dir is not None else self.session_dir / "kb"
        self.entries_path = self.kb_dir / "entries.jsonl"
        self.insights_path = self.kb_dir / "insights.jsonl"
        self.conflicts_path = self.kb_dir / "conflicts.jsonl"
        self.personas_dir = self.session_dir / "personas"

    # ------------------------------------------------------------------
    # filesystem hygiene
    # ------------------------------------------------------------------
    def _ensure_dirs(self) -> None:
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.personas_dir.mkdir(parents=True, exist_ok=True)

    def _iter_entries(self) -> list[dict[str, Any]]:
        if not self.entries_path.is_file():
            return []
        out: list[dict[str, Any]] = []
        try:
            for line in self.entries_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate corruption
                if not isinstance(rec, dict):
                    continue
                out.append(rec)
        except OSError as exc:
            raise KBError(f"failed to read {self.entries_path}: {exc}") from exc
        return out

    # ------------------------------------------------------------------
    # cold-start logic (DESIGN §6.2)
    # ------------------------------------------------------------------
    def count_entries(self, model_family: str) -> int:
        """Count entries that match ``user_id × model_family``."""
        target = _model_family(model_family) if "/" in model_family or "-" in model_family else str(model_family).lower()
        n = 0
        for rec in self._iter_entries():
            if rec.get("user_id") not in (self.user_id, None):
                continue
            fam = str(rec.get("model_family", "")).lower()
            if fam == target:
                n += 1
        return n

    def is_warm_start_eligible(self, model_family: str) -> bool:
        """Warm start when ≥1 prior entry exists for the same family.

        DESIGN §6.2 ADR-21: first time we see a family we *only* write
        entries; we don't read them. The caller is expected to gate
        ``recall_for_model`` on this predicate.
        """
        return self.count_entries(model_family) >= 1

    # ------------------------------------------------------------------
    # recall + ingest
    # ------------------------------------------------------------------
    def recall_for_model(
        self,
        model_name: str,
        agent_name: str,
        top_k: int = 5,
        *,
        timeout_s: float = 30.0,
        kb_query_argv: list[str] | None = None,
    ) -> str:
        """Return markdown bullets of the top-k relevant lessons.

        Cold-start short-circuits to the empty string (DESIGN §6.2 ADR-21).
        Otherwise we shell out to ``inference_optimizer.kb.kb_query`` so
        that the same retrieval logic powers external sage callers.
        """
        family = _model_family(model_name)
        if not self.is_warm_start_eligible(family):
            return ""
        argv = kb_query_argv or [
            sys.executable,
            "-m", "inference_optimizer.kb.kb_query",
            f"{model_name} {agent_name}",
            "--kb-dir", str(self.kb_dir),
            "--top-k", str(top_k),
            "--compact",
        ]
        try:
            out = subprocess.check_output(
                argv, timeout=timeout_s, text=True
            )
        except subprocess.TimeoutExpired:
            return ""
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""
        return out.strip()

    def ingest(
        self,
        category: str,
        model: str,
        action: str,
        lesson: str,
        tags: list[str],
        gain: float,
        status: str,
        *,
        ts: float | None = None,
    ) -> KBEntry:
        """Append-only write. Returns the persisted :class:`KBEntry`."""
        self._ensure_dirs()
        rec = KBEntry(
            category=str(category),
            user_id=self.user_id,
            model=str(model),
            model_family=_model_family(model),
            action=str(action),
            lesson=str(lesson),
            tags=list(tags),
            gain=float(gain),
            status=str(status),
            ts=float(ts) if ts is not None else time.time(),
        )
        with self.entries_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.to_dict(), default=str) + "\n")
        return rec

    # ------------------------------------------------------------------
    # personas (per-agent narrative memory) — §6 / §10.5.7
    # ------------------------------------------------------------------
    def read_persona(self, agent_name: str) -> str:
        """Return the full persona body, or an empty string if unset."""
        path = self.personas_dir / f"{agent_name}.md"
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise KBError(f"failed to read persona {path}: {exc}") from exc

    def append_persona(self, agent_name: str, note: str) -> Path:
        """Append a timestamped note to ``personas/<agent>.md``."""
        if not note or not note.strip():
            return self.personas_dir / f"{agent_name}.md"
        self._ensure_dirs()
        path = self.personas_dir / f"{agent_name}.md"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n\n<!-- ts={time.time()} -->\n{note.strip()}\n")
        return path

    # ------------------------------------------------------------------
    # marathon-only synthesis
    # ------------------------------------------------------------------
    def cross_run_synthesize(
        self,
        *,
        max_lookback: int = 200,
        ts: float | None = None,
    ) -> dict[str, Any]:
        """Append a summarising entry to ``insights.jsonl`` (Sage 6h cadence).

        The output is a *summary record* — not a backend call. Real LLM
        synthesis is delegated to the Sage reactor; this routine just
        owns the file-format guarantees.
        """
        self._ensure_dirs()
        recent = self._iter_entries()[-max_lookback:]
        by_family: dict[str, list[dict[str, Any]]] = {}
        for rec in recent:
            fam = str(rec.get("model_family", "unknown"))
            by_family.setdefault(fam, []).append(rec)
        summary = {
            "kind": "cross_run_synthesis",
            "ts": float(ts) if ts is not None else time.time(),
            "samples": len(recent),
            "by_family": {
                fam: {
                    "count": len(rs),
                    "kept_count": sum(1 for r in rs if r.get("status") == "keep"),
                    "mean_gain": (
                        sum(float(r.get("gain", 0.0)) for r in rs) / max(1, len(rs))
                    ),
                }
                for fam, rs in sorted(by_family.items())
            },
        }
        with self.insights_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, default=str) + "\n")
        return summary

    def detect_conflicts(self) -> list[Conflict]:
        """Return entries that disagree on the same ``(model, action)``.

        We flag a conflict when two records have the same model_family +
        action but opposite ``status`` (one ``keep`` and one ``revert``).
        """
        rows = self._iter_entries()
        conflicts: list[Conflict] = []
        # bucket by (family, action)
        bucket: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for r in rows:
            key = (str(r.get("model_family", "")), str(r.get("action", "")))
            bucket.setdefault(key, []).append(r)
        for key, recs in bucket.items():
            if len(recs) < 2:
                continue
            kept = [r for r in recs if r.get("status") == "keep"]
            reverted = [r for r in recs if r.get("status") == "revert"]
            for k in kept:
                for v in reverted:
                    conflicts.append(
                        Conflict(
                            entry_a=k,
                            entry_b=v,
                            reason=(
                                f"{key[0]}/{key[1]}: keep@gain={k.get('gain')} "
                                f"vs revert@gain={v.get('gain')}"
                            ),
                        )
                    )
        # mirror to disk so external review can grep them
        if conflicts:
            self._ensure_dirs()
            with self.conflicts_path.open("a", encoding="utf-8") as fh:
                for c in conflicts:
                    fh.write(
                        json.dumps(
                            {
                                "kind": "kb_conflict",
                                "ts": time.time(),
                                "reason": c.reason,
                                "entry_a": c.entry_a,
                                "entry_b": c.entry_b,
                            },
                            default=str,
                        )
                        + "\n"
                    )
        return conflicts
