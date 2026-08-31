"""Producer-aware kernel recipe SDK records under a canonical ``kernel:`` id.

This is KernelForge's rewrite knowledge and nothing else's. A rewrite resolves
its own identity, writes its own candidate and owns its own champion pointer; it never
reaches into the inference document that a parent assembles. A run under
Hyperloom and a run from the command line therefore record the same way.

The agent hands over the complete picture of one port plus the files that
belong to it, and the SDK owns the envelope around it -- the identity, the
candidate id and the champion policy -- so a caller never re-derives any of
them. Normal reads restore JSON and every file for the selected Top-N into
separate session directories::

    identity = KernelRecipeIdentity(
        producer="flydsl",
        kernel_name="softmax",
        framework="vllm",
        framework_version="0.10.0",
        backend="flydsl",
        gpu="mi355x",
    )
    kb = KernelRecipeKB.open_identity(identity, config)
    prior = kb.read_best(workspace / "prior-recipes")
    outcome = kb.write_candidate({"metric": {...}}, [kernel_path], 1.4)

The canonical id includes ``producer``. KB Store support for that canonical
dimension is still a live deployment blocker; this client does not emulate it
with a producer-neutral fallback.

``speedup`` is a parameter rather than part of the payload because it decides
the champion pointer. A port that does not beat its source baseline is still
recorded: it is what saves the next run from repeating PORT. Only the pointer
is gated.

Wiring is unconditional. A run with no configured store leaves the SDK
inactive and turns every call into a no-op, so a caller never has to branch on
whether the KB is on. Nothing here raises into the agent: knowledge is
advisory, and a failure to record must not fail a rewrite.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kernelforge.config import Config
from kernelforge.knowledge.experience_reader import sanitize_read_error
from kernelforge.knowledge.experience_store import knowledge_config_from_runtime
from kernelforge.knowledge.kernel_identity import (
    KernelRecipeIdentity,
    kernel_recipe_canonical_id,
)
from kernelforge.rewrite_by_flydsl.identity import session_id as candidate_session_id
from kernelforge.rewrite_by_flydsl.record_store import (
    RewriteRecordStore,
    create_rewrite_record_store,
    safe_rel_path,
)

_DIGEST_LEN = 32


@dataclass(frozen=True)
class CandidateMetadata:
    """Ranking metadata for one previously recorded port.

    ``speedup`` is the recorded claim; ``measured_speedup`` is set only once a
    consumer applied this port and measured it, and it is what ranking trusts.
    """

    session_id: str
    value: dict[str, Any]
    speedup: float | None
    is_champion: bool
    measured_speedup: float | None


@dataclass(frozen=True)
class CandidateBundle(CandidateMetadata):
    """One selected port fully materialized in its own local directory."""

    bundle_dir: Path
    recipe_path: Path
    files_dir: Path


def kb_store_secrets(config: Config) -> tuple[str, ...]:
    """The credentials a store failure's text must never be allowed to keep.

    Public because the callers that wrap this facade report their own store
    errors, and a reason is only as redacted as the secret list it was given, so
    every one of them redacts against the same configured credential.
    """
    knowledge = knowledge_config_from_runtime(config)
    return tuple(value for value in (knowledge.kb_store_token,) if value)


def _named_files(files: Any) -> dict[str, Path]:
    """Accept either ``{rel_path: source}`` or a plain list of file paths."""
    if isinstance(files, Mapping):
        return {safe_rel_path(str(rel)): Path(source) for rel, source in files.items()}
    if isinstance(files, (str, Path)):
        files = [files]
    if not isinstance(files, Iterable):
        return {}
    return {Path(str(source)).name: Path(str(source)) for source in files}


def _port_digest(knowledge: Mapping[str, Any], files: Mapping[str, Path]) -> str:
    """Fingerprint one port so re-recording it updates a candidate, not adds one."""
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(knowledge), ensure_ascii=False, sort_keys=True).encode())
    for rel_path in sorted(files):
        digest.update(rel_path.encode())
        try:
            digest.update(files[rel_path].read_bytes())
        except OSError:
            digest.update(b"<unreadable>")
    return digest.hexdigest()[:_DIGEST_LEN]


class KernelRecipeKB:
    """Read and write one producer's candidates for one kernel recipe identity.

    The producer owns candidate ranking and its champion pointer. ``backend`` is
    separate: it describes the final implementation type produced by that
    system, not the system that authored the recipe.
    """

    def __init__(
        self,
        store: RewriteRecordStore | None,
        identity: KernelRecipeIdentity | None = None,
        canonical_id: str = "",
        *,
        config: Config | None = None,
        reason: str = "",
    ) -> None:
        self._store = store
        self._identity = identity
        self.canonical_id = canonical_id
        self._config = config
        self.reason = reason

    @classmethod
    def open_identity(
        cls,
        identity: KernelRecipeIdentity,
        config: Config,
    ) -> "KernelRecipeKB":
        """Open the Rewrite SDK directly for any validated backend identity."""
        store = create_rewrite_record_store(config)
        if store is None:
            return cls(None, reason="not_configured")
        try:
            canonical_id = kernel_recipe_canonical_id(identity)
        except Exception as error:  # noqa: BLE001 - an invalid identity cold-starts
            return cls(
                None,
                reason=sanitize_read_error(error, secrets=kb_store_secrets(config)),
            )
        return cls(store, identity, canonical_id, config=config)

    @classmethod
    def open_canonical_id(
        cls,
        canonical_id: str,
        config: Config,
    ) -> "KernelRecipeKB":
        """Open the SDK on an address a prior read already resolved.

        Amending a record needs the address the candidate came from and nothing
        else, so the identity dimensions are not re-derived here; that keeps a
        write-back from being filed anywhere but the record it measured.
        """
        if not str(canonical_id or "").strip():
            return cls(None, reason="missing_canonical_id")
        store = create_rewrite_record_store(config)
        if store is None:
            return cls(None, reason="not_configured")
        return cls(store, None, canonical_id, config=config)

    @property
    def active(self) -> bool:
        """True when an identity resolved and the calls below do something."""
        return self._store is not None and bool(self.canonical_id)

    # -- read ----------------------------------------------------------------

    def read_best(self, destination: str | Path) -> CandidateBundle | None:
        """Materialize and return this producer's Top1, or ``None`` when cold."""
        ranked = self.read_top_n(destination, limit=1)
        return ranked[0] if ranked else None

    def list_candidates(self, limit: int = 3) -> list[CandidateMetadata]:
        """Return ranking metadata without downloading candidate artifacts."""
        if not self.active or limit <= 0:
            return []
        try:
            found = self._store.candidates(self.canonical_id, limit=limit)
        except Exception as error:  # noqa: BLE001 - a KB read must cold-start
            self.reason = sanitize_read_error(error, secrets=self._config_secrets())
            return []
        summaries: list[CandidateMetadata] = []
        for candidate in found:
            value = candidate.knowledge.get("value")
            summaries.append(
                CandidateMetadata(
                    session_id=candidate.session_id,
                    value=dict(value) if isinstance(value, Mapping) else {},
                    speedup=candidate.speedup,
                    is_champion=candidate.is_champion,
                    measured_speedup=candidate.measured_speedup,
                )
            )
        return summaries

    def read_top_n(
        self,
        destination: str | Path,
        limit: int = 3,
    ) -> list[CandidateBundle]:
        """Materialize this producer's recorded recipes, best evidence first.

        Candidates a consumer already measured come first, ranked by that
        measurement; the rest follow ranked by the speedup they claim.
        A candidate that lost to its source baseline is included: the caller
        decides whether to replay it or read it as reference material. Each
        selected candidate gets ``<destination>/<session-id>/recipe.json`` and
        its own ``files/`` tree; candidates outside Top-N are never downloaded.
        """
        if not self.active or limit <= 0:
            return []
        try:
            found = self._store.candidates(self.canonical_id, limit=limit)
            bundles: list[CandidateBundle] = []
            for candidate in found:
                bundle_dir = self._store.materialize(
                    self.canonical_id,
                    candidate,
                    destination,
                )
                value = candidate.knowledge.get("value")
                bundles.append(
                    CandidateBundle(
                        session_id=candidate.session_id,
                        value=dict(value) if isinstance(value, Mapping) else {},
                        speedup=candidate.speedup,
                        is_champion=candidate.is_champion,
                        measured_speedup=candidate.measured_speedup,
                        bundle_dir=bundle_dir,
                        recipe_path=bundle_dir / "recipe.json",
                        files_dir=bundle_dir / "files",
                    )
                )
            return bundles
        except Exception as error:  # noqa: BLE001 - a KB read must cold-start
            self.reason = sanitize_read_error(error, secrets=self._config_secrets())
            return []

    def prior_file(self, session_id: str, rel_path: str) -> bytes:
        """Fetch one artifact's byte-exact contents on demand.

        The result is artifact bytes without decoding or newline conversion.
        Normal Top-N consumers should use the materialized :class:`Path`
        objects returned by :meth:`read_top_n`.
        """
        if not self.active:
            return b""
        try:
            return self._store.read_bytes(self.canonical_id, session_id, rel_path)
        except Exception:  # noqa: BLE001 - an unreadable artifact is just a miss
            return b""

    # -- write ---------------------------------------------------------------

    def write_candidate(
        self,
        knowledge: Mapping[str, Any],
        files: Any = (),
        speedup: float | None = None,
    ) -> dict[str, Any]:
        """Record the complete picture of one port, plus the files it needs.

        ``files`` is either a list of paths, whose basenames become the
        artifact names, or a ``{rel_path: source}`` mapping when the names
        matter. The champion pointer moves only when this port both improves on
        its source baseline and beats the identity's incumbent.

        Never raises: a refusal is returned, and the caller persists that reason
        in the run's result JSON, so a store exception is redacted and bounded
        before it is handed back. The exception type leads the message, so the
        cap can only cut the tail of a long error body.
        """
        if not self.active:
            return {"written": False, "reason": self.reason or "not_configured"}
        if not isinstance(knowledge, Mapping):
            return {"written": False, "reason": "knowledge_not_a_mapping"}
        try:
            named = _named_files(files)
            document = {
                "producer": self._identity.producer,
                "speedup": round(speedup, 4) if speedup is not None else None,
                "identity": asdict(self._identity),
                "value": dict(knowledge),
            }
            session_id = candidate_session_id(
                self.canonical_id,
                self._identity.kernel_name,
                _port_digest(knowledge, named),
            )
            with tempfile.TemporaryDirectory(prefix="rewrite-agent-kb-") as temporary:
                staged = self._stage(named, Path(temporary))
                self._store.write(self.canonical_id, session_id, document, staged)
            promoted = self._maybe_promote(session_id, speedup)
        except Exception as error:  # noqa: BLE001 - a KB write never breaks a rewrite
            return {
                "written": False,
                "reason": sanitize_read_error(error, secrets=self._config_secrets()),
            }
        return {
            "written": True,
            "canonical_id": self.canonical_id,
            "session_id": session_id,
            "solution": f"{self.canonical_id}/{session_id}",
            "speedup": speedup,
            "champion": promoted,
            "files": sorted(named),
        }

    def record_measured_speedup(
        self,
        session_id: str,
        measured_speedup: float,
    ) -> dict[str, Any]:
        """Amend one recorded candidate with the speedup this run measured.

        A recorded measurement is what lets the next run rank this candidate on
        evidence instead of on the number it claims. Never raises: the caller
        reports the returned reason rather than losing the run over it, and that
        reason is persisted into the run's result JSON, so a store exception is
        redacted and bounded before it is handed back.
        """
        if not self.active:
            return {"recorded": False, "reason": self.reason or "not_configured"}
        try:
            self._store.record_measured_speedup(
                self.canonical_id,
                session_id,
                measured_speedup,
            )
        except Exception as error:  # noqa: BLE001 - reported, never raised at a run
            return {
                "recorded": False,
                "reason": sanitize_read_error(error, secrets=self._config_secrets()),
            }
        return {
            "recorded": True,
            "canonical_id": self.canonical_id,
            "session_id": session_id,
            "measured_speedup": measured_speedup,
        }

    def _stage(self, named: Mapping[str, Path], root: Path) -> dict[str, Path]:
        """Copy validated artifacts aside for a byte-consistent upload."""
        staged: dict[str, Path] = {}
        for index, rel_path in enumerate(sorted(named)):
            safe = safe_rel_path(rel_path)
            target = root / str(index) / Path(*safe.split("/"))
            try:
                target.resolve(strict=False).relative_to(root.resolve())
            except ValueError as error:
                raise ValueError(f"staged artifact escapes temporary root: {safe!r}") from error
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.resolve(strict=False).relative_to(root.resolve())
            except ValueError as error:
                raise ValueError(f"staged artifact escapes temporary root: {safe!r}") from error
            target.write_bytes(named[rel_path].read_bytes())
            staged[safe] = target
        return staged

    def _maybe_promote(self, session_id: str, speedup: float | None) -> bool:
        if speedup is None or speedup <= 1.0:
            return False
        champion = self._store.champion_speedup(self.canonical_id)
        if champion is not None and speedup <= champion:
            return False
        self._store.promote(self.canonical_id, session_id, speedup)
        return True

    def _config_secrets(self) -> tuple[str, ...]:
        return kb_store_secrets(self._config) if self._config is not None else ()


__all__ = [
    "CandidateBundle",
    "CandidateMetadata",
    "KernelRecipeKB",
    "kb_store_secrets",
]
