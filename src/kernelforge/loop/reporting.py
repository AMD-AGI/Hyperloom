# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Incremental publication of the best canonically verified Forge result."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

from kernelforge.llm.git import git
from kernelforge.loop.scoring import aggregate_regression_detail
from kernelforge.durable_io import atomic_write_text, fsync_directory

# v2 adds `aggregate_regression` and derives `total_improved` from it, so a v1
# manifest is missing a field this publisher always writes. Publication identity
# is a whole-dict comparison, which reads that difference as a conflicting
# publication of the same iteration rather than as the upgrade it is.
MANIFEST_SCHEMA_VERSION = 2

# What every published best version must contain. Named once because two places
# ask the question and they have to agree: publication refuses to accept a
# version missing any of these, and reconciliation reads the same list to decide
# a bundle is already whole and needs no republishing. Answering differently
# would let reconciliation call a bundle complete that publication would reject.
BEST_BUNDLE_FILES = ("forge.patch", "validation.txt", "benchmark.json")


def _round_budget_lines(summary: object) -> list[str]:
    """Render what the campaign's rounds cost, for the report's reader.

    Planning is the largest single thing a round buys and was, until it was
    measured here, the one part of the budget nobody could see without reading
    a log. A campaign that ended because no round fit the time left says so
    here too: from the outside that is indistinguishable from a campaign that
    ran out of ideas, and the two call for opposite responses.

    Every duration here is campaign-cumulative -- it spans every session the
    campaign has run, not the one that wrote this report -- and is labelled so.
    The share is read from the summary rather than computed here, and this
    renderer deliberately has no clock to compute one from: the numerator and
    the denominator have to describe the same span, and only the writer of the
    summary knows they do. A summary carrying no share is rendered without one.
    Dividing cumulative planning by whatever span was nearest to hand is how a
    resumed 10-minute session against 45 minutes of cumulative planning came to
    publish "450% of the run".
    """
    if not isinstance(summary, dict) or not summary:
        return []
    lines = ["", "## Round Budget", ""]
    rounds = int(summary.get("rounds", 0) or 0)
    planning_sec = float(summary.get("planning_total_sec", 0.0) or 0.0)
    total_sec = float(summary.get("total_sec", 0.0) or 0.0)
    campaign_sec = float(summary.get("campaign_sec", 0.0) or 0.0)
    lines.append(f"- Rounds planned (campaign total): {rounds}")
    lines.append(f"- Planning wall-clock (campaign total): {planning_sec / 60:.1f} min")
    lines.append(f"- Round wall-clock (campaign total): {total_sec / 60:.1f} min")
    if campaign_sec > 0:
        # Printed beside the share so a reader can check the division that
        # produced it against the two numbers it was made from.
        lines.append(f"- Campaign wall-clock: {campaign_sec / 60:.1f} min")
    share = summary.get("planning_share_pct")
    if isinstance(share, (int, float)) and not isinstance(share, bool):
        lines.append(f"- Planning share of campaign wall-clock: {float(share):.0f}%")
    refusal = summary.get("refused")
    if refusal:
        lines.append(f"- Stopped: no round fit the remaining budget ({refusal})")
    return lines


class BestResultPublisher:
    """Publish immutable best versions behind one atomic manifest."""

    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir).resolve()
        self.root = self.workspace / "forge_experiments"
        self.best_root = self.root / "best"
        self.manifest_path = self.best_root / "manifest.json"
        self.result_path = self.root / "best_result.json"
        self.report_path = self.root / "optimization_report.md"
        self.history_path = self.root / "optimization_history.md"

    @staticmethod
    def _write_text_durable(path: Path, text: str) -> None:
        """Write ``text`` and fsync the file so it survives a crash."""
        with open(path, "w") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())

    @classmethod
    def _fsync_tree(cls, root: Path) -> None:
        """fsync every file and directory under ``root`` (bottom of the bundle
        must be durable before the top-level rename makes it visible)."""
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                file_path = Path(dirpath) / name
                fd = os.open(str(file_path), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            fsync_directory(Path(dirpath))

    def _copy_changed_files(
        self,
        destination: Path,
        source_files: dict[str, bytes],
    ) -> list[str]:
        copied: list[str] = []
        for raw_path, payload in source_files.items():
            relative = Path(raw_path)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            copied.append(str(relative))
        return copied

    def _source_files(
        self,
        changed_files: list[str],
        *,
        commit_hash: str,
    ) -> dict[str, bytes]:
        """Read publishable files from the immutable KEEP commit."""
        sources: dict[str, bytes] = {}
        for raw_path in changed_files:
            relative = Path(raw_path)
            if relative.is_absolute():
                try:
                    relative = relative.resolve().relative_to(self.workspace)
                except ValueError:
                    continue
            if ".." in relative.parts:
                continue
            result = git(
                "show",
                f"{commit_hash}:{relative.as_posix()}",
                cwd=self.workspace,
                check=False,
                text=False,
            )
            if result.returncode == 0:
                sources[str(relative)] = result.stdout
        return sources

    # Manifest keys that describe when a publication was made rather than what
    # it is. Both move on their own while the KEEP behind them does not, so
    # comparing them would report a conflict every time a campaign republishes
    # the same best result.
    _VOLATILE_MANIFEST_KEYS = frozenset({"published_at", "round_budget"})

    @classmethod
    def _same_publication(cls, left: dict, right: dict) -> bool:
        """Compare publication identity, ignoring what is not part of it."""
        return {key: value for key, value in left.items() if key not in cls._VOLATILE_MANIFEST_KEYS} == {
            key: value for key, value in right.items() if key not in cls._VOLATILE_MANIFEST_KEYS
        }

    @staticmethod
    def _load_json(path: Path, *, label: str) -> dict:
        try:
            value = json.loads(path.read_text())
        except Exception as error:
            raise ValueError(f"invalid {label}: {path}") from error
        if not isinstance(value, dict):
            raise ValueError(f"invalid {label}: {path}")
        return value

    def _validate_existing_bundle(
        self,
        version_dir: Path,
        *,
        expected: dict,
        validation_text: str,
        benchmark: dict,
        patch: str,
        source_files: dict[str, bytes],
    ) -> dict:
        """Accept only a complete immutable bundle for the same publication."""
        missing = [name for name in BEST_BUNDLE_FILES if not (version_dir / name).is_file()]
        if missing:
            raise ValueError(f"incomplete best artifact version {version_dir}: missing {', '.join(missing)}")

        publication_path = version_dir / "publication.json"
        if publication_path.is_file():
            publication = self._load_json(
                publication_path,
                label="best artifact publication",
            )
        elif self.manifest_path.is_file():
            current_manifest = self._load_json(
                self.manifest_path,
                label="best manifest",
            )
            publication = (
                current_manifest
                if current_manifest.get("artifact_dir") == expected["artifact_dir"]
                else {
                    **expected,
                    "published_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        else:
            publication = {
                **expected,
                "published_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

        expected_with_time = {
            **expected,
            "published_at": publication.get("published_at"),
        }
        if not self._same_publication(publication, expected_with_time):
            raise ValueError(f"inconsistent best artifact version: {version_dir}")
        if (version_dir / "forge.patch").read_text() != patch:
            raise ValueError(f"inconsistent best artifact patch: {version_dir}")
        if (version_dir / "validation.txt").read_text() != validation_text:
            raise ValueError(f"inconsistent best artifact validation: {version_dir}")
        stored_benchmark = self._load_json(
            version_dir / "benchmark.json",
            label="best artifact benchmark",
        )
        if stored_benchmark != benchmark:
            raise ValueError(f"inconsistent best artifact benchmark: {version_dir}")

        files_root = version_dir / "files"
        stored_files = (
            {str(path.relative_to(files_root)) for path in files_root.rglob("*") if path.is_file()}
            if files_root.is_dir()
            else set()
        )
        if stored_files != set(source_files):
            raise ValueError(f"inconsistent best artifact files: {version_dir}")
        for relative, source in source_files.items():
            if (files_root / relative).read_bytes() != source:
                raise ValueError(f"inconsistent best artifact file: {relative}")
        return publication

    @staticmethod
    def _render_report(manifest: dict) -> str:
        """Render the human-facing view of one published manifest.

        The manifest may withhold the improvement badge over a contradiction
        between the score and the aggregate wall times, and this report is the
        artifact an operator opens, so the verdict and its reason are stated
        here rather than left to whoever reads the JSON.
        """
        changed = manifest.get("changed_files") or []
        aggregate_regression = str(manifest["aggregate_regression"])
        lines = [
            "# Forge Optimization Report",
            "",
            f"- Campaign: `{manifest['campaign_id']}`",
            "- Status: best verified result",
            f"- mean case speedup: {manifest['mean_case_speedup']:.6f}x",
            f"- Improved overall: {'yes' if manifest['total_improved'] else 'no'}",
            *([f"- Aggregate regression: {aggregate_regression}"] if aggregate_regression else []),
            f"- Baseline raw mean: {manifest['baseline_wall_ms']:.4f} ms",
            f"- Search-start raw mean: {manifest['search_start_ms']:.4f} ms",
            (
                "- Selected candidate raw mean (diagnostic; not monotonic, but "
                "it withdraws the improvement above when it contradicts the "
                f"score): {manifest['best_wall_ms']:.4f} ms"
            ),
            "- Correctness: PASS",
            f"- Best iteration: {manifest['iteration']}",
            f"- Commit: `{manifest['commit_hash']}`",
            f"- Optimization: {manifest['plan'] or 'unspecified'}",
            "",
            "## Changed Files",
            "",
        ]
        lines.extend(f"- `{path}`" for path in changed)
        lines.extend(_round_budget_lines(manifest.get("round_budget")))
        lines.extend(
            [
                "",
                "## Artifacts",
                "",
                f"- Patch: `{manifest['patch_path']}`",
                f"- Validation: `{manifest['validation_path']}`",
                f"- Benchmark: `{manifest['benchmark_path']}`",
                f"- Bundle: `{manifest['artifact_dir']}`",
                "",
            ]
        )
        return "\n".join(lines)

    def publish(
        self,
        *,
        campaign_id: str,
        session_index: int,
        experiment_id: str,
        iteration: int,
        commit_hash: str,
        plan: str,
        baseline_wall_ms: float,
        search_start_ms: float | None = None,
        best_wall_ms: float,
        mean_case_speedup: float,
        search_start_mean_case_speedup: float,
        snr_db: float | None,
        validation_text: str,
        benchmark: dict,
        changed_files: list[str],
        patch: str,
        round_budget: dict | None = None,
    ) -> dict:
        """Publish one KEEP and atomically point the campaign at it.

        ``round_budget`` is what the campaign's rounds have cost so far. It
        describes the run rather than this result, so it is written into the
        manifest but kept out of publication identity.
        """
        self.best_root.mkdir(parents=True, exist_ok=True)
        version_name = f"iter_{iteration:03d}"
        sources = self._source_files(
            changed_files,
            commit_hash=commit_hash,
        )
        search_start = search_start_ms if search_start_ms is not None else baseline_wall_ms
        resolved_mean_case_speedup = float(mean_case_speedup)
        resolved_search_start_speedup = float(search_start_mean_case_speedup)
        if (
            not math.isfinite(resolved_mean_case_speedup)
            or resolved_mean_case_speedup <= 0.0
            or not math.isfinite(resolved_search_start_speedup)
            or resolved_search_start_speedup <= 0.0
        ):
            raise ValueError("mean case speedups must be finite and positive")
        # The manifest is the artifact downstream reporting reads, so it has to
        # carry the same contradiction the CLI result already names: a KEEP
        # decided on the mean of per-case speedups can still be slower in
        # aggregate wall time, and that must not ship as an improvement.
        aggregate_regression = aggregate_regression_detail(
            baseline_ms=baseline_wall_ms,
            best_ms=best_wall_ms,
            mean_case_speedup=resolved_mean_case_speedup,
        )
        expected = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "session_index": session_index,
            "experiment_id": experiment_id,
            "iteration": iteration,
            "commit_hash": commit_hash,
            "plan": (plan or "").strip(),
            "baseline_wall_ms": baseline_wall_ms,
            "pristine_baseline_ms": baseline_wall_ms,
            "search_start_ms": search_start,
            "best_wall_ms": best_wall_ms,
            "mean_case_speedup": resolved_mean_case_speedup,
            "search_start_mean_case_speedup": (resolved_search_start_speedup),
            "speedup": round(resolved_mean_case_speedup, 6),
            "total_speedup": round(resolved_mean_case_speedup, 6),
            "incremental_speedup": round(
                resolved_mean_case_speedup / resolved_search_start_speedup,
                6,
            ),
            "aggregate_regression": aggregate_regression,
            "total_improved": (resolved_mean_case_speedup > 1.0 and not aggregate_regression),
            "incremental_improved": (resolved_mean_case_speedup > resolved_search_start_speedup),
            "improved_during_search": (resolved_mean_case_speedup > resolved_search_start_speedup),
            "correctness_passed": True,
            "snr_db": snr_db,
            "changed_files": list(sources),
        }

        manifest: dict | None = None
        candidates = [
            self.best_root / version_name,
            *sorted(self.best_root.glob(f"{version_name}.generation-*")),
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            relative_dir = candidate.relative_to(self.root)
            candidate_expected = {
                **expected,
                "artifact_dir": str(relative_dir),
                "patch_path": str(relative_dir / "forge.patch"),
                "validation_path": str(relative_dir / "validation.txt"),
                "benchmark_path": str(relative_dir / "benchmark.json"),
            }
            try:
                manifest = self._validate_existing_bundle(
                    candidate,
                    expected=candidate_expected,
                    validation_text=validation_text,
                    benchmark=benchmark,
                    patch=patch,
                    source_files=sources,
                )
                break
            except ValueError:
                continue

        if manifest is None:
            base_dir = self.best_root / version_name
            if not base_dir.exists():
                version_dir = base_dir
            else:
                generation = 1
                while True:
                    candidate = self.best_root / (f"{version_name}.generation-{generation:03d}")
                    if not candidate.exists():
                        version_dir = candidate
                        break
                    generation += 1
            relative_dir = version_dir.relative_to(self.root)
            expected = {
                **expected,
                "artifact_dir": str(relative_dir),
                "patch_path": str(relative_dir / "forge.patch"),
                "validation_path": str(relative_dir / "validation.txt"),
                "benchmark_path": str(relative_dir / "benchmark.json"),
            }
            manifest = {
                **expected,
                "published_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            temporary = Path(
                tempfile.mkdtemp(
                    dir=str(self.best_root),
                    prefix=f".{version_name}.",
                )
            )
            try:
                self._write_text_durable(temporary / "forge.patch", patch)
                self._write_text_durable(temporary / "validation.txt", validation_text)
                self._write_text_durable(
                    temporary / "benchmark.json",
                    json.dumps(benchmark, indent=2, sort_keys=True) + "\n",
                )
                copied = self._copy_changed_files(
                    temporary / "files",
                    sources,
                )
                if copied != list(sources):
                    raise ValueError("changed files became unavailable during publication")
                self._write_text_durable(
                    temporary / "publication.json",
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                )
                # Make the whole bundle durable before it becomes visible, then
                # fsync the parent so the rename itself survives a crash
                # (mirrors archive.record's fsync discipline).
                self._fsync_tree(temporary)
                os.replace(temporary, version_dir)
                fsync_directory(self.best_root)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)

        if self.manifest_path.is_file():
            current = self._load_json(self.manifest_path, label="best manifest")
            current_iteration = int(current.get("iteration", 0) or 0)
            if current_iteration > iteration:
                raise ValueError(f"best manifest is ahead of iteration {iteration}: {current_iteration}")
            # Only manifests written under the same schema are comparable: an
            # older one differs by construction, so comparing it would report a
            # conflict on every republish across an upgrade and leave the stale
            # manifest -- and the verdict it was written with -- published.
            if (
                current_iteration == iteration
                and int(current.get("schema_version", 0) or 0) == MANIFEST_SCHEMA_VERSION
                and not self._same_publication(current, manifest)
            ):
                raise ValueError(f"best manifest conflicts with iteration {iteration}")
        if round_budget:
            manifest = {**manifest, "round_budget": dict(round_budget)}
        payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"

        # The manifest is the atomic commit point. Derived human/machine views
        # are regenerated only after it points at a complete immutable bundle.
        atomic_write_text(self.manifest_path, payload)
        atomic_write_text(self.result_path, payload)
        atomic_write_text(self.report_path, self._render_report(manifest))
        return manifest

    def refresh_round_budget(self, round_budget: dict) -> bool:
        """Restate the campaign's round costs on an already-published best.

        The last KEEP of a campaign is usually published well before the run
        ends, so the totals it carried were the totals at that moment. This
        rewrites them once the campaign is over -- including the refusal that
        ended it, which nothing published earlier could have known about.

        Best-effort by contract: nothing here changes which result is
        published, so a workspace that cannot take the rewrite keeps the
        report it already had.
        """
        if not round_budget or not self.manifest_path.is_file():
            return False
        try:
            manifest = self._load_json(self.manifest_path, label="best manifest")
            manifest = {**manifest, "round_budget": dict(round_budget)}
            payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            atomic_write_text(self.manifest_path, payload)
            atomic_write_text(self.result_path, payload)
            atomic_write_text(self.report_path, self._render_report(manifest))
        except (ValueError, OSError, KeyError):
            return False
        return True

    def describes_current_best(self, *, iteration: int, commit_hash: str) -> bool:
        """Report whether the published manifest already is this best.

        Reconciliation rebuilds and republishes run_state.best to repair a
        manifest that a crash left behind. On a resumed session it recomputes
        fields the stored manifest does not carry identically -- session_index,
        experiment_id -- so republishing an already-current best tripped the
        same-iteration conflict guard and set persistence_degraded, while the
        KEEP, the git state and run_state.best were all intact. Two consecutive
        resumed sessions in the 12-hour run ended degraded for exactly that
        harmless divergence. Skipping the republish when the manifest already
        names the same (iteration, commit_hash) behind a complete bundle mirrors
        the fresh path's idempotency and keeps a clean resume clean.
        """
        if not self.manifest_path.is_file():
            return False
        try:
            manifest = self._load_json(self.manifest_path, label="best manifest")
        except ValueError:
            return False
        if int(manifest.get("schema_version", 0) or 0) != MANIFEST_SCHEMA_VERSION:
            return False
        # iteration 0 is a legitimate best (a warm-started baseline), so it must
        # not be read through an "or -1" default that a falsy 0 would trip.
        try:
            stored_iteration = int(manifest["iteration"])
        except (KeyError, TypeError, ValueError):
            return False
        if stored_iteration != iteration:
            return False
        if manifest.get("commit_hash") != commit_hash:
            return False
        artifact_dir = str(manifest.get("artifact_dir") or "")
        if not artifact_dir:
            return False
        bundle = self.root / artifact_dir
        return all((bundle / name).is_file() for name in BEST_BUNDLE_FILES)

    @staticmethod
    def _changed_files_from_metadata(metadata: dict) -> list[str]:
        explicit = metadata.get("changed_files") or []
        if explicit:
            return [str(path) for path in explicit]
        changed: list[str] = []
        for line in str(metadata.get("change_diff") or "").splitlines():
            if not line.startswith("diff --git a/"):
                continue
            parts = line.split()
            if len(parts) >= 4 and parts[2].startswith("a/"):
                changed.append(parts[2][2:])
        return changed

    def publish_history(
        self,
        *,
        events: list[dict],
        candidate_metadata: dict[int, dict],
    ) -> None:
        """Regenerate the complete human-readable iteration history."""
        iteration_events = sorted(
            (event for event in events if event.get("type") == "iteration_result"),
            key=lambda event: int(event.get("iter", 0) or 0),
        )
        lines = ["# Forge Optimization History", ""]
        for event in iteration_events:
            iteration = int(event.get("iter", 0) or 0)
            decision = str(event.get("decision") or "UNKNOWN")
            metadata = candidate_metadata.get(iteration, {})
            lines.extend(
                [
                    f"## Iteration {iteration} — {decision}",
                    "",
                    f"- Session: {event.get('session_index', 0)}",
                    f"- Experiment: `{event.get('experiment_id', '')}`",
                    f"- Session end: `{event.get('session_end_reason', '')}`",
                    f"- Turns: {event.get('turns', '')}",
                    f"- Plan: {event.get('plan', '') or 'unspecified'}",
                    f"- Canonical correctness: {'PASS' if metadata.get('validation_passed') else 'FAIL/NOT RUN'}",
                    f"- Candidate mean case speedup: {event.get('mean_case_speedup', '')}",
                    f"- Best mean case speedup before: {metadata.get('best_mean_case_speedup_before', '')}",
                    f"- Best mean case speedup after: {event.get('best_after_mean_case_speedup', '')}",
                    f"- Candidate raw mean ms: {event.get('wall_ms', '')}",
                    f"- Commit: `{metadata.get('commit_hash', '')}`",
                    f"- Archive: `{metadata.get('archive_path', '')}`",
                    "",
                    "Changed files:",
                ]
            )
            changed_files = self._changed_files_from_metadata(metadata)
            lines.extend([f"- `{path}`" for path in changed_files] or ["- (none)"])
            lines.append("")
        atomic_write_text(self.history_path, "\n".join(lines))
