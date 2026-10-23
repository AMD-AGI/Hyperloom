# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tree-strategy search control for the iteration loop: where each iteration starts, and the tree it grows."""

from __future__ import annotations

import time

from kernelforge.llm.git import git
from kernelforge.loop.run_state import BestRecord, RunState, make_event
from kernelforge.loop.search_policy import (
    OBJECTIVE_IMMEDIATE_CANONICAL_GAIN,
    SEARCH_MODE_EXPLOIT,
    SearchPolicyDecision,
)
from kernelforge.loop.search_tree import (
    KEEP_DECISION,
    NODE_SOURCE_ITERATION,
    ROOT_SOURCE_BASELINE,
    SearchStrategy,
    SearchTree,
    SearchTreeStore,
    TreeAttempt,
    TreeBranch,
    TreeNode,
    TreeSeat,
    render_tree_view,
)
from kernelforge.orchestrator.start_selector import StartSelection
from kernelforge.tools.durable_io import atomic_write_text

_TREE_POLICY = SearchPolicyDecision(
    mode=SEARCH_MODE_EXPLOIT,
    reason_codes=("TREE_START_SELECTED",),
    objective_kind=OBJECTIVE_IMMEDIATE_CANONICAL_GAIN,
)


def _best_record_of(node: TreeNode) -> BestRecord:
    """The run-state incumbent a node stands for; the baseline root is the state a fresh campaign begins in."""
    if node.source == ROOT_SOURCE_BASELINE:
        return BestRecord()
    return BestRecord(
        iteration=node.iteration,
        wall_ms=node.wall_ms,
        mean_case_speedup=node.mean_case_speedup,
        commit_hash=node.commit,
        plan=node.plan[:120],
        source=node.source,
    )


def _seat_record(state: RunState, node: TreeNode) -> None:
    """Make ``node`` the incumbent ``state`` records."""
    state.best = _best_record_of(node)
    state.head_commit = node.commit
    state.best_case_times = dict(node.case_times)


class TreeSearchRuntimeMixin:
    """Own the search tree: the seat each iteration starts on, the attempts it records, and the global best."""

    #: The campaign's tree once it is loaded or rooted; always ``None`` for a linear campaign.
    _search_tree: SearchTree | None = None

    @property
    def _search_tree_store(self) -> SearchTreeStore | None:
        if self.ic.search_strategy is not SearchStrategy.TREE:
            return None
        return SearchTreeStore(self.ic.workspace_dir)

    def _loaded_search_tree(self) -> SearchTree | None:
        store = self._search_tree_store
        if self._search_tree is None and store is not None:
            self._search_tree = store.load()
        return self._search_tree

    def _save_search_tree(self) -> None:
        self._search_tree_store.save(self._search_tree)

    def _active_branch(self) -> str:
        """The branch the workspace is expected on: the tree's current branch, else the campaign's own."""
        tree = self._loaded_search_tree()
        return tree.current_branch if tree is not None else self.ic.git_branch

    def _history_scope(self) -> frozenset[int] | None:
        """The iterations whose history the seated iteration reads; ``None`` reads every iteration."""
        tree = self._search_tree
        if tree is None or tree.seat is None:
            return None
        return tree.history_scope(tree.seat.start_commit)

    def _tree_search_policy(self) -> SearchPolicyDecision | None:
        """The fixed policy of a tree campaign, whose start selector owns every change of direction."""
        return _TREE_POLICY if self._search_tree_store is not None else None

    def _is_new_global_best(self, result) -> bool:
        """Whether a KEEP is the campaign's best, before or after the tree records it."""
        tree = self._search_tree
        if tree is None:
            return True
        best = tree.global_best()
        if result.commit_hash == best.commit:
            return True
        return tree.node(result.commit_hash) is None and (result.mean_case_speedup or 0.0) > best.mean_case_speedup

    def _published_best_record(self) -> BestRecord:
        """The incumbent the best artifact describes: the global best of a tree, else the run-state best."""
        tree = self._search_tree
        return self.run_state.best if tree is None else _best_record_of(tree.global_best())

    def _begin_search_tree(self) -> None:
        """Root a fresh campaign's tree at the incumbent, or fold a resumed campaign's unfinished seat into it."""
        if self._search_tree_store is None:
            return
        if self._loaded_search_tree() is None:
            root = TreeNode(
                commit=self._workspace_head(),
                parent="",
                branch=self.ic.git_branch,
                depth=0,
                iteration=0,
                mean_case_speedup=self.best_mean_case_speedup or 1.0,
                wall_ms=self.best_wall_ms,
                case_times=dict(self._best_case_times),
                plan=self.run_state.best.plan,
                source=self.run_state.best.source or ROOT_SOURCE_BASELINE,
            )
            self._search_tree = SearchTree.create(root)
            self._save_search_tree()
            return
        self._settle_resumed_seat()
        self._reconcile_best_publication()

    def _settle_resumed_seat(self) -> None:
        """Close the seat an earlier session left open from its recorded outcome, or drop it if it has none."""
        tree = self._search_tree
        seat = tree.seat
        if seat is None:
            return
        event = self._iteration_result_events().get(seat.iteration)
        if event is None:
            tree.seat = None
        else:
            decision = str(event.get("decision") or "")
            node = self._kept_node(seat, str(event.get("commit_hash") or "")) if decision == KEEP_DECISION else None
            tree.close_seat(
                self._attempt_of(seat, decision, str(event.get("plan") or ""), event.get("mean_case_speedup"), node),
                node,
            )
        self._save_search_tree()

    def _reconcile_tree_seat(self, state: RunState) -> None:
        """Put the workspace and ``state`` back on the tree's current branch after a session that died mid-seat.

        A pending KEEP is left to its own recovery, which is measured against the seat it was made on.
        """
        tree = self._loaded_search_tree()
        if tree is None or self._pending_keep_path.exists():
            return
        branch = tree.branch(tree.current_branch)
        self._checkout_tree_branch(branch)
        node = tree.node(self._workspace_head())
        if node is not None:
            _seat_record(state, node)

    async def _seat_tree_start(self, iteration: int, selector) -> None:
        """Choose this iteration's start, put the workspace on it, and make it the incumbent a candidate must beat."""
        tree = self._search_tree
        if tree is None:
            return
        started_at = time.time()
        view = render_tree_view(tree, archive_dir=str(self.archive.root), lessons_dir=str(self.lessons.root))
        selection = await self._select_start(view, tree, selector)
        self._round_planning_sec += max(0.0, time.time() - started_at)
        start = selection.commit or self._fallback_start(tree)
        branch = tree.branch_headed_by(start) or tree.allocate_branch(start, iteration)
        tree.open_seat(TreeSeat(iteration, start, branch, selection.reason, selection.failure))
        self._save_search_tree()
        self._record_selection(tree.seat, view, selection.answer)
        self._checkout_tree_branch(tree.branch(branch))
        self._seat_incumbent(tree.node(start))
        self.state_store.append_event(
            make_event(
                "tree_start_seated",
                iteration,
                start_commit=start,
                branch=branch,
                selector_fallback=selection.failure or None,
            )
        )
        node = tree.node(start)
        print(f"  [tree] iteration {iteration} starts from {start[:12]} on {branch} ({node.mean_case_speedup:.6f}x)")
        if selection.failure:
            print(f"  [tree] start selector failed ({selection.failure}); continuing the last branch")

    async def _select_start(self, view: str, tree: SearchTree, selector) -> StartSelection:
        if selector is None:
            return StartSelection(failure="no start selector is configured")
        try:
            return await selector.select(
                workspace=self.ic.workspace_dir,
                tree_view=view,
                resolve=tree.resolve,
                usage=self._usage,
            )
        finally:
            self._checkpoint_llm_usage()

    def _record_selection(self, seat: TreeSeat, view: str, answer: str) -> None:
        """Keep exactly what the selector was shown and what it answered, beside the seat that answer produced."""
        record = "\n".join(
            [
                f"# Start selection, iteration {seat.iteration}",
                "",
                f"- Start: {seat.start_commit}",
                f"- Branch: {seat.branch}",
                f"- Fallback: {seat.selector_fallback or 'none'}",
                "",
                "## Tree view given to the selector",
                "",
                view,
                "",
                "## Selector answer",
                "",
                answer or "(no answer)",
                "",
            ]
        )
        directory = self._search_tree_store.path.parent / "tree_selection"
        atomic_write_text(directory / f"iter_{seat.iteration:03d}.md", record)

    @staticmethod
    def _fallback_start(tree: SearchTree) -> str:
        """The current HEAD of the branch the last attempt ran on, or of the root branch before any attempt."""
        name = tree.attempts[-1].branch if tree.attempts else tree.root_branch
        return tree.branch(name).head

    def _workspace_head(self) -> str:
        return self._git("rev-parse", "HEAD").splitlines()[0]

    def _checkout_tree_branch(self, branch: TreeBranch) -> None:
        """Put the workspace on ``branch``, creating it at its start commit when git does not have it yet."""
        workspace = self.ic.workspace_dir
        current = self._git("branch", "--show-current").splitlines()
        if current and current[0] == branch.name:
            return
        known = git("rev-parse", "--verify", "--quiet", f"refs/heads/{branch.name}", cwd=workspace, check=False)
        if known.returncode == 0:
            git("checkout", branch.name, cwd=workspace)
        else:
            git("checkout", "-b", branch.name, branch.start_commit, cwd=workspace)

    def _seat_incumbent(self, node: TreeNode) -> None:
        """Make ``node`` the incumbent every measurement, plan and KEEP bar of the loop is taken against."""
        head = self._workspace_head()
        if head != node.commit:
            raise RuntimeError(f"workspace HEAD {head} is not the tree node {node.commit} being seated")
        moved = self._canonical_commit() != node.commit
        _seat_record(self.run_state, node)
        self.best_wall_ms = node.wall_ms
        self.best_mean_case_speedup = node.mean_case_speedup
        self._best_case_times = dict(node.case_times)
        if moved:
            self._adopt_nearest_evidence(node.commit, self._tree_node_score)
        self._persist_scoring_state()

    def _tree_node_score(self, commit: str) -> float | None:
        node = self._search_tree.node(commit)
        return node.mean_case_speedup if node is not None else None

    def _rest_on_global_best(self) -> None:
        """Between iterations, keep the workspace and run state on the global best, so every reader sees it."""
        tree = self._search_tree
        if tree is None:
            return
        tree.seat = None
        best = tree.global_best()
        branch = tree.branch(tree.branch_headed_by(best.commit))
        self._checkout_tree_branch(branch)
        tree.current_branch = branch.name
        self._save_search_tree()
        self._seat_incumbent(best)

    def _record_tree_attempt(self, result, *, decision: str, plan: str) -> None:
        """Record the seated iteration's outcome, adding its node when it was kept."""
        tree = self._search_tree
        if tree is None or tree.seat is None:
            return
        seat = tree.seat
        node = self._kept_node(seat, result.commit_hash) if result.kept else None
        tree.close_seat(self._attempt_of(seat, decision, plan, result.mean_case_speedup, node), node)
        self._save_search_tree()

    def _kept_node(self, seat: TreeSeat, commit: str) -> TreeNode:
        """The node a KEEP on ``seat`` added, read off the incumbent the KEEP was promoted to."""
        best = self.run_state.best
        if not commit or best.commit_hash != commit:
            raise ValueError(f"KEEP {commit or '(no commit)'} of iteration {seat.iteration} is not the incumbent")
        parent = self._search_tree.node(seat.start_commit)
        return TreeNode(
            commit=commit,
            parent=parent.commit,
            branch=seat.branch,
            depth=parent.depth + 1,
            iteration=seat.iteration,
            mean_case_speedup=best.mean_case_speedup,
            wall_ms=best.wall_ms,
            case_times=dict(self._best_case_times),
            plan=best.plan,
            source=NODE_SOURCE_ITERATION,
        )

    @staticmethod
    def _attempt_of(seat: TreeSeat, decision: str, plan: str, mean_case_speedup, node: TreeNode | None) -> TreeAttempt:
        return TreeAttempt(
            iteration=seat.iteration,
            start_commit=seat.start_commit,
            branch=seat.branch,
            decision=decision,
            plan=" ".join(str(plan or "").split())[:240],
            mean_case_speedup=mean_case_speedup,
            child_commit=node.commit if node is not None else "",
            selector_reason=seat.selector_reason,
            selector_fallback=seat.selector_fallback,
        )


__all__ = ["TreeSearchRuntimeMixin"]
