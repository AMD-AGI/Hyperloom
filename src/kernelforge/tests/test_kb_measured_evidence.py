"""KB integrity: rank on measured evidence and write measurements back.

Reproduces the MI355X kernel-arena regression on ``vllm-kimi-k3-kda-attn``: a
record claiming 6.7608x ranked first, warm start adopted it because it was the
first candidate whose patch applied, and it measured 5.106x once applied. The
verified 5.8452x result sat at rank 2 and was never tried, and the inflated
claim was never corrected, so the same wrong candidate kept winning.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kernelforge.config import Config
from kernelforge.knowledge import experience_integration as integration
from kernelforge.knowledge import experience_sink as sink
from kernelforge.loop.scoring import passes_keep_threshold
from kernelforge.knowledge.experience_store import (
    REMOTE_BACKEND_KB_STORE,
    KnowledgeConfig,
    knowledge_config_from_runtime,
)
from kernelforge.rewrite_by_flydsl import record_store
from kernelforge.rewrite_by_flydsl.agent_kb import KernelRecipeKB
from kernelforge.rewrite_by_flydsl.record_store import (
    LocalRewriteRecords,
    RewriteCandidate,
    RewriteRecordError,
)

from kernelforge.tests.test_rewrite_by_flydsl_kb import InMemoryKBStore, _remote_config

PRODUCER_KERNEL_PATH = Path("packages/src/aiter_meta/ops/triton/deterministic_kernel.py")
CONSUMER_KERNEL_PATH = Path("src/aiter/ops/triton/deterministic_kernel.py")

PRISTINE_SOURCE = """\
import triton

BLOCK_SIZE = 32

@triton.jit
def deterministic_kernel(x):
    return x
"""

#: The revision whose claim survives measurement (10.0 / 5.0 == 2.0x).
HONEST_SOURCE = PRISTINE_SOURCE.replace("BLOCK_SIZE = 32", "BLOCK_SIZE = 64")
#: The revision whose claim collapses under measurement (10.0 / 8.0 == 1.25x).
INFLATED_SOURCE = PRISTINE_SOURCE.replace("BLOCK_SIZE = 32", "BLOCK_SIZE = 128")

#: Two further revisions, used only to fill a candidate field past the bound.
WIDE_SOURCE = PRISTINE_SOURCE.replace("BLOCK_SIZE = 32", "BLOCK_SIZE = 256")
WIDEST_SOURCE = PRISTINE_SOURCE.replace("BLOCK_SIZE = 32", "BLOCK_SIZE = 512")

#: A revision published after the field was already measured (10.0 / 4.0 == 2.5x).
BEST_SOURCE = PRISTINE_SOURCE.replace("BLOCK_SIZE = 32", "BLOCK_SIZE = 16")

#: The revision that wins the per-case mean and loses the suite (see
#: ``_DRIVER_CASE_MS``).
LOPSIDED_SOURCE = PRISTINE_SOURCE.replace("BLOCK_SIZE = 32", "BLOCK_SIZE = 8")
#: The revision that halves every case, so both measures agree it is faster.
BALANCED_SOURCE = PRISTINE_SOURCE.replace("BLOCK_SIZE = 32", "BLOCK_SIZE = 4")

#: The revision that is slower than pristine (10.0 / 12.5 == 0.8x), so it misses
#: the keep threshold outright instead of losing on the suite total.
SLOWER_SOURCE = PRISTINE_SOURCE.replace("BLOCK_SIZE = 32", "BLOCK_SIZE = 1024")

#: Per-case wall clock the two-case driver double reports for each revision. The
#: lopsided revision speeds one cheap case up fourfold and lets the expensive
#: case slip to 0.8x, which averages to 2.4x while the suite total rises from
#: 101.0 ms to 125.25 ms.
_DRIVER_CASE_MS = {
    "BLOCK_SIZE = 8": {"case-cheap": 0.25, "case-expensive": 125.0},
    "BLOCK_SIZE = 4": {"case-cheap": 0.5, "case-expensive": 50.0},
    "BLOCK_SIZE = 32": {"case-cheap": 1.0, "case-expensive": 100.0},
}

#: Wall clock the driver double reports for each revision of the kernel.
_DRIVER_MS = {
    "BLOCK_SIZE = 16": 4.0,
    "BLOCK_SIZE = 512": 6.0,
    "BLOCK_SIZE = 256": 7.0,
    "BLOCK_SIZE = 128": 8.0,
    "BLOCK_SIZE = 64": 5.0,
    "BLOCK_SIZE = 32": 10.0,
    "BLOCK_SIZE = 1024": 12.5,
}

SUMMARY = {
    "category": "elementwise",
    "strategy": "widen the deterministic block",
    "recipe": "Raise BLOCK_SIZE.",
    "lessons": "Wider blocks are not always faster.",
}

CANONICAL_ID = "kernel:forge-loop:deterministic:aiter:unspecified:triton:mi355x"

#: Set per test by the autouse fixture; producer and consumer share one store.
_KNOWLEDGE_ROOT: Path | None = None


def _run_config() -> Config:
    knowledge = KnowledgeConfig.from_env({}, mode="local", local_root=_KNOWLEDGE_ROOT)
    return Config.from_env(
        workspace=str(_KNOWLEDGE_ROOT),
        gpu_target="gfx950",
        gpu_type="mi355x",
        knowledge_config=knowledge,
        agent_precheck=False,
    )


def _remote_config_with_token(tmp_path: Path, token: str) -> Config:
    """A KB Store run configuration whose credential is a recognizable string."""
    knowledge = KnowledgeConfig.from_env(
        {},
        mode="remote",
        local_root=tmp_path / "remote-knowledge",
        kb_store_url="http://in-memory",
        kb_store_token=token,
        remote_backend=REMOTE_BACKEND_KB_STORE,
    )
    return Config.from_env(
        workspace=str(tmp_path),
        gpu_target="gfx950",
        gpu_type="mi355x",
        knowledge_config=knowledge,
        agent_precheck=False,
    )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _initialize_workspace(root: Path, name: str, kernel_path: Path) -> tuple[Path, Path, str]:
    workspace = root / name
    workspace.mkdir()
    _git(workspace, "init", "-b", "main")
    _git(workspace, "config", "user.email", "kb-evidence@example.com")
    _git(workspace, "config", "user.name", "KB Evidence")
    kernel = workspace / kernel_path
    kernel.parent.mkdir(parents=True)
    kernel.write_text(PRISTINE_SOURCE)
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-m", "pristine")
    return workspace, kernel, _git(workspace, "rev-parse", "HEAD")


def _publish_candidate(
    root: Path,
    name: str,
    *,
    optimized_source: str,
    claimed_speedup: float,
) -> dict:
    """Record one producer solution that claims ``claimed_speedup``."""
    workspace, kernel, base = _initialize_workspace(root, name, PRODUCER_KERNEL_PATH)
    kernel.write_text(optimized_source)
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-m", "optimize deterministic kernel")
    patch = subprocess.run(
        ["git", "diff", base, "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert patch
    status = sink.write_run_experience(
        config=_run_config(),
        workspace=str(workspace),
        kernel_path=str(kernel),
        kernel_source=optimized_source,
        kernel_backend="triton",
        gpu_target="gfx950",
        experiment_id=f"producer-{name}",
        baseline_wall_ms=10.0,
        best_wall_ms=10.0 / claimed_speedup,
        mean_case_speedup=claimed_speedup,
        cumulative_diff=patch,
        digest="deterministic producer result",
        summary_override=SUMMARY,
    )
    assert status["written"] is True
    assert status["kernel"] == CANONICAL_ID
    return status


def _install_driver_doubles(monkeypatch: pytest.MonkeyPatch, kernel: Path) -> list[float]:
    """Drive correctness and benchmark results from the patched kernel source."""
    measured: list[float] = []

    def benchmark(_driver: str, *_args, **_kwargs) -> dict:
        source = kernel.read_text()
        wall_ms = next(value for marker, value in _DRIVER_MS.items() if marker in source)
        measured.append(wall_ms)
        return {
            "success": True,
            "median_ms": wall_ms,
            "case_times": {"case-1": wall_ms},
        }

    def correctness(_driver: str, _snr_threshold: float) -> bool:
        return "BLOCK_SIZE" in kernel.read_text()

    monkeypatch.setattr(integration, "_bench_once", benchmark)
    monkeypatch.setattr(integration, "_correctness_once", correctness)
    return measured


def _install_suite_driver_doubles(
    monkeypatch: pytest.MonkeyPatch,
    kernel: Path,
) -> list[float]:
    """Report a two-case suite whose total can disagree with its case mean.

    ``median_ms`` is the suite's total wall time, the aggregate a real driver
    prints for the whole run, while ``case_times`` carries the per-case timings
    the mean case speedup is computed from. Returns the list of suite totals the
    double reported, in call order.
    """
    measured: list[float] = []

    def benchmark(_driver: str, *_args, **_kwargs) -> dict:
        source = kernel.read_text()
        case_times = next(times for marker, times in _DRIVER_CASE_MS.items() if marker in source)
        total_ms = sum(case_times.values())
        measured.append(total_ms)
        return {
            "success": True,
            "median_ms": total_ms,
            "case_times": dict(case_times),
        }

    def correctness(_driver: str, _snr_threshold: float) -> bool:
        return "BLOCK_SIZE" in kernel.read_text()

    monkeypatch.setattr(integration, "_bench_once", benchmark)
    monkeypatch.setattr(integration, "_correctness_once", correctness)
    return measured


def _warm_start(workspace: Path, kernel: Path) -> dict:
    return integration.kb_warmstart(
        config=_run_config(),
        kernel=str(kernel),
        driver="unused-driver.py",
        workspace_dir=str(workspace),
        kernel_backend="triton",
    )


def _records() -> LocalRewriteRecords:
    return LocalRewriteRecords(knowledge_config_from_runtime(_run_config()).rewrite_root)


def _stored(canonical_id: str = CANONICAL_ID) -> dict[str, RewriteCandidate]:
    """Every recorded candidate for one identity, keyed by session id."""
    return {candidate.session_id: candidate for candidate in _records().candidates(canonical_id, limit=50)}


def _session_id(status: dict) -> str:
    return str(status["session_id"])


def _index_status(workspace: Path, rank: int) -> str:
    line = next(
        line
        for line in (workspace / "forge_experiments" / "kb_references" / "index.md").read_text().splitlines()
        if line.startswith(f"- Rank {rank}:")
    )
    return line.split("status `", 1)[1].rstrip("`")


class MergingKBStore(InMemoryKBStore):
    """In-memory KB Store that honors the SDK's documented merge mode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.knowledge_writes: list[tuple[str, str, dict, str]] = []

    def put_knowledge(self, canonical_id, knowledge, *, session_id="", mode="merge"):
        self.knowledge_writes.append((canonical_id, session_id, dict(knowledge), mode))
        key = (canonical_id, session_id)
        if mode == "merge" and key in self.knowledge:
            merged = dict(self.knowledge[key])
            merged.update(knowledge)
            self.knowledge[key] = merged
            return {"session_id": session_id, "mode": mode}
        return super().put_knowledge(canonical_id, knowledge, session_id=session_id, mode=mode)


@pytest.fixture(autouse=True)
def knowledge_root(tmp_path_factory):
    """One empty on-disk store per test, so runs never inherit each other."""
    global _KNOWLEDGE_ROOT
    _KNOWLEDGE_ROOT = tmp_path_factory.mktemp("kb-evidence")
    yield _KNOWLEDGE_ROOT
    _KNOWLEDGE_ROOT = None


def test_candidate_ranking_prefers_a_measured_speedup_over_a_higher_claim():
    claimed = RewriteCandidate(
        session_id="claims-more",
        knowledge={"speedup": 6.7608},
        speedup=6.7608,
        is_champion=True,
    )
    measured = RewriteCandidate(
        session_id="measured-less",
        knowledge={"speedup": 5.8452, "measured_speedup": 5.8452},
        speedup=5.8452,
        is_champion=False,
        measured_speedup=5.8452,
    )

    ranked = record_store._rank([claimed, measured], 2)

    assert [candidate.session_id for candidate in ranked] == [
        "measured-less",
        "claims-more",
    ]


def test_candidate_ranking_orders_equal_evidence_by_session_id():
    first = RewriteCandidate(
        session_id="aaa-session",
        knowledge={"speedup": 2.0},
        speedup=2.0,
        is_champion=False,
    )
    second = RewriteCandidate(
        session_id="bbb-session",
        knowledge={"speedup": 2.0},
        speedup=2.0,
        is_champion=False,
    )

    assert [item.session_id for item in record_store._rank([second, first], 2)] == [
        "aaa-session",
        "bbb-session",
    ]


def test_local_measured_write_back_amends_the_record_without_losing_the_claim(tmp_path):
    store = _records()
    artifact = tmp_path / "kernel.py"
    artifact.write_text("kernel\n")
    store.write(
        CANONICAL_ID,
        "kda-attn-session",
        {"speedup": 6.7608, "value": {"tag": "inflated"}},
        {"kernel.py": artifact},
    )

    store.record_measured_speedup(CANONICAL_ID, "kda-attn-session", 5.106)

    candidate = _stored()["kda-attn-session"]
    assert candidate.speedup == 6.7608
    assert candidate.measured_speedup == 5.106
    assert candidate.knowledge["value"] == {"tag": "inflated"}
    assert store.read_bytes(CANONICAL_ID, "kda-attn-session", "kernel.py") == b"kernel\n"


def test_local_measured_write_back_fails_loudly_for_an_unknown_session():
    with pytest.raises(RewriteRecordError, match="candidate knowledge"):
        _records().record_measured_speedup(CANONICAL_ID, "never-recorded", 2.0)


def test_remote_measured_write_back_merges_into_the_candidate_session(
    tmp_path,
    monkeypatch,
):
    store = MergingKBStore()
    monkeypatch.setattr(record_store, "KBStoreClient", lambda *a, **k: store)
    kb = KernelRecipeKB.open_canonical_id(CANONICAL_ID, _remote_config(tmp_path))
    store.put_knowledge(
        CANONICAL_ID,
        {"producer": "forge-loop", "speedup": 6.7608, "value": {"tag": "inflated"}},
        session_id="kda-attn-session",
        mode="replace",
    )

    outcome = kb.record_measured_speedup("kda-attn-session", 5.106)

    assert outcome["recorded"] is True
    assert store.knowledge[(CANONICAL_ID, "kda-attn-session")] == {
        "producer": "forge-loop",
        "speedup": 6.7608,
        "measured_speedup": 5.106,
        "value": {"tag": "inflated"},
    }
    assert store.knowledge_writes[-1] == (
        CANONICAL_ID,
        "kda-attn-session",
        {"measured_speedup": 5.106},
        "merge",
    )


def test_a_refused_measured_write_back_redacts_and_bounds_the_store_error(
    tmp_path,
    monkeypatch,
):
    """The refusal reason is persisted, so it may not carry a credential.

    ``record_measured_speedup`` reports a refusal instead of raising, and that
    reason travels through ``measured_writebacks`` and
    ``measured_writeback_failures`` into the run's result JSON. A KB Store
    exception can quote the bearer token the client authenticated with, a
    credentialed URL and an unbounded response body, so this path sanitizes and
    bounds its text at 240 characters exactly like every read path beside it.
    """
    token = "kb-store-secret-9f3c"
    store = MergingKBStore()
    monkeypatch.setattr(record_store, "KBStoreClient", lambda *a, **k: store)

    def refuse(*_args, **_kwargs):
        raise record_store.KBStoreError(
            f"PUT https://forge:{token}@kb.example/knowledge failed "
            f"(sent Bearer {token}); the store said {token} expired" + " and returned an unbounded body" * 20
        )

    monkeypatch.setattr(store, "put_knowledge", refuse)
    kb = KernelRecipeKB.open_canonical_id(
        CANONICAL_ID,
        _remote_config_with_token(tmp_path, token),
    )

    outcome = kb.record_measured_speedup("kda-attn-session", 5.106)

    assert outcome["recorded"] is False
    assert token not in outcome["reason"]
    assert outcome["reason"].startswith("KBStoreError: PUT https://[REDACTED]@")
    assert "Bearer [REDACTED]" in outcome["reason"]
    assert "the store said [REDACTED] expired" in outcome["reason"]
    assert len(outcome["reason"]) == 240


def test_warm_start_adopts_the_best_measured_candidate_not_the_first_applied(
    monkeypatch,
    tmp_path,
):
    inflated = _publish_candidate(
        tmp_path,
        "producer-inflated",
        optimized_source=INFLATED_SOURCE,
        claimed_speedup=3.0,
    )
    honest = _publish_candidate(
        tmp_path,
        "producer-honest",
        optimized_source=HONEST_SOURCE,
        claimed_speedup=2.0,
    )
    consumer, kernel, base = _initialize_workspace(tmp_path, "consumer", CONSUMER_KERNEL_PATH)
    measured = _install_driver_doubles(monkeypatch, kernel)

    warm = _warm_start(consumer, kernel)

    assert warm["applied"] is True
    assert warm["num_references"] == 2
    # Rank 1 claims 3.0x but measures 1.25x; rank 2 claims 2.0x and holds it.
    assert warm["solution_slug"] == honest["solution"]
    assert warm["applied_rank"] == 2
    assert warm["mean_case_speedup"] == 2.0
    assert warm["keep_baseline_ms"] == 5.0
    assert kernel.read_text() == HONEST_SOURCE
    assert _index_status(consumer, 1) == "rejected:outperformed_by_rank_2"
    assert _index_status(consumer, 2) == "applied"
    # Both candidates measured once each, after the three pristine measurements.
    assert measured == [10.0] * 3 + [8.0] * 3 + [5.0] * 3
    assert warm["applied_commit"] == _git(consumer, "rev-parse", "HEAD")
    assert warm["applied_commit"] != base
    assert _git(consumer, "log", "-1", "--pretty=%s") == (f"kb warm-start: apply {honest['solution']}")
    assert _git(consumer, "status", "--porcelain=v1", "--untracked-files=no") == ""
    assert _git(consumer, "diff", "--name-only", base, "HEAD").splitlines() == [CONSUMER_KERNEL_PATH.as_posix()]
    assert _session_id(inflated) in _stored()


def test_warm_start_writes_every_measured_speedup_back_to_the_kb(
    monkeypatch,
    tmp_path,
):
    inflated = _publish_candidate(
        tmp_path,
        "producer-inflated",
        optimized_source=INFLATED_SOURCE,
        claimed_speedup=3.0,
    )
    honest = _publish_candidate(
        tmp_path,
        "producer-honest",
        optimized_source=HONEST_SOURCE,
        claimed_speedup=2.0,
    )
    consumer, kernel, _base = _initialize_workspace(tmp_path, "consumer", CONSUMER_KERNEL_PATH)
    _install_driver_doubles(monkeypatch, kernel)

    warm = _warm_start(consumer, kernel)

    stored = _stored()
    inflated_record = stored[_session_id(inflated)]
    honest_record = stored[_session_id(honest)]
    assert inflated_record.speedup == 3.0
    assert inflated_record.measured_speedup == 1.25
    assert honest_record.speedup == 2.0
    assert honest_record.measured_speedup == 2.0
    assert warm["measured_writebacks"] == [
        {
            "rank": 1,
            "solution_slug": inflated["solution"],
            "measured_mean_case_speedup": 1.25,
            "recorded": True,
            "reason": "",
        },
        {
            "rank": 2,
            "solution_slug": honest["solution"],
            "measured_mean_case_speedup": 2.0,
            "recorded": True,
            "reason": "",
        },
    ]


def test_warm_start_ranks_a_corrected_record_on_its_measured_value(
    monkeypatch,
    tmp_path,
):
    """The second run must start from the verified winner, at rank 1 and once."""
    _publish_candidate(
        tmp_path,
        "producer-inflated",
        optimized_source=INFLATED_SOURCE,
        claimed_speedup=3.0,
    )
    honest = _publish_candidate(
        tmp_path,
        "producer-honest",
        optimized_source=HONEST_SOURCE,
        claimed_speedup=2.0,
    )
    first_consumer, first_kernel, _first_base = _initialize_workspace(tmp_path, "consumer-first", CONSUMER_KERNEL_PATH)
    _install_driver_doubles(monkeypatch, first_kernel)
    _warm_start(first_consumer, first_kernel)

    second_consumer, second_kernel, _second_base = _initialize_workspace(
        tmp_path, "consumer-second", CONSUMER_KERNEL_PATH
    )
    measured = _install_driver_doubles(monkeypatch, second_kernel)
    warm = _warm_start(second_consumer, second_kernel)

    assert warm["applied"] is True
    assert warm["applied_rank"] == 1
    assert warm["solution_slug"] == honest["solution"]
    assert warm["mean_case_speedup"] == 2.0
    assert measured == [10.0] * 3 + [5.0] * 3
    assert _index_status(second_consumer, 2) == "not_attempted_after_apply"


def test_warm_start_evaluates_a_later_record_that_outclaims_the_leader(
    monkeypatch,
    tmp_path,
):
    """A measured leader must not freeze out records published after it.

    Ranking puts every measured candidate ahead of every merely claimed one, and
    a record only earns a measurement by being adopted, so every solution
    published later starts behind. Ending the search on a confirmed leader alone
    would pin warm start to the first record ever measured.
    """
    _publish_candidate(
        tmp_path,
        "producer-honest",
        optimized_source=HONEST_SOURCE,
        claimed_speedup=2.0,
    )
    first_consumer, first_kernel, _first_base = _initialize_workspace(tmp_path, "consumer-first", CONSUMER_KERNEL_PATH)
    _install_driver_doubles(monkeypatch, first_kernel)
    _warm_start(first_consumer, first_kernel)

    best = _publish_candidate(
        tmp_path,
        "producer-best",
        optimized_source=BEST_SOURCE,
        claimed_speedup=2.5,
    )

    second_consumer, second_kernel, _second_base = _initialize_workspace(
        tmp_path, "consumer-second", CONSUMER_KERNEL_PATH
    )
    measured = _install_driver_doubles(monkeypatch, second_kernel)
    warm = _warm_start(second_consumer, second_kernel)

    assert warm["applied"] is True
    assert warm["applied_rank"] == 2
    assert warm["solution_slug"] == best["solution"]
    assert warm["mean_case_speedup"] == 2.5
    assert measured == [10.0] * 3 + [5.0] * 3 + [4.0] * 3


def test_warm_start_stops_evaluating_once_the_top_claim_is_confirmed(
    monkeypatch,
    tmp_path,
):
    honest = _publish_candidate(
        tmp_path,
        "producer-honest",
        optimized_source=HONEST_SOURCE,
        claimed_speedup=2.0,
    )
    _publish_candidate(
        tmp_path,
        "producer-inflated",
        optimized_source=INFLATED_SOURCE,
        claimed_speedup=1.5,
    )
    consumer, kernel, _base = _initialize_workspace(tmp_path, "consumer", CONSUMER_KERNEL_PATH)
    measured = _install_driver_doubles(monkeypatch, kernel)

    warm = _warm_start(consumer, kernel)

    assert warm["applied"] is True
    assert warm["applied_rank"] == 1
    assert warm["solution_slug"] == honest["solution"]
    assert measured == [10.0] * 3 + [5.0] * 3
    assert _index_status(consumer, 2) == "not_attempted_after_apply"
    assert len(warm["measured_writebacks"]) == 1


def test_warm_start_evaluates_no_more_candidates_than_the_bound(
    monkeypatch,
    tmp_path,
):
    """Bound the driver cost even when no claim survives its measurement."""
    for name, source, claim in (
        ("widest", WIDEST_SOURCE, 10.0),
        ("wide", WIDE_SOURCE, 9.0),
        ("inflated", INFLATED_SOURCE, 8.0),
        ("honest", HONEST_SOURCE, 7.0),
    ):
        _publish_candidate(
            tmp_path,
            f"producer-{name}",
            optimized_source=source,
            claimed_speedup=claim,
        )
    monkeypatch.setattr(integration, "_WARMSTART_TOP_K", 4)
    consumer, kernel, _base = _initialize_workspace(tmp_path, "consumer", CONSUMER_KERNEL_PATH)
    measured = _install_driver_doubles(monkeypatch, kernel)

    warm = _warm_start(consumer, kernel)

    assert warm["num_references"] == 4
    assert integration._WARMSTART_MAX_MEASURED_CANDIDATES == 3
    assert len(warm["measured_writebacks"]) == 3
    # Three measured candidates, then the field is closed: the fourth is never
    # built or benchmarked even though no claim was confirmed.
    assert measured == [10.0] * 3 + [6.0] * 3 + [7.0] * 3 + [8.0] * 3
    assert warm["applied"] is True
    assert warm["applied_rank"] == 1
    assert warm["mean_case_speedup"] == pytest.approx(10.0 / 6.0)
    assert _index_status(consumer, 2) == "rejected:outperformed_by_rank_1"
    assert _index_status(consumer, 3) == "rejected:outperformed_by_rank_1"
    assert _index_status(consumer, 4) == "not_attempted_after_apply"
    assert kernel.read_text() == WIDEST_SOURCE
    assert _git(consumer, "status", "--porcelain=v1", "--untracked-files=no") == ""


def test_warm_start_reports_a_failed_measured_write_back_and_still_applies(
    monkeypatch,
    tmp_path,
):
    honest = _publish_candidate(
        tmp_path,
        "producer-honest",
        optimized_source=HONEST_SOURCE,
        claimed_speedup=2.0,
    )
    consumer, kernel, base = _initialize_workspace(tmp_path, "consumer", CONSUMER_KERNEL_PATH)
    _install_driver_doubles(monkeypatch, kernel)

    def refuse(*_args, **_kwargs):
        raise RewriteRecordError("store rejected the amendment")

    monkeypatch.setattr(
        LocalRewriteRecords,
        "record_measured_speedup",
        refuse,
    )
    warm = _warm_start(consumer, kernel)

    assert warm["applied"] is True
    assert warm["solution_slug"] == honest["solution"]
    assert warm["measured_writebacks"] == [
        {
            "rank": 1,
            "solution_slug": honest["solution"],
            "measured_mean_case_speedup": 2.0,
            "recorded": False,
            "reason": "RewriteRecordError: store rejected the amendment",
        },
    ]
    assert _stored()[_session_id(honest)].measured_speedup is None
    assert _git(consumer, "rev-parse", "HEAD") != base
    assert _git(consumer, "status", "--porcelain=v1", "--untracked-files=no") == ""


def test_warm_start_restores_the_worktree_when_no_candidate_is_adoptable(
    monkeypatch,
    tmp_path,
):
    _publish_candidate(
        tmp_path,
        "producer-inflated",
        optimized_source=INFLATED_SOURCE,
        claimed_speedup=3.0,
    )
    _publish_candidate(
        tmp_path,
        "producer-honest",
        optimized_source=HONEST_SOURCE,
        claimed_speedup=2.0,
    )
    consumer, kernel, base = _initialize_workspace(tmp_path, "consumer", CONSUMER_KERNEL_PATH)
    _install_driver_doubles(monkeypatch, kernel)
    monkeypatch.setattr(integration, "_correctness_once", lambda *_a, **_k: False)

    warm = _warm_start(consumer, kernel)

    assert warm["applied"] is False
    assert warm["reference_reason"] == "correctness_failed"
    assert warm["measured_writebacks"] == []
    assert kernel.read_text() == PRISTINE_SOURCE
    assert _git(consumer, "rev-parse", "HEAD") == base
    assert _git(consumer, "status", "--porcelain=v1", "--untracked-files=no") == ""


def test_the_lopsided_suite_clears_the_keep_threshold_it_regresses_against():
    """Pin the arithmetic the aggregate rejection below depends on.

    The lopsided revision has to clear the keep gate, otherwise the rejection
    proves nothing new: the mean beats the pristine baseline by the required
    margin and the suite total still rises.
    """
    pristine = _DRIVER_CASE_MS["BLOCK_SIZE = 32"]
    lopsided = _DRIVER_CASE_MS["BLOCK_SIZE = 8"]
    case_speedups = [pristine[case_id] / lopsided[case_id] for case_id in pristine]
    mean_case_speedup = sum(case_speedups) / len(case_speedups)

    assert min(case_speedups) < 1.0
    # Three identical measurements carry no spread, so the gate falls back to
    # its floor -- the weakest bar this revision could be asked to clear.
    assert passes_keep_threshold([mean_case_speedup] * 3, best_mean_case_speedup=1.0)
    assert sum(lopsided.values()) > sum(pristine.values())


def test_warm_start_refuses_a_candidate_that_is_slower_over_the_whole_suite(
    monkeypatch,
    tmp_path,
):
    """An adopted warm start becomes the run's incumbent and iteration-0 best.

    The per-case mean is unbounded above and bounded at 0 below, so one cheap
    case improving fourfold outvotes one expensive case collapsing and the mean
    reads 2.4x while the suite takes 125.25 ms against a pristine 101.0 ms.
    Starting there hands the run a worse baseline than doing nothing.
    """
    lopsided = _publish_candidate(
        tmp_path,
        "producer-lopsided",
        optimized_source=LOPSIDED_SOURCE,
        claimed_speedup=3.0,
    )
    consumer, kernel, base = _initialize_workspace(tmp_path, "consumer", CONSUMER_KERNEL_PATH)
    measured = _install_suite_driver_doubles(monkeypatch, kernel)

    warm = _warm_start(consumer, kernel)

    assert warm["applied"] is False
    # Named apart from performance_failed: this candidate cleared the threshold.
    assert warm["reference_reason"] == "aggregate_regression"
    assert _index_status(consumer, 1) == ("rejected:aggregate_regression (measured 2.400000x recorded)")
    # The suite was benchmarked, so the record is amended even though the
    # candidate lost: it claimed 3.0x and this consumer measured 2.4x.
    assert warm["measured_writebacks"] == [
        {
            "rank": 1,
            "solution_slug": lopsided["solution"],
            "measured_mean_case_speedup": 2.4,
            "recorded": True,
            "reason": "",
        },
    ]
    assert _stored()[_session_id(lopsided)].measured_speedup == 2.4
    assert warm["pristine_ms"] == 101.0
    assert warm["keep_baseline_ms"] == 101.0
    assert warm["mean_case_speedup"] == 1.0
    assert warm["applied_commit"] == ""
    assert measured == [101.0] * 3 + [125.25] * 3
    # Measured once, then restored: no adoption and nothing left in the tree.
    assert kernel.read_text() == PRISTINE_SOURCE
    assert _git(consumer, "rev-parse", "HEAD") == base
    assert _git(consumer, "status", "--porcelain=v1", "--untracked-files=no") == ""


def test_warm_start_tries_the_next_rank_after_an_aggregate_regression(
    monkeypatch,
    tmp_path,
):
    """Rejecting the leader falls through to the field, it does not end warm start.

    This is the incident's shape: rank 1 carries the higher claim and loses over
    the suite, rank 2 is faster on both measures and was never tried.
    """
    lopsided = _publish_candidate(
        tmp_path,
        "producer-lopsided",
        optimized_source=LOPSIDED_SOURCE,
        claimed_speedup=3.0,
    )
    balanced = _publish_candidate(
        tmp_path,
        "producer-balanced",
        optimized_source=BALANCED_SOURCE,
        claimed_speedup=2.0,
    )
    consumer, kernel, base = _initialize_workspace(tmp_path, "consumer", CONSUMER_KERNEL_PATH)
    measured = _install_suite_driver_doubles(monkeypatch, kernel)

    warm = _warm_start(consumer, kernel)

    assert warm["applied"] is True
    assert warm["num_references"] == 2
    assert warm["applied_rank"] == 2
    assert warm["solution_slug"] == balanced["solution"]
    assert warm["mean_case_speedup"] == 2.0
    assert warm["pristine_ms"] == 101.0
    assert warm["keep_baseline_ms"] == 50.5
    assert _index_status(consumer, 1) == ("rejected:aggregate_regression (measured 2.400000x recorded)")
    assert _index_status(consumer, 2) == "applied"
    assert measured == [101.0] * 3 + [125.25] * 3 + [50.5] * 3
    # Both candidates were benchmarked, so both records are amended in rank
    # order. The leader's inflated 3.0x claim is corrected to the 2.4x this
    # consumer measured even though it was rejected, which is what stops it
    # winning rank 1 and being re-measured on every later run.
    assert [item["rank"] for item in warm["measured_writebacks"]] == [1, 2]
    assert _stored()[_session_id(lopsided)].measured_speedup == 2.4
    assert _stored()[_session_id(balanced)].measured_speedup == 2.0
    # Writing that measurement back must not make the leader adoptable: 2.4x is
    # the higher measurement of the two, so a rejected candidate leaking into the
    # adoption field would win it and rank 1 would be the incumbent above.
    assert kernel.read_text() == BALANCED_SOURCE
    assert warm["applied_commit"] == _git(consumer, "rev-parse", "HEAD")
    assert warm["applied_commit"] != base
    assert _git(consumer, "status", "--porcelain=v1", "--untracked-files=no") == ""


def test_warm_start_writes_back_a_candidate_that_missed_the_keep_threshold(
    monkeypatch,
    tmp_path,
):
    """A candidate can be measured and rejected without an aggregate regression.

    performance_failed means the patch applied, correctness passed and the whole
    driver suite was benchmarked; the candidate simply came out slower. That
    measurement is exactly as valid as an adopted one, and the record claiming
    2.0x for something this consumer measures at 0.8x is the record the KB most
    needs corrected.
    """
    slower = _publish_candidate(
        tmp_path,
        "producer-slower",
        optimized_source=SLOWER_SOURCE,
        claimed_speedup=2.0,
    )
    consumer, kernel, base = _initialize_workspace(tmp_path, "consumer", CONSUMER_KERNEL_PATH)
    measured = _install_driver_doubles(monkeypatch, kernel)

    warm = _warm_start(consumer, kernel)

    assert warm["applied"] is False
    assert warm["reference_reason"] == "performance_failed"
    assert _index_status(consumer, 1) == ("rejected:performance_failed (measured 0.800000x recorded)")
    [writeback] = warm["measured_writebacks"]
    assert writeback["rank"] == 1
    assert writeback["solution_slug"] == slower["solution"]
    # The published figure is now the mean of the measurements rather than their
    # minimum, so it carries the mean's rounding rather than a sample's exact
    # value.
    assert writeback["measured_mean_case_speedup"] == pytest.approx(0.8)
    assert writeback["recorded"] is True
    assert writeback["reason"] == ""
    assert _stored()[_session_id(slower)].speedup == 2.0
    assert _stored()[_session_id(slower)].measured_speedup == pytest.approx(0.8)
    # Measured once, then restored: the run starts from pristine.
    assert measured == [10.0] * 3 + [12.5] * 3
    assert warm["keep_baseline_ms"] == 10.0
    assert warm["mean_case_speedup"] == 1.0
    assert kernel.read_text() == PRISTINE_SOURCE
    assert _git(consumer, "rev-parse", "HEAD") == base
    assert _git(consumer, "status", "--porcelain=v1", "--untracked-files=no") == ""


def test_warm_start_reports_a_rejected_candidate_write_back_the_store_refused(
    monkeypatch,
    tmp_path,
):
    """A refusal has to be as visible for a rejected candidate as an adopted one.

    The refusal leaves the KB ranking a claim this consumer just contradicted,
    so it travels the same route: into measured_writebacks, out of
    kb_read_status as a measured_writeback_failure, and onto the reference index
    the operator reads.
    """
    lopsided = _publish_candidate(
        tmp_path,
        "producer-lopsided",
        optimized_source=LOPSIDED_SOURCE,
        claimed_speedup=3.0,
    )
    consumer, kernel, _base = _initialize_workspace(tmp_path, "consumer", CONSUMER_KERNEL_PATH)
    _install_suite_driver_doubles(monkeypatch, kernel)

    def refuse(*_args, **_kwargs):
        raise RewriteRecordError("store rejected the amendment")

    monkeypatch.setattr(LocalRewriteRecords, "record_measured_speedup", refuse)
    warm = _warm_start(consumer, kernel)

    assert warm["applied"] is False
    assert warm["measured_writebacks"] == [
        {
            "rank": 1,
            "solution_slug": lopsided["solution"],
            "measured_mean_case_speedup": 2.4,
            "recorded": False,
            "reason": "RewriteRecordError: store rejected the amendment",
        },
    ]
    assert integration.kb_read_status(warm)["measured_writeback_failures"] == [
        "RewriteRecordError: store rejected the amendment"
    ]
    assert _index_status(consumer, 1) == ("rejected:aggregate_regression (measured 2.400000x write-back refused)")
    assert _stored()[_session_id(lopsided)].measured_speedup is None
    assert kernel.read_text() == PRISTINE_SOURCE
    assert _git(consumer, "status", "--porcelain=v1", "--untracked-files=no") == ""


def test_warm_start_adopts_a_candidate_that_wins_the_suite_and_the_case_mean(
    monkeypatch,
    tmp_path,
):
    """The aggregate gate must not cost the warm start its reason to exist."""
    balanced = _publish_candidate(
        tmp_path,
        "producer-balanced",
        optimized_source=BALANCED_SOURCE,
        claimed_speedup=2.0,
    )
    consumer, kernel, base = _initialize_workspace(tmp_path, "consumer", CONSUMER_KERNEL_PATH)
    measured = _install_suite_driver_doubles(monkeypatch, kernel)

    warm = _warm_start(consumer, kernel)

    assert warm["applied"] is True
    assert warm["applied_rank"] == 1
    assert warm["solution_slug"] == balanced["solution"]
    assert warm["reference_reason"] == ""
    assert warm["mean_case_speedup"] == 2.0
    assert warm["pristine_ms"] == 101.0
    assert warm["keep_baseline_ms"] == 50.5
    assert _index_status(consumer, 1) == "applied"
    assert measured == [101.0] * 3 + [50.5] * 3
    assert kernel.read_text() == BALANCED_SOURCE
    assert warm["applied_commit"] == _git(consumer, "rev-parse", "HEAD")
    assert warm["applied_commit"] != base
    assert _git(consumer, "status", "--porcelain=v1", "--untracked-files=no") == ""
