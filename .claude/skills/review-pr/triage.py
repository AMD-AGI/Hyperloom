#!/usr/bin/env python3
"""Deterministic half of the review-pr skill: rule derivation and step gates.

`derive` decides which rule families a diff earns, reading only the Step 1 artifacts in
$WORK; `expand` prints the bodies of exactly those rules; `mapping` regenerates MAPPING.md
from the family table so the two cannot drift.

The `gate` subcommands exist because a review step that leaves no checkable trace is a step
that degrades silently: a review that skipped Step 2 reads identically to one that did it.
Each gate reads one artifact and rejects work that was not actually done -- an unanswered
question, a verdict with no reason, a card finding no verdict adjudicated. `ledger.txt`
records the sha256 of every artifact at the moment its gate passed and every gate
re-verifies it, so a step cannot be written after the fact to justify a later finding.
"""

import argparse
import fnmatch
import hashlib
import re
import sys
from pathlib import Path

# Family table for `triage.py derive`. One entry per diff shape the deriver can detect;
# `rules` is the checklist that shape earns in Step 4. MAPPING.md is generated from this
# table by `triage.py mapping` -- edit here, never there.
# Detection reads only the Step 1 artifacts: files.txt, numstat.txt, diff.txt, title.txt,
# body.txt, testfiles.txt, commits.txt, base.txt. ADD means a line starting
# with "+", DEL a line starting with "-", both outside the hunk headers.
FAMILIES = [
    {
        "name": "fix-title",
        "rules": ["C1", "C2", "T1", "T4", "X1"],
        "why": (
            "title.txt starts with a fix or perf conventional prefix, or with a bracketed [FIX]; "
            "the regression-test half applies when testfiles.txt is empty."
        ),
    },
    {
        "name": "unwired-or-unhonoured",
        "rules": ["C1", "C3"],
        "why": (
            "ADD declares an uppercase status literal or enum member, a dataclass or TypedDict field, or a "
            "keyword parameter with a default; or an identifier first introduced by ADD occurs exactly once in "
            "the head tree; or DEL removes a call whose callee is not deleted; or files.txt touches "
            "orchestrator/loop/coordinator.py, orchestrator/loop/writeback.py, **/state/shared_state.py, or a "
            "KNOBS / verdict / targets validation table."
        ),
    },
    {
        "name": "signature-or-doc-drift",
        "rules": ["C1", "X3"],
        "why": (
            "diff.txt pairs '-def NAME(' with '+def NAME(' for the same NAME; or a hunk whose context contains "
            "'def ' adds non-comment body lines while the adjacent docstring stays unchanged context; or DEL "
            "removes a comment or docstring line."
        ),
    },
    {
        "name": "explicit-field-contract",
        "rules": ["C1", "C5"],
        "why": (
            "files.txt touches src/hyperloom/inference_optimizer/protocol/** or orchestrator/policy/gate.py; or "
            "DEL removes an infer_/detect_/guess_/derive_/sniff_ helper while ADD introduces a mode, kind, type "
            "or format field with a string-literal default, merges a 'params' envelope layer, or validates a key "
            "only when present."
        ),
    },
    {
        "name": "operator-knob-added",
        "rules": ["C3", "C4", "S6", "X5"],
        "why": (
            "ADD calls add_argument with a '--' option string, or reads an uppercase environment name that does "
            "not appear in the base tree; or files.txt touches src/hyperloom/inference_optimizer/cli/** or "
            "src/kernelforge/cli.py."
        ),
    },
    {
        "name": "resolver-precedence",
        "rules": ["C4", "D8", "X3"],
        "why": (
            "ADD calls os.environ.setdefault, sets GIT_AUTHOR_/GIT_COMMITTER_/user.name/user.email, reads "
            "ROCR_/HIP_/CUDA_VISIBLE_DEVICES, reads an environment variable inside a non-CLI helper, defines a "
            "_select/_resolve/_pick sources helper, nests a key loop inside a source loop, or claims a precedence "
            "in a comment or docstring; or files.txt touches orchestrator/knowledge/remote_recipe/values.py, "
            "common/env_safety.py or **/agents/kernel/tools/**."
        ),
    },
    {
        "name": "executor-change",
        "rules": ["C2", "D1", "P1", "P4"],
        "why": (
            "files.txt touches src/hyperloom/orchestrator/actions/**, in particular exactly one file under "
            "actions/executors/ while sibling executors in the same directory are untouched."
        ),
    },
    {
        "name": "default-changed",
        "rules": ["C2", "S4", "S6", "T4"],
        "why": (
            "ADD sets default or default_enabled to True, passes a non-None default to add_argument, compares a "
            "parsed value against a DEFAULT_ constant, or calls an _apply profile/budget/preset or restore(args) "
            "helper."
        ),
    },
    {
        "name": "removal-or-tightening",
        "rules": ["C3", "T4", "X4"],
        "why": (
            "DEL removes an add_argument call, a [project.scripts] entry, an os.environ read, or an uppercase "
            "enum member; or a deleted '<' or '<=' comparison comes back as an added '==' or '!=' or an all(...) "
            "requirement."
        ),
    },
    {
        "name": "observable-effect-unstated",
        "rules": ["X1", "X2"],
        "why": (
            "files.txt has a src/**/*.py path outside **/tests/**, so the diff can carry an operator-visible "
            "effect the description has to state. Whether it does is read from body.txt at Step 4; no diff "
            "shape can settle it."
        ),
    },
    {
        "name": "description-drift",
        "rules": ["X2", "X3"],
        "why": (
            "commits.txt has more than one line; or title.txt is shaped like a branch name (two or more "
            "slash-separated segments); or body.txt is empty, still holds a template comment, or has an "
            "unchecked '- [ ]'; or body.txt states a percentage, a numeric default, or the word "
            "'unchanged'."
        ),
    },
    {
        "name": "prompt-surface",
        "rules": ["C5", "X3", "X5"],
        "why": (
            "files.txt matches src/hyperloom/**/prompts/**, **/assets/system_prompts/**, **/SKILL.md or "
            "docs/reference/**; also fires in the inverse direction, when ADD changes a flag, action name, env "
            "knob or artifact path and none of those paths are in files.txt."
        ),
    },
    {
        "name": "persisted-schema",
        "rules": ["C1", "R4", "X6"],
        "why": (
            "ADD or DEL mentions SCHEMA_VERSION, from_dict, ensure_schema or CREATE TABLE; or files.txt touches "
            "src/hyperloom/inference_optimizer/breakdown/** or src/kernelforge/durable_io.py; or a field is added "
            "or removed in a class in a file that contains SCHEMA_VERSION."
        ),
    },
    {
        "name": "pin-bump",
        "rules": ["X2", "X7"],
        "why": (
            "ADD or DEL changes a VLLM/SGLANG/ATOM/AITER/TRACELENS/MAGPIE/GEAK _VERSION, _REF, _SHA or _COMMIT "
            "literal; or files.txt touches **/framework_deps.py, **/framework_registry.py, assets/install*.sh or "
            "docs/compatibility.rst; or title.txt is a chore bump."
        ),
    },
    {
        "name": "tests-touched",
        "rules": ["T1", "T3"],
        "why": (
            "testfiles.txt is non-empty and files.txt also has a path outside **/tests/**; or ADD under "
            "**/tests/** builds MagicMock/Mock or ctx=object(), calls an underscore-prefixed internal, authors a "
            "fixture with write_text/write_bytes, or reaches requests, httpx, openai, boto3, subprocess, a "
            "provider API key, rocm-smi, /sys/ or probe=True."
        ),
    },
    {
        "name": "coverage-reduced",
        "rules": ["T1", "T2"],
        "why": (
            "DEL removes a 'def test_' or a critic_agent_e2e / targeted_build_e2e marker; or numstat.txt has a "
            "**/tests/** path with zero added lines; or a range(N) bound changes inside **/tests/**; or "
            "numstat.txt adds a src/** file over 300 lines with nothing in testfiles.txt."
        ),
    },
    {
        "name": "broad-kill",
        "rules": ["D8", "R1"],
        "why": (
            "ADD runs pkill, killall, kill -9, kill -KILL, scancel or docker rm/kill/stop, touches "
            "HIP_/ROCR_/CUDA_VISIBLE_DEVICES, or mentions os.killpg, start_new_session, setsid, preexec_fn or "
            "process_group; or files.txt touches inference_optimizer/assets/slurm/**, **/_subprocess_kill.py, "
            "**/multi_node/scripts/kill_multinode.py or **/assets/install*.sh."
        ),
    },
    {
        "name": "destructive-selfheal",
        "rules": ["R2", "R3"],
        "why": (
            "ADD calls shutil.rmtree, shutil.move, os.unlink, Path.unlink, git clean or git reset --hard; or "
            "defines an audit, reconcile, invalidate, repair, self_heal or purge step; or resolves an artifact "
            "through '.parent /' or an rglob over a runs, workspace or session root; or files.txt touches "
            "orchestrator/phases/kernel.py, **/_aiter_jit.py or **/framework/paths.py."
        ),
    },
    {
        "name": "cleanup-teardown",
        "rules": ["R3", "R4", "S1"],
        "why": (
            "ADD runs 'git checkout REF -- PATH', passes check=False, defines a teardown, rollback, cleanup, "
            "restore, release, revert or reclaim step, or reconstructs file contents inside a patch module; or "
            "files.txt touches executors/integrate_patch.py, orchestrator/kernel/patch_lifecycle.py, "
            "orchestrator/kernel/patch_landing.py or common/git_safety.py."
        ),
    },
    {
        "name": "state-transaction",
        "rules": ["C3", "P4", "P5", "R4"],
        "why": (
            "ADD defines a record_/read_/seal_ helper, __enter__/__exit__ or @contextmanager; calls "
            "recover/resume/reclaim_stale/adopt; sets 'mutated ='; assigns to shared_state or self.current_best; "
            "calls _emit_lifecycle or .save(); builds a spec, manifest or recipe; wraps a mutation in 'except "
            "Exception'; or calls sys.exit, parse_known_args or parser.error outside a CLI module. Also fires on "
            "orchestrator/state/shared_state.py, orchestrator/loop/writeback.py, cli/recover.py, common/io.py, "
            "common/jsonio.py, executors/_workload_envs.py, executors/_server_argv.py."
        ),
    },
    {
        "name": "untrusted-patch",
        "rules": ["D2", "R5", "S7"],
        "why": (
            "ADD runs git apply, patch -pN, --unsafe-paths, extractall, shutil.unpack_archive or tar -x; or "
            "files.txt touches src/kernelforge/data/serving_patches/**, **/_nogit_patch.py, "
            "**/_patch_sentinel.py, **/*_patcher.py or **/kernelforge/agent_backends/**."
        ),
    },
    {
        "name": "added-work-hot-path",
        "rules": ["P1", "P3", "P4"],
        "why": (
            "ADD inside an async def calls subprocess.run/check_output/call, .communicate(), time.sleep, "
            "requests, rglob, iterdir, glob, open, read_text or read_bytes without asyncio.to_thread or "
            "run_in_executor; or an added parse, hash or walk (json.load, gzip.open, yaml.safe_load, read_text, "
            "hashlib, rglob, os.walk) lands in a hunk whose context holds a lock (with ... lock, _file_lock, "
            "fcntl, flock) or inside a _promote_, _commit, writeback, record_keep, _wait_for_, _poll or retry "
            "function; or files.txt touches orchestrator/loop/** or **/_file_lock.py."
        ),
    },
    {
        "name": "budget-overrun",
        "rules": ["P2", "P6"],
        "why": (
            "an added append or extend lands in a hunk whose context mentions a limit, a budget or a budget "
            "slice; or a per-item cost estimate changes (_COST_S, cost_s =); or rows, shapes, tasks or "
            "candidates are appended; or files.txt touches src/kernelforge/gemm_tune/**, "
            "orchestrator/kernel/lane_budget.py or orchestrator/phases/machine_state.py."
        ),
    },
    {
        "name": "deadline-and-budget-scope",
        "rules": ["C2", "M3", "P2", "T4"],
        "why": (
            "ADD sets stream=True, passes timeout= to subprocess.run or create_subprocess_exec, or mentions "
            "KERNEL_AGENT_GPU_PLACEMENT, ssh, ray or geak_submit; or ADD/DEL changes an asyncio.wait_for or "
            "async timeout scope, a TIMEOUT / DEADLINE / GRACE / BUDGET constant, or a budget forwarded to a "
            "child (budget_min, a --*-budget flag)."
        ),
    },
    {
        "name": "metric-definition",
        "rules": ["M1", "M2", "X3"],
        "why": (
            "files.txt touches common/perf_metric.py, common/gain_math.py, orchestrator/scoring/** or "
            "inference_optimizer/breakdown/**; or ADD/DEL mentions speedup, gain_pct, throughput, ttft, itl, "
            "tpot, p99, baseline_ or denominator; or a quoted dict key is deleted and a differently spelled one "
            "added; or title.txt or body.txt mentions rename, metric, denominator or axis."
        ),
    },
    {
        "name": "cross-system-compare",
        "rules": ["D1", "M1", "S2"],
        "why": (
            "files.txt touches executors/baseline.py or orchestrator/kernel/conc_sweep.py; or ADD mentions "
            "self_report, revalidat, current_best or incumbent, names a vendor result field (geak, magpie, "
            "aiperf, vendor or external followed by _result, _ms, _score or _speedup), or compares against "
            "current_best; or title.txt or body.txt mentions revalidation, re-bench, promote or cross-check."
        ),
    },
    {
        "name": "threshold-units",
        "rules": ["M3", "S4", "T4"],
        "why": (
            "ADD assigns a THRESHOLD, LIMIT, RATE, PCT, RATIO or BUDGET constant to a number, compares against "
            "such a constant, or names a unit-bearing metric (error_rate, latency_ms, _ms, _s, p99, percent)."
        ),
    },
    {
        "name": "silent-failure",
        "rules": ["S1", "S2"],
        "why": (
            "ADD calls contextlib.suppress, passes ignore_errors=True, opens a narrowed 'except (' tuple, opens "
            "an except whose next two added lines are pass, continue or a return of None, False, an empty "
            "collection or an empty string, or returns an ok-true dict from a recovery path."
        ),
    },
    {
        "name": "unverified-success",
        "rules": ["S2", "S3"],
        "why": (
            "an added subprocess.run sits in a hunk with check=False and no returncode read; or an added "
            "mkdir(parents=True) is followed in the same hunk by write_text, a copy or json.dump; or ADD states "
            "a correctness literal (allclose or correctness set to True or 'pass') or a measurement literal "
            "(snr, speedup, latency_ms, tflops set to a number); or a compile_only or fallback branch appears in "
            "a hunk that also emits result, verdict or status."
        ),
    },
    {
        "name": "degraded-default",
        "rules": ["C3", "D5", "S3"],
        "why": (
            "ADD defines or returns a Noop, Disabled or Null implementation, especially from a **/factory.py; "
            "guards a warning on an _enabled config whose default is None; or writes a HYPERLOOM_ STATUS env "
            "value on one branch only; or ADD/DEL changes the unconfigured default backend among openai, "
            "anthropic, azure and bedrock in a resolver or factory path."
        ),
    },
    {
        "name": "empty-vs-absent",
        "rules": ["S1", "S5", "S7"],
        "why": (
            "a guard on one name flips between 'is None' and 'if not'; or ADD builds Path of an empty string, "
            "defaults an argument to frozenset(), set(), an empty dict or an empty list, coalesces with an "
            "or-empty or or-zero expression, or puts a presence test in front of a parser (parse, split, _ids); "
            "or an added isinstance(x, Mapping/dict/list/Sequence/tuple) or 'if x is not None:' returns within "
            "three added lines and sits above an existing fallback; or files.txt touches **/_gpu*.py, "
            "**/session/paths.py or src/kernelforge/**/kb.py."
        ),
    },
    {
        "name": "fail-closed-gate",
        "rules": ["C3", "S4", "S5"],
        "why": (
            "a deleted 'if not X:' is followed within two deleted lines by a return or raise; or ADD supplies a "
            "passing numeric default to an accessor (.get(key, 0), 0.0, 1, 1.0 or True), sets default=0, 0.0, "
            "1.0 or True, coalesces with an or-empty or or-zero expression, or widens a comparable, allowed or "
            "accepted set; or files.txt touches orchestrator/policy/gate.py, orchestrator/scoring/**, "
            "**/mapping.py, common/perf_metric.py or common/gain_math.py."
        ),
    },
    {
        "name": "duplicate-or-moved-code",
        "rules": ["D1", "D2", "D4", "X3"],
        "why": (
            "title.txt or body.txt mentions consolidate, extract shared, re-home, move, relocate, de-dup or a "
            "parallel path; or numstat.txt has one file whose added count is within 10% of another file's "
            "deleted count; or a 'def NAME(' added in one file is deleted in a sibling; or a DEFAULT_ constant "
            "relocates from a deleted inline literal; or ADD defines an apply, deploy, revert, snapshot, "
            "cleanup, revalidate or publish helper, or an enable_ flag set to False as a dispatch switch; or "
            "src/hyperloom/common/** gains a function body."
        ),
    },
    {
        "name": "fast-path-branch",
        "rules": ["D2", "S2"],
        "why": (
            "ADD introduces pre_applied, already_applied or a skip_apply/build/rebuild/invalidate/verify/post "
            "branch, or narrows a multi-file set by basename, '.name ==', startswith or _scope_diff_to_target; "
            "or files.txt touches executors/integrate_patch.py, an executors patcher module or "
            "orchestrator/kernel/patch_lifecycle.py."
        ),
    },
    {
        "name": "fallback-removed",
        "rules": ["D3", "T2", "X4"],
        "why": (
            "numstat.txt has a non-test src/** entry with deletions and no additions, or files.txt drops a "
            "**/bypass*.py, **/*_fallback*.py or **/*_legacy*.py; or DEL mentions bypass, fallback, legacy or "
            "alternative route; or title.txt or body.txt says drop, remove, delete or retire a route, path, "
            "fallback, mode, backend or analysis."
        ),
    },
    {
        "name": "backend-selection",
        "rules": ["C2", "D5", "T3"],
        "why": (
            "files.txt touches src/kernelforge/agent_backends/**, src/hyperloom/agents/** or "
            "common/llm_config.py; or ADD mentions ANTHROPIC_, OPENAI_, CODEX_, CLAUDE_MODEL, claude_agent_sdk, "
            "an anthropic or openai client call, calls resolve_ llm_model without default=, or sets api_key, "
            "credential or provider=."
        ),
    },
    {
        "name": "heuristic-matching",
        "rules": ["C1", "D6", "D7", "S5"],
        "why": (
            "ADD resolves identity by bidirectional containment (a in b or b in a), startswith, endswith, find "
            "or fnmatch, tests membership in trace_name, kernel_name, op_name, symbol, route, key or verdict, "
            "matches quant_method, config.json, _is_mx_fp4 or checkpoint, or returns from inside a for loop "
            "(first wins); or ADD makes an attribute per-item (per_file, per_entry, rebuild_mode) while a set, "
            "seen, dedup, sorted or comprehension key is added or deleted; or files.txt touches "
            "orchestrator/kernel/patch_*.py or executors/_*.py."
        ),
    },
    {
        "name": "argv-assembly",
        "rules": ["C1", "D9", "X5"],
        "why": (
            "ADD or DEL mentions an EXTRA_ ARGS list, extra_server_args, extra_args or passthrough, or builds a "
            "regex over a '--' flag; or ADD calls build_server_command, a build argv helper or argv "
            "extend/append, defines a _remove/_strip/_parse arg helper, guards a retry on a substring of stderr "
            "or stdout, or names --profiler-config, --moe-runner-backend or torch_profiler_dir."
        ),
    },
]

# Rules with no diff trigger: they govern how the review itself is run and published, so
# `derive` emits them for every PR instead of deriving them from a family.
ALWAYS = ["V1", "V2", "V3", "V4", "V5", "V6"]

# Why each always-on rule has no family: what it reads is not a shape of the diff. Kept
# beside ALWAYS so a rule cannot be moved into a family without this note going stale.
ALWAYS_WHY = {
    "V1": "scoped to every candidate finding and checked against base.txt; no diff shape selects it",
    "V2": "governs how a finding is reproduced and published, not what the diff contains",
    "V3": "applies to the cause the PR states, which is prose in body.txt rather than a diff shape",
    "V4": "read from openprs.txt, which describes other branches rather than this diff",
    "V5": "read from ci.txt and the mergeable field of meta.txt, not from the diff",
    "V6": "read from the closing keywords in body.txt and the linked issue, not from the diff",
}

LEDGER = "ledger.txt"
RULES_MD = "rules.md"
MIN_REASON = 40
MIN_OWN_TOKENS = 5

# Step 2. The text is the gate's restatement baseline: an answer whose only content is the
# question's own words is not an answer.
QUESTIONS = {
    "Q1": "behaviour: the behaviour this diff changes, in one sentence, as the new observable state",
    "Q2": "mechanism: how the diff produces that behaviour, cited at path:line",
    "Q3": "root cause: the cause the change removes, not the symptom it narrows",
    "Q4": "blast radius: what else must move for the change to hold, cited at path:line or a sha",
    "Q5": "falsification: what would have to be true for this reading of the diff to be wrong",
}

# Step 5. Named structural checks, because an unnamed diagnostic is a paragraph nobody can
# tell was skipped.
# Keys are the line prefixes Step 5 of SKILL.md asks for; the text is what the gate quotes
# back when one is missing. Keep the two in step -- a renamed key here silently stops
# requiring the check there.
DIAGNOSTIC_CHECKS = {
    "wiring": "an added symbol nothing consumes, or a removed caller whose helper survives",
    "twins": "mirrored code half-adapted: sibling executors, per-framework patchers, sync and async variants",
    "claims": "description, docstring, comment or prompt asserting what the code does not enforce",
    "silent-failure": "a guard, try or default converting a failed measurement, apply or write into a plausible value",
    "test-falsifies": "a test that restates the implementation or would pass unchanged on the merge base",
    "constants": "an underived timeout, budget or threshold, or an unbounded task, blocking await or subprocess",
}

# Backbone files, from the repo's tier table. Tier 1 is what kills the run as a whole or
# silently inverts its conclusion; Tier 2 is a contract with a second owner.
TIER1_FILES = (
    "src/hyperloom/inference_optimizer/cli/__init__.py",
    "src/hyperloom/orchestrator/loop/coordinator.py",
    "src/hyperloom/orchestrator/loop/writeback.py",
    "src/hyperloom/orchestrator/loop/dispatcher.py",
    "src/hyperloom/orchestrator/state/shared_state.py",
    "src/hyperloom/orchestrator/phases/machine_state.py",
    "src/hyperloom/orchestrator/policy/gate.py",
)
TIER2_FILES = (
    "src/hyperloom/orchestrator/kernel/request_handlers.py",
    "src/hyperloom/orchestrator/actions/executors/baseline.py",
    "src/hyperloom/orchestrator/actions/executors/integrate_patch.py",
    "src/hyperloom/orchestrator/actions/executors/_workload_envs.py",
    "src/hyperloom/orchestrator/actions/executors/_grid_runner.py",
    "src/hyperloom/orchestrator/phases/framework.py",
    "src/hyperloom/orchestrator/phases/kernel.py",
    "src/hyperloom/orchestrator/framework/paths.py",
    "src/hyperloom/inference_optimizer/protocol/action_surfaces.py",
    "src/hyperloom/inference_optimizer/protocol/intent.py",
    "src/hyperloom/inference_optimizer/breakdown/schema.py",
    "src/hyperloom/common/llm_config.py",
    "src/kernelforge/cli.py",
    "src/kernelforge/config.py",
)
# Q1b (name-resolved wiring) and Q2 (the number a KEEP/REVERT is made from) reach Tier 1;
# Q3 (two-owner contract) and Q4 (survives the round that caused it) reach Tier 2.
TIER1_PATTERNS = (
    "src/hyperloom/orchestrator/loop/*",
    "src/hyperloom/orchestrator/phases/*",
    "src/hyperloom/orchestrator/state/*",
    "src/hyperloom/orchestrator/scoring/*",
    "src/hyperloom/common/perf_metric.py",
    "src/hyperloom/common/gain_math.py",
)
TIER2_PATTERNS = (
    "src/hyperloom/inference_optimizer/protocol/*",
    "src/hyperloom/inference_optimizer/breakdown/*",
    "src/hyperloom/inference_optimizer/session/*",
    "src/hyperloom/inference_optimizer/multi_node/*",
    "src/hyperloom/inference_optimizer/assets/slurm/*",
    "src/hyperloom/orchestrator/actions/executors/*",
    "src/hyperloom/orchestrator/kernel/*",
    "src/hyperloom/orchestrator/framework/*",
    "*/assets/install*.sh",
)

PATH_LINE = re.compile(r"[\w./+-]+\.[A-Za-z]{1,5}:\d+")
SHA = re.compile(r"\b[0-9a-f]{7,40}\b")
CITATION = re.compile(r"[\w./+-]+\.(?:py|md|toml|yaml|yml|json|sh|rst|txt|cfg|ini|sql|in)\b")
ANCHOR_IDENT = re.compile(r"\b[a-z]+_[a-z_0-9]{2,}\b|\b[A-Z][A-Z0-9_]{3,}\b|\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b")
SURFACE = re.compile(r"^(n/?a|none|no|yes|ok|nothing|same|unchanged|see above|tbd|clean|fine|done|-+)\.?$", re.I)
STOP = frozenset(
    "the a an is are was were be been this that it its of to in on for with and or not no as at by from into "
    "than then so if but which what when where who whom how why does do did done has have had will would can "
    "could should must may might one two both each every any some all none here there their they them we i you".split()
)
ACTION_VERBS = frozenset(
    "wire drop delete add move gate rename require bump scope replace restore reject pass route remeasure "
    "document split revert fail keep set guard propagate surface reserve persist read validate escape pin "
    "reuse emit return raise test cover migrate sync anchor compare measure narrow widen forward plumb "
    "extend inline extract fold bound clamp catch log record".split()
)


def _read(work, name):
    try:
        return (Path(work) / name).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _lines(text):
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _is_comment(line):
    s = line.strip()
    return not s or s.startswith(("#", '"""', "'''", "//"))


def _tokens(text):
    return {t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in STOP and len(t) > 2}


class Artifacts:
    """The Step 1 collection, parsed once. Detection reads nothing else -- no network, no
    working tree -- so a derivation can be reproduced from $WORK alone."""

    def __init__(self, work):
        self.work = Path(work)
        self.diff = _read(work, "diff.txt")
        self.title = _read(work, "title.txt").strip()
        self.body = _read(work, "body.txt")
        self.commits = _lines(_read(work, "commits.txt"))
        self.files = _lines(_read(work, "files.txt"))
        self.testfiles = _lines(_read(work, "testfiles.txt"))
        self.numstat = []
        for ln in _lines(_read(work, "numstat.txt")):
            parts = ln.split("\t") if "\t" in ln else ln.split()
            if len(parts) >= 3:
                added = int(parts[0]) if parts[0].isdigit() else 0
                deleted = int(parts[1]) if parts[1].isdigit() else 0
                self.numstat.append((added, deleted, parts[2].strip()))
        self.hunks = parse_hunks(self.diff)
        self.per_file = {}
        for h in self.hunks:
            entry = self.per_file.setdefault(h["path"], {"add": [], "del": []})
            entry["add"].extend(h["add"])
            entry["del"].extend(h["del"])
        self.add = "\n".join(ln for h in self.hunks for ln in h["add"])
        self.dele = "\n".join(ln for h in self.hunks for ln in h["del"])
        self.text = self.add + "\n" + self.dele
        self.tokens = set(ANCHOR_IDENT.findall(self.text))

    def nontest_files(self):
        return [p for p in self.files if "/tests/" not in p and not p.startswith("tests/")]

    def src_files(self):
        return [p for p in self.nontest_files() if p.startswith("src/") and p.endswith(".py")]


def parse_hunks(diff_text):
    hunks, path, cur = [], None, None
    for line in diff_text.splitlines():
        m = re.match(r"^diff --git a/(\S+) b/(\S+)", line)
        if m:
            path, cur = m.group(2), None
            continue
        if line.startswith("@@"):
            cur = {"path": path or "", "add": [], "del": [], "ctx": []}
            hunks.append(cur)
            continue
        if cur is None or line.startswith(("+++", "---")):
            continue
        if line.startswith("+"):
            cur["add"].append(line[1:])
        elif line.startswith("-"):
            cur["del"].append(line[1:])
        elif line.startswith(" "):
            cur["ctx"].append(line[1:])
    return hunks


def _touch(a, *needles):
    return any(n in p for p in a.files for n in needles)


def _glob(a, *patterns):
    return any(fnmatch.fnmatch(p, pat) for p in a.files for pat in patterns)


def _hunk_text(h):
    return "\n".join(h["ctx"] + h["add"])


def d_fix_title(a):
    return bool(re.match(r"(?i)^(fix|perf)(\([^)]*\))?!?:", a.title) or re.match(r"(?i)^\[fix\]", a.title))


def d_unwired(a):
    if _touch(a, "orchestrator/loop/coordinator.py", "orchestrator/loop/writeback.py", "/state/shared_state.py"):
        return True
    if re.search(r"\bKNOBS\b|\bdef verdict\(|\btargets\b[^\n]*\bvalid|\bvalid[^\n]*\btargets\b", a.text):
        return True
    if re.search(r"^\s*[A-Z][A-Z0-9_]{2,}\s*(?::[^=]+)?=\s*[\"'\w]", a.add, re.M):
        return True
    if re.search(r"^\s{4,}[a-z_]\w*\s*:\s*[A-Za-z_\[\"']", a.add, re.M):
        return True
    if re.search(r"def \w+\([^)]*\b\w+\s*(?::[^=,)]+)?=\s*[^,)\s]", a.add):
        return True
    for name in set(re.findall(r"^\s*def (\w+)\(", a.add, re.M)) | set(re.findall(r"^[A-Z][A-Z0-9_]{2,}", a.add, re.M)):
        if len(re.findall(rf"(?<![\w.]){re.escape(name)}\b", a.text)) == 1:
            return True
    deleted_defs = set(re.findall(r"^\s*def (\w+)\(", a.dele, re.M))
    for call in set(re.findall(r"(?<![\w.])([a-z_]\w{3,})\(", a.dele)):
        if call not in deleted_defs and call not in ("print", "range", "len", "return", "if", "while"):
            return True
    return False


def d_signature_drift(a):
    added = set(re.findall(r"^\s*def (\w+)\(", a.add, re.M))
    if added & set(re.findall(r"^\s*def (\w+)\(", a.dele, re.M)):
        return True
    if any(_is_comment(ln) for ln in a.dele.splitlines()):
        return True
    for h in a.hunks:
        ctx = "\n".join(h["ctx"])
        if "def " in ctx and '"""' in ctx and any(not _is_comment(ln) for ln in h["add"]):
            return True
    return False


def d_explicit_field(a):
    if _touch(a, "inference_optimizer/protocol/", "orchestrator/policy/gate.py"):
        return True
    if not re.search(r"def (infer|detect|guess|derive|sniff)_\w+\(", a.dele):
        return False
    return bool(
        re.search(r"\b(mode|kind|type|format)\b\s*[:=][^=\n]*[\"']\w+[\"']", a.add)
        or re.search(r"[\"']params[\"']", a.add)
        or re.search(r"if\s+[\"']\w+[\"']\s+in\s+\w+", a.add)
    )


ENV_READ = re.compile(r"os\.environ(?:\.get\(|\[)\s*[\"']([A-Z][A-Z0-9_]{2,})|getenv\(\s*[\"']([A-Z][A-Z0-9_]{2,})")


def _new_env_names(a):
    seen = {m for pair in ENV_READ.findall(a.add) for m in pair if m}
    known = {m for pair in ENV_READ.findall(a.dele) for m in pair if m}
    ctx = "\n".join(ln for h in a.hunks for ln in h["ctx"])
    known |= {m for pair in ENV_READ.findall(ctx) for m in pair if m}
    return seen - known


def d_operator_knob(a):
    if _touch(a, "inference_optimizer/cli/", "src/kernelforge/cli.py"):
        return True
    return bool(re.search(r"add_argument\(\s*[\"']--", a.add)) or bool(_new_env_names(a))


def d_resolver_precedence(a):
    if _touch(a, "knowledge/remote_recipe/values.py", "common/env_safety.py", "/agents/kernel/tools/"):
        return True
    if re.search(
        r"os\.environ\.setdefault|GIT_AUTHOR_|GIT_COMMITTER_|user\.name|user\.email|ROCR_|HIP_VISIBLE|CUDA_VISIBLE_DEVICES",
        a.add,
    ):
        return True
    if re.search(r"def _?(select|resolve|pick)\w*\(", a.add) and re.search(r"sources?\b", a.add):
        return True
    if re.search(r"(?i)precedence|takes priority|outranks|wins over", a.add):
        return True
    for h in a.hunks:
        for i, ln in enumerate(h["add"]):
            if re.match(r"\s*for \w*(?:source|src|provider|origin)\w*\b", ln):
                if any(re.match(r"\s+for \w*(?:key|name|var)\w*\b", nxt) for nxt in h["add"][i + 1 : i + 8]):
                    return True
    cli_only = all("/cli" in p or p.endswith("parser.py") for p in a.files) if a.files else False
    return bool(ENV_READ.search(a.add)) and not cli_only


def d_executor_change(a):
    return _touch(a, "src/hyperloom/orchestrator/actions/")


def d_default_changed(a):
    return bool(
        re.search(r"default(?:_enabled)?\s*=\s*True", a.add)
        or re.search(r"add_argument\([^\n]*default\s*=\s*(?!None)[^\s,)]", a.add)
        or re.search(r"[=!]=\s*DEFAULT_[A-Z0-9_]+|DEFAULT_[A-Z0-9_]+\s*[=!]=", a.add)
        or re.search(r"_apply_\w*(profile|budget|preset)\w*\(|restore\(\s*args\b", a.add)
    )


def d_removal_or_tightening(a):
    if re.search(r"add_argument\(|\[project\.scripts\]|^\s*[A-Z][A-Z0-9_]{2,}\s*=", a.dele, re.M):
        return True
    if ENV_READ.search(a.dele):
        return True
    return bool(re.search(r"[<]=?[^=]", a.dele)) and bool(re.search(r"[=!]=|\ball\(", a.add))


def d_observable_effect_unstated(a):
    return bool(a.src_files())


def d_description_drift(a):
    if len(a.commits) > 1:
        return True
    if a.title.count("/") >= 1 and " " not in a.title:
        return True
    if not a.body.strip() or "<!--" in a.body or "- [ ]" in a.body:
        return True
    return bool(re.search(r"\d+\s*%|default[^\n]*\b\d|\bunchanged\b", a.body, re.I))


def d_prompt_surface(a):
    if _glob(a, "src/hyperloom/*prompts/*", "*/assets/system_prompts/*", "*SKILL.md", "docs/reference/*"):
        return True
    changes_surface = bool(
        re.search(r"add_argument\(\s*[\"']--|ACTION_|action_name|[\"'][\w./-]+\.(?:json|md|txt|ya?ml)[\"']", a.add)
        or _new_env_names(a)
    )
    return changes_surface


def d_persisted_schema(a):
    if _touch(a, "inference_optimizer/breakdown/", "src/kernelforge/durable_io.py"):
        return True
    if re.search(r"SCHEMA_VERSION|from_dict|ensure_schema|CREATE TABLE", a.text):
        return True
    for path, side in a.per_file.items():
        body = "\n".join(side["add"] + side["del"])
        if "SCHEMA_VERSION" in body and re.search(r"^\s+\w+\s*:\s*[A-Za-z_\[]", body, re.M):
            return bool(path)
    return False


def d_pin_bump(a):
    if re.search(r"\b(VLLM|SGLANG|ATOM|AITER|TRACELENS|MAGPIE|GEAK)_\w*(VERSION|REF|SHA|COMMIT)\b", a.text):
        return True
    if _glob(a, "*framework_deps.py", "*framework_registry.py", "*/assets/install*.sh", "docs/compatibility.rst"):
        return True
    return bool(re.match(r"(?i)^chore(\([^)]*\))?!?:.*\b(bump|pin|upgrade|update)\b", a.title))


def d_tests_touched(a):
    if a.testfiles and a.nontest_files():
        return True
    test_add = "\n".join(ln for p, s in a.per_file.items() if "/tests/" in p for ln in s["add"])
    return bool(
        re.search(r"MagicMock|\bMock\(|ctx\s*=\s*object\(\)|write_text\(|write_bytes\(", test_add)
        or re.search(r"(?<![\w.])_\w+\(", test_add)
        or re.search(r"requests\.|httpx|openai|boto3|subprocess|API_KEY|rocm-smi|/sys/|probe\s*=\s*True", test_add)
    )


def d_coverage_reduced(a):
    if re.search(r"def test_\w+|critic_agent_e2e|targeted_build_e2e", a.dele):
        return True
    if any(added == 0 and "/tests/" in path for added, _, path in a.numstat):
        return True
    for path, side in a.per_file.items():
        if "/tests/" in path:
            before = set(re.findall(r"range\((\d+)", "\n".join(side["del"])))
            after = set(re.findall(r"range\((\d+)", "\n".join(side["add"])))
            if before and after and before != after:
                return True
    big = any(added > 300 and path.startswith("src/") for added, _, path in a.numstat)
    return big and not a.testfiles


def d_broad_kill(a):
    if _glob(a, "*/assets/slurm/*", "*_subprocess_kill.py", "*/kill_multinode.py", "*/assets/install*.sh"):
        return True
    return bool(
        re.search(r"\bpkill\b|\bkillall\b|kill\s+-9|kill\s+-KILL|\bscancel\b|docker\s+(rm|kill|stop)", a.add)
        or re.search(r"HIP_VISIBLE|ROCR_|CUDA_VISIBLE_DEVICES", a.add)
        or re.search(r"os\.killpg|start_new_session|\bsetsid\b|preexec_fn|process_group", a.add)
    )


def d_destructive_selfheal(a):
    if _glob(a, "*/phases/kernel.py", "*_aiter_jit.py", "*/framework/paths.py"):
        return True
    return bool(
        re.search(r"shutil\.(rmtree|move)|os\.unlink|\.unlink\(|git\s+clean|git\s+reset\s+--hard", a.add)
        or re.search(r"def \w*(audit|reconcile|invalidate|repair|self_heal|purge)\w*\(", a.add)
        or re.search(r"\.parent\s*/|rglob\([^)]*(runs|workspace|session)", a.add)
    )


def d_cleanup_teardown(a):
    if _glob(a, "*/integrate_patch.py", "*/patch_lifecycle.py", "*/patch_landing.py", "*/common/git_safety.py"):
        return True
    return bool(
        re.search(r"git\s+checkout\s+\S+\s+--\s|check\s*=\s*False", a.add)
        or re.search(r"def \w*(teardown|rollback|cleanup|restore|release|revert|reclaim)\w*\(", a.add)
    )


def d_state_transaction(a):
    if _glob(
        a,
        "*/state/shared_state.py",
        "*/loop/writeback.py",
        "*/cli/recover.py",
        "*/common/io.py",
        "*/common/jsonio.py",
        "*/executors/_workload_envs.py",
        "*/executors/_server_argv.py",
    ):
        return True
    if re.search(r"def (record|read|seal)_\w+\(|__enter__|__exit__|@contextmanager", a.add):
        return True
    if re.search(
        r"\b(recover|resume|reclaim_stale|adopt)\w*\(|mutated\s*=|shared_state\.\w+\s*=|self\.current_best", a.add
    ):
        return True
    if re.search(r"_emit_lifecycle|\.save\(\)|\bspec\s*=|\bmanifest\s*=|\brecipe\s*=", a.add):
        return True
    if re.search(r"except Exception", a.add):
        return True
    non_cli = "\n".join(ln for p, s in a.per_file.items() if "/cli" not in p for ln in s["add"])
    return bool(re.search(r"sys\.exit\(|parse_known_args\(|parser\.error\(", non_cli))


def d_untrusted_patch(a):
    if _glob(a, "src/kernelforge/data/serving_patches/*", "*_nogit_patch.py", "*_patch_sentinel.py", "*_patcher.py"):
        return True
    if _touch(a, "kernelforge/agent_backends/"):
        return True
    return bool(re.search(r"git\s+apply|patch\s+-p\d|--unsafe-paths|extractall|unpack_archive|tar\s+-\w*x", a.add))


BLOCKING_CALL = re.compile(
    r"subprocess\.(run|check_output|call)|\.communicate\(|time\.sleep\(|requests\.|\.rglob\(|\.iterdir\(|\bglob\(|\bopen\(|read_text\(|read_bytes\("
)
PARSE_CALL = re.compile(r"json\.load|gzip\.open|yaml\.safe_load|read_text\(|hashlib|\.rglob\(|os\.walk")
HOT_CONTEXT = re.compile(
    r"with [^\n]*lock|_file_lock|fcntl|flock|def _?(promote|commit|writeback|record_keep|wait_for|poll|retry)\w*\("
)


def d_added_work_hot_path(a):
    if _touch(a, "src/hyperloom/orchestrator/loop/") or _glob(a, "*_file_lock.py"):
        return True
    for h in a.hunks:
        added = "\n".join(h["add"])
        if "async def" in _hunk_text(h) and BLOCKING_CALL.search(added):
            if not re.search(r"to_thread|run_in_executor", added):
                return True
        if PARSE_CALL.search(added) and HOT_CONTEXT.search(_hunk_text(h)):
            return True
    return False


def d_budget_overrun(a):
    if _touch(a, "src/kernelforge/gemm_tune/", "orchestrator/kernel/lane_budget.py", "phases/machine_state.py"):
        return True
    if re.search(r"_COST_S|cost_s\s*=", a.text):
        return True
    for h in a.hunks:
        added = "\n".join(h["add"])
        if re.search(r"\.(append|extend)\(", added):
            if re.search(r"(?i)limit|budget|slice", _hunk_text(h)):
                return True
            if re.search(r"(rows|shapes|tasks|candidates)\.(append|extend)\(", added):
                return True
    return False


def d_deadline_budget_scope(a):
    return bool(
        re.search(r"stream\s*=\s*True", a.add)
        or re.search(r"(subprocess\.run|create_subprocess_exec)[^\n]*timeout\s*=", a.add)
        or re.search(r"KERNEL_AGENT_GPU_PLACEMENT|\bssh\b|\bray\b|geak_submit", a.add)
        or re.search(
            r"asyncio\.wait_for|async_timeout|\b[A-Z_]*(TIMEOUT|DEADLINE|GRACE|BUDGET)[A-Z_]*\b|budget_min|--\w+-budget",
            a.text,
        )
    )


def d_metric_definition(a):
    if _touch(
        a, "common/perf_metric.py", "common/gain_math.py", "orchestrator/scoring/", "inference_optimizer/breakdown/"
    ):
        return True
    if re.search(r"speedup|gain_pct|throughput|\bttft\b|\bitl\b|\btpot\b|\bp99\b|baseline_|denominator", a.text):
        return True
    del_keys = set(re.findall(r"[\"'](\w{3,})[\"']\s*:", a.dele))
    add_keys = set(re.findall(r"[\"'](\w{3,})[\"']\s*:", a.add))
    if (del_keys - add_keys) and (add_keys - del_keys):
        return True
    return bool(re.search(r"(?i)rename|metric|denominator|axis", a.title + "\n" + a.body))


def d_cross_system_compare(a):
    if _touch(a, "executors/baseline.py", "orchestrator/kernel/conc_sweep.py"):
        return True
    if re.search(r"self_report|revalidat|current_best|incumbent", a.add):
        return True
    if re.search(r"(?i)\b(geak|magpie|aiperf|vendor|external)\w*_(result|ms|score|speedup)\b", a.add):
        return True
    return bool(re.search(r"(?i)revalidation|re-?bench|promote|cross-?check", a.title + "\n" + a.body))


def d_threshold_units(a):
    return bool(
        re.search(r"\b[A-Z][A-Z0-9_]*_(THRESHOLD|LIMIT|RATE|PCT|RATIO|BUDGET)\b", a.add)
        or re.search(r"\b\w*(THRESHOLD|LIMIT|RATE|PCT|RATIO|BUDGET)\w*\s*=\s*[\d.]+", a.add)
        or re.search(r"\berror_rate\b|\blatency_ms\b|\w+_ms\b|\b\w+_s\b\s*[=<>]|\bp99\b|percent", a.add)
    )


def d_silent_failure(a):
    if re.search(r"contextlib\.suppress|ignore_errors\s*=\s*True|except\s*\(", a.add):
        return True
    for h in a.hunks:
        for i, ln in enumerate(h["add"]):
            if re.match(r"\s*except\b", ln):
                for nxt in h["add"][i + 1 : i + 3]:
                    if re.match(r"^\s*(pass|continue|return\s*(None|False|\{\}|\[\]|\(\)|set\(\)|\"\"|'')?\s*$)", nxt):
                        return True
    return bool(re.search(r"[\"']ok[\"']\s*:\s*True", a.add))


def d_unverified_success(a):
    for h in a.hunks:
        added = "\n".join(h["add"])
        whole = _hunk_text(h)
        if "subprocess.run" in added and "check=False" in whole and "returncode" not in whole:
            return True
        if "mkdir(parents=True" in added and re.search(r"write_text\(|copy\w*\(|json\.dump", added):
            return True
        if re.search(r"compile_only|fallback", added) and re.search(r"\bresult\b|\bverdict\b|\bstatus\b", whole):
            return True
    return bool(
        re.search(r"(allclose|correctness)\w*\s*[:=]\s*(True|[\"']pass)", a.add)
        or re.search(r"\b(snr|speedup|latency_ms|tflops)\w*\s*[:=]\s*[\d.]+", a.add)
    )


def d_degraded_default(a):
    if re.search(r"class \w*(Noop|NoOp|Disabled|Null)\w*|return \w*(Noop|NoOp|Disabled|Null)\w*", a.add):
        return True
    if re.search(r"\w+_enabled\b[^\n]*None|if [^\n]*_enabled[^\n]*:\s*$", a.add) and "warning" in a.add:
        return True
    if re.search(r"HYPERLOOM_\w*STATUS", a.add):
        return True
    return bool(re.search(r"[\"'](openai|anthropic|azure|bedrock)[\"']", a.text)) and _glob(
        a, "*factory.py", "*resolver*.py", "*llm_config.py"
    )


def d_empty_vs_absent(a):
    if _glob(a, "*_gpu*.py", "*/session/paths.py", "src/kernelforge/*kb.py"):
        return True
    none_guard = set(re.findall(r"if\s+(?:not\s+)?(\w+)\s+is\s+(?:not\s+)?None", a.dele))
    falsy_guard = set(re.findall(r"if\s+not\s+(\w+)\s*[:)]", a.add))
    if none_guard & falsy_guard:
        return True
    if set(re.findall(r"if\s+not\s+(\w+)\s*[:)]", a.dele)) & set(
        re.findall(r"if\s+(?:not\s+)?(\w+)\s+is\s+(?:not\s+)?None", a.add)
    ):
        return True
    if re.search(
        r"Path\(\s*[\"']{2}\s*\)|=\s*(frozenset\(\)|set\(\)|\{\}|\[\])|\bor\s*(\{\}|\[\]|0\b|[\"']{2})", a.add
    ):
        return True
    if re.search(r"if\s+\w+\s*:\s*[^\n]*\b(parse|split|_ids)\w*\(", a.add):
        return True
    for h in a.hunks:
        for i, ln in enumerate(h["add"]):
            if re.search(r"isinstance\([^)]*(Mapping|dict|list|Sequence|tuple)\)|if \w+ is not None:", ln):
                if any("return" in nxt for nxt in h["add"][i + 1 : i + 4]):
                    return True
    return False


def d_fail_closed_gate(a):
    if _touch(
        a, "orchestrator/policy/gate.py", "orchestrator/scoring/", "common/perf_metric.py", "common/gain_math.py"
    ):
        return True
    if _glob(a, "*mapping.py"):
        return True
    deleted = a.dele.splitlines()
    for i, ln in enumerate(deleted):
        if re.match(r"\s*if not \w", ln) and any(re.search(r"\b(return|raise)\b", n) for n in deleted[i + 1 : i + 3]):
            return True
    return bool(
        re.search(r"\.get\([^)]*,\s*(0|0\.0|1|1\.0|True)\s*\)|default\s*=\s*(0|0\.0|1\.0|True)\b", a.add)
        or re.search(r"\bor\s*(\{\}|\[\]|0\b)", a.add)
        or re.search(r"(comparable|allowed|accepted)\w*\s*=.*[,|]", a.add)
    )


def d_duplicate_or_moved(a):
    if re.search(
        r"(?i)consolidate|extract shared|re-?home|\bmove\b|relocate|de-?dup|parallel path", a.title + "\n" + a.body
    ):
        return True
    for added, _, pi in a.numstat:
        for _, deleted, pj in a.numstat:
            if pi != pj and added >= 20 and deleted >= 20 and abs(added - deleted) <= 0.1 * max(added, deleted):
                return True
    for path, side in a.per_file.items():
        added_defs = set(re.findall(r"^\s*def (\w+)\(", "\n".join(side["add"]), re.M))
        for other, oside in a.per_file.items():
            if other != path and added_defs & set(re.findall(r"^\s*def (\w+)\(", "\n".join(oside["del"]), re.M)):
                return True
    if re.search(r"^\s*DEFAULT_[A-Z0-9_]+\s*=", a.add, re.M) and re.search(r"=\s*[\d\"']", a.dele):
        return True
    if re.search(
        r"def \w*(apply|deploy|revert|snapshot|cleanup|revalidate|publish)\w*\(|enable_\w+\s*=\s*False", a.add
    ):
        return True
    return any(
        "src/hyperloom/common/" in p and re.search(r"^\s*def \w+\(", "\n".join(s["add"]), re.M)
        for p, s in a.per_file.items()
    )


def d_fast_path_branch(a):
    if _glob(a, "*/integrate_patch.py", "*/executors/*patcher*.py", "*/patch_lifecycle.py"):
        return True
    return bool(
        re.search(r"pre_applied|already_applied|skip_(apply|build|rebuild|invalidate|verify|post)", a.add)
        or re.search(r"\.name\s*==|\.startswith\(|_scope_diff_to_target|os\.path\.basename|\.basename\b", a.add)
    )


def d_fallback_removed(a):
    if any(
        deleted > 0 and added == 0 and path.startswith("src/") and "/tests/" not in path
        for added, deleted, path in a.numstat
    ):
        return True
    if _glob(a, "*bypass*.py", "*_fallback*.py", "*_legacy*.py"):
        return True
    if re.search(r"(?i)bypass|fallback|legacy|alternative route", a.dele):
        return True
    return bool(
        re.search(
            r"(?i)\b(drop|remove|delete|retire)\b[^\n]*\b(route|path|fallback|mode|backend|analysis)\b",
            a.title + "\n" + a.body,
        )
    )


def d_backend_selection(a):
    if _touch(a, "src/kernelforge/agent_backends/", "src/hyperloom/agents/", "common/llm_config.py"):
        return True
    if re.search(r"ANTHROPIC_|OPENAI_|CODEX_|CLAUDE_MODEL|claude_agent_sdk", a.add):
        return True
    if re.search(r"\b(anthropic|openai)\.\w+\(|\.messages\.create\(|\.chat\.completions\.create\(", a.add):
        return True
    if re.search(r"resolve_\w*llm_model\((?![^)]*default=)", a.add):
        return True
    return bool(re.search(r"\bapi_key\s*=|\bcredential\w*\s*=|\bprovider\s*=", a.add))


def d_heuristic_matching(a):
    if _glob(a, "*/orchestrator/kernel/patch_*.py", "*/executors/_*.py"):
        return True
    if re.search(r"\b(\w+)\s+in\s+(\w+)\s+or\s+\2\s+in\s+\1\b", a.add):
        return True
    if re.search(r"\.startswith\(|\.endswith\(|\.find\(|fnmatch", a.add):
        return True
    if re.search(r"\bin\s+(trace_name|kernel_name|op_name|symbol|route|key|verdict)\b", a.add):
        return True
    if re.search(r"quant_method|config\.json|_is_mx_fp4|checkpoint", a.add):
        return True
    for h in a.hunks:
        for i, ln in enumerate(h["add"]):
            if re.match(r"\s*for \w+", ln) and any(re.match(r"\s+return\b", n) for n in h["add"][i + 1 : i + 6]):
                return True
    per_item = re.search(r"per_file|per_entry|rebuild_mode", a.add)
    key_moved = re.search(r"\bset\(|\bseen\b|\bdedup|\bsorted\(|\bkey\s*=", a.text)
    return bool(per_item and key_moved)


def d_argv_assembly(a):
    if re.search(r"EXTRA_\w*ARGS|extra_server_args|extra_args|passthrough", a.text):
        return True
    if re.search(r"re\.(compile|sub|search|match)\([^\n]*--", a.add):
        return True
    if re.search(
        r"build_server_command|build_\w*argv|argv\.(extend|append)\(|def _?(remove|strip|parse)_\w*arg", a.add
    ):
        return True
    if re.search(r"(stderr|stdout)[^\n]*\bin\b[^\n]*:|if [\"'][^\"']+[\"'] in (stderr|stdout)", a.add):
        return True
    return bool(re.search(r"--profiler-config|--moe-runner-backend|torch_profiler_dir", a.add))


DETECT = {
    "fix-title": d_fix_title,
    "unwired-or-unhonoured": d_unwired,
    "signature-or-doc-drift": d_signature_drift,
    "explicit-field-contract": d_explicit_field,
    "operator-knob-added": d_operator_knob,
    "resolver-precedence": d_resolver_precedence,
    "executor-change": d_executor_change,
    "default-changed": d_default_changed,
    "removal-or-tightening": d_removal_or_tightening,
    "observable-effect-unstated": d_observable_effect_unstated,
    "description-drift": d_description_drift,
    "prompt-surface": d_prompt_surface,
    "persisted-schema": d_persisted_schema,
    "pin-bump": d_pin_bump,
    "tests-touched": d_tests_touched,
    "coverage-reduced": d_coverage_reduced,
    "broad-kill": d_broad_kill,
    "destructive-selfheal": d_destructive_selfheal,
    "cleanup-teardown": d_cleanup_teardown,
    "state-transaction": d_state_transaction,
    "untrusted-patch": d_untrusted_patch,
    "added-work-hot-path": d_added_work_hot_path,
    "budget-overrun": d_budget_overrun,
    "deadline-and-budget-scope": d_deadline_budget_scope,
    "metric-definition": d_metric_definition,
    "cross-system-compare": d_cross_system_compare,
    "threshold-units": d_threshold_units,
    "silent-failure": d_silent_failure,
    "unverified-success": d_unverified_success,
    "degraded-default": d_degraded_default,
    "empty-vs-absent": d_empty_vs_absent,
    "fail-closed-gate": d_fail_closed_gate,
    "duplicate-or-moved-code": d_duplicate_or_moved,
    "fast-path-branch": d_fast_path_branch,
    "fallback-removed": d_fallback_removed,
    "backend-selection": d_backend_selection,
    "heuristic-matching": d_heuristic_matching,
    "argv-assembly": d_argv_assembly,
}


def derive_families(a):
    return [fam for fam in FAMILIES if DETECT[fam["name"]](a)]


def cmd_derive(args):
    a = Artifacts(args.work)
    if not a.diff.strip() and not a.files:
        print(f"{args.work}/diff.txt and files.txt are both empty -- run fetch.sh first", file=sys.stderr)
        return 2
    fired = derive_families(a)
    rules = []
    for fam in fired:
        for rid in fam["rules"]:
            if rid not in rules:
                rules.append(rid)
    rules += [r for r in ALWAYS if r not in rules]
    rules.sort(key=lambda r: (r[0], int(r[1:])))
    out = ["# derived by triage.py derive -- do not edit; rerun derive"]
    for fam in fired:
        out.append(f"FAMILY {fam['name']} {' '.join(fam['rules'])}")
    out.append(f"FAMILY always-on {' '.join(ALWAYS)}")
    out.append("RULES " + " ".join(rules))
    (Path(args.work) / "rules.txt").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    width = max([len(f["name"]) for f in fired] + [len("always-on")])
    print(f"{len(fired)} families derived from {len(a.files)} changed files, {len(rules)} rules to adjudicate")
    for fam in fired:
        print(f"  {fam['name']:<{width}}  {' '.join(fam['rules'])}")
    print(f"  {'always-on':<{width}}  {' '.join(ALWAYS)}")
    print(f"wrote {Path(args.work) / 'rules.txt'}")
    return 0


def derived_rules(work):
    ids = []
    for ln in _read(work, "rules.txt").splitlines():
        if ln.startswith("RULES "):
            ids += [r for r in ln.split()[1:] if re.fullmatch(r"[A-Z]\d+", r)]
    return ids


def cmd_rules(args):
    ids = derived_rules(args.work)
    if not ids:
        print(
            f"{args.work}/rules.txt has no RULES line -- run `triage.py derive --work {args.work}` first",
            file=sys.stderr,
        )
        return 2
    print("\n".join(ids))
    return 0


def rule_bodies():
    """{rule id: (category heading, body lines)} sliced out of rules.md by heading."""
    path = Path(__file__).resolve().parent / RULES_MD
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    starts, category, category_of = [], None, {}
    for i, ln in enumerate(lines):
        if ln.startswith("## ") and not ln.startswith("### "):
            category = ln
        m = re.match(r"^### ([A-Z]\d+)\b", ln)
        if m:
            starts.append((i, m.group(1)))
            category_of[m.group(1)] = category
    out = {}
    for (i, rid), (j, _) in zip(starts, starts[1:] + [(len(lines), "")]):
        k = j
        while k > i and not lines[k - 1].strip():
            k -= 1
        out[rid] = (category_of[rid], lines[i:k])
    return out


def cmd_expand(args):
    want = derived_rules(args.work)
    if not want:
        print(
            f"{args.work}/rules.txt has no RULES line -- run `triage.py derive --work {args.work}` first",
            file=sys.stderr,
        )
        return 2
    bodies = rule_bodies()
    if not bodies:
        print(
            f"cannot read {Path(__file__).resolve().parent / RULES_MD} -- the rule bodies live there", file=sys.stderr
        )
        return 2
    out, last = [], None
    for rid in want:
        if rid not in bodies:
            continue
        category, body = bodies[rid]
        if category and category != last:
            last = category
            out += [category, ""]
        out += body + [""]
    missing = [r for r in want if r not in bodies]
    print("\n".join(out).rstrip())
    if missing:
        print(
            f"\nNOT IN {RULES_MD}: {' '.join(missing)} -- the deriver names a rule the rule file does not define",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_mapping(_args):
    rows = [(f["name"], " ".join(f["rules"]), f["why"]) for f in FAMILIES]
    ids = sorted({r for _, rules, _ in rows for r in rules.split()} | set(ALWAYS), key=lambda r: (r[0], int(r[1:])))
    print("# Family -> rule mapping")
    print()
    print("Generated by `triage.py mapping`. Do not edit -- regenerate with `python3 triage.py mapping > MAPPING.md`")
    print("and diff, which is how a disagreement between this file and the deriver is caught.")
    print()
    print(f"{len(rows)} families, {len(ids)} distinct rule ids, {len(ALWAYS)} of them always on.")
    print()
    print("| family | rules | fires when |")
    print("|---|---|---|")
    for name, rules, why in rows:
        print(f"| {name} | {rules} | {why} |")
    print(f"| always-on | {' '.join(ALWAYS)} | every review; these govern how the review is run and published |")
    print()
    print("Always on, never derived from a family:")
    print()
    for rule in ALWAYS:
        print(f"- **{rule}** -- {ALWAYS_WHY[rule]}.")
    bodies = rule_bodies()
    if bodies:
        unreachable = sorted(set(bodies) - set(ids), key=lambda r: (r[0], int(r[1:])))
        print()
        if unreachable:
            print(
                f"Unreachable: {' '.join(unreachable)} -- defined in {RULES_MD} and emitted by no family, so no review ever reads them."
            )
        else:
            print(f"Every rule in {RULES_MD} is reachable from at least one family.")
    return 0


def _sha256(path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def ledger_entries(work):
    out = []
    for ln in _read(work, LEDGER).splitlines():
        parts = ln.split(None, 1)
        if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            out.append((parts[0], parts[1].strip()))
    return out


def ledger_record(work, name):
    digest = _sha256(Path(work) / name)
    if digest is None or any(n == name for _, n in ledger_entries(work)):
        return
    try:
        with open(Path(work) / LEDGER, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(f"{digest}  {name}\n")
    except OSError as exc:
        print(f"WARNING: ledger.txt not written for {name}: {exc}", file=sys.stderr)


def ledger_drift(work):
    """Artifacts whose bytes no longer match the hash recorded when their gate passed.

    First entry wins: a most-recent-wins ledger would leave the back door open in a new
    shape -- edit the artifact, rerun its gate, walk through on a fresh hash."""
    problems, seen = [], set()
    for digest, name in ledger_entries(work):
        if name in seen:
            continue
        seen.add(name)
        path = Path(work) / name
        if not path.is_file():
            problems.append((name, "recorded in ledger.txt and now missing"))
        elif _sha256(path) != digest:
            problems.append(
                (
                    name,
                    "bytes changed after its gate passed -- a finding that arrived late is a free-form finding that goes through refutation, not an edit to an artifact that already went green",
                )
            )
    return problems


def _require(work, name):
    path = Path(work) / name
    if not path.is_file():
        return None, [(name, f"missing -- {name} is the artifact this gate reads; the step that writes it did not run")]
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return None, [(name, "empty -- an empty artifact is a step that did not happen, not a step that found nothing")]
    return text, []


def _anchored(reason, a):
    if CITATION.search(reason) or PATH_LINE.search(reason):
        return True
    if any(p.rsplit("/", 1)[-1] in reason for p in a.files):
        return True
    return bool(set(ANCHOR_IDENT.findall(reason)) & a.tokens)


def gate_answers(work):
    text, problems = _require(work, "answers.txt")
    if problems:
        return problems
    seen = {}
    for ln in text.splitlines():
        m = re.match(r"^\s*(Q[1-5])\s*[:\-]\s*(.+?)\s*$", ln, re.I)
        if m:
            seen.setdefault(m.group(1).upper(), []).append(m.group(2).strip())
    for key, prompt in QUESTIONS.items():
        if key not in seen:
            problems.append((key, f"unanswered -- write `{key}: <answer>` covering {prompt}"))
            continue
        answer = " ".join(seen[key])
        if SURFACE.match(answer):
            problems.append((key, f"answered with `{answer[:30]}` -- {prompt}"))
            continue
        own = _tokens(answer) - _tokens(prompt)
        if len(own) < MIN_OWN_TOKENS:
            problems.append((key, f"restates the question and adds nothing -- {prompt}"))
            continue
        if key in ("Q2", "Q4") and not (PATH_LINE.search(answer) or SHA.search(answer)):
            problems.append((key, "cites no path:line and no sha -- name where in the tree you read this"))
    return problems


def tier_of(path):
    if path in TIER1_FILES:
        return "1"
    if path in TIER2_FILES:
        return "2"
    if any(fnmatch.fnmatch(path, pat) for pat in TIER1_PATTERNS):
        return "1"
    if any(fnmatch.fnmatch(path, pat) for pat in TIER2_PATTERNS):
        return "2"
    return None


def gate_corefiles(work):
    text, problems = _require(work, "core_files.txt")
    if problems:
        return problems
    a = Artifacts(work)
    required = {p: tier_of(p) for p in a.nontest_files() if tier_of(p)}
    seen, assessed = {}, 0
    # A docs-, CI- or test-only PR has no file to tier. It still records that, so the empty
    # artifact cannot be confused with Step 3 never having run.
    if not a.src_files():
        if CORE_NONE.search(text):
            return problems
        return [
            (
                "core_files.txt",
                "no non-test file under src/ in this diff -- write one `NONE -- <reason naming what the PR does touch>` line",
            )
        ]
    for ln in text.splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        m = re.match(r"^\s*([\w./+=-]+)\s+TIER([123])\s+(COVERED|GAP|N/A)\s*(?:--|:)\s*(.+?)\s*$", ln)
        if not m:
            problems.append((ln.strip()[:70], "malformed -- expected `<path> TIER<1|2|3> COVERED|GAP|N/A -- <reason>`"))
            continue
        path, tier, verdict, reason = m.groups()
        assessed += 1
        seen[path] = tier
        if not any(t == path or t.endswith("/" + path) for t in a.files):
            problems.append(
                (
                    path,
                    "assessed but absent from files.txt -- a risk assessment of a file the PR does not change is stale or invented",
                )
            )
            continue
        if path in required and tier != required[path]:
            problems.append(
                (
                    path,
                    f"recorded TIER{tier}; the backbone table and the tiering questions put it at TIER{required[path]} -- downgrading a tier is not a way past the checks that tier requires",
                )
            )
        if len(reason) < MIN_REASON or SURFACE.match(reason):
            problems.append(
                (path, f"marked {verdict} with no reason -- what breaks, and what in this PR makes you say it does not")
            )
        elif not _anchored(reason, a):
            problems.append(
                (
                    path,
                    "the reason names no file, symbol or line this PR changes -- it would read the same against any PR touching this file",
                )
            )
    for path, tier in sorted(required.items()):
        if path not in seen:
            problems.append(
                (path, f"TIER{tier} file in this diff with no assessment line -- Step 3 was not performed for it")
            )
    want = len([p for p in a.src_files()])
    if assessed < want:
        problems.append(
            (
                "core_files.txt",
                f"{assessed} assessment line(s) for {want} changed non-test file(s) under src/ -- every one needs a tier, including the Tier 3 ones",
            )
        )
    return problems


def parse_verdicts(work):
    out = {}
    for ln in _read(work, "verdicts.txt").splitlines():
        m = re.match(r"^\s*([A-Z]\d+)\s+(FIRE|CLEAR|N/A)\s*(?:--|:)\s*(.+?)\s*$", ln)
        if m:
            out[m.group(1)] = (m.group(2), m.group(3))
    return out


def gate_verdicts(work):
    text, problems = _require(work, "verdicts.txt")
    if problems:
        return problems
    want = derived_rules(work)
    if not want:
        return [("rules.txt", f"no RULES line -- run `triage.py derive --work {work}` before adjudicating")]
    a = Artifacts(work)
    seen = {}
    for ln in text.splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        m = re.match(r"^\s*([A-Z]\d+)\s+(FIRE|CLEAR|N/A)\s*(?:--|:)\s*(.+?)\s*$", ln)
        if not m:
            problems.append((ln.strip()[:70], "malformed -- expected `<RULE-ID> FIRE|CLEAR|N/A -- <reason>`"))
            continue
        rid, verdict, reason = m.groups()
        if rid in seen:
            problems.append((rid, "adjudicated twice -- one line per rule"))
            continue
        seen[rid] = (verdict, reason)
        if rid not in want:
            problems.append(
                (rid, "not in rules.txt -- adjudicating a rule the diff did not earn hides which ones it did")
            )
            continue
        if len(reason) < MIN_REASON or SURFACE.match(reason):
            problems.append(
                (
                    rid,
                    f"marked {verdict} with a {len(reason)}-character reason -- say what in the diff you read and what it showed",
                )
            )
        elif verdict == "CLEAR" and not _anchored(reason, a):
            problems.append(
                (
                    rid,
                    "CLEAR with a reason that names nothing from the diff -- a clearance that would read the same on any PR is not one",
                )
            )
    for rid in want:
        if rid not in seen:
            problems.append((rid, "derived and never adjudicated -- Step 4 skipped it"))
    return problems


def gate_diagnostic(work):
    text, problems = _require(work, "ai_diagnostic.txt")
    if problems:
        return problems
    a = Artifacts(work)
    seen = {}
    for ln in text.splitlines():
        m = re.match(r"^\s*([a-z-]+)\s*[:=]\s*(CLEAN|HIT)\s*(?:--|:)\s*(.+?)\s*$", ln)
        if m:
            seen[m.group(1)] = (m.group(2), m.group(3))
    for name, what in DIAGNOSTIC_CHECKS.items():
        if name not in seen:
            problems.append((name, f"not reported -- write `{name}: CLEAN|HIT -- <reason>`; the check is {what}"))
            continue
        _verdict, reason = seen[name]
        if len(reason) < MIN_REASON or SURFACE.match(reason):
            problems.append((name, f"reported with a {len(reason)}-character reason -- name what you looked at"))
        elif not _anchored(reason, a):
            problems.append(
                (name, "the reason names nothing from the diff -- say which added symbol, file or line you checked")
            )
    return problems


def parse_blocks(text):
    """`FINDING: <key>` blocks, each holding the lines up to the next FINDING."""
    blocks, key = {}, None
    for ln in text.splitlines():
        m = re.match(r"^\s*FINDING\s*:\s*(\S+)\s*$", ln)
        if m:
            key = m.group(1)
            blocks[key] = []
            continue
        if key:
            blocks[key].append(ln)
    return blocks


def _refute_problems(key, body, a, label):
    problems = []
    attempt = ""
    outcome = None
    for ln in body:
        m = re.match(r"^\s*ATTEMPT\s*:\s*(.+?)\s*$", ln)
        if m:
            attempt = m.group(1)
        m = re.match(r"^\s*OUTCOME\s*:\s*(SURVIVES|DROPPED)\s*(?:--|:)\s*(.+?)\s*$", ln)
        if m:
            outcome = m.group(1)
    if not attempt:
        problems.append(
            (key, f"no `ATTEMPT: <what you did to kill it>` line -- {label} is the attempt, not the conclusion")
        )
    elif len(attempt) < MIN_REASON or SURFACE.match(attempt) or not _anchored(attempt, a):
        problems.append(
            (key, "the attempt names no file, symbol or command -- a refutation that cannot be repeated is not one")
        )
    if outcome is None:
        problems.append((key, "no `OUTCOME: SURVIVES|DROPPED -- <reason>` line"))
    return problems, outcome


def gate_refutations(work):
    text, problems = _require(work, "refutations.txt")
    if problems:
        return problems
    a = Artifacts(work)
    fired = [rid for rid, (verdict, _) in parse_verdicts(work).items() if verdict == "FIRE"]
    blocks = parse_blocks(text)
    for key, body in blocks.items():
        found, _outcome = _refute_problems(key, body, a, "self-refutation")
        problems += found
    for rid in fired:
        if rid not in blocks:
            problems.append(
                (
                    rid,
                    "adjudicated FIRE with no refutation block -- write `FINDING: <id>` with an ATTEMPT and an OUTCOME, or change the verdict",
                )
            )
    if not blocks and not problems:
        # No candidate to refute is only credible when nothing fired. Saying so costs one
        # line and keeps an empty file from reading the same as a skipped step.
        if fired:
            problems.append(("refutations.txt", "no FINDING block -- Step 6 writes one per candidate finding"))
        elif not NO_CANDIDATES.search(text):
            problems.append(
                (
                    "refutations.txt",
                    "no FINDING block and no `FINDING: none -- <reason>` line -- an empty file reads the same as a skipped step",
                )
            )
    return problems


def gate_independent(work):
    text, problems = _require(work, "independent.txt")
    if problems:
        return problems
    a = Artifacts(work)
    survived = [k for k, v in parse_blocks(_read(work, "refutations.txt")).items() if _survives(v)]
    fired = {rid for rid, (verdict, _) in parse_verdicts(work).items() if verdict == "FIRE"}
    blocks = parse_blocks(text)
    for key, body in blocks.items():
        found, _outcome = _refute_problems(key, body, a, "independent refutation")
        problems += found
    for key in survived:
        if key in fired or key.startswith("free:"):
            if key not in blocks:
                problems.append(
                    (
                        key,
                        "survived self-refutation and no independent reader judged it -- Step 7 needs a block per surviving finding, or the finding comes off the card",
                    )
                )
    return problems


def _survives(body):
    return any(re.match(r"^\s*OUTCOME\s*:\s*SURVIVES\b", ln) for ln in body)


def gate_ledger(work):
    entries = ledger_entries(work)
    if not entries:
        return [
            (
                LEDGER,
                "no entries -- gates record an artifact hash as they pass; an empty ledger means no gate has passed yet",
            )
        ]
    return [(name, why) for name, why in ledger_drift(work)]


CARD_HEAD = re.compile(r"^\s*(\d+)\.\s*\[([^\]]+)\]\s*(.+?)\s*$")
CARD_NONE = re.compile(r"^\s*Blocking issues\s*:\s*none\s*$", re.I | re.M)
NO_CANDIDATES = re.compile(r"^\s*FINDING\s*:\s*none\s*(?:--|:)\s*\S", re.I | re.M)
CARD_PART = re.compile(r"^\s*(Problem|Impact|Action)\s*:", re.I)
CORE_NONE = re.compile(r"^\s*NONE\s*(?:--|:)\s*\S", re.I | re.M)
CARD_TEMPLATE = "1. [<RULE-ID|free:slug>] <one-line summary> [verified|inferred]\n   Problem: <what is wrong> path/to/file.py:123\n   Impact: <what it costs at runtime>\n   Action: <verb> <what to change>"


def gate_card(work):
    text, problems = _require(work, "card.md")
    if problems:
        return problems
    a = Artifacts(work)
    verdicts = parse_verdicts(work)
    self_ref = parse_blocks(_read(work, "refutations.txt"))
    indep = parse_blocks(_read(work, "independent.txt"))
    findings, order = {}, []
    current = None
    for ln in text.splitlines():
        m = CARD_HEAD.match(ln)
        if m:
            current = m.group(2).strip()
            order.append(current)
            findings[current] = [ln]
            continue
        if current is not None and re.match(r"^\s*#{1,3} ", ln):
            current = None
            continue
        if current is not None:
            findings[current].append(ln)
    deferred = set(re.findall(r"^\s*deferred:\s*(\S+)\s*(?:--|:)\s*.+$", text, re.M))
    if not order:
        # A clean review is a real outcome, not a skipped step: the card says so in as many
        # words, and the surviving-but-unreported scan below still runs, so "none" cannot be
        # used to drop a finding that survived both refutation passes.
        if not CARD_NONE.search(text):
            problems.append(
                (
                    "card.md",
                    f"no finding parsed -- write `Blocking issues: none` if the review is clean, otherwise a finding starts with\n{CARD_TEMPLATE}",
                )
            )
            return problems
    if len(order) > 5:
        problems.append(
            (
                "card.md",
                f"{len(order)} findings -- the card holds at most 5; the rest are `deferred: <key> -- <reason>` lines",
            )
        )
    for key in order:
        body = "\n".join(findings[key])
        if not PATH_LINE.search(body):
            problems.append((key, "cites no path:line -- a finding a reader cannot open is not actionable"))
        for label in ("Problem", "Impact"):
            if not re.search(rf"^\s*{label}\s*:\s*\S", body, re.M):
                problems.append(
                    (key, f"no `{label}:` line -- every finding states the problem, the impact and the action")
                )
        action = re.search(r"^\s*Action\s*:\s*(\w+)", body, re.M)
        if not action:
            problems.append((key, "no `Action:` line -- say what to change"))
        elif action.group(1).lower().rstrip("s") not in ACTION_VERBS and action.group(1).lower() not in ACTION_VERBS:
            problems.append(
                (
                    key,
                    f"`Action: {action.group(1)}` does not start with an action verb -- name the edit, not the concern",
                )
            )
        # The head line carries the tag for the Problem/Impact/Action trio it introduces. Any
        # further line citing its own path:line is a second claim and is tagged separately.
        for ln in findings[key][1:]:
            if CARD_PART.match(ln) or not PATH_LINE.search(ln):
                continue
            if not re.search(r"\[(verified|inferred)\]", ln):
                problems.append((key, f"claim `{ln.strip()[:50]}` carries no [verified] or [inferred] tag"))
        if not re.search(r"\[(verified|inferred)\]", body):
            problems.append((key, "no [verified] or [inferred] tag anywhere in the finding"))
        if key.startswith("free:"):
            if key not in self_ref or not _survives(self_ref[key]):
                problems.append(
                    (
                        key,
                        "free-form finding with no surviving block in refutations.txt -- it goes through the same two refutation passes as a rule finding",
                    )
                )
            if key not in indep or not _survives(indep[key]):
                problems.append((key, "free-form finding no independent reader cleared"))
            continue
        if verdicts.get(key, ("", ""))[0] != "FIRE":
            problems.append(
                (
                    key,
                    "reported without a FIRE verdict in verdicts.txt -- mark it `free:<slug>` and refute it like one, or adjudicate the rule",
                )
            )
            continue
        if key not in self_ref or not _survives(self_ref[key]):
            problems.append((key, "did not survive self-refutation in refutations.txt"))
        if key not in indep or not _survives(indep[key]):
            problems.append((key, "did not survive independent refutation in independent.txt"))
    for key, body in self_ref.items():
        if not _survives(body):
            continue
        if key in order or key in deferred:
            continue
        if key in indep and not _survives(indep[key]):
            continue
        problems.append(
            (key, "survived both refutation passes and is on neither the card nor a `deferred: <key> -- <reason>` line")
        )
    if not _anchored(text, a):
        problems.append(("card.md", "names no file or symbol this PR changes"))
    return problems


GATES = {
    "answers": ("answers.txt", gate_answers),
    "corefiles": ("core_files.txt", gate_corefiles),
    "verdicts": ("verdicts.txt", gate_verdicts),
    "diagnostic": ("ai_diagnostic.txt", gate_diagnostic),
    "refutations": ("refutations.txt", gate_refutations),
    "independent": ("independent.txt", gate_independent),
    "ledger": (LEDGER, gate_ledger),
    "card": ("card.md", gate_card),
}


def cmd_gate(args):
    work = Path(args.work)
    if not work.is_dir():
        print(f"{work} is not a directory -- $WORK holds the artifacts every gate reads", file=sys.stderr)
        return 1
    artifact, fn = GATES[args.name]
    problems = []
    if args.name != "ledger":
        problems += ledger_drift(work)
    problems += fn(work)
    if problems:
        print(f"GATE {args.name} FAILED: {len(problems)} problem(s) in {artifact}", file=sys.stderr)
        for what, why in problems:
            print(f"  {what}: {why}", file=sys.stderr)
        return 1
    if args.name == "ledger":
        print(
            f"GATE ledger PASSED: {len(ledger_entries(work))} artifact(s) still match the hash recorded at their gate"
        )
        return 0
    ledger_record(work, artifact)
    print(f"GATE {args.name} PASSED: {artifact} recorded in {LEDGER}")
    return 0


def main(argv=None):
    # rules.md and PR bodies carry non-ASCII; a Windows console defaults to cp1252 and would
    # abort `expand` mid-stream on the first arrow.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        prog="triage.py", description="Derive the rule set for a PR and gate each step of the review."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, help_text in (
        ("derive", "detect the families this diff earns and write rules.txt"),
        ("expand", "print the bodies of the derived rules"),
        ("rules", "print the derived rule ids, one per line"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--work", required=True, help="the $WORK directory fetch.sh filled")
    sub.add_parser("mapping", help="print MAPPING.md, generated from the family table")
    p = sub.add_parser("gate", help="check that one review step actually happened")
    p.add_argument("name", choices=sorted(GATES))
    p.add_argument("--work", required=True, help="the $WORK directory fetch.sh filled")
    args = parser.parse_args(argv)
    return {"derive": cmd_derive, "expand": cmd_expand, "rules": cmd_rules, "mapping": cmd_mapping, "gate": cmd_gate}[
        args.cmd
    ](args)


if __name__ == "__main__":
    sys.exit(main())
