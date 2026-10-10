# Hyperloom review rules

Rule bodies for the `review-pr` skill. Do not read this file top to bottom during a review: use the
index below to pick the rules the diff at hand can actually break, then read only those bodies.

Every rule here was clustered from real review history on this repository. `Seen in` cites the
PR the rule was learned from.

Severity is `blocking` under one of the three kinds SKILL.md's output contract names: the current
behaviour is wrong, something that had to move in the same PR did not (title, description,
docstring), or the diff breaks a rule `AGENTS.md` states outright. Everything else is `advisory`:
it is adjudicated in `verdicts.txt` like any other rule and then **not published**, because a
review that ships it has stopped being two parts. An advisory `FIRE` is a note for the reviewer
answering a later question from the author, not a line on the card.

Rules that duplicate a static gate (ruff, pylint, bandit, CodeQL, gitleaks, REUSE) or that restate
[`AGENTS.md`](../../../AGENTS.md) are deliberately absent. See
[`.github/copilot-instructions.md`](../../../.github/copilot-instructions.md) for the prose
statement of the same review posture.

---

## Index

Pick every row whose left column matches the changed files and a skim of the diff, then read only
the rules listed. Rows overlap; a rule listed twice is read once.

| The diff... | Read |
|---|---|
| changes any non-test file under `src/` | X1 X3 D12 |
| adds or changes a value that crosses a boundary: status literal, enum member, dataclass or TypedDict field, keyword argument, signature, return semantics | C1 C3 C5 |
| adds or changes a knob or a pin: CLI flag, env var, config key, default value, a pinned external version, ref or sha, an install script, `docs/compatibility.rst`, or the argv or extra-args list one is assembled into | C3 C4 S6 T4 X5 X7 D8 D9 D11 |
| removes a flag, env var, enum member, test, fallback/legacy/bypass route or whole file, or tightens a comparison (`<` returns as `==`, a new `all(...)`) | C3 T2 X4 D3 |
| fixes one site of an operation that has siblings (executors, per-framework patchers, sync and async twins), or moves, copies or consolidates code | C2 T4 D1 D2 D4 D11 |
| adds a second implementation of an operation the repo already owns (patch deploy, revert, snapshot, cleanup, revalidation), or a `pre_applied`/`skip_*`/already-done branch that short-circuits one | D1 D2 D10 |
| defines, outside the module that owns the concern, a constant, precedence list, parser or client constructor the owner exports: an LLM model, SDK client, API key, base URL or header read outside `llm_config.py`, a backend registered outside `agent_backends/registry.py` | D5 D10 |
| repairs a value where it is consumed: a local re-merge of `os.environ`, a second parse of a handoff blob or serialized mapping, or any re-derivation of something another module already publishes | D11 V3 |
| touches persisted or shared state: `SCHEMA_VERSION`, `from_dict`, `CREATE TABLE`, a `record_*`/`read_*`/`seal_*` pair, `.save()`, a spec, manifest or recipe, a context manager, recovery or resume | X6 R4 P4 P5 |
| touches a prompt, `SKILL.md`, `docs/**`, `*.md` or `*.rst` | X3 X5 |
| adds error handling or a default: `except`, `contextlib.suppress`, `ignore_errors=True`, `.get(k, 0)`, `or {}`, an early `isinstance` guard, a noop or degraded implementation | S1 S2 S3 S4 S5 S7 |
| touches tests, or adds a large `src/` module with no matching test file | T1 T2 T3 |
| touches async code, a lock, a subprocess, a poll or retry loop, a per-candidate step, a timeout or a budget | P1 P2 P3 P6 |
| touches a metric, score, speedup or gain denominator, or a threshold compared against a number this repo did not produce | M1 M2 M3 |
| matches or selects by name: substring, `startswith`, `fnmatch`, a first-wins loop, a dedup/grouping/sort key, or an LLM backend, model or credential choice | D5 D6 D7 |
| runs a destructive or privileged command: `pkill`/`kill -9`/`scancel`/`docker rm`, `rmtree`/`unlink`/`move`, `git reset --hard`/`git checkout -- <path>`, `git apply`, `tar -x`/`extractall`, or a cleanup, teardown or self-heal step | R1 R2 R3 R5 |
| imports from another package, adds an `_internal`/`_`-prefixed import, reads or writes another object's `_attr`, uses `getattr(obj, "_x")`, or adds an exemption to an import check | A1 A2 |
| adds a string the runtime imports (`import_module`, `-m`, `"pkg.mod:attr"`), a module-level hook a higher layer sets, or an import under `if TYPE_CHECKING:` | A2 |
| adds a helper that uses only stdlib (io, json, stat, hashing, subprocess, retry, time), or edits a module whose name says utility (`*utils*`, `io`, `jsonio`, `proctree`, `timeutil`, `_subprocess_kill`) | A3 |
| adds a new module, or adds methods, fields or imports to a module or class already over a size trigger | A5 |
| extracts a function from a long one, or cites the complexity or size trigger as the reason for a split | A6 |
| branches on a framework, backend, arch, decision, status or phase name, or adds a dict keyed by one | A7 A10 |
| adds or changes a function signature or return annotation, writes a key or attribute on a parameter, or changes call sites that unpack one object | A8 A9 A11 |
| adds a `Protocol`, `ABC`, base class, factory, registry, or a function whose body is one forwarding call | A12 |
| adds an `isinstance`/`is None`/`or {}` guard on an internal value, a renaming local, or an `except ...: raise` | A13 |
| adds or renames a function, parameter, module binding or import alias | A14 |
| reaches through two or more collaborators (`a.b.c.method()`), or adds a module function that mostly reads one parameter's fields | A4 |
| adds a `global` statement, or a query-named function (`is_`/`get_`/`resolve_`/property) that writes | A11 |

V1-V6 apply to every PR: they govern how the review is run and published, not what the diff
contains. X2 does too — it reads `title.txt`, `body.txt` and `commits.txt`, so no row can trigger
it from the file list, and a docs- or CI-only PR whose description went stale is the case a
file-shaped row would miss.

---

## C -- Change completeness

### C1 -- A new or changed value must reach and be honoured by every consumer

**Severity:** blocking
**Fires when:** the diff adds a status literal, enum member, dataclass field or keyword argument, or changes a signature or return semantics, and the symbol has call sites the diff does not touch.
**The rule:** every consumer honours the new value: membership tests and accept-sets that were exhaustive before now admit it, every production call site passes the new argument, and caller-side path or index arithmetic matches the redefined return. A value no consumer accepts is inert, and the tests stay green because they exercise the producer.
**Seen in:** PR #1527 -- the `grading` argument added to `build_agentx_workload_spec` was never passed at its only production call site, so that part of the fix did not take effect.
**Not a finding when:** the untouched consumers are tests or dead code, the accept-set rejects the new value deliberately (an explicit reject branch or raise counts as honouring it), or the consumer dispatches by string key so the grep misses it -- resolve the key by name before firing.
**Evidence:** `$WORK/diff.txt` for the added value and the changed signature; `$WORK/files.txt` to confirm which consumers lie outside the diff.
**Report as:** `C1 <file>:<line> -- <new value> is not honoured by <consumer>; <observable effect>`

### C2 -- A fix must cover every path that can produce the condition

**Severity:** blocking
**Fires when:** `title.txt` carries a `fix` prefix and the diff changes one invocation site, handler or executor of an operation that has siblings elsewhere in the tree.
**The rule:** the fix covers every producer of the condition -- direct injection, inheritance from a baseline recipe, authored variants, sweep and explore loops, and the alternate runtimes (Ray, CLI, SSH placement, pod-side runner). A per-invocation fix such as `git -c`, one env override or one executor leaves the next equivalent call failing on the same host, and the PR title then claims a fix the diff does not deliver. When the diff also flips a feature from opt-in to default-on, the fallback branches it just made common are part of the change and the PR must state which population of inputs now hits them.
**Seen in:** PR #1501 -- the identity fix covered one commit site while the next commit in the same repo still failed on the same host, because `-c` is per-invocation.
**Not a finding when:** the PR body names the uncovered path and says why it is out of scope, or the sibling paths cannot reach the condition (different input type, gated off, already removed).
**Evidence:** `$WORK/title.txt` and `$WORK/body.txt` for the claimed scope; `$WORK/diff.txt` and `$WORK/files.txt` for the delivered scope.
**Report as:** `C2 <file>:<line> -- fix covers <path> only; <sibling path> still produces <condition>`

### C3 -- Nothing may be left unwired: no writer without a reader, no reader without a writer, no knob nothing enforces

**Severity:** blocking
**Fires when:** the diff adds a CLI flag, env var, config key, record field or status whose identifier occurs only in the added lines; or removes a call site while the callee survives; or adds a validation entry whose target set cannot fail or cannot pass.
**The rule:** both directions are blocking. A new knob is traced to the code that consumes it and the documented effect is proved to occur -- a value computed, reported and never checked on the relevant branch is a dead knob, so wire the consumer or drop the knob. After a rewiring or deletion, a helper whose only remaining references are its own tests describes behaviour the system deliberately no longer performs and goes with the PR, together with the prompt text that still promises it (`AGENTS.md` covers the deletion hygiene; the writer/reader pairing is what this rule adds). A check whose target set makes it unable to fail is not a check -- demote it to record-only or give it a basis.
**Seen in:** PR #1491 -- `--phase-budget-enablement-pct` was a new operator flag nothing enforced: the cap was computed, reported exceeded, and no transition followed.
**Not a finding when:** the consumer lands in the same PR behind indirect dispatch (string key, `getattr`, config schema, entry point), or the field is declared record-only for later analysis and the PR says so.
**Evidence:** `$WORK/diff.txt` plus a head-tree grep of the new identifier; `$WORK/files.txt` for the removed caller's module.
**Report as:** `C3 <file>:<line> -- <identifier> has no <reader|writer>; wire <consumer> or delete it`

### C4 -- An explicitly supplied parameter must not be re-derived deeper in the call chain

**Severity:** blocking
**Fires when:** the diff adds a CLI flag or function parameter, and a helper below the entry point reads an environment variable or a presence/ordering heuristic for the same value.
**The rule:** the caller-supplied value wins all the way down. A deep helper that reads `os.environ` instead of the argument, or falls back to source-presence ordering, discards the caller's intent, and the symptom is a wrong selection rather than a dropped argument, so it does not look like a plumbing bug.
**Seen in:** PR #626 -- `tracelens_analysis.py` accepted `--framework` and the handler forwarded it, but `_expand_op_fanout()` read only `HYPERLOOM_FRAMEWORK`, and with that unset `_select_sources()` could prefer `sglang` sources for a `vllm` trace.
**Not a finding when:** the env read is the documented default and is consulted only when the argument is `None`, or the helper is reached by a second caller that has no such argument and the argument-carrying path is threaded separately.
**Evidence:** `$WORK/diff.txt` -- the flag definition, the forwarding frame, and the helper that re-derives.
**Report as:** `C4 <file>:<line> -- <helper> re-derives <param> from <env or heuristic> and ignores the supplied value`

### C5 -- Replacing inference with a caller-supplied field needs enforcement for out-of-tree producers

**Severity:** blocking
**Fires when:** the diff deletes shape or heuristic inference in favour of an explicit field, and the field has a default or its validator checks the value only when present.
**The rule:** the in-tree producers are not the whole producer set -- an orchestration LLM, a plugin or an external client can emit the same request. The field is denied when absent rather than silently defaulted, and the validator merges the same envelope layers the router reads (top level plus `params`); a gate that validates "if present" does not catch the omission.
**Seen in:** PR #1410 -- the mode became caller-supplied with a `patch` default and all three in-tree producers were updated, but nothing enforced it for an `integrate` the orchestration LLM writes itself, and `policy/gate.py` validated the value only when present.
**Not a finding when:** the request object is constructible only in-tree (no LLM, plugin or HTTP entry point reaches the constructor), or absence has a documented safe meaning that the default expresses.
**Evidence:** `$WORK/diff.txt` for the deleted inference and the new default; `$WORK/files.txt` to see whether the protocol, gate or schema moved with it.
**Report as:** `C5 <file>:<line> -- <field> defaults silently when absent; an out-of-tree producer gets <wrong behaviour>`

## S -- Silent failure and fail-open defaults

### S1 -- No swallowed failure, and no error return indistinguishable from a valid negative

**Severity:** blocking
**Fires when:** added lines introduce `contextlib.suppress`, an `except` whose body is `pass` / `continue` / a falsy return, `ignore_errors=True`, a narrowed `except (...)` tuple, or a wrapper returning `""` / `None` / `False` on its error branch.
**The rule:** a suppressed failure does not continue as success. The success log sits outside the suppressed region, the `except` tuple covers what the callee actually raises (SDK transports and resolvers raise outside the obvious set), and an error return is distinguishable from a legitimate negative answer. Where the caller emits a verdict, the failure reaches the status or the exit code. `AGENTS.md` already bans new broad catches; this rule is about the narrow ones that still swallow.
**Seen in:** PR #1564 -- a failed archive was swallowed with no trace because the `log.warning` sat inside the suppressed block and fired only on success.
**Not a finding when:** the suppressed call is genuinely advisory (best-effort telemetry, cache warm) and the caller has an independent signal for the same condition, or the handler logs and re-raises.
**Evidence:** `$WORK/diff.txt` -- the guarded region, the position of the log, and the caller's use of the return value.
**Report as:** `S1 <file>:<line> -- <failure> is swallowed; <caller> cannot distinguish it from <valid negative>`

### S2 -- Do not report success for work that produced no effect or verified nothing

**Severity:** blocking
**Fires when:** added lines contain `subprocess.run(..., check=False)` with no `returncode` inspection, a `mkdir(parents=True, ...)` followed by a write into the created path, a literal correctness or performance value on a compile-only or fallback branch, or a success dict returned from a branch that restored or wrote nothing.
**The rule:** a path that did no work does not report success. Creating the write target proves nothing reads it; an unchecked non-zero exit is a failure reported as a pass; a number the code did not measure is fabricated, whether it is `allclose: True`, an SNR literal or a timing derived from object size. The path validates against an existing target, surfaces the exit code, or emits an explicit `UNVERIFIED` marker that downstream gates can read.
**Seen in:** PR #1418 -- the bogus directory was created, the file written and the action reported success, while the vLLM runtime never read that path, so the tuned `fused_moe` config had no effect and no error signal appeared anywhere.
**Not a finding when:** the exit code is inspected one frame up by the caller, or the destination is the documented output location that this code owns and creates by design.
**Evidence:** `$WORK/diff.txt`; `$WORK/commits.txt` when a commit message claims a check the diff adds on one branch only.
**Report as:** `S2 <file>:<line> -- reports success without performing or verifying <work>`

### S3 -- A feature that disables or degrades itself must say so on the operator-facing line

**Severity:** blocking
**Fires when:** the diff adds a Noop or disabled implementation, a factory branch returning a degraded engine, a change of the unconfigured-default backend, or a status value written on one branch only.
**The rule:** degradation is visible where an operator looks. A warning gated on a config value that defaults to `None` never fires, so the component turns itself off with no log line; a status written only on the failure path cannot separate "never attempted" from "attempted and failed" and must be written on every branch; and a startup line that prints `configured` for a deployment with zero credentials inverts the one signal the operator has.
**Seen in:** PR #1472 -- RCA degraded to `NoopRcaEngine` on a subscription-token host, and because `llm_rca_enabled` defaults to `None` the guarding `log.warning` was skipped entirely.
**Not a finding when:** the degraded path logs unconditionally at warning or above, or the true state is published in a status artifact or health line the operator already reads.
**Evidence:** `$WORK/diff.txt` -- the fallback construction, the guard on the warning, and the status write sites.
**Report as:** `S3 <file>:<line> -- <component> degrades to <fallback> with no operator-visible signal`

### S4 -- Do not open a fail-closed gate; a missing input must fail closed

**Severity:** blocking
**Fires when:** the diff deletes an early `return` or `raise` on a validity check and replaces it with log-and-continue, widens an accept-set, adds a fallback substituting a different metric or axis, or feeds a gate from an accessor with a numeric default such as `.get(key, 0.0)`, `or {}`, or `default=1.0`.
**The rule:** a missing input fails closed. An accessor that coalesces an absent metric to `0.0` error or `1.0` accuracy makes the documented fail-closed branch dead code and lets a run that reported nothing pass, so gate inputs use `default=None` with an explicit absent branch. Where the accepted value becomes the reference, anchor or session state, the degradation cascades into every later decision, and the PR must show it cannot propagate.
**Seen in:** PR #1444 -- `mapping.py` read `stat(m, "request_error_rate")`, which coalesces a missing metric to `0.0`, so `parse_agentx_error_rate` never returned `None` and the documented fail-closed path was dead.
**Not a finding when:** the default is the identity of a value the upstream provably always produces, or a separate presence check runs before the accessor and diverts the absent case.
**Evidence:** `$WORK/diff.txt` for the removed refusal and the accessor default; `$WORK/base.txt` to confirm the guard existed on the merge base.
**Report as:** `S4 <file>:<line> -- missing <input> coalesces to <passing default>; <gate> can no longer fail closed`

### S5 -- Absent, empty, falsy and explicitly-cleared must stay distinguishable

**Severity:** blocking
**Fires when:** a guard flips between `if not x` and `x is None`, a neutral value (`""`, `{}`, `[]`, `frozenset()`, `Path("")`) is passed to mean "cleared", or a presence test accepts a string the parser can return empty for.
**The rule:** absent, empty, falsy and explicitly-cleared are four states and stay separable at every hop. `Path("")` normalises to `Path(".")`, whose `is_dir()` is `True`, so a recursive glob scans the process working directory; a default `frozenset()` inventory reads to a reaper as "everything untracked here is mine"; an empty `extra_envs` dict takes the opposite branch from an absent one; a resolver reading `None` as "not supplied" re-resolves the default instead of honouring the clear. A non-blank but unparseable value (a UUID, a stringified YAML sequence) passes the presence test, parses to empty, and lands the caller on exactly the default the change existed to eliminate.
**Seen in:** PR #1030 -- in the standard-Roofline fallback `Path("")` became `Path(".")`, so the shape extractor recursively globbed the process working directory and fed wrong shapes into tuning.
**Not a finding when:** the value is validated at the system boundary so the states provably collapse there (a parser that raises on unparseable input, a schema that forbids empty), or the consumer treats empty and absent identically by documented design.
**Evidence:** `$WORK/diff.txt` -- the flipped guard, the default value, and the consumer's branch.
**Report as:** `S5 <file>:<line> -- <empty/absent/cleared> are conflated; <input> takes <wrong branch>`

### S6 -- A sentinel default must be distinguishable from an explicitly supplied equal value

**Severity:** blocking
**Fires when:** added lines compare a parsed option against its default constant (`== DEFAULT_...`, `is DEFAULT_...`), or an argparse argument keeps a non-`None` `default=` while later code branches on whether the user supplied it.
**The rule:** equality with the default cannot separate "the user did not pass the flag" from "the user passed the default value". The argument takes `default=None`, the default is settled after all restore, merge and profile logic, and every reader between the parse and the settling point tolerates `None` -- a profile applied before the settling point inverts the intended precedence.
**Seen in:** PR #1491 -- `DEFAULT_MAX_HOURS` is `2.0`, so `== DEFAULT_MAX_HOURS` could not tell `--max-hours 2` from an absent flag, and `_apply_agentx_budget_profile` ran before the new default settling.
**Not a finding when:** the default is a sentinel the parser cannot produce from user input, or nothing downstream branches on supplied-ness and the two cases are intended to behave identically.
**Evidence:** `$WORK/diff.txt` -- the argument definition, the equality test, and the order of the profile/restore steps.
**Report as:** `S6 <file>:<line> -- <option> infers absence from equality with <default>; an explicit <value> takes the absent branch`

### S7 -- A new guard must not make the fallback below it unreachable

**Severity:** blocking
**Fires when:** the diff adds an early `return`, an `isinstance(...)` check or a type guard directly above an existing fallback branch.
**The rule:** the guard is evaluated against the values the producers in this PR actually emit, not against the type it names. `isinstance({}, Mapping)` is `True`, so an empty dict returns early and the recovery branch below becomes dead code for every record the change produces; a truthiness test belongs in that position. The new step must also not already be covered by the fallback it sits in front of. `AGENTS.md` bans layered belt-and-braces fallbacks; this rule is about the dead branch the new layer creates.
**Seen in:** PR #1341 -- `isinstance({}, Mapping)` returned early for an empty `observed_server_identity`, so the `server_log_path` fallback was dead code for every measurement the PR produced.
**Not a finding when:** the producers cannot emit the empty or degenerate value (a schema default, a constructor that raises), or the fallback is still reached by another caller that does not pass through the guard.
**Evidence:** `$WORK/diff.txt` -- the inserted guard, the fallback below it, and the producer sites in the same diff.
**Report as:** `S7 <file>:<line> -- new guard makes <fallback> unreachable for <value>`

---

## X -- Cross-artifact sync

### X1 -- The description must state every observable effect in the diff

**Severity:** blocking
**Fires when:** a non-test file under `src/**/*.py` changes and `body.txt` names no operator-visible effect and claims no exemption, or it names one and the diff carries further effects it does not mention.
**The rule:** A description covering only part of the diff counts as missing, in particular when it omits the step carrying the destructive or operator-visible side effect -- a forced full rebuild, a removed fallback, a renamed exported field, a newly populated report bucket, a new failure mode. It is verified against the diff, not against the PR title, and it states the new observable state and what the old behaviour cost, rather than restating the change. `AGENTS.md` makes the statement mandatory and the release cut aggregates these descriptions into the GitHub release, so an omission here is lost for good; the completeness test is the reviewer's addition.
**Seen in:** PR #1532 -- the entry omitted the KERNEL-entry audit, the one step that forces a full JIT rebuild for the operator. Raised then against the `CHANGELOG.md` entry, which #1589 retired in favour of the description carrying the same obligation.
**Not a finding when:** every changed path is under `docs/`, `*.md`, `*.rst` or a `tests/` directory, or the PR body names the exemption it claims (pure refactor, nothing operator-observable) and the diff supports that claim.
**Evidence:** `$WORK/body.txt` and `$WORK/diff.txt` -- the description against the behaviour set; `$WORK/files.txt` for the docs-only or test-only exemption.
**Report as:** `X1 <file>:<line> -- "<what an operator now sees>" is stated nowhere in the description`

### X2 -- Title and description must match the diff at the current head

**Severity:** blocking
**Fires when:** `commits.txt` has more than one commit, or `title.txt` matches the branch-name shape `^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/`, or `body.txt` still contains template markers (`<!--`, unchecked `- [ ]`) or is empty, or `body.txt` states a number, percentage, config value or an "unchanged" claim, or `title.txt` or a commit subject is typed `refactor`.
**The rule:** The title and body are read against the head commit, not the first commit: a follow-up commit that added a behaviour change, a file or a test count the description does not mention is a desync. A narrow title (`chore: pin ...`) over a diff touching unrelated modules is the same defect. Factual claims in the body -- tooling behaviour, config values, a percentage table, "strategies are unchanged" -- are verified by reading the referenced file and summing the numbers, not by trusting the prose. A `refactor` PR title is a claim that nothing observable changed, and it is verified like any other claim: a refactor-titled PR whose diff changes a flag, exit code, env var, default, argv, persisted field or decision branch that the description does not state is a title/diff desync. Retitle it, or state the effect under X1. A refactor-typed commit subject over such a change, under an accurate title and description, is advisory.
**Seen in:** PR #1567 -- the description still described two changes after the branch had grown a third commit. PR #1297 -- under an accurate `feat(keep)` title and description, a `refactor(keep): ...` commit made the CLI reject `--max-latency-ms` with exit 2 for a non-scriptable framework: the advisory commit-subject case.
**Not a finding when:** the extra commits are merges of the base branch, lint fixes or review-feedback edits inside code the description already covers, and no claim in the body contradicts the head tree.
**Evidence:** `$WORK/title.txt`, `$WORK/body.txt`, `$WORK/commits.txt`, `$WORK/numstat.txt` -- prose against the commit list and the per-file counts.
**Report as:** `X2 -- description says "<claim>"; at head <sha> the diff does <actual>`

### X3 -- Docstrings, comments and prompts must follow the behaviour they describe, in every copy

**Severity:** blocking
**Fires when:** a function body changes while its docstring or inline comments do not, or the diff touches `**/prompts/**`, `**/SKILL.md`, `docs/reference/**`. Deciding whether a surviving sentence is now false requires grepping it across the whole tree; do that for every sentence stating the changed contract.
**The rule:** A docstring asserting the opposite of the new behaviour is worse than none, especially when the stated property -- ordering, gate condition, return value, metric name, when a budget is reserved -- is what callers depend on. A correction applied to one copy and not the duplicated ones is the same defect. A prompt is the agent's spec: a schema line still marking a now-required field optional drives the agent into a guaranteed denial, so stale prompt text is a behaviour bug, not a doc nit. `AGENTS.md` constrains comment quality; the re-sync obligation is this rule.
**Seen in:** PR #1410 -- `orchestration.md:272` documented `mode` as optional in the same prompt that had just made it required, so every agent following the schema line was denied at the gate.
**Not a finding when:** the prose is vague rather than contradicted, or the sentence describes a neighbouring function whose behaviour did not change.
**Evidence:** `$WORK/diff.txt` -- changed function bodies against their unchanged prose; `$WORK/docfiles.txt` against `$WORK/files.txt` for the prose that did not move at all; the head tree for duplicated copies of the same sentence.
**Report as:** `X3 <file>:<line> -- doc/comment/prompt states "<old contract>"; code now does <new contract>`

### X4 -- A removal must be swept for remaining references and declared as a compatibility break

**Severity:** blocking
**Fires when:** deleted lines match `add_argument\(`, `\[project\.scripts\]`, `os\.environ`, an enum member `^\s*[A-Z][A-Z0-9_]*\s*=\s*['"]`, or a comparison is tightened (a deleted `<`/`<=` returning as `!=`/`==`, or a new `all\(`).
**The rule:** The parser option set, console scripts, env vars read and enum members are diffed between the released version and the branch. Every removal needs an upgrade-note entry and a description naming the failure mode -- argparse rejects the flag loudly, a dropped env var is silently ignored -- and the version chosen must match the declared compatibility policy; being a runtime no-op does not make a removal compatible. The removed name is then grepped across warning strings, docs and templates: a message pointing at a route that no longer exists turns a diagnostic into a dead end. A tightened comparison is the same class of break and must state what previously-accepted input is now rejected.
**Seen in:** PR #1519 -- removing a runtime no-op flag made every existing launch and resume command fail in argparse, and seven v1.1.0 environment variables disappeared with five of them silently ignored.
**Not a finding when:** the removed name keeps a working alias or deprecation shim added in the same PR, or the symbol is private (leading underscore, not exported, no reference outside the changed module).
**Evidence:** `$WORK/diff.txt` for the removals; `$WORK/body.txt` for the declared break; the head tree for surviving references in strings, docs and templates.
**Report as:** `X4 <file>:<line> -- removing <name> breaks <caller/config>; not declared in the description`

### X5 -- A new operator knob must land in the reference docs, the env template, the prompts and the launcher argv

**Severity:** blocking
**Fires when:** an added line matches `os\.environ(\.get)?[\[\(]\s*['"][A-Z][A-Z0-9_]{2,}` or `add_argument\(\s*['"]--` for a name absent from the base tree, or `files.txt` touches `src/hyperloom/inference_optimizer/cli/**` or `src/kernelforge/cli.py`.
**The rule:** Nothing enforces the sync between `os.environ` keys and `docs/reference/environment-variables.md` / `.env.template`, so the reviewer is the gate. Each new env var or flag is one of two kinds and the PR must say which: operator-facing config, documented with its unset default, or internal subprocess handoff, deliberately undocumented. An added or renamed flag must also reach the launcher that assembles the argv, the SKILL.md and examples that tell the agent to use it, and the how-to docs. Whether the knob should exist at all is `AGENTS.md`'s question, not this rule's.
**Seen in:** PR #1472 -- `docs/reference/environment-variables.md`, touched by the same PR, still said `OPENAI_BASE_URL` was required together with `OPENAI_API_KEY`.
**Not a finding when:** the variable is only written by a parent process and read by its own child (pure subprocess handoff) and the PR says so, or the flag is already present in the base tree and only its default moved.
**Evidence:** `$WORK/diff.txt` and `$WORK/files.txt` -- the new name against the documentation and launcher paths; `$WORK/body.txt` for the operator-facing versus internal declaration.
**Report as:** `X5 <file>:<line> -- new knob <NAME> is absent from <docs/reference/environment-variables.md | .env.template | launcher argv>`

### X6 -- A persisted schema change needs a version bump and a migration

**Severity:** blocking
**Fires when:** a field is added, removed or retyped on a type carrying `SCHEMA_VERSION`, `from_dict`, `ensure_schema`, or `files.txt` touches `src/hyperloom/inference_optimizer/breakdown/**` or `src/kernelforge/durable_io.py`, or the diff contains `CREATE TABLE`.
**The rule:** Any added, removed or renamed field in a persisted record -- run state, session state, sqlite schema, SBD document -- comes with a schema-version bump and a migration step following the file's existing convention. The check is to load a payload written at the previous version: strict field-set validation in `from_dict` and `CREATE TABLE IF NOT EXISTS` both fail to save you, and the symptom is that every existing workspace dies at startup, `--resume` included.
**Seen in:** PR #1497 -- `search_start_mean_case_speedup` was added to `RunState` with `SCHEMA_VERSION` left at 19 and no migration, killing every campaign with an existing workspace at startup.
**Not a finding when:** the field is optional with a default and the loader tolerates unknown or missing keys on the old-version path -- verify by reading the loader, not by assuming `total=False` is enough.
**Evidence:** `$WORK/diff.txt` -- the field change against the `SCHEMA_VERSION` literal and the migration block in the same file.
**Report as:** `X6 <file>:<line> -- <field> changes the persisted schema; SCHEMA_VERSION unchanged and no migration`

### X7 -- An external-component pin moves in all four places at once

**Severity:** blocking
**Fires when:** an added or deleted line matches `(VLLM|SGLANG|ATOM|AITER|TRACELENS|MAGPIE|GEAK)_(VERSION|REF|SHA|COMMIT)`, or `files.txt` touches `**/framework_deps.py`, `**/framework_registry.py`, `assets/install*.sh` or `docs/compatibility.rst`.
**The rule:** A component pin lives in code, in `docs/compatibility.rst` and in the install scripts, and all three move together, stated in the description, with the platform gate travelling with the bump. The pin must be an immutable tag or commit rather than a branch name or branch head, and a commit is preferred where the compatibility doc records the tag as known-insufficient. A quoted previous value is verified against the merge base rather than taken from the description, and a fix derives from the value effective at runtime rather than from the old constant.
**Seen in:** PR #768 -- the description's "previous value" was 120 min while `origin/main` had 130, and the fix was written against the stale constant instead of the budget actually forwarded to the child.
**Not a finding when:** the pin is only referenced in one place in the head tree (grep confirms no compatibility-doc or install-script mention), or the bump is confined to a test fixture.
**Evidence:** `$WORK/files.txt` -- which of the three locations moved; `$WORK/base.txt` to check any quoted previous value; `$WORK/body.txt` for the declared bump.
**Report as:** `X7 -- <COMPONENT> pin moves in <files touched> but not in <files missing>`

## T -- Tests and coverage

### T1 -- A test must drive the production entry point, with inputs the real path can produce

**Severity:** blocking
**Fires when:** `testfiles.txt` is non-empty alongside a non-test change, or an added test line matches `ctx\s*=\s*object\(\)`, `MagicMock\(\)`, a call to an underscore-prefixed helper, or authors a fixture inline (`write_bytes\(`, `write_text\(`, a literal payload) for a predicate added in the same commit.
**The rule:** Each added test is read against the code path it is named for. A test that reaches the behaviour only by calling an internal directly with hand-built input the real path cannot produce, that passes a stub context so a wrong-container read goes unnoticed, or that picks arguments the code filters out before the changed line, asserts a branch that is dead in production while the PR's real regression stays invisible. A new predicate inspecting real-world artifacts -- traces, compiled `.so`, tuned CSVs, logs -- is run against artifacts from an actual run before it is trusted: fixtures authored alongside the gate encode the author's assumption, and a permanently-false predicate ships green.
**Seen in:** PR #1457 -- `serving_modules_cover_csv()` could never return True for the real CSVs, so the skip path was dead while unit tests passed on synthetic `.so` bytes containing the literal names.
**Not a finding when:** the internal under test is the production entry point for that behaviour (the public caller only forwards), or the end-to-end path is already covered by another test in the same file that the PR leaves intact.
**Evidence:** `$WORK/testfiles.txt` and `$WORK/diff.txt` -- the added test bodies against the changed production path.
**Report as:** `T1 <test file>:<line> -- exercises <internal> with <input the real path cannot produce>; <changed line> stays uncovered`

### T2 -- Coverage may not be quietly reduced, and deleted end-to-end tests must be migrated

**Severity:** blocking
**Fires when:** a deleted line matches `^\-\s*(async\s+)?def test_`, a test file is deleted, an existing test's `range\(\d+`, iteration count or timing constant changes, an `@pytest.mark.(critic_agent_e2e|targeted_build_e2e)` is removed, or `numstat.txt` shows a new `src/**` module over 300 added lines with no path in `testfiles.txt`.
**The rule:** Edits to existing tests preserve the amount of work exercised: a loop rewrite that cuts iterations while the assertions stay unchanged is a silent loss -- the test still goes green while verifying much less. When a PR deletes tests along with the code they covered, the deleted test is named and something picks up its scenario at the same depth; a real end-to-end test replaced by tests that stub the expensive step is a coverage loss, not a migration. A large new module performing privileged or security-relevant actions needs at least a pure-function test for its decision logic, whatever the convention of its directory. `docs/contributing/style-guide.md` requires a replacement test to carry the original's contract and failure-mode assertions across and to keep them in the default CI selection, from which the `*_e2e` markers are excluded; T2 is the check that the PR did so, because the 90% floor cannot show that a specific assertion survived.
**Seen in:** PR #1475 -- a loop change cut the exercised work 12.5x with the assertions untouched, so the loss was invisible in a green run.
**Not a finding when:** the deleted tests covered code deleted in the same PR and no surviving route reaches that scenario, or the iteration count moved because the production loop bound moved with it.
**Evidence:** `$WORK/diff.txt` for the deleted or shrunk tests; `$WORK/numstat.txt` for new modules without a test file; `$WORK/testfiles.txt` for what was added in exchange.
**Report as:** `T2 <test file>:<line> -- <deleted/shrunk test> removes coverage of <scenario>; nothing replaces it at that depth`

### T3 -- Tests must not depend on live network, credentials or hardware

**Severity:** blocking
**Fires when:** an added line under a `tests/` path matches `requests\.|httpx\.|openai\.|boto3\.`, `os\.environ\[['"](OPENAI|ANTHROPIC|HF|AWS)`, `subprocess\.(run|Popen)`, `rocm-smi|rocminfo|/sys/`, or `probe\s*=\s*True`; or a non-test change adds such a call inside a builder or factory that existing tests already reach.
**The rule:** A test that falls back to a real client when an argument is `None` or a credential happens to be exported passes on CI and fails on a configured developer box. In the other direction, new code hanging off a function many existing tests already reach must not perform real sysfs reads, device probes or CLI spawns under test: an unconditional probe inside a widely-covered builder makes unrelated tests slow and environment-dependent. The probe must be injectable.
**Seen in:** PR #1528 -- `test_overlapping_edits_are_left_to_a_resolver` made a live network call and failed whenever a credential was exported.
**Not a finding when:** the call is monkeypatched or behind a fixture in the same file, or the test carries an e2e marker that CI skips and the marker is registered in `pyproject.toml`.
**Evidence:** `$WORK/testfiles.txt` and `$WORK/diff.txt` -- the call site against the surrounding fixtures and markers.
**Report as:** `T3 <file>:<line> -- test reaches <network/credential/device> when <condition>; passes on CI, fails on a configured box`

### T4 -- A correctness fix needs a regression test that fails on the merge base and pins the exact discrimination

**Severity:** blocking
**Fires when:** `title.txt` starts with `fix` or `perf` and `testfiles.txt` is empty, or the diff changes a default (`default=`), a timeout selection, or adds a gate together with a runtime fallback.
**The rule:** Each correctness defect gets a test encoding the precise thing the code got wrong -- the same input with two competing candidates, asserting the right one is picked -- not a happy-path test, and it must fail on the merge base. Where a PR changes a default, a timeout selection, or adds a gate plus a fallback, the crossing cases are required: operator-pinned value present and the gate fires; gate alone sufficient so the fallback never runs; the value arriving from each source (task params, env var, reference recipe, YAML base). Testing each layer's happy path separately is insufficient.
**Seen in:** PR #626 -- no regression covered the same op carrying both `vllm` and `sglang` entries with different sources, which is the discrimination the fix turned on.
**Not a finding when:** the fix is not observable from any test seam (a log string, a comment, a type annotation), or an existing test already fails on `base.txt` for this defect -- check it before asking for a new one.
**Evidence:** `$WORK/title.txt`, `$WORK/testfiles.txt`, `$WORK/base.txt` -- the claimed fix against the tests added and the merge base they must fail on.
**Report as:** `T4 -- fix for <defect> has no test that fails on <base sha>; missing case: <the discrimination>`

---

## R -- Resource safety and destructive operations

### R1 -- Destructive action scoped to a global pattern instead of to this run

**Severity:** blocking
**Fires when:** an added line calls a node-wide reaper or device sweep -- `pkill`, `killall`, `kill -9`, `scancel`, `docker rm|kill|stop` with a name pattern rather than an owned id, or sets a visible-device mask copied from the host environment.
**The rule:** a destructive primitive may only reach processes, containers and GPUs this run owns. The spawned tree runs in its own process group or session and only that tree is killed; containers and GPUs are reclaimed by an owned label or cgroup; the container's visible-device mask is derived for the container, not inherited from the host. A broad sweep is acceptable only as an explicit single-tenant opt-in, never as the default, and "this is only a manual experiment script" does not hold once an autonomous path calls the same primitive.
**Seen in:** PR #703 -- `pkill -f` on the server pattern, reachable from autonomous `combined_e2e`, could kill another tenant's sglang/vLLM servers, and a startup reaper blind-killed `hipcc|ninja|clang++` on a shared node.
**Not a finding when:** the pattern is matched against an identifier minted by this run (session id, campaign id, container label, cgroup path) and cannot match a process the run did not start, or the call is inside a branch reachable only from a documented single-tenant flag.
**Evidence:** `$WORK/diff.txt` for the added call and the identifier it matches on; `$WORK/files.txt` to see whether `assets/slurm/**` or a kill/reclaim module is in scope.
**Report as:** `R1: <file>:<line> <primitive> matches <pattern>, which can reach another tenant's <processes|containers|GPUs>; scope it to <owned identifier>`

### R2 -- Destructive self-heal outside its reachable input set, branch or directory

**Severity:** blocking
**Fires when:** an added audit, reconcile, invalidate or repair step deletes, unlinks or moves build artifacts, or an added artifact lookup walks to a parent or sibling directory (`Path(workspace).parent / ...`, a glob over a shared runs root).
**The rule:** the input set the step scans must equal the set the runtime can actually reach, and the branch it sits in must be reachable under the default configuration. A scan that globs every shipped artifact fires on a healthy install and forces a full rebuild; a step placed below an unconditional `return` in the default branch does nothing its description promises. An artifact resolver stays inside its own run directory -- for a run root the parent is the shared runs directory, so a different run's artifact gets attributed, hashed into identity and published.
**Seen in:** PR #1532 -- the audit moved the whole `jit/build` aside on a healthy install, and the fix's stated effect sat below `if geak_enabled: ... return` so it never ran on the default backend. PR #1341 -- a `server.log` probe escaped the measurement's own directory and scraped a sibling run's launch argv.
**Not a finding when:** the scan is restricted to paths this run created, the enclosing branch is the configured default in `assets/configs`, and the parent-directory walk lands in a directory owned by the same run rather than a shared root.
**Evidence:** `$WORK/diff.txt` for the glob and the enclosing control flow; `$WORK/body.txt` for the effect the description claims, which the placement must actually produce.
**Report as:** `R2: <file>:<line> <step> scans <set>/sits under <branch>, so it <destroys healthy state|never runs|reads another run's artifact>`

### R3 -- Cleanup handles only what the operation modified, not what it created

**Severity:** blocking
**Fires when:** an added teardown, rollback, release, restore or revert path uses per-path `git checkout <base> -- <path>`, `check=False` on a restore command, or reconstructs a patch by writing file contents.
**The rule:** the cleanup path must be checked against a file the operation created and committed, not only one it edited. Path-by-path `git checkout <base> -- <path>` no-ops for paths unknown at the base commit, and with `check=False` that failure is swallowed, so created files survive the restore. An ownership inventory is captured before the operation and the removal step runs after the index and HEAD are restored. The same completeness applies to patch reconstruction: handling only text modify and add silently skips pure deletes, renames, mode-only and binary changes, so the A/B measurement is not measuring the real patch.
**Seen in:** PR #1420 -- a file the campaign created stayed in the working tree because `git checkout <base> -- <path>` failed with `pathspec did not match` under `check=False`. PR #703 -- pure deletes were skipped, so the live tree kept files the optimizer patch had deleted.
**Not a finding when:** the restore is a whole-tree operation (`git reset --hard` plus a clean of untracked paths from a pre-operation snapshot) rather than path-by-path, or the patch kinds outside modify/add are rejected loudly at parse time instead of silently dropped.
**Evidence:** `$WORK/diff.txt` for the restore command and its `check=` argument; `$WORK/testfiles.txt` for whether a test covers a created-then-reverted file.
**Report as:** `R3: <file>:<line> cleanup misses <created files|deletes|renames|mode-only|binary>, leaving <contamination> for the next round`

### R4 -- Rollback state written, read or recovered in the wrong order

**Severity:** blocking
**Fires when:** the diff adds or changes a persisted-record pair (`record_*` / `read_*` / `seal_*`), a recovery entry point, or a context manager whose `__exit__` guards on a handle assigned in `__enter__`.
**The rule:** the record a resume path depends on is persisted before the first mutation, is not gated on a condition the target configuration cannot satisfy, and carries the pessimistic flag -- a pre-mutation record cannot know what the round touched, so it must read as written-to. In a context manager the handle `__exit__` guards on is assigned as soon as the resource is identified, before any mutation; an early `return self` between the two leaves the resource live with teardown skipped, and a skipped or failed cleanup must make the exit code non-zero. Recovery runs at entry, before the state is read or sealed. Every field the writer persists is read back by the reader.
**Seen in:** PR #1563 -- a session dying between the first write and the return left `pending` holding `mutated=False`, restored nothing and returned `{"ok": True}`. PR #1420 -- `seal_campaign_baseline` ran before anything reclaimed, so a dead campaign's rewrite became the next session's serving baseline, and `read_campaign_baseline` dropped the `base_commit` the writer persisted.
**Not a finding when:** the persisted field is genuinely write-only by design and the diff says so, or recovery already runs at process entry on a path the parent-killed case also reaches.
**Evidence:** `$WORK/diff.txt` for the write/mutate/read ordering and the reader's return tuple; `$WORK/files.txt` for `state/shared_state.py` and `cli/recover.py` in scope.
**Report as:** `R4: <file>:<line> <record> is <written after the first mutation|never read back|recovered only after the worker exits>, so <resume outcome>`

### R5 -- Patches and archives from a backend, agent or LLM treated as trusted input

**Severity:** blocking
**Fires when:** an added line calls `git apply`, `patch -p`, `tar` extraction, `zipfile.extractall` or `shutil.unpack_archive` on content produced by a kernel backend, agent or LLM, or passes `--unsafe-paths` or any option that disables path containment.
**The rule:** this content is untrusted and must be validated before application. A malformed or malicious diff header with an absolute path or a `..` component escapes the extraction directory even during a verification-only reconstruction. The diff headers are parsed first, restricted to a single expected target, with absolute paths and `..` components rejected; both cases carry a regression test, because green CI over happy-path reconstruction proves nothing here.
**Seen in:** PR #681 -- `git apply --unsafe-paths` on GEAK/OOB/LLM-produced patch artifacts let a crafted header escape the temporary extraction directory.
**Not a finding when:** the archive or patch is produced in-process by this repo from content it already controls, or a validator in the same diff rejects absolute paths and `..` before the call.
**Evidence:** `$WORK/diff.txt` for the apply/extract call and any validation added alongside it; `$WORK/testfiles.txt` for the absolute-path and `..` regression tests.
**Report as:** `R5: <file>:<line> applies backend-produced <patch|archive> without header validation (<flag>), allowing escape to <path>`

## P -- Concurrency, budgets and cost

### P1 -- Blocking filesystem or subprocess work on the event loop

**Severity:** blocking
**Fires when:** an added line inside an `async def`, or inside a sync closure an `async def` calls directly, matches `subprocess.run`, `.communicate()`, `time.sleep(`, `requests.`, `rglob(`, `glob(`, `iterdir(` or `open(` without `asyncio.to_thread` or `run_in_executor`.
**The rule:** a coroutine may not perform a synchronous filesystem walk, glob, process probe or sleep. If the same file already wraps a comparable probe in `asyncio.to_thread`, the new one does the same. An unbounded `rglob` over a directory holding run output is blocking regardless of how often it runs -- the loop is stalled for the whole walk, starving every concurrently scheduled lane and the bus heartbeat.
**Seen in:** PR #1526 -- `_server_liveness_probe` did an unbounded blocking filesystem walk on the event loop, stalling every concurrently scheduled task and the bus heartbeat.
**Not a finding when:** the call is on a bounded, known-small path (a single `stat` or a read of one small file) and the enclosing coroutine is not the coordinator loop, or the diff already routes it through `to_thread`/`run_in_executor`.
**Evidence:** `$WORK/diff.txt` for the call and its enclosing `async def`; `$WORK/files.txt` for `orchestrator/loop/**` in scope, where the starvation reaches every lane.
**Report as:** `P1: <file>:<line> <call> blocks the event loop for <duration/scope>; wrap in asyncio.to_thread as <sibling probe> does`

### P2 -- Deadline does not cover the whole operation, or a forwarded budget mismatches the outer timeout

**Severity:** blocking
**Fires when:** an added line introduces `stream=True`, moves an awaited call out of a `wait_for`/`timeout` block, or changes a `*_BUDGET`/`*_TIMEOUT`/`*_budget_min` constant that is forwarded into a child process.
**The rule:** when a call becomes streaming, the existing timeout must still cover consumption of the body, not only the call that opens the stream -- a proxy that opens the stream then stalls hangs the caller forever. When a per-backend budget is raised and forwarded to a child, the wrapping subprocess timeout must use the same effective budget; a parent timeout derived from a different, smaller variable hard-kills the child before its own budget expires and defeats the change entirely.
**Seen in:** PR #710 -- `asyncio.wait_for()` wrapped only `client.chat.completions.create(...)`, so `_score_one_model()` could hang indefinitely while the proxy stalled mid-stream. PR #768 -- the GEAK budget was raised to 180 min but the outer subprocess timeout still used `backend_budget_min` (default 60), killing the child early.
**Not a finding when:** the stream is consumed inside the same `wait_for` scope, or the outer timeout is derived from the same variable the child receives (trace both to a single source before firing).
**Evidence:** `$WORK/diff.txt` for the `wait_for` scope and both sides of the budget expression; `$WORK/body.txt` if the timeout value is operator-observable.
**Report as:** `P2: <file>:<line> deadline covers <opening call> but not <body consumption>` / `outer timeout uses <var A> while the child receives <var B>`

### P3 -- Cost of work added to a locked, hot or per-iteration path not accounted for

**Severity:** blocking
**Fires when:** an added line performs `json.load`, `gzip.open(...).read()`, `read_text()`, `yaml.safe_load`, directory hashing, or constructs a library object that probes hardware or fetches a model config, inside a lock-protected section, a promotion or commit path, a poll or retry loop, or a per-candidate step.
**The rule:** the added work must be costed on a production-sized input before it lands. File reads, log scans, directory hashing and hardware-probing constructors are seconds of blocking I/O added to the section a lock test exists to protect. The same value is not computed twice on the same input -- compute once and pass it down. Fully loading a payload to answer a boolean predicate is a regression when a streaming scan answers it.
**Seen in:** PR #1447 -- `traces_complete` full-`json.load`ed every trace just to find one kernel event: 4.2 s and 2607 MB peak RSS per file, inside a loop polling every 10 s. PR #1341 -- `ServerArgs.__post_init__` reads the HF model config and probes the GPU, run inside the promotion path. PR #1321 -- the recipe YAML was read and parsed twice per handoff.
**Not a finding when:** the call runs once per session outside any lock, or the input is bounded small by construction and the diff states the bound. A measurement on a real artifact beats an estimate -- if no production-sized input is available, report the concern with the size you could not obtain rather than a number you guessed.
**Evidence:** `$WORK/diff.txt` for the call and its enclosing lock/loop; `$WORK/testfiles.txt` for whether a lock or concurrency test covers the section.
**Report as:** `P3: <file>:<line> adds <cost> to <locked section|poll loop|per-candidate step>, measured <time/RSS> on <input>`

### P4 -- BaseException escapes a half-applied state transaction

**Severity:** blocking
**Fires when:** the diff adds a call inside a promotion or commit path that mutates shared state before returning, and the enclosing handler is `except Exception`; especially when the added call reaches argparse (`parse_known_args`, `parser.error()`) or `sys.exit`.
**The rule:** every failure mode of work added after shared state is assigned but before the transaction completes must be contained. `except Exception` does not catch `SystemExit` or `KeyboardInterrupt`, and argparse-based parsing raises `SystemExit` on malformed input, leaving the transaction half-applied -- one field written, its companion unset. Containment means catching `BaseException` at that boundary or not parsing with argparse there at all. A new broad catch is itself constrained by `AGENTS.md`; this rule is about the ordering, not the breadth.
**Seen in:** PR #1341 -- `parse_known_args` raised `SystemExit` on a malformed value, which `except Exception` did not catch, leaving `current_best` written and `current_best_measurement` unset.
**Not a finding when:** the mutation happens after the risky call rather than before it, or the shared state is assigned atomically at the end of the transaction so no partial ordering exists.
**Evidence:** `$WORK/diff.txt` for the assignment-then-call ordering and the handler's exception type; `$WORK/files.txt` for `loop/writeback.py` or an executor in scope.
**Report as:** `P4: <file>:<line> <call> can raise SystemExit past <handler>, leaving <field A> written and <field B> unset`

### P5 -- Ordering around whole-object saves and late-resolved values

**Severity:** blocking
**Fires when:** the diff adds a step that refreshes fields on shared state near an `_emit_lifecycle` or `.save()` call, or constructs a spec, manifest or recipe that copies fields from a mutable object.
**The rule:** every whole-object save that can run between a field refresh and its persistence overwrites the refresh, because `SharedState.save` is a whole-object overwrite -- the merge must run before the terminal lifecycle save, and list-valued fields must be unioned rather than overwritten. The mirror case: a spec handed to another component is constructed after the last write to every field it copies, or it ships defaults while the system runs on operator values, and nothing downstream notices because the identity fields still look right.
**Seen in:** PR #1030 -- `_emit_lifecycle(status="END")` unconditionally saved the live `shared_state` and overwrote the refreshed fields, surfacing only as a skipped-merge warning. PR #1365 -- `build_agentx_workload_spec` had to be written into the recipe at the end of `apply_agentx_switch`, after all derived values landed in `envs`.
**Not a finding when:** no save can be reached between the refresh and the persistence (trace the call graph, not just the file), or the field is already unioned by the save path.
**Evidence:** `$WORK/diff.txt` for the placement relative to the save/emit call; `$WORK/files.txt` for `state/shared_state.py` and `executors/_workload_envs.py` in scope.
**Report as:** `P5: <file>:<line> <refresh> is overwritten by <whole-object save> at <line>` / `<spec> is built before <field> is written`

### P6 -- Work added after a budget cut is not charged against that budget

**Severity:** blocking
**Fires when:** the diff appends rows, shapes or tasks after a `limit=`/budget-based selection has already been applied, inside a function governed by a timeout constant.
**The rule:** if exceeding the enclosing deadline is a hard kill that discards partial results, the additions must be reserved before the cut, not appended after it. Appending with no time accounting lets the run overrun its hard timeout and return nothing. The overrun must be reproduced with realistic input sizes rather than the one case in the PR body -- a per-shape cost estimate that happens to be pessimistic for the author's model hides it.
**Seen in:** PR #1454 -- decode rows were appended with no time accounting: 30 rows at ~74 s each is 2220 s against a `timeout_s=1216` hard deadline, while the 4-group Qwen3-14B case in the PR body fit only because `_DEMAND_PER_SHAPE_COST_S` was ~1.85x pessimistic for that op.
**Not a finding when:** the deadline is soft (partial results are kept and graded on overrun), or the appended items are reserved out of the budget before the selection cut.
**Evidence:** `$WORK/diff.txt` for the append site relative to the budget cut and the timeout constant; `$WORK/body.txt` for the input size the author validated against.
**Report as:** `P6: <file>:<line> appends <N items> after the budget cut; <N x cost> exceeds the hard timeout <T>, discarding the run`

---

## M -- Measurement and grading

### M1 -- Never compare numbers produced by two measurement systems

**Severity:** blocking
**Fires when:** a comparison added by the diff has one operand from an external optimizer,
self-reported or vendor-tool field and the other from a value this repo measured
(`current_best`, a stored baseline, a previous round's result).
**The rule:** a value must be compared only against a value produced by the same code path,
so the comparison is warm-vs-warm or cold-vs-cold by construction. Median-of-N, tolerance
bands, config digests, workload signatures and protocol reconstruction do not make a
cross-system comparison sound; they patch around it. A measurement standard raised on one
path only -- extra repeats, a confirmation round -- creates the same defect, because the two
paths stop being commensurate.
**Seen in:** PR #1258 -- a cold revalidation number from GEAK's harness was compared against
a hot `current_best` from Hyperloom's, producing a false `no_promote`.
**Not a finding when:** the diff re-measures the candidate through the incumbent's own path
before comparing, or both operands are written by the same function -- open the writer of
each field and confirm rather than trusting the field names.
**Evidence:** `$WORK/diff.txt` -- the added comparison and the assignment of each operand.
**Report as:** `M1: <file>:<line> compares <field> measured by <system A> against <field>
measured by <system B> -- re-measure the candidate through <path>`

### M2 -- One field, one definition: a published metric must not silently change meaning

**Severity:** blocking
**Fires when:** the diff makes a published metric computable two ways depending on which
inputs are present (an `if <source>: ... else: ...` around a reported field), renames a
metric key, or changes the denominator of a speedup or gain.
**The rule:** a reported key carries exactly one definition. When an input the definition
needs is missing, the code fails loudly or withholds the value; it does not publish a
differently-denominated number under the same key. The axis a result is graded on is the
axis it is reported on, and a renamed field is renamed in every producer, reader and
docstring in the same diff.
**Seen in:** PR #1497 -- an empty `source_case_ms` flipped the published `speedup` back to a
port denominator, reintroducing the two-denominator defect the PR existed to remove;
PR #1527 left `record_workload`'s docstring naming the pre-rename `intvty_p90`.
**Not a finding when:** the second computation writes a different key, or the fallback
branch raises or returns no value instead of publishing.
**Evidence:** `$WORK/diff.txt` for the branch and the rename; `$WORK/body.txt` when the
metric is operator-visible.
**Report as:** `M2: <file>:<line> publishes <key> with <denominator A> when <input> is
present and <denominator B> when it is not -- one key, one definition`

### M3 -- Read threshold units off the metric's producer

**Severity:** blocking
**Fires when:** the diff adds or changes a numeric constant compared against a metric
produced outside this repo (`*_THRESHOLD`, `*_LIMIT`, `*_RATE`, a literal in a gate).
**The rule:** the unit of the constant must be read from the function that produces the
metric -- percent or fraction, ms or s, bytes or KiB -- not inferred from the metric's name.
A unit mismatch makes the gate orders of magnitude too strict or too loose and is invisible
in tests, which are written against the same wrong assumption. Once settled, the unit
belongs in the constant's name.
**Seen in:** PR #1444 -- `AGENTX_ERROR_RATE_THRESHOLD = 0.10` was compared against aiperf's
`request_error_rate`, which `RequestErrorRateMetric._derive_value` returns as
`100.0 * errors / total`, so the gate rejected anything above 0.1% where upstream allows 10%.
**Not a finding when:** the producer is inside this repo and the diff touches both sides, or
the constant name already encodes the unit and matches the producer.
**Evidence:** `$WORK/diff.txt` for the constant and the comparison; the producer must be read
in the head tree, so state that you opened it.
**Report as:** `M3: <constant> = <value> is compared against <metric>, which <producer>
returns in <unit> -- gate is <N>x too <strict|loose>`

## D -- Design, duplication and matching

### D1 -- Route through the canonical path instead of adding or hardening a parallel one

**Severity:** blocking
**Fires when:** the diff adds a function that performs an operation the repo already owns
(patch deploy, revert, snapshot, cleanup, revalidation), or adds guards, digests, signatures
or protocol reconstruction to a branch that runs in parallel with the default executor;
or introduces a second way to express a concern its sibling modules already express one way
(how a helper reports failure), without migrating
the siblings.
**The rule:** one operation has one implementation. A second one means two sets of
semantics, and the newer, weaker one becomes the one the real path uses. A fix that hardens
a special path is answering the wrong question: the case belongs on the default path, with
the path-specific dispatch flags dropped, unless the general path provably cannot serve it.
`AGENTS.md` *Clean design* ("a second copy of a behaviour is a bug you will later fix once
and miss elsewhere") is the bullet a finding cites, and the finding must name the existing
owner module -- without that name it is taste and goes. `.github/copilot-instructions.md`
states the duplication and pipeline-bypass principle; this rule is the concrete check.
The same holds for a code shape, under *Finish the replacement* ("migrate the callers and
delete what it replaced"): agents copy the file they are in, so two shapes for one concern
become a growing population of both. Migrate the siblings in the same PR, use the
existing shape, or name the remaining sites under *Finish the replacement*'s staging clause.
**Seen in:** PR #703 -- a direct `git apply` flow for combined E2E created a second patch
deployment semantics alongside the byte-exact atomic snapshot deploy already in the tree.
PR #1658 -- of three `_git_head_sha`
copies, the dead one reporting failure as `(None, stderr)` was deleted and the two returning
`""` converged in `_git`.
**Not a finding when:** the existing helper cannot serve the case for a reason the diff or
the body states, or the new code is the canonical path and the diff deletes the old one. A
private copy of a helper another module exports is D10's finding, an empty base class or a pure
forwarder is A12's, and a tuple, dict-bag or second-channel result is A9's, not this one.
**Evidence:** `$WORK/diff.txt` for the added implementation; `$WORK/files.txt` to confirm the
existing owner module is untouched.
**Report as:** `D1: <file>:<line> re-implements <operation> already owned by <module> --
route through it, or state why it cannot serve this case`

### D2 -- A fast-path branch must run every stage the canonical path runs

**Severity:** blocking
**Fires when:** the diff adds a `pre_applied` / `skip_*` / already-done branch that
short-circuits an established pipeline and synthesizes its result, or narrows a multi-file
input to one target by basename or prefix.
**The rule:** enumerate the stages the real path performs -- cache invalidation, rebuild,
multi-node fan-out, post-verify, manifest -- and the branch runs the same sequence,
target-gated where needed, rather than re-picking a subset. A stage missing from a branch
whose output feeds a measurement or a keep/revert decision produces a wrong decision with no
error. A heuristic that drops companion files means the artifact downstream is not what the
producer emitted, while it is reported as if it were.
**Seen in:** PR #1501 -- the synthesized `apply_result` skipped aiter JIT and cpp_itfs
invalidation, so an aiter publication was re-baselined against the stale binary.
**Not a finding when:** the skipped stage is provably a no-op for the state the branch
requires, and the branch asserts that state rather than assuming it.
**Evidence:** `$WORK/diff.txt` -- the branch and the canonical sequence it bypasses.
**Report as:** `D2: the <branch> path skips <stage> that <canonical path> runs -- <what the
downstream consumer then reads>`

### D3 -- Demand evidence before deleting a fallback path

**Severity:** blocking
**Fires when:** the diff removes one of several alternative implementations of the same
capability -- a module, route, backend or branch the body describes as fallback, bypass,
alternative or legacy.
**The rule:** the surviving path must be shown to cover every case the removed one handled.
Removing the only path free of some dependency promotes "that dependency is sufficient for
every input" into a load-bearing claim, and that claim needs evidence about arbitrary real
inputs, not about the narrower question the cited corpus answers. The review asks which real
inputs were run end to end on the surviving path and what the degraded behaviour is for an
input the dependency cannot handle.
**Seen in:** PR #1368 -- dropping the `bypass` route removed the only TraceLens-free analysis
path while the `custom` framework depended on it, and the cited corpus validated the idle
guard rather than TraceLens's coverage of operator-supplied traces.
**Not a finding when:** the removed path is unreachable in the head tree, or the body cites
runs of the surviving path on the input classes the removed one served.
**Evidence:** `$WORK/diff.txt` and `$WORK/numstat.txt` for the deletion; `$WORK/body.txt`
for the evidence offered.
**Report as:** `D3: removing <path> leaves <surviving path> as the only route for <input
class>; the PR's evidence covers <X>, not <that class>`

### D4 -- Moved or copied code must be diffed against the version it replaces

**Severity:** blocking
**Fires when:** the diff re-homes or duplicates a helper -- a large added block in a shared
module paired with deletions elsewhere, or a title or body containing re-home, consolidate,
extract or de-duplicate.
**The rule:** the new copy is diffed against the current implementation at its old home, not
against an older revision: copying a pre-fix version reverts whatever landed since the fork,
and a performance revert must be quantified. When the same block already exists in two
sibling modules, check whether the copies have diverged -- divergence at the time of the PR
proves the next schema change will miss one site. Relocated constants must equal the
literals they replaced, and the untouched path must be shown byte-identical afterwards. A
rule that exists in two languages needs a parity test exercising the real other side, not a
re-spelled copy of it.
**Seen in:** PR #1368 -- the new `_trace_reader.stream_events` was byte-identical to the
pre-`c6cbccbff` implementation and so reverted that fix, measured ~5x slower on a 32MB trace
(0.28s to 1.46s); PR #1341's copied `_attach_grid_launch_evidence` body had already diverged
on `warm_reuse`.
**Not a finding when:** the diff shows the moved block unchanged against the current head
version, or the divergence is intentional and named in the body.
**Evidence:** `$WORK/diff.txt`, `$WORK/base.txt` -- compare the added block against the old
home at the merge base.
**Report as:** `D4: <file>:<line> copies the pre-<sha> version of <symbol>, reverting <fix>
-- <measured cost>`

### D5 -- Backend, model and credential selection goes through the repo's single resolver

**Severity:** blocking
**Fires when:** the diff adds an LLM call, a model id, a provider constant or a credential
check (`ANTHROPIC_*`, `OPENAI_*`, `CODEX_*`, `CLAUDE_MODEL`, a direct SDK client).
**The rule:** backend selection defers to the existing helper and the model default that
helper implies. A pinned provider means the feature does not exist on a deployment
configured for the other one while sibling paths in the same component keep working -- two
backend policies inside one component. The model default lives with the resolver, so a caller
must not reintroduce one of its own beside it. A retired key must not satisfy a credential
check, and no same-provider silent model fallback may be reintroduced.
**Seen in:** PR #1528 -- rung 4 hardcoded Anthropic instead of following
`llm_config.preferred_agent_backend()`, so on a Codex-configured box that rung did not exist
while forge's rewrite lane ran normally, and `resolve_forge_llm_model("claude")` was called
without `default=`.
**Not a finding when:** the code path is provider-specific by contract (a backend adapter
implementing one provider) rather than a feature that should run on any configured backend.
**Evidence:** `$WORK/diff.txt` -- the added selection and whether the backend and model id come from the resolver.
**Report as:** `D5: <file>:<line> pins <provider> instead of <resolver> -- <feature> is
absent on a <other provider> deployment`

### D6 -- No substring or keyword heuristics for identity; mirror the real predicate and treat ambiguity as unresolved

**Severity:** blocking
**Fires when:** an added line resolves a name with `a in b or b in a`, `startswith`, `find`,
a keyword scan over kernel/op/symbol/route names, a substring test against `config.json` or
`quant_method` values, or a `for ... : if match: return` first-wins lookup.
**The rule:** identity is resolved by exact match or by an explicit glob contract with
literal symbols escaped -- prefixed, truncated and nested names all resolve to the wrong
target under containment tests. A glob must be evaluated at every root and depth it claims
to cover, because `fnmatch` anchors both ends. A gate that reproduces a decision an upstream
library makes internally mirrors that library's predicate field for field, including its
strict comparisons. Human-readable explanatory strings never enter a matcher; keep the prose
in a render-only field. Where more than one candidate can match, first-wins is a defect:
report ambiguous or unresolved so the caller fails loudly.
**Seen in:** PR #626 -- `_kernel_matches()` returned
`trace_name == key or trace_name in key or key in trace_name` and took the first hit;
PR #1097's `verdict()` matched `t in v` against the literal
`"Performance (or Power at a uniform bin)"`, which contains `power`;
PR #1412's `fnmatch(".../aiter_cache/sources/abc/flydsl_cache/launch_1", "aiter_cache/*")`
is `False`.
**Not a finding when:** both operands are already-canonicalized identifiers from one
producer, and the diff shows the canonicalization.
**Evidence:** `$WORK/diff.txt` -- the matcher and the value space it is applied to.
**Report as:** `D6: <file>:<line> resolves <name> by <substring|prefix|keyword> match --
<concrete value> resolves to <wrong target>`

### D7 -- Re-check dedup, grouping and ordering keys when an attribute stops being uniform

**Severity:** blocking
**Fires when:** the diff makes a previously-uniform attribute vary per item (per-file mode,
per-entry flag) and a dedup key, `set`/`dict` collapse, sort key or first-wins lookup keyed
on the coarser attribute is left unchanged.
**The rule:** every key that collapsed items on the old attribute must be re-keyed on the
tuple that now distinguishes them. An order-dependent survivor drops work while the caller
still reports success, which is a silent wrong-result path rather than a failure -- nothing
downstream can tell the dropped item from an item that was never requested.
**Seen in:** PR #1567 -- `_multi_root_strategies` deduped by root after the SGLang rebuild
mode became per-file, so only the first strategy in descriptor order survived while apply
still reported `status: ok`; the fix keys on `(root, rebuild_mode)`.
**Not a finding when:** the collapsed items are genuinely interchangeable after the change,
or the loser is reported rather than silently dropped.
**Evidence:** `$WORK/diff.txt` -- the newly per-item attribute and every key expression that
still ignores it.
**Report as:** `D7: <file>:<line> dedups by <old key> while <attribute> is now per-<item> --
<which item> is dropped and <caller> still reports ok`

### D8 -- Documented precedence must match evaluation order, and injected defaults must lose to configuration

**Severity:** blocking
**Fires when:** the diff adds or edits a resolver looping over sources and keys, adds a
docstring claiming a precedence, or injects a default identity, credential, path or config
value (`os.environ.setdefault`, `GIT_AUTHOR_*`, `GIT_COMMITTER_*`).
**The rule:** with an outer source loop and an inner key loop, source precedence dominates
key precedence, so a docstring's key ordering holds only within one source -- read the loop
nesting rather than the docstring, and confirm the source that wins in production is the one
the author intended, remembering that an autofilled or synthesized value is not an authored
one. An injected default must lose to existing configuration: environment variables outrank
config files, so setting `GIT_AUTHOR_*` rewrites the identity of every commit in a repository
that already has one. Probe, then inject.
**Seen in:** PR #1321 -- loop nesting made source precedence dominate, so the docstring's
"`ROCR_VISIBLE_DEVICES` before `HIP`/`CUDA`" held only within a single source, and a recipe
mask that had been autofilled rather than authored won;
PR #1501 set `GIT_AUTHOR_*` over a configured `user.name`/`user.email`.
**Not a finding when:** the injected value is written into a process the diff also creates
and no outer configuration reaches it, or the docstring is corrected in the same diff.
**Evidence:** `$WORK/diff.txt` -- the loop nesting, the docstring, and the injection site.
**Report as:** `D8: <file>:<line> evaluates <source> before <key>, contradicting the
docstring's <claimed order>` / `D8: <file>:<line> setdefault of <var> overrides configured
<setting>`

### D9 -- Flag edits must survive the real argv assembly of every backend

**Severity:** blocking
**Fires when:** the diff appends to or strips from an extra-args / env-passthrough list
(`EXTRA_*_ARGS`, `extra_server_args`), edits an argument-removal regex, or adds an error
substring that triggers an automatic retry or fallback.
**The rule:** run the actual command-build function of each backend that consumes the list
rather than reasoning from one backend's ordering: a backend that emits its own copy of the
same flag and appends the extra args last lets the injected value win, silently redirecting
output to a path nothing reads. A removal pattern is checked against four shapes -- the flag
with a value, the bare flag at the end of the string, the bare flag alone, and the bare flag
followed by another flag -- and needs a negative lookahead so the next flag is not swallowed
as a value. Error-signature matching needs a second independent context marker; a single
generic substring costs a full re-run on an unrelated failure.
**Seen in:** PR #1565 -- `bypass_engine.build_server_command` emits its own
`--profiler-config.torch_profiler_dir` and appends `EXTRA_VLLM_ARGS` last, the opposite of
the Magpie ordering the change assumed, so the round reported no trace files.
**Not a finding when:** the diff exercises each consuming backend's build function in a test
or in the body, and the ordering is shown.
**Evidence:** `$WORK/diff.txt` for the edit; `$WORK/files.txt` to enumerate which backends'
argv builders exist and whether any were touched.
**Report as:** `D9: <flag> injected via <list> loses to <backend>'s own copy in
<build function> -- <what breaks>` / `D9: the strip pattern misses <shape>`

### D10 -- A concern belongs in the module that owns it, not beside its caller

**Severity:** blocking
**Fires when:** the diff adds or edits, outside the module that owns the concern, a constant,
precedence list, parser or client constructor that the owner already exports: LLM model, SDK
client, API key, base URL or header resolution anywhere but `src/hyperloom/common/llm_config.py`;
backend registration outside `agent_backends/registry.py`; tier, path or schema constants beside
the code that reads them; a private re-implementation of a `hyperloom.common` helper --
`common/io.py` (`atomic_write_*`, `safe_mtime`, `append_jsonl`), `common/jsonio.py`
(`read_json`, `read_jsonl`), `common/timeutil.py` (`now_iso`), `common/proctree.py`
(`group_alive`, `kill_tree`). The shape is an `os.environ` read, a hardcoded tuple of names, or a
second `def` whose body matches an exported one.
**The rule:** a second home for a concern means two answers to the same question, and the one
the reader is looking at is not necessarily the one that runs. Import from the owner. If the
owner cannot serve the case, the fix is to extend the owner, not to keep a copy near the caller
-- and if there is genuinely no owner yet, the PR says which module becomes one. `AGENTS.md`
*Clean design* ("one boundary rule per concern, owned by one module ... duplicated state or a
duplicated decision is the same problem") is the bullet a finding cites.
**Seen in:** PR #1338 -- vendoring KernelForge brought
`src/kernelforge/llm/gateway.py:29 _ANTHROPIC_KEY_ENVS` alongside the byte-identical
`llm_config.py:63 ANTHROPIC_SYNTHESIZABLE_KEY_ENVS`, and a second `parse_custom_headers` at
`gateway.py:81` beside the exported one at `llm_config.py:382`, while four `kernelforge` modules
already import from `llm_config` -- so no layering barrier explained the copy. PR #1348 added
`orchestrator/kernel/gemm_shape_coverage.py:483 _safe_mtime` beside `common/io.py:safe_mtime`,
and PR #1408 added `orchestrator/kernel/controller_submit.py:64 _read_json` beside
`common/jsonio.py:read_json`.
**Not a finding when:** the owner's export cannot reach the caller without a dependency cycle or
a layering inversion, and the diff or the body says which; the two look alike but answer to
different contracts (`AGENTS.md` *Simplify by removing a mechanism* names this case); or the
value is a test fixture deliberately pinned so the test fails when the owner moves.
**Evidence:** `$WORK/diff.txt` for the added definition; the owner module in the head tree for
the export that already serves it; a grep proving the caller's package already imports the owner,
which is what removes the layering excuse.
**Report as:** `D10: <file>:<line> defines <name>, already owned and exported by <owner>:<line>
-- import it, or state the barrier that prevents it`

### D11 -- Repair a wrong value at its producer, not by re-deriving it inside one consumer

**Severity:** blocking
**Fires when:** the diff makes a value correct by re-reading, re-parsing, re-merging or
re-deriving it inside a consumer, while the single place that produces or publishes that value
to every reader is untouched. The shape is a local `{**os.environ, **something}`, a second
`json.loads` of a handoff blob, or a helper that reassembles upstream's output before using it.
**The rule:** count the readers before accepting the repair. When a value is wrong because its
producer never published it in the form readers expect, repairing one reader leaves the rest on
the old answer and adds a second resolution path -- the defect stops being "this value is
missing" and becomes "this value means two things", which is strictly harder to find and is the
state the repair was supposed to end. One repaired reader out of N blocks; changing the producer
fixes all N without touching any of them. This is D10's dual: D10 is a concern copied beside its
caller while an owner exports it, D11 is an owner that exists but does not publish, compensated
for downstream. `AGENTS.md` *Clean design* ("a second copy of a behaviour is a bug you will later
fix once and miss elsewhere") is the bullet, and the finding names the producer module that
should have changed instead.
**Seen in:** PR #1797 -- `--extra-env` pins reached only the `INFERENCE_OPTIMIZER_EXTRA_ENV`
blob and never the individual names, so the first version re-merged that blob inside
`agentx_env_for_conc`. Every other reader kept its bare `os.environ.get`, and a pinned
`HYPERLOOM_AGENTIC_BACKEND` then selected the MLPerf client while the seeded grading axis, the
persisted backend identity and the benchmark timeout all still resolved `aiperf` -- a session
that REVERTed every round. Exporting each pin at the producer, `_export_operator_launch_shape`,
corrected all of them without changing a single reader.
**Not a finding when:** the producer is out of tree, or cannot carry the value for a reason the
diff or body states; the consumer is provably the only reader and the diff says so; the
re-derivation is a declared rung in a documented precedence ladder, which is one resolution
point rather than a second; or the producer is being changed in a stated follow-up and the body
names the readers left on the old answer until then.
**Evidence:** `$WORK/diff.txt` for the added re-derivation; a grep of the value's name across
the head tree for the readers the diff did not touch -- that count is the finding.
**Report as:** `D11 <file>:<line> -- re-derives <value> at 1 of <N> readers; <producer module>
publishes it and is untouched, so <other reader>:<line> still resolves it as <old answer>`

### D12 -- A function this diff adds, or whose complexity it raises, may not exceed cyclomatic complexity 20

**Severity:** blocking
**Fires when:** the diff adds a `def` or `async def` under `src/` outside a `tests/` directory,
or adds branches to an existing one, and the measurement below puts that function above 20 at
the head while the merge base had it at 20 or below, absent, or at a lower number.
**The rule:** measure, do not estimate -- the verdict is a comparison of two numbers, and a
function that reads long is not the same claim. Run the ceiling against the changed files at the
head, then against the same paths at `base.txt`, and match per function name:

```bash
ruff check --select C901 --config "lint.mccabe.max-complexity=20" --force-exclude \
  --output-format concise $(cut -f3 "$WORK/numstat.txt" | grep '^src/.*\.py$' | grep -v '/tests/')

git show "$(cat "$WORK/base.txt")":<path> > "$WORK/base-<name>.py"
ruff check --select C901 --config "lint.mccabe.max-complexity=20" --isolated \
  --output-format concise "$WORK/base-<name>.py"
```

`--force-exclude` keeps Ruff's `extend-exclude` (vendored SDK copies, `src/kernelforge/examples`)
out when paths are passed explicitly; `--isolated` is what lets the base copy be checked from
outside the tree. Present at head and absent at base, or a higher number at head than at base,
fires. The same number on both sides is backlog and is silent. This is not a rule duplicating a
static gate: `C901` is absent from `[tool.ruff.lint] select` and Pylint runs `--errors-only`, so
nothing in CI measures it and review is the only place it surfaces. `AGENTS.md` *Size is a design
signal* carries the gate and the style guide § *Complexity ceiling* carries the two cases; the
finding names the function, both numbers, and the branch the diff added.
**Seen in:** no single PR -- this is the gate written into the style guide's *Size and
complexity* table rather than a shape clustered from review history. The tree held 124 units
above 20 when it was introduced, which is the backlog the comparison exists to keep out of the
finding.
**Not a finding when:** the base number equals the head number (the unit was already over and
this diff did not grow it); the file is a test, or sits in `extend-exclude`, which
`--force-exclude` already settles; the added branches are a flat dispatch or validation table
whose arms share no state and the body says so -- that is the trigger's accepted answer, and it
survives at the ceiling only when the arms are genuinely independent.
**Evidence:** both Ruff invocations above, with the head and base numbers quoted; `$WORK/diff.txt`
for the branches that moved the number.
**Report as:** `D12 <file>:<line> -- <function> is complexity <head> (base: <base|absent>), over
the ceiling of 20; split it or keep <added branch> out of it`

## A -- Abstraction, boundaries and shape

`AGENTS.md` *Code shape* states the principles this family checks; each rule is the concrete
test for one of its sentences, or for a sentence of *Clean design*, *Trust the caller* or
*Simplify by removing a mechanism*. A finding names the owner, the type or the split that would
satisfy it; one that cannot is taste and goes.

### A1 -- Cross a module boundary only through the names its owner exports

**Severity:** blocking (`AGENTS.md` *Code shape*: "A module is used through the names its owner exports; an `_`-name or an `_internal` package is not an interface"; *Clean design*: "a minimal typed interface outward"). A finding that cannot name the owner module and the export to add is taste and goes.
**Fires when:** an added line imports `_name` from, or imports through an `_`-prefixed module segment (`._internal.`, `executors._multi_node_env`) of, a package other than the importer's own (package = `hyperloom.<layer>.<subpkg>`); reads or assigns another object's `_attr` where the receiver is not `self`/`cls`; probes it with `getattr(obj, "_x", None)` or `hasattr(obj, "_x")`; or hangs an undeclared attribute on someone else's object.
**The rule:** a leading underscore, an `_internal` package and an undeclared attribute all mean "not part of the interface". Reaching for one couples the caller to the owner's implementation, and the owner can no longer change it without breaking a caller it cannot see. When the caller genuinely needs the value, the fix is in the owner: promote the name to a public export, add a method or property that answers the question, or move the state onto the collaborator that owns it (#1794). A move that relocates a module without promoting the names its callers already used is the replacement left unfinished; re-exporting the private name under the same alias (`_X as _X`) across packages is the same reach, laundered. The `self._coord` handle a `CoordinatorCollaborator` holds is the layer's declared collaboration pattern and is not itself a finding; a hop past it into another object's `_field` is.
**Seen in:** PR #1794 -- `ConversationCollaborator` wrote `self._coord._kb_last_read = evidence` and `IntentRouter` read it back with `getattr(router._coord, "_kb_last_read", None)`; the fix moved the state to its owner (`self.kb_last_read = evidence`). PR #1650 moved `_grid_server_args.py` down to `inference_optimizer/grid_server_args.py` without promoting its names, so `orchestrator/actions/executors/_grid_runner.py:86-96` now imports `_SPACE_VALUE_FLAGS as _SPACE_VALUE_FLAGS` and `_shell_safe_dedupe as _shell_safe_dedupe` across packages. `orchestrator/actions/executors/_multi_node_server_lifecycle.py:19-20` imports `inference_optimizer.multi_node._internal.env_safety` and `.server_args_safety`, plus function-level `_internal` imports at :293, :659, :876, :935 and :1161, although `_internal/__init__.py` says "the CLI is the only public surface".
**Not a finding when:** importer and imported module are in the same owning package and the underscore means package-private; the `_attr` access is in the module or owning package that defines the receiver's class; the code is that package's own tests or conftest; the name belongs to stdlib or a third-party library with no public equivalent (`sys._getframe`, namedtuple `_replace`/`_asdict`); the identifier lives in a standalone script shipped to a remote host or injected as source text (`multi_node/scripts/`, patcher payloads); the access predates the diff (V1).
**Evidence:** `$WORK/diff.txt` for the added import or attribute access; the owner module in the head tree to confirm the name is private and what public export would serve.
**Report as:** `A1 <file>:<line> -- reaches <owner>.<_name>; export <public name> from <owner> (or add <method>) and use that`

### A2 -- Dependencies point one way down the layers; an upward or cyclic edge is not made legal by deferring or hiding it

**Severity:** blocking (`AGENTS.md` *Clean design*: "dependencies pointing one way down the layers — no cycles").
**Fires when:** the diff adds, in non-test code, an import -- at module or function level -- that points up the layers or closes a cycle between two packages; or adds, in non-test code under `hyperloom.common`, `inference_optimizer`, `agents` or `kernelforge`, a string the runtime resolves into a higher layer (`importlib.import_module("hyperloom.orchestrator...")`, `[sys.executable, "-m", "hyperloom.orchestrator..."]`, `"pkg.mod:attr"`); or adds a module-level writable hook or registry in a lower layer (`_HOOK = None`, `set_*_hook`) whose only writer is a higher layer and whose type the lower layer does not declare; or moves an import that runs at runtime under `if TYPE_CHECKING:`; or adds an exemption to an import check (a baseline or ignore entry, a skip in `test_common_import_lint.py`) to let such an import through.
**The rule:** dependencies point one way down `orchestrator > agents > inference_optimizer > common` -- the direction nearly every import already takes, and the one PR #1650 moved code to keep -- and `kernelforge` uses `hyperloom.common` only. An import deferred into a function is still an edge: it hides a cycle from a module-level scan and from the reader without removing it. A string the runtime imports, a subprocess `-m` target, an untyped global hook a higher layer assigns and an exemption added to an import check are the same edge with the evidence removed: the lower module can no longer be loaded, tested or reused without the higher one, and the next refactor of the higher layer breaks it at runtime instead of at lint. Move the code down to the lowest layer whose dependencies it already satisfies (#1650), or invert the dependency with a protocol or parameter the lower layer declares and the caller passes in (that is dependency inversion, not a finding). An exemption list only shrinks: a diff that adds an entry to get an import through is the defect, not the fix.
**Seen in:** PR #1650 -- the fix shape: it "removes every agents -> orchestrator import that only reached for one of these modules", by moving pure-domain modules down (`_grid_server_args.py -> inference_optimizer/grid_server_args.py`). `orchestrator/state/shared_state.py:1633` imports `..kernel._kernel_decisions` inside a method while `kernel/_kernel_decisions.py:28` imports `..state.kernel_decision_settings`: a state <-> kernel cycle hidden from a module-level import scan.
**Not a finding when:** the string names a third-party or optional package (`claude_agent_sdk`, `sglang` for `__version__`); the string is user-facing text or argparse `prog=`; the callback's type is declared by the lower layer and the higher layer only implements it; the diff removes or reformats exemption entries; the `TYPE_CHECKING` import is used only in annotations; the upward import predates the diff and was only moved (V1); the importer is the `inference_optimizer/cli` entry point wiring orchestrator components into a command (a composition root: check it for a cycle, not for direction); the import is a direct one from `hyperloom.common`, which `test_common_import_lint.py` already rejects.
**Evidence:** `$WORK/diff.txt` for the added import, string target, hook or exemption entry; the head tree to confirm which layer the target lives in.
**Report as:** `A2 <file>:<line> -- <lower module> depends on <higher module> via <import|string|hook|exemption>; move <code> down to <layer> or declare <protocol> in <lower>`

### A3 -- A utility module names no business concept, and a business module does not host a generic helper

**Severity:** blocking (`AGENTS.md` *Code shape*: "A business-agnostic helper that more than one module needs lives in `hyperloom.common` and names no business concept"; for a business rule added to a generic utility module, *Clean design*: "A module should read as one job: cohesive inside"). A private copy of a helper `hyperloom.common` already exports is D10's case, reported under D10, not here.
**Fires when:** the diff adds a business rule -- a business constant or threshold, an import of a business module, a function whose job is a business decision (benchmark timeouts, server readiness, recipe or KEEP handling) -- to a module whose name says utility (`*_utils.py`, `io.py`, `jsonio.py`, `timeutil.py`, `proctree.py`, `_subprocess_kill.py`) and whose other functions are generic, so the module now holds two jobs; or adds, in a business package, a second caller-facing copy of a stdlib-only generic helper (atomic write, JSON io, stat, hashing, process group, retry, timestamp, truncation) that has no owner yet.
**The rule:** code that would read the same in any project is a utility, and a utility has one home, `hyperloom.common`, for both `hyperloom` and `kernelforge`. A generic helper that gets its second caller moves there, not beside the first caller, because a utility parked in an upper-layer business package forces every lower-layer reuser into an upward import (A2). The converse holds too: once a module named for a mechanism also knows benchmark timeouts and server-log markers, it is a business module with a misleading name, and lower layers that need the business half end up importing a "utility" module from the orchestrator. Lift the generic half into `common`, and give the business half a module named for its job, at the lowest layer that needs it.
**Seen in:** PR #1611 -- "move the unified-diff parser out of the framework agent ... The canonical location is now common/ since the module has no framework-agent imports" (`common/unified_diff.py`). PR #1655 -- "common/github_urls.py: new module with public repo_slug() extracted from sources/_shared._repo_slug. The function was a pure GitHub-URL parser with no dependency on the PR-source backend layer". `orchestrator/actions/executors/_subprocess_kill.py` -- named for process killing, it holds `new_session_kwargs` (:55), `_process_group_alive` (:62) and `_signal_group` (:75) next to `resolve_benchmark_timeouts` (:281), `server_log_death_excerpt` (:455) and `stamp_server_ready` (:559), and imports `from .bypass_analysis import parse_server_log_throughput` (:31). `inference_optimizer/cli/preflight.py:1697` and `inference_optimizer/cli/__init__.py:1723` import the business function `resolve_benchmark_timeouts` upward out of it. PR #1348 (`orchestrator/kernel/gemm_shape_coverage.py:483 _safe_mtime`) and PR #1408 (`orchestrator/kernel/controller_submit.py:64 _read_json`) added copies beside `common/io.py:safe_mtime` and `common/jsonio.py:read_json` -- the D10 half, cited to show the drift this rule prevents.
**Not a finding when:** the helper must run standalone on a remote node or be injected as source text (`multi_node/scripts/`, `assets/host_probe/`, patcher payloads); the local version has a different contract and its name says so (`_magpie_patcher.atomic_write_text` adds a log prefix and returns bool); the helper's semantics depend on business constants; it is a private three-line helper with a single caller and no `common` equivalent; the module is named for one business mechanism and every function serves it (`enablement/runtime/build_utils.py`); the business word appears only in a message or docstring.
**Evidence:** `$WORK/diff.txt` for the added function or import; `git grep` over `src/hyperloom/common` for an existing export of the same shape; the module's other importers for which layer needs it.
**Report as:** `A3 <file>:<line> -- <generic helper|business code> in <utility|business> module <module>; move <name> to <common module|module named for its job>`

### A4 -- Ask the owner: no navigation through a collaborator's collaborators, and logic beside the state it reads

**Severity:** blocking (`AGENTS.md` *Code shape*: "Ask the owner of a state rather than navigating its collaborators, and keep logic beside the state it reads"). The finding names the accessor to add or the class to move onto; without it the item goes. Pure forwarding layers are A12; private hops are A1.
**Fires when:** the diff adds an attribute chain that starts from a non-`self` receiver and passes two or more repo-owned collaborators before reading a field or calling a method (`a.b.c.method()`; a third-party SDK's fluent API such as `client.chat.completions.create` is not navigation); or adds a module-level function that reads five or more fields of one parameter, writes at least one of them, and uses almost none of its own module's state; or reaches past a collaborator to a store it owns (its connection, table or file) instead of calling it.
**The rule:** a caller that walks three objects to read one value depends on the shape of all three. Give the first object a method or property that answers the question, or pass the needed value in. A function that mostly reads and writes one parameter's fields belongs on that parameter's class, next to the data, rather than in a sibling module reached through a forwarding shim and a function-level import. When moving it back would push the owner over a size trigger, the size problem is solved by splitting the owner along a real seam (A5), not by keeping its logic outside it.
**Seen in:** PR #1662 -- "the conversation builder and the sub-agent runner reached past their collaborators to the shared connection and wrote their own SQL against messages and tasks, so a schema each module owns had readers it could not see. Both now ask the owner": the queries moved behind named `MessageBus` and `TaskRegistry` methods. `orchestrator/state/shared_state.py` carries 11 methods documented "Forwarding shim -- implementation in `._kernel_decisions`" (the first at :1628), while `orchestrator/kernel/_kernel_decisions.py:502 record_kernel_integrate_result(state, ...)` reads and writes `state.kernel_integrate_attempts` and `state.pending_kernel_integrations` and calls `state._is_integrate_fault`: the logic sits one module away from its data.
**Not a finding when:** the intermediate objects are immutable values or parsed config (`spec.server.args`); the function only reads the record to render or project it into an output format its own module owns; a collaborator calls another collaborator's public method through `self._coord.<collaborator>` (one hop past the handle is the layer's pattern), or a module-level helper that takes the collaborator as its first argument does the same through `router._coord.<collaborator>`; the envious function was placed elsewhere to break an import cycle and the diff says so (fix the direction under A2 first); the chain predates the diff and was only moved (V1).
**Evidence:** `$WORK/diff.txt` for the added chain or function; the owner class in the head tree for an existing accessor.
**Report as:** `A4 <file>:<line> -- navigates <a.b.c> to read <x>; add <owner>.<accessor>` or `A4 <file>:<line> -- <function> works on <param>'s fields; move it onto <Class>`

### A5 -- One reason to change per unit: split along the seam the changes follow, never into a grab-bag

**Severity:** blocking (`AGENTS.md` *Clean design*: "A module should read as one job: cohesive inside"). Both shapes put a second job in one unit.
**Fires when:** the diff (a) adds a new module whose functions share only where they came from ("moved off the executor", "no self state") or what shape they have, rather than one job -- a name like `utils`/`helpers`/`misc`/`manager`/`_*_helpers`/`_*_evidence` makes the check due but is not itself the finding; or (b) adds to a module, class or function already over a style-guide size trigger a new public function, a new state field or an import from a package the unit did not import before, serving a concern the unit's name and docstring do not describe.
**The rule:** a unit changes for one reason. Name that reason before accepting a new home, and place a function there only if it changes when that reason changes. A module grouped by provenance ("helpers moved off `IntegratePatchExecutor`") or by shape ("things that run a subprocess") is a grab-bag: each function goes to the module that already owns its concern, or the PR names which new module becomes its owner and why no existing one is. A `utils` module is acceptable only when every function in it is business-agnostic and they form one toolkit (A3). The finding names the two reasons to change and the seam to split along.
**Seen in:** PR #1703 -- its first commit moved seven executor-state-free methods into a new `executors/_enablement_keep_evidence.py`; its second commit "Delete executors/_enablement_keep_evidence.py and give each function the module that already owns its concern" (`_accuracy_gate.py`, `enablement/recipe/keep_records.py`, `keep_probe.py`). `src/kernelforge/gemm_tune/utils.py` (263 lines) holds file hashing, tuner-script lookup, GPU status, build parallelism caps (cgroup memory, job limits), subprocess running and result emission.
**Not a finding when:** the new module is a private helper of one owner package and its name states that owner's job (`_accuracy_gate.py`); a `utils` module whose functions are all business-agnostic and form one toolkit; a business rule added to a generic utility module, which is A3's; the added function is another instance of the unit's single job (one more phase handler in a phase dispatcher); the PR is the staged first half of a split and the body names the owner module and the follow-up.
**Evidence:** `$WORK/files.txt` for added modules; `$WORK/diff.txt` for added defs and imports; the head tree for the unit's size against the style-guide triggers.
**Report as:** `A5 <file>:<line> -- <unit> gains <concern> beside <existing job>; split along <seam>` or `A5 <file> -- groups functions by <provenance|shape>; move <fn> to <owner>`

### A6 -- Extract a function by purpose, not by line range

**Severity:** blocking (`AGENTS.md` *Simplify by removing a mechanism*: "Extract a helper when it names a complete job", judged "including the helpers, interfaces and parameter plumbing the change itself introduces"). A single caller is not the finding -- the same bullet says a single-caller helper can earn its name; the finding is a slice that names no job.
**Fires when:** the diff adds a function called from exactly one site in a function it shrinks, and the new function either carries a position suffix (`_part1`, `_rest`, `_impl`, `_inner`, `_core`) instead of a purpose, takes six or more parameters that are the caller's locals passed through unchanged, or returns three or more values the caller immediately unpacks back into its locals. Weighted up when the body or a commit cites the complexity or size trigger as the reason.
**The rule:** an extracted function is independently nameable and independently testable: its name states the job or decision it performs, its inputs are what that job needs rather than the caller's local variable table, and its result is not a write-back of the caller's locals. Slicing a long body at line boundaries to pass D12 moves the complexity into the parameter list and the unpack site and decomposes nothing. When the slice needs many of the caller's locals, the real seam is a parameter object, a small class that owns that state, or a different cut.
**Seen in:** PR #1366 -- `src/kernelforge/fusion/command.py:1476 _run_multi_patch_nomination` takes 26 keyword parameters, returns `tuple[list[dict[str, Any]], Optional[CompilePassOutcome], Optional[LoopResult], int]`, and is called once from `run` (:1083) as `patches_out, compile_pass_outcome, loop_result, withheld = _run_multi_patch_nomination(...)`. Counter-example: PR #1702 split the 981-line `RooflineExecutor._execute` into `_profile_until_usable`, `_analyze` and `_compute_bound_reprofile`, each a nameable step.
**Not a finding when:** the extracted function names a complete decision with a narrow signature even if it has one caller (PR #1751's `orchestrator/phases/machine_state.py:1703 _skip_to_close_reason(global_inputs, hint, unrecorded_baseline_tput=None) -> tuple[str, dict[str, Any]]`, a `(reason, evidence)` pair); the parameters are a typed parameter object; the long signature is a public entry point mirroring a CLI; the outer wrapper adds a named contract and the inner is also called elsewhere.
**Evidence:** `$WORK/diff.txt` for the new def, its single call site and the caller's shrinkage.
**Report as:** `A6 <file>:<line> -- <fn> is a line-range slice of <caller> (threads <N> locals|returns <N>-tuple of locals); cut along <job> or introduce <Type>`

### A7 -- A variation point lives on the owner's spec, not in a switch repeated at each consumer

**Severity:** blocking (`AGENTS.md` *Code shape*: "A per-framework, per-backend or per-arch fact is a field on the owner's spec, not a branch at each consumer"). A switch whose default arm silently maps an unknown or newly registered value to another one is also wrong behaviour; say so in the finding.
**Fires when:** the diff adds an `if`/`elif`/`match`/conditional expression, or a dict literal, keyed on a framework name (`"vllm"`, `"sglang"`, `"atom"`, `"xdit"`), kernel backend, agent backend or GPU arch, in a module other than the dimension's owner (`framework_registry`, `agent_backends/registry.py`, the kernel backend package), and the same dimension is already switched on in another module, or the owner has no field for the fact.
**The rule:** a per-framework, per-backend or per-arch fact is a field (or method) on the owner's spec, read by every consumer; it is not an `if fw == "atom"` written again wherever the fact is needed. Adding a framework should mean filling in one spec, not finding every switch. Branches inside one function that differ only in how a single probe script is assembled are local detail, not a variation point.
**Seen in:** PR #1705 -- "Adding a framework meant editing tables that restated what framework_registry already knows ... profile.py chose profile_<fw>.yaml through an if/elif chain; _workload_envs mapped each framework to baseline_<fw>.yaml through a dict. Both fell back to sglang for an unknown name", with the concrete bug that `resolve_framework_tree("xdit")` always returned `""`; `FrameworkSpec` gained `python_package` and `source_root`. `inference_optimizer/cli/preflight.py` branches on the framework name in four functions, at :778, :816, :823, :880 and :1032 (PR #1182, PR #1615).
**Not a finding when:** the module is the owner of that dimension's behaviour (`bypass_engine.py` building each framework's server argv); the branch is the single dispatch point that selects a strategy object; the branches answer different contracts that only share a key (*Simplify by removing a mechanism*); the code is a standalone shipped script that cannot import the registry.
**Evidence:** `$WORK/diff.txt` for the added branch; `git grep` over the head tree for other switches on the same dimension; the owner spec for a matching field.
**Report as:** `A7 <file>:<line> -- branches on <dimension> == "<value>" for <fact>; add <field> to <Spec> (other switches: <sites>)`

### A8 -- Parameters are few, cohesive and non-redundant: no flag argument, no data clump, no derivable or unread parameter

**Severity:** blocking for (a), (c) and (d) (`AGENTS.md` *Code shape*: "no flag argument that selects between two bodies, no parameter that is derivable, unread or set only by tests"; *Simplify by removing a mechanism*: "rather than become one function with modes"). Advisory for (b): no `AGENTS.md` sentence names a clump.
**Fires when:** the diff adds a parameter to, or adds a `def` under `src/` with: (a) a `bool` (or two-value `Literal`) whose first use is `if <param>:` ending in `return`, or whose callers on one value pass dummies (`""`, `0`, `None`) for parameters the other branch reads; (b) a group of three or more names passed together in the same order to more than one function, or unpacked from one object at every call site; (c) a parameter every production caller derives from another argument, or that only tests set to a non-default value; (d) a parameter the body never reads.
**The rule:** a signature states the callee's real inputs once. A flag that selects a behaviour is two functions sharing a name: split them, so neither caller supplies dummies for the other branch. Values that always travel together become one frozen, keyword-only dataclass built once where they are resolved (`_SweepRun`, `_Round` are the house pattern). When every caller unpacks the same object to feed the callee, pass the object -- unless taking it would make the callee import a type from a layer above it, in which case pass the fields. A parameter that is derivable, unread or varied only by tests is plumbing the change introduced. Parameter count alone is not this rule's finding.
**Seen in:** PR #1660 -- "_sweep_one_arm_single_server took 28 parameters ... and 18 of them were the same sweep-wide values threaded through every call" (clump -> `_SweepRun`); "has_budget was always total_budget_sec is not None" (derivable); "The partial flush's partial flag had no caller that passed False" and "write_reports=False, passed only by tests" (test-only). PR #1707 -- "_reject took eleven positional parameters, four consecutive str among them"; the callers in `enablement/lane.py` now pass `round_row` instead of unpacking it. `inference_optimizer/multi_node/_internal/infera_support.py:113 build_node_launch_args(..., kill_only=False)` (the flag dates from PR #548) branches at :132 `if kill_only: ... return`, and `multi_node/commands/infera.py:498` passes `model=""`, `tp=0`, `kill_only=True` to satisfy the other branch's signature.
**Not a finding when:** the function is a CLI entry point whose parameters are the operator's flags; the bool tunes a detail inside one behaviour (`verbose`, `exist_ok`); passing the owning object would invert a layer; the parameter is an existing test seam on the merge base; the diff only forwards an existing parameter; the parameter is required by a Protocol or callback signature; a static gate in CI already reports the unread parameter.
**Evidence:** `$WORK/diff.txt` for the signature; `git grep` for the production call sites and what they pass.
**Report as:** `A8 <file>:<line> -- <fn> <takes flag <name> selecting <two behaviours>|repeats clump <a,b,c> across <N> signatures|takes <param> derivable from <other>|takes <param> only tests set|never reads <param>>; <split|introduce <Type>|pass <object>|drop it>`

### A9 -- Return one named shape: no positional tuple of loosely related values, no dict bag for a type, no second channel through an argument

**Severity:** blocking (`AGENTS.md` *Code shape*: "A result is one named type: no positional tuple of loosely related values, no dict whose optional keys stand for states, no second channel written into an argument"). A second channel that makes a caller misread the outcome is also wrong behaviour; say so in the finding. None-versus-empty on a return is S1/S5's concern.
**Fires when:** the diff adds or widens a return so that a function under `src/` (a) returns a tuple of three or more elements where two share a type or the elements are not one value's coordinates, or adds an element to such a tuple; (b) returns a `dict[str, Any]` built from literal keys, some present only on some paths, to an in-repo caller that branches on them; (c) returns a value while also writing a key or attribute on a parameter that a caller reads to interpret the return.
**The rule:** a function answers with one shape the reader can name. Several loosely related results become a frozen dataclass or `NamedTuple` whose fields carry the meaning the position carried; adding an element to a positional tuple is the moment to make that change, not to widen it. A dict passed around inside the repo whose optional keys stand for states becomes a dataclass or `TypedDict` with the states explicit. A result does not travel in two channels: when the outcome needs more than the return value holds (refused vs reverted, failed vs empty), widen the return type (`Verdict`, a result dataclass) instead of stamping the caller's argument.
**Seen in:** PR #1739 -- "Promoters now return Verdict (adopted/refused/reverted/failed/recorded) instead of the bool-plus-PROMOTION_REFUSED_KEY side-channel ... so a lift-refused integrate_patch ... [is] no longer journalled as KEEP"; before it, `orchestrator/loop/writeback.py` set `result[PROMOTION_REFUSED_KEY] = True` and `inference_optimizer/session/optimization_journal.py` read it back with `if result.get(PROMOTION_REFUSED_KEY):`. PR #1120 widened `orchestrator/roles/claude.py:850 _invoke_and_collect` to `tuple[list[Intent], str, int, dict[str, Any], str | None, str | None]` (:858); the two `str | None` are told apart only by position. `orchestrator/phases/prelude.py:396 _resolve_kernel_target_paths(self, entry) -> list[str]` (PR #1199) writes `entry["resolution_error"] = reason` (:400) and returns `[]`, and :806 reads `entry.get("resolution_error")` to tell failure from empty.
**Not a finding when:** the dict is a JSON, subprocess, LLM or persisted payload at a system boundary and nothing branches on its optional keys; the tuple is one value's coordinates (`(m, n, k)`, `(lo, hi)`) or a `(value, reason)` pair unpacked by one caller; the argument mutation is the function's declared job and it returns nothing or the same object (`apply_*`, `stamp_*`, `fill_*`); the diff only forwards an existing return.
**Evidence:** `$WORK/diff.txt` for the return annotation and statements; the callers in the head tree for how they read it.
**Report as:** `A9 <file>:<line> -- <fn> returns <N-tuple|dict bag with optional <keys>|<value> and writes <arg>[<key>]>; return <Type> carrying <fields>`

### A10 -- A closed vocabulary gets a type where it enters, not a raw string compared at every reader

**Severity:** blocking (`AGENTS.md` *Code shape*: "a closed vocabulary is one Enum or registry entry, converted where the value enters"). A raw-string branch that takes the wrong arm for a valid member, or falls silently to a default on an unknown value, is also wrong behaviour. A re-spelled copy of a vocabulary the owner exports is D10's finding, and a branch on a framework, backend or arch is A7's, not this one.
**Fires when:** the diff, outside the vocabulary's owner and outside the parser that converts it, compares a value to a string literal from a closed set the repo already names (decision `"KEEP"`/`"kept"`, status, phase, role, intent type) while an Enum, registry entry or constant exists (`Verdict`, `IntentType`, `OUTCOME_*`); normalises case before such a comparison away from the boundary; or introduces a new set of two or more literals that branches dispatch on, typed as `str`.
**The rule:** a value from a closed set is converted to its type once, where it enters (record parse, CLI argument, LLM output), and travels as that type. Readers compare against `Verdict.ADOPTED` or a spec field, never a literal they spelled themselves; an `.upper()` before a comparison in a reader is the reader repairing a type nobody gave the value. A new vocabulary that code branches on is a `str` Enum in the module that owns the concept, so an added member breaks the non-exhaustive readers instead of falling into the `else`.
**Seen in:** PR #1739 -- "Introduce settlement Verdict to replace the kept boolean": it deleted `_JOURNAL_KEEP_STATUSES = frozenset({"kept"})` and `_JOURNAL_REVERT_STATUSES = frozenset({"reverted", "accuracy_unavailable_reject", "regression"})` for `class Verdict(str, Enum)`. `orchestrator/knowledge/remote_recipe/values.py:508`: `str(last.get("decision") or "").upper() == "KEEP" or str(last.get("status") or "").lower() == "kept"` while `inference_optimizer/session/optimization_journal.py:28` exports `OUTCOME_KEEP`.
**Not a finding when:** the literal belongs to an external format the repo does not own (a framework CLI flag, an SDK `stop_reason`) and only one adapter branches on it; the comparison is inside the parser or adapter that performs the conversion; the vocabulary is open (model ids, paths, task ids).
**Evidence:** `$WORK/diff.txt` for the comparison; the owner module for the Enum, constant or registry that already names the value.
**Report as:** `A10 <file>:<line> -- compares <var> to raw "<literal>"; <Type>.<MEMBER> (<owner>:<line>) names it -- convert at <boundary> and compare the type`

### A11 -- A query-named function changes nothing, and a function that changes state is named for the change

**Severity:** blocking for a query-named function with an effect (`AGENTS.md` *Code shape*: "A function named as a query changes nothing"). Advisory for module-level mutable state written outside one named setter: no `AGENTS.md` sentence names it. An unstated observable side effect in the description is X1's finding.
**Fires when:** the diff adds or edits a function named as a query (`is_`/`has_`/`should_`/`can_`/`get_`/`find_`/`derive_`/`compute_`, or a property) whose body writes a parameter, assigns `self.<attr>`, writes module state, starts a process that changes something (a write, a kill, a launch; a read-only probe such as `grep` or `git rev-parse` is not), or exits; or a `resolve_*`/`parse_*` function that mutates its arguments or module state (reading files is not the trigger); or adds a `global` statement or a module-level mutable written by functions other than one named setter.
**The rule:** a query reads state and returns a value; calling it twice, or not at all, changes nothing. When a decision also has to be applied, split it into the pure computation and a command whose verb names the effect (`spend_on_session_reserve`, `write_*`, `record_*`, `apply_*`), and let the caller sequence them. Module-level mutable state is written only through one named setter in its owner, with the reason it cannot be threaded through a signature stated where it is declared; per-action state uses a `ContextVar`. The pure core then gets a unit test without a filesystem, and the effect has one place to fail.
**Seen in:** PR #1660 -- "session_closing() looked like a predicate but" wrote the budget state; after the fix "session_closing() is now a pure query" and callers spend the reserve through a new `spend_on_session_reserve()`. `inference_optimizer/cli/__init__.py:754 _resolve_models_for_run(args, ...) -> None` (PR #1112) rewrites `args.claude_model`/`args.codex_model`, probes the model catalog and exits; its own comment at :773 says "mutates args.claude_model on fallback; sys.exit(2) on failure". Counter-example: `common/llm_attribution.py:54 set_current_phase` writes a module global under a setter name with the reason stated beside it, and `_current_action` (:67) is a `ContextVar`.
**Not a finding when:** the function returns a value and writes an argument the caller reads to interpret it, which is A9's second channel; the write is a memo cache keyed by the arguments and invisible to callers (`functools.cache`, a lazy import of an optional dependency); a lazily-created, idempotent, process-private resource whose docstring says so (`_patch_snapshot._isolation_root`); logging or metrics inside a query; the verb already names the effect (`apply_`, `stamp_`, `ensure_`, `record_`, `observe_`), even when it returns its outcome as a `bool`; the mutated object was created by the function itself; a `warnings.append` collector the caller passes in for that purpose.
**Evidence:** `$WORK/diff.txt` for the function body; callers in the head tree for whether they call it more than once or conditionally.
**Report as:** `A11 <file>:<line> -- <fn> is named as a query but <writes <arg>|mutates <module state>|exits>; split into <pure fn> and <effect fn>`

### A12 -- No abstraction before its second real implementation, and no layer that only forwards

**Severity:** blocking. A new pure forwarding function, and a base class that declares nothing, rest on `AGENTS.md` *Simplify by removing a mechanism* ("a forwarding layer with no job of its own"); a single-implementation abstraction rests on *Code shape* ("An abstraction — Protocol, ABC, base class, factory, registry — arrives with its second implementation, not before"). A config knob with one value is C3's.
**Fires when:** the diff adds a `Protocol`, `ABC`, factory whose job is to choose an implementation (`select_*`, or a `make_*`/`build_*` typed to return an abstraction), registry or strategy dict with fewer than two production implementations or entries in the head tree; or adds a base class that declares no members of its own, whatever its subclass count; or adds a `def` whose body after the docstring is one `return`/`await` of another callable with the same arguments in the same order -- an `_impl` twin, a private alias of an imported function, a method forwarding to one collaborator method.
**The rule:** an abstraction is paid for by its second user. A Protocol, base class, factory or registry with one production implementation gives the reader an extra hop that decides nothing, and agents copying the file add more of the same seam. Use the concrete class, and introduce the seam in the PR that adds the second implementation. A function that only renames and forwards its arguments is a forwarding layer: call the target by its own name. The finding names the abstraction or forwarder, the single implementation or target, and the call sites that would bind to it directly.
**Seen in:** PR #1665 deleted `phases/base.py`'s `class PhaseHandler(CoordinatorCollaborator)`, which declared no members, so "every phase that named it reached its collaborator through one hop that decided nothing". PR #1642 -- "`_write_variant_abort_marker` forwarded to `_write_variant_abort_marker_impl` with the same signature", and the one-line `BaselineExecutor._resolve_lifecycle_params` was inlined into its only caller. `orchestrator/bringup/reap.py:93,185` (PR #1409): `class ReapBackend(Protocol)` with one implementation, `ProcessGroupReaper`, and `select_reaper() -> ReapBackend` that returns it; nothing under `src/` outside its tests uses either. `orchestrator/actions/executors/profile.py:699` (PR #1249): `_is_capture_trace(path, root=None)` only returns `_shared_is_capture_fragment(path, root)`, with the same private wrapper at `orchestrator/trace_analysis/_bypass_trace_reader.py:103` and `tracelens_analysis.py:1287`.
**Not a finding when:** the memberless base is an exception family that callers catch to handle its subclasses together; a builder does real construction work (assembles a parser, a config, an argv) rather than choosing an implementation; the seam is an exported contract consumed out of tree or by plugins (`agent_backends/registry.py`, a client interface with HTTP and in-memory implementations); the second implementation lands in the same PR; the Protocol is declared by a lower layer so a higher layer can implement it without an upward import (dependency inversion, A2); the forwarder adapts something (adds a default, converts or checks a type at runtime, binds `self` state, converts units, computes an argument) -- narrowing only the annotation is not adapting; it is a staged-migration adapter that names its remaining consumer (*Finish the replacement*); it is a documented test monkeypatch point.
**Evidence:** `$WORK/diff.txt` for the new class or def; `git grep` over the head tree, excluding tests, for implementations and callers.
**Report as:** `A12 <file>:<line> -- <abstraction> has one implementation <impl>|<fn> only forwards to <target>; use <impl|target> directly at <call sites>`

### A13 -- No ceremony the types or the callers already rule out

**Severity:** blocking for a guard on an internal value that its annotation and every in-tree caller make impossible (`AGENTS.md` *Trust the caller*: "redundant re-checks, layered fallbacks, and belt-and-braces defaults buy nothing"); the finding names the guarantee, and without it the item goes. Advisory for renaming locals and re-raise-only handlers. An unread parameter is A8's.
**Fires when:** added lines (1) re-check with `isinstance`, `is None`, `or {}`/`or ""` or a membership fallback a value that its non-Optional annotation and every in-tree caller already guarantee, or that the same function has already dereferenced unconditionally; (2) bind a local that only renames a parameter, import or local and is never rebound (`x_fn = x`, `filtered = list(items)` with no filter); (3) add `except E: raise` or `except E: log(...); raise` with no cleanup.
**The rule:** validate at the boundary, then trust internal callers. A guard for a state the types and callers make impossible is not safety: its else-branch is dead code a reader has to reason about, and agents copy the guard into the next function. Delete it with the defaulted branch it feeds. A local that only renames, a copy that filters nothing and a handler that only re-raises are the same ceremony. This rule is the concrete check for *Trust the caller*; a broad `except` is *Don't grow the debt*'s.
**Seen in:** PR #1658 deleted `if level not in _LEVELS:` from `inference_optimizer/multi_node/_internal/log.py` (its only callers pass a level they own) and `filtered_patches: list[Path] = list(patches)`, which filtered nothing, from `_server_patcher._discover_sglang_plan`. PR #1642 deleted the one-use alias `_tracelens_patch_enabled_fn = _tracelens_patch_enabled`. PR #1620 -- "getattr-with-default, dict.get, isinstance, str/bool/int coercions ... cannot raise. The except handlers in these try blocks were dead code."
**Not a finding when:** the value crosses a system boundary in this function (JSON/YAML/`from_dict`, an LLM or agent payload, CLI or operator input, subprocess output, a plugin, a file read from disk, even if annotated `dict`), or is a field of a record built from raw or persisted data without per-field coercion (`cls(**raw)`), wherever it was deserialized; the annotation is wider than the guard (`dict | None`, `Any`, `Mapping`) and some caller passes the other case; the earlier dereference is conditional (inside a short-circuit or a branch); the alias narrows a type for mypy, pins a value before a rebinding loop, or names a non-obvious expression; the handler cleans up before re-raising.
**Evidence:** `$WORK/diff.txt` for the guard; the annotation and the callers in the head tree for the guarantee.
**Report as:** `A13 <file>:<line> -- <guard|alias|handler> can never change the result: <annotation|unconditional dereference at :<line>|callers <set>> guarantees <fact>; delete it`

### A14 -- A name states what the code does, and one name keeps one contract within a package

**Severity:** blocking (`AGENTS.md` *Code shape*: "A name states what the code does; an alias or a shared role such as `log` keeps the meaning its package gives it"). When a colliding name has already sent a caller to the wrong behaviour in the diff, report the wrong behaviour under C1 or D10. A query name on a function with an effect is A11's.
**Fires when:** the diff adds or renames a function, method, parameter, module-level binding or import alias, and the name describes something the body does not do (a phase, state or effect that does not exist); or it gives a name a second contract where a reader takes one for the other: an import alias that takes a name the package already uses for a different contract (`import a as b` where `b` already means something else), or a module-level binding of a role its sibling modules all bind one way (a `log` that is not a `logging.Logger`).
**The rule:** agents pick a function by its name, not its body. A name that states a different job, or that collides with a sibling holding the opposite contract, sends the next caller to the wrong behaviour with no error. Name the function for the decision it returns or the effect it has, and keep one meaning for an alias or a shared role, such as a module-level `log`, within a package. Comment quality is *Comment below the local average*'s, not this rule's.
**Seen in:** PR #1658 -- "should_reloop_to_explore decides whether SWEEP opens another macro-cycle ... there is no EXPLORE phase in PHASE_NAMES"; "baseline imported _patch_touched_paths_from_text under the name _patch_touched_paths, which _patch_snapshot already uses for the opposite contract"; "Every sibling module in multi_node/_internal binds log to a logging.Logger; this one bound it to a function taking (level, msg)"; and the misspelt method `_find_frameworkoposal`.
**Not a finding when:** the name is fixed by an external contract (a CLI subcommand, persisted field, SDK hook, Protocol method); the two same-named symbols are private to different modules, neither is imported under the other's name, and no sibling convention binds the name (two private `_is_patched` helpers in two patchers); the rename is staged and the body names the follow-up.
**Evidence:** `$WORK/diff.txt` for the name and body; `git grep` within the package for an existing symbol of the same name.
**Report as:** `A14 <file>:<line> -- <name> <says <X> but does <Y>|already means <contract> at <file>:<line>>; rename to <name>`

## V -- Review method and PR hygiene

### V1 -- Confirm the finding is introduced by this diff, and that the diff contains nothing it did not intend

**Severity:** blocking
**Fires when:** always. This rule governs every candidate finding and every review.
**The rule:** a finding is reported only after confirming the line is added by this PR and is
not pre-existing context around an added line; compare against the merge base, because
behaviour that predates the branch is out of scope however wrong it looks. In the other
direction, the diff is scanned for changes the PR did not intend -- a file re-saved as
double-encoded UTF-8 turns em dashes and arrows into mojibake, and when a corrupted string is
a runtime value rather than a comment, persisted identity or dedup keys change shape and stop
merging with rows written before the change. Prove it with occurrence counts against the base.
**Seen in:** PR #1447 -- `writeback.py` and `integrate_patch.py` were re-saved as
double-encoded UTF-8, corrupting a persisted KB string, at 70 and 37 occurrences against 0 in
the base; PR #1038 reported an unconditional sync that the diff showed as unchanged context.
**Not a finding when:** the line appears in `$WORK/diff.txt` only as context (no leading `+`),
or the mojibake predates the merge base.
**Evidence:** `$WORK/diff.txt`, `$WORK/base.txt` -- added lines only, counted against the base.
**Report as:** `V1: <finding> is pre-existing at <base sha> -- dropped` / `V1: <file> re-saved
with <encoding defect>, <N> occurrences vs <M> at base`

### V2 -- Reproduce before reporting, and publish the verification depth

**Severity:** blocking
**Fires when:** always. This rule governs every candidate finding and the shape of the card.
**The rule:** the PR description's account of the defect and of the fix is not accepted as
evidence. The changed function is called directly on a constructed case, the matrix of shapes
the fix must cover is enumerated, and only the tests scoped to the change are run -- CI covers
the full suite. The published review has two parts: what the PR does and how, then blocking
issues only. When there are none, that is stated explicitly, with the head commit, the merge
base, the cases run and the tests run, so the author sees the depth rather than a bare LGTM.
Non-blocking observations are not appended in any form.
**Seen in:** PR #1412 -- the review listed what was verified rather than taking the description
at face value; PR #1420 ran four cases through the real `_seal_repository` on a repository a
killed session had left on `forge/controller/dead-1`.
**Not a finding when:** n/a -- this is a precondition for writing the card, not a finding.
**Evidence:** `$WORK/comments.txt` to avoid repeating a point already made;
`$WORK/testfiles.txt` for the tests scoped to the change.
**Report as:** the card's verification-depth section: `head <sha>, base <sha>; cases run:
<list>; tests run: <list>`

### V3 -- Name the root cause in one sentence before judging the fix

**Severity:** blocking
**Fires when:** always. This rule governs every review, and its correction half fires whenever
the body or a commit message states a causal claim.
**The rule:** the root cause is stated in one sentence and the diff is checked against it.
Accept that the symptom is real, then ask whether the change removes the cause or only narrows
the failure window -- a description listing several mitigations rather than one causal change
is the tell. Causal claims in the body and commit messages are verified against the data flow:
if the consumer named does not read that value, or the described failure cannot occur in this
system's execution model, the text must be corrected, because a wrong stated motivation
distorts the cost/benefit of the change for every future maintainer. `AGENTS.md` tells the
author to treat review feedback as a hypothesis; this is the reviewer-side obligation.
**Seen in:** PR #1258 -- the false `no_promote` was real but the cause was a cross-system
comparison, not the guard the PR added; PR #1097's body attributed the KB hardware dimension
to `detect_gfx_arch()` when it comes from `kb_hardware_slug(gpu_type, ...)`.
**Not a finding when:** the body's stated cause matches the data flow, even if the fix is
narrower than you would have written -- narrowness on its own is taste. It stops being taste
when the narrow fix leaves the cause in place and adds a second way to resolve something the
tree already resolves elsewhere: that is D11, and it blocks on its own terms.
**Evidence:** `$WORK/body.txt`, `$WORK/commits.txt`, `$WORK/diff.txt`.
**Report as:** `V3: the body states <claim>, but <consumer> reads <actual source> -- correct
the description` / `V3: the diff narrows <window> without removing <cause>`

### V4 -- Check for a conflicting or overlapping in-flight PR

**Severity:** advisory
**Fires when:** always. Escalate when another open PR touches more than three of the same
files, or when both PRs rewrite the same call contract.
**The rule:** before approving a large refactor or deletion, establish whether another open PR
touches the same files and whether the two disagree about what survives. Report the
file-overlap count and any place where one PR's premise is destroyed by the other -- that is
what gets resolved wrongly in a merge. Recommend a landing order, or ask for the PR to be
rebased on or explicitly stacked on the other, rather than letting the conflict be settled
silently.
**Seen in:** PR #1368 -- PR #1397 touched 19 files, 18 of them also touched here, and both
rewrote the `record_kernel_discovery` call contract, so landing this one first would erase
#1397's premise.
**Not a finding when:** the overlapping files are only lockfiles or test
fixtures, or the other PR is a draft the author has already stacked on this one.
**Evidence:** `$WORK/openprs.txt` -- other open PRs and the files they share.
**Report as:** `V4: #<N> touches <K> of the same files and <how the premises conflict> --
land <which> first`

### V5 -- CI must be green at the current head, proportional to the diff, and free of base conflicts

**Severity:** blocking
**Fires when:** always. Escalate when a check's conclusion is not success, when a success is
attached to a head sha older than the current one, when only a narrow subset of workflows
reported, or when the PR is reported as conflicting with its base.
**The rule:** the commit each passing check ran against is checked, not just its colour --
successes attached to an older head do not cover commits pushed afterwards, and a rerun on the
current head is required when later commits rewrote the code those checks validated. If a
narrow subset of workflows reported while the diff touches runtime logic, the full matrix is
required. A failing required status, docs builds included, must be resolved or shown to be
non-required with logs. A PR reporting conflicts with its base is rebased before review, not
reviewed around the markers.
**Seen in:** PR #703 -- at head `7881d481` only `Analyze (python)` passed and ReadTheDocs
failed, while the `Tests with Coverage` and `Lint` successes belonged to older heads and did
not cover the later `apply_and_bench` rewrite.
**Not a finding when:** the unreported workflows are path-filtered out by the doc-only
`paths-ignore` contract and the diff is genuinely doc-only.
**Evidence:** `$WORK/ci.txt` for the check runs and their head, `$WORK/meta.txt` for
`mergeable`, `$WORK/commits.txt` for what landed after the last green run.
**Report as:** `V5: <check> passed at <old sha>, current head is <sha>; <what landed since>
is uncovered` / `V5: PR reports CONFLICTING with <base> -- rebase before review`

### V6 -- Do not close an issue the change only partially fixes

**Severity:** blocking
**Fires when:** always. Escalate when the body contains `Fixes #N`, `Closes #N` or
`Resolves #N`.
**The rule:** the linked closing issue must be fully resolved by the diff. If the PR fixes one
call site or one deployment shape while the issue describes several, the issue stays open or
is split, and the body's closing keyword is removed. A workaround that does not meet a
constraint the issue explicitly states is not a fix.
**Seen in:** PR #710 -- the change fixed ProposalScorer only while #709 covered both call
sites; PR #703's pre-run cleanup workaround did not resolve #720, which explicitly calls out
that a blind `pkill -f hipcc|ninja|clang++` is unsafe on shared nodes.
**Not a finding when:** the issue's remaining scope is already tracked in another open issue
the body names.
**Evidence:** `$WORK/body.txt` for the closing keyword; `$WORK/diff.txt` for which of the
issue's cases the change actually covers.
**Report as:** `V6: <body> closes #<N>, but the diff covers <X> while the issue also
describes <Y> -- keep #<N> open or split it`
