---
myst:
    html_meta:
        "description": "Security compatibility variables and agent isolation reference for Hyperloom: credential inheritance switch, specialist user account, bubblewrap sandbox details, and isolation chooser."
        "keywords": "Hyperloom, environment variables, security, HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV, HYPERLOOM_SPECIALIST_USER, bubblewrap, agent isolation, bwrap, sandbox"
---
# Hyperloom Security compatibility switches

These switches keep production-compatible behavior by default while still
allowing operators to turn off credential/env persistence in hardened
deployments.

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV` | Unset (`1`) | Specialist subprocesses inherit the limited provider credential set by default: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_CUSTOM_HEADERS`, `CLAUDE_CODE_OAUTH_TOKEN`, `OPENAI_API_KEY`, `OPENAI_CUSTOM_HEADERS`, and AWS Bedrock credential/config vars. Set to `0` only when the `claude` CLI is authenticated through its own config and env credentials must be suppressed; a specialist in a bubblewrap sandbox cannot read that config, so where bubblewrap would sandbox it the launch runs unconfined, or is refused where it must be isolated (see [Agent isolation](#agent-isolation)). Unrelated secrets such as GitHub and KB tokens remain blocked. |
| `HYPERLOOM_SPECIALIST_USER` | Unset (no change) | Account name to run specialist subprocesses as (the `claude`, `codex` and `dsh` CLIs), so they cannot read the app's files, `/proc/<pid>/environ` or key files. Needs a Hyperloom process that can start a child as another user and later stop it: root, or all of `CAP_SETUID`, `CAP_SETGID` and `CAP_KILL` (`CAP_KILL` is what lets the deadline, stale-heartbeat, done-file and cancel reaps signal the child; without it none of them could be confirmed). The launch fails with a clear error, and no specialist runs, if one of those is missing, the account does not exist, is uid 0, is the app's own user, or belongs to group 0, or its uid, its group or a supplementary group is not mapped in the app's user namespace (`/proc/self/uid_map`, `/proc/self/gid_map`: in a container that maps only its root, a switch to any other id fails with `EINVAL`, or under a fakeroot `LD_PRELOAD` appears to succeed while the child stays root); so does a GPU specialist on the Ray actor path, which cannot be demoted, unless bubblewrap isolates it (see [Agent isolation](#agent-isolation)). The child gets that account's `HOME`, `USER` and `LOGNAME`, supplementary groups, and `umask 007` (group full access, others none); the Hyperloom process's own umask is left alone, and each launch re-shares the workspace and worktree, so files the app rebuilt in between become group-writable again. The child starts through `setpriv` (util-linux, required), which clears its inheritable and ambient capabilities and sets `no_new_privs`, so a non-root app that holds `CAP_SETUID` as an ambient capability does not hand it to the child. Each task's workspace and worktree stay owned by the app but get the account's group with group read-write (setgid directories); the app must be root or a member of that group to do so, and the account must be able to traverse every parent directory of the session directory (the app never changes the modes of those parents), read the checkout the worktree is branched from, and execute the agent CLI. The worktree's `.git` file and its admin directory under the repository's `.git/worktrees` are shared read-only (git inside the worktree works for status, diff and log, not for writes), and a copy of the `--mcp-config` file is staged in the workspace. The app reads and writes the paths the child can rewrite (`specialist_done.json`, `patches/`, the harvested patch, the Codex config, `prompt.md`, `process.log` and its redaction copy, the `dsh` home and session logs) without following symlinks. The agent runner (`LocalAgentRunner`, `orchestrator/specialists/agent_runner.py`) demotes the same way, sharing the session's working directory as the worktree; there an unusable account fails the launch with `AgentConfigError`. For the isolation chooser below, setting the variable also makes a specialist's isolation required (see [Agent isolation](#agent-isolation)). |
| `HL_ALLOW_DANGEROUS_AGENT_PERMISSIONS` | Unset (`0`) | Slurm carrier only. Set to `1` only in dedicated internal containers to re-enable legacy Claude/Codex approval and sandbox bypass flags. |

## Agent isolation

`hyperloom.common.agent_isolation` decides how an agent is launched; there is no variable that turns it on or off.
`choose_agent_isolation()` (a specialist) takes the first tier that works:

1. **Account**: `HYPERLOOM_SPECIALIST_USER` names an account that resolves and passes the checks in the table above,
   and the process holds `CAP_SETUID`, `CAP_SETGID` and `CAP_KILL`. Never on the Ray GPU actor path, and never for an
   agent sandbox's commands (`choose_sandbox_isolation()`), which run as the app's own account.
2. **bubblewrap**: the agent runs in a sandbox that starts from an empty root (`hyperloom_common/bwrap.py`): only the
   paths its confinement binds exist inside, read-only unless they are its work trees, and the root takes no new name;
   the credential locations inside a bound tree are hidden (the app's home and XDG directories, every path an
   environment variable with a credential-shaped name points at, Hugging Face's login files, made yet or not, where
   `huggingface_hub` puts them (`HF_TOKEN_PATH` and `stored_tokens` beside it, else under `HF_HOME`,
   `XDG_CACHE_HOME` or the home), and the files the app registers): a directory is masked, and a read-only tree that
   holds a file, or the first missing directory on the way to a login not made yet, is bound entry by entry without
   it, so a login the app replaces or makes there after the launch is not shown either. A credential file in a
   work tree refuses the launch, naming it. It has its own
   user, pid and IPC namespaces, its own `/proc`, no capability and `no_new_privs`, and shares the app's network
   (loopback MCP). A program's own directory is bound, and its parent only away from those credential locations, so
   a native install under `~/.local/bin` does not make the rest of `~/.local` readable. Nothing is bound, run or
   read for the confinement through a tree the agent writes (its work trees and scratch): a symlink it leaves there
   never becomes a bind of its target at the next launch, and a read-only path inside such a tree refuses the launch.
   A work path reached through another work tree is not bound on its own (that tree shows it), and where work paths
   are reached through each other's trees, which takes a link in one of them, none of them is bound, and each still
   counts as a tree the agent writes. The masks come from the app's own environment and home as well as the child's. Everything under the readable
   system roots (`/etc`, `/opt`, `/usr`) is readable: an image must not keep a credential there.

Only an agent launched through the chooser is isolated: every specialist (the dispatcher, its Ray GPU actor path and the
agent runner). Agents started elsewhere run as the app and can read what it can, in a hosted run too:

- the orchestration role (`inference_optimizer/cli/backends.py`, `_build_orchestration_backend`), on whichever agent
  CLI it runs: on Codex, the backend every non-critic role gets when the only credential is OpenAI's
  (`orchestrator/roles/codex.py`, `CodexBackend._session_for`), its shell commands run in Codex's own sandbox, which
  limits writes, not reads, and takes `bwrap` from `PATH`; on the DeepSeek Harness (`orchestrator/roles/deepseek.py`,
  named only by `INFERENCE_OPTIMIZER_AGENT_BACKEND=deepseek`) the harness's read-only mode likewise limits writes; on Claude it
  has no shell, but its `Read` (`orchestrator/mcp_tool_names.py`, `ORCHESTRATION_BUILTIN_TOOL_NAMES`) is limited to
  no tree;
- the TraceLens skill runner's agent (`orchestrator/trace_analysis/tracelens_skill_runner.py`): Claude with `Bash`
  (`DEFAULT_ALLOWED_TOOLS`), Codex (`_run_tracelens_skill_codex`) or the DeepSeek Harness;
- GEAK's agents (`orchestrator/kernel/geak_runner.py`, which starts GEAK's `run_e2e`);
- kernelforge's agents, each in a `python -m kernelforge.cli` child: `kernel-rewrite-controller`'s forge loops
  (`orchestrator/kernel/controller_submit.py`), `forge-fuse`'s kernel author (`orchestrator/kernel/forge_fusion.py`)
  and `gemm-tune`'s generated-tuner author (`orchestrator/kernel/forge_gemm_tuning.py`; `kernelforge/workflows/gemm_tune/tier3`);
- the quantization agent (`agents/quantization/driver/runner.py`: Claude with `Bash`, or the DeepSeek Harness), which
  runs only with `--quantize` and `HYPERLOOM_QUANTIZE_ENABLED` set; no hosted spec field sets `--quantize`.

The critic and the one-shot Claude calls start no agent with tools. A hosted run whose spec's `flow` is `multibranch`
is handed to `hyperloom-multibranch` before the entry's isolation check (below); `hyperloom-multibranch hosted run`
makes the same check at its start, with the same layer bubblewrap, and also ends the run, naming which and why, when
the agent runner would start its specialists unconfined or refuse them.

Specialists in a bubblewrap sandbox (`orchestrator/specialists/launch_env.py`; the dispatcher, its Ray GPU actor
path and the agent runner) write their workspace and worktree, and a private `/tmp` and `/dev/shm` (directories of
mode 0700 on the host's `/tmp` and `/dev/shm`; with a bubblewrap older than 0.12, a fresh tmpfs where the workspace
lies below `/tmp`). `HOME`, and `CLAUDE_CONFIG_DIR`, `CODEX_HOME`, `XDG_CACHE_HOME` and the compiler caches
(`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `AITER_JIT_DIR`, `FLYDSL_RUNTIME_CACHE_DIR`) when they name a
directory elsewhere, point into the workspace, and `TMPDIR` at `/tmp`; `AITER_JIT_DIR` and `FLYDSL_RUNTIME_CACHE_DIR`
are set into the workspace when unset too, since AITER otherwise builds its JIT modules in its installed tree, which
is read-only there (a GPU specialist so compiles them afresh). With neither `HF_HOME` nor `HF_HUB_CACHE` set, Hugging
Face's default directory under the app's home (`$XDG_CACHE_HOME/huggingface`, else `~/.cache/huggingface`) becomes the
child's `HF_HOME`, read-only and without its login files, so a model downloaded there stays visible once `HOME` has
moved (where that directory is a link, `HF_HOME` names the directory it leads to, which is what the sandbox shows). They read their toolchain, the git metadata of the worktree's repository (the one whose admin directory the
repository records for the worktree, never the one the worktree's `.git` file names: the agent can rewrite that file,
and runs as the app's own uid, so no ownership check tells a directory it made from the app's), the framework source
roots, the task's model when it is a local path, the programs of the stdio MCP servers they start (each named in one
WARNING: nothing else such a server reads is bound unless the confinement shows it), and, for an agent runner launch,
`AgentLaunchSpec.readable` (a multi-branch specialist's: the session's `MB-reference/<name>` directories, the
directory its workload's model is served from, the files its worktree links and its branch's own dev lease area,
`branch-sandboxes/<branch>` of a local toolbox or the branch's `dev` directory of a hosted one, so it applies a Forge
patch from its lease's root and sees no other branch's lease); a GPU task gets `/dev/kfd` and `/dev/dri`. The paths `HYPERLOOM_STATE_DIR`,
`HYPERLOOM_BUILD_STORE` and `HYPERLOOM_BUILD_TRUST` name are masked wherever a bind would show them (a hosted run's
entry, which takes `HYPERLOOM_STATE_DIR` out of its environment, registers its state directory so it stays masked). The `claude` CLI
runs there only with a login in its environment: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` or
`CLAUDE_CODE_OAUTH_TOKEN` (`claude setup-token` prints one), or `CLAUDE_CODE_USE_BEDROCK=1` with `AWS_ACCESS_KEY_ID`
and `AWS_SECRET_ACCESS_KEY`, or with `AWS_SHARED_CREDENTIALS_FILE` or `AWS_CONFIG_FILE` naming a file, which is then
bound read-only. Its own login file is not shown, since that would show it to the agent: without one of those, a launch
that must be isolated (below) is refused, and any other runs unconfined, as on a host without bubblewrap, with the
WARNING and the session report's line below naming why (`a claude CLI with no login in its environment runs
unconfined: ...`). `codex` runs with its
own sandbox off whatever `HYPERLOOM_CODEX_SANDBOX_MODE` says, since bubblewrap confines it and codex's own sandbox
cannot map a user namespace inside. bubblewrap ends a sandbox on a SIGTERM of its own without passing it on, so a
deadline, stale-heartbeat, done-file or cancel kill first TERMs the sandbox's processes that carry the launch's reap
mark and waits for them, then kills what is left. A sandbox bubblewrap could not set up fails the task with
bubblewrap's line (`specialist sandbox could not be set up: bwrap: ...`) rather than as the agent's exit, read from
its `--json-status-fd`; an agent killed by signal `n` exits `128 + n`. A Ray GPU actor's sandbox is built from the
driver's probe and paths, so it is for a single-node Ray cluster only, and its status goes to a file a shell in the
actor opens for bubblewrap; the sandbox has `--die-with-parent` there, so it ends when the actor dies (an OOM kill,
say). The app creates the launch's `process.log` fresh, as for a local launch, and the actor only appends to that
file: a link or anything else put at the name is refused, never written through.

A specialist's sandbox has a network of its own (`--unshare-net`): nothing that listens on this host or beyond is
reachable from it, the run's Ray head included (its dashboard Jobs API, client server and workers run what they are
sent as the app). What it needs is relayed in over Unix sockets in its scratch `/tmp` by the app
(`orchestrator/specialists/sandbox_network.py`): each loopback endpoint its launch names (an HTTP MCP server, an LLM
gateway on this host) at the same port on the sandbox's own loopback, and an HTTP proxy, which the sandbox's
`HTTPS_PROXY` and `HTTP_PROXY` name (loopback exempt), that connects only to the other LLM endpoints and MCP
servers its launch names: `ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL` and any other `*_BASE_URL` variable, the value
of codex's `-c model_providers.<name>.base_url=` setting (no other text on the command line, a dsh task say, names
an endpoint), each provider's API host where no base URL names one (`api.anthropic.com`, `api.openai.com`), with
`CLAUDE_CODE_USE_BEDROCK` the AWS hosts a Bedrock login uses (`bedrock-runtime`, `sts`, `portal.sso` and `oidc` in
each region named by `AWS_REGION`, `AWS_DEFAULT_REGION`, `ANTHROPIC_SMALL_FAST_MODEL_AWS_REGION` and the `region`
and `sso_region` keys of `AWS_CONFIG_FILE`, `us-east-1` where none of the first two is set, and the global
`sts.amazonaws.com`), and the HTTP servers of the MCP config (for an agent runner launch, the one it writes from
`AgentLaunchSpec.mcp_servers`, for codex as for claude). A `CONNECT` reaches any of those; a plain `http` request only
one named with `http://`. Through the app's own `HTTPS_PROXY`/`HTTP_PROXY` (with any login in its URL), unless
`NO_PROXY` exempts the host, the bridge opens its own `CONNECT` to the checked host and port and carries the
specialist's bytes only inside that tunnel, so nothing it sends later on the connection reaches that proxy as a
request; the app's proxy settings are not shown inside. Anything else gets a 403:
a sandboxed specialist downloads no model or package; what it reads must be on the host already (a Hugging Face
model in the cache shown to it, say). A loopback endpoint below port 1024 cannot be served inside (a WARNING names
it). The relay is a `python -I -S` of the app's interpreter that starts ahead of the agent and ends with the
sandbox.

The app's own git over a specialist's worktree (the commit it starts from and the harvested patch) goes through the
admin directory the repository recorded when the worktree was made, in every tier; a worktree the repository does not
record (a directory left at its path, say) is not used, and the specialist runs in its workspace. After the run the
app collects patches, by git or from `patches/`, only while the worktree's path still names the directory it had when
the specialist started: a specialist that writes its workspace can leave a link to any directory of the host there,
which the app's git and patch write would otherwise follow. Otherwise no patch, harvested, scanned or claimed, is
taken from the run, a link at that path is removed, and the task's error says `worktree_refused: <path> ...`. While
the specialist runs, its checkpoint in the worktree (`specialist_done.partial.json`) is read only through that same
directory, and none is read while the path names another; the session's resource record gives no disk use
(`worktree_disk_bytes: null`) for a worktree path that is a link.

Where neither works, a launch that must be isolated refuses, naming the reasons of both tiers: every launch through the
chooser in a hosted run (its entry calls `require_agent_isolation()` once at start-up, so a host where neither works refuses before
anything runs), and a specialist's wherever `HYPERLOOM_SPECIALIST_USER` is set. Any other launch runs unconfined as
before; `announce_agent_isolation()`, which the optimize command calls at start-up, then logs one WARNING
(`agents run unconfined on this host, as before: they can read this account's files and credentials (<reasons>)`), and
the session report (`reports/final.md` and `final.json`'s `agent_isolation`) carries its line (`agent isolation:
specialists <account|bwrap|none|refused>, sandboxes <...> (bubblewrap X.Y.Z PATH): <reasons>`). A launch that runs
unconfined although start-up found bubblewrap usable (its probe gave no answer in time on a loaded host, say, or a
`claude` CLI with no login in its environment) logs that WARNING itself, once per process, and the report adds
`agent isolation: N agent launches ran unconfined: <reasons>`.

A multi-branch session's local agent sandboxes (autonomous branches, the dossier pass, dev leases) run every command
of one sandbox in one bubblewrap sandbox whose user and pid namespaces all its commands share, so a server one command
starts is seen, signalled and stopped by a later one (`pkill`, or its PID file against the sandbox's own `/proc`). They
share the sandbox's private `/tmp` and full-size `/dev/shm`, each command has its own IPC namespace (System V IPC does
not outlive a command; POSIX shared memory in `/dev/shm` does), and the network is the app's. A command writes the
lease's root and, for an agent, its directory and worktree (a dev lease's worktree and what its links point at are
read-only); the image's installed frameworks are read-only but aiter, which writes its JIT modules and tuning files
beside itself. Its environment is not the app's: the search paths, locale, models, the image's own settings and the
LLM gateway with its credentials (`ANTHROPIC_*`, `OPENAI_*`, `CLAUDE_CODE_OAUTH_TOKEN`, the proxies and CA bundle),
with `HOME`, `CLAUDE_CONFIG_DIR` and `TRITON_CACHE_DIR` of the sandbox's own (an agent's home is `sandbox/home` in its
branch's session directory, where its Claude CLI's transcripts are read from). A Claude CLI there logs in only through
those variables, never the app's `~/.claude`. pip installs to `PIP_TARGET`, `site-packages` in that home, which is on
no interpreter's import path, so no server a command starts imports what was installed, nor the dependencies an install
brought, ahead of the image's (the image's venv takes no user install and is read-only); an agent's adapter may put
it on the path of the agent's own commands only. `AITER_JIT_DIR` and `FLYDSL_RUNTIME_CACHE_DIR` are aiter's
defaults, its own writable tree, where the image's prebuilt JIT modules and FlyDSL kernels are. A sandbox whose paths
bubblewrap cannot bind safely (before 0.12, nothing may be bound below the shared `/tmp`, so a session directory under
`/tmp` cannot be) runs unconfined, as before, with a WARNING; a hosted run refuses it.

A multi-branch session measured on this host's cards runs each test's runner, and the server it starts with the
branch's patches, in a local sandbox (`hyperloom.sandbox.confined`) under the same rule: where its provider names no
account for it (`ProcessUsers`), bubblewrap where `choose_test_isolation()` gives it, else as before with one WARNING
(`tests run unconfined on this host, as before: they can read this account's files and credentials (<reasons>)`); a
hosted run refuses to build the toolbox instead. Each sandbox writes its own `fs/` (`/in`, `/out`, `/work`), a private
`/tmp` and a full-size `/dev/shm`, and a sealed one its own `private/`, where the runner reads its signing key and
removes it; it reads its toolchain, its interpreter, the model and what its entries link to (each link's target
itself, never the directory it is in; a hub snapshot's repository `blobs`), the Hugging Face hub cache, and the trees
its template names (the framework snapshot, the staged rule pack, the compiled-code store, the runner cache, the native
build store and trust file, a local InferenceX repository); the app's credential locations those show are
masked, and a sandbox one of whose read paths is itself a credential location (a home, a login file) runs as before
with a WARNING, refused in a hosted run. So does one whose model links to or below a credential location, or into a
hidden entry of a home (`~/.ssh`, `~/.claude`, `~/.config`, `~/.aws`, `~/.netrc`), where no mask could hide the logins;
a link to anything else in the home (`~/tokdir`, `~/config.json`), outside it, or into the hub cache binds as before,
except into a credential location the hub cache holds (one set to the home's parent), whose mask the link would open.
Nothing else of the host is there: not the session's other files, another sandbox's directory or key, the app's state,
nor any process of the app's (its own pid namespace; a sandbox that runs several commands shares one holder's, as an
agent sandbox does). A sandbox with cards gets `/dev/kfd` and `/dev/dri`, `/sys`
is read-only, the network is the app's (its server listens on loopback for the benchmark client). The sandbox's id
(`E2B_SANDBOX_ID`) is set inside only, so the reap, the card guard and the tag sweep find its processes and never
bubblewrap's own, and a reap's SIGTERM reaches them with their grace. Before bubblewrap 0.12 a sealed runner whose
session directory is below `/tmp` gets a fresh `/tmp` of its own instead of the bound one. A probe that gives no answer
in time leaves only that launch's test unconfined: the next sandbox is chosen for again. The app opens a sandbox's files
(the runner's output and logs) by one walk below its `fs/` that follows no link out of it, so a process of the sandbox
cannot swap a link to the app's files in between a check and the open.

A hosted run uses only its hosting layer's bubblewrap, by the path its entry names, never one found on `PATH` (the
framework image may carry its own); a local run uses the `bwrap` on `PATH`. `hyperloom hosted run` and
`hyperloom-multibranch hosted run` name `bin/bwrap` in the directory `HYPERLOOM_HOST_LAYER` names (whitespace around it
ignored), which the layer's `start.sh` sets to its own; with the variable unset, blank or relative, no launch of the run
uses bubblewrap. `hyperloom hosted run` checks before its installer or `optimize` runs: where neither tier
works, the run ends with a final `exit.json` (return code 2) whose `error` names both reasons; a probe that gave no
answer in time ends the leg without one, so the host starts it again. Where the account is chosen, it probes the layer's
bubblewrap too: the Ray GPU actor path never starts a specialist as an account, so where that bubblewrap cannot confine
one, one WARNING says so (`specialists run as the account <name>, but the Ray GPU actor path cannot run the specialist
as another user and bubblewrap cannot confine one here, so that path refuses its specialists: <reason>`), and the run
goes on. The probe (`python -I -m
hyperloom_common.bwrap probe [--bwrap PATH]`) refuses a setuid bubblewrap and one older than 0.5.0, then runs a canary
sandbox and checks from inside that its masks, read-only binds, pid namespace, capabilities and `no_new_privs` hold. It
runs once per process for each binary, interpreter, ids, namespaces and user-namespace sysctls; one that gives no
answer within 10 s is not kept, and refuses a hosted start-up as transient. A failed probe names the step with
bubblewrap's own line; the usual causes:

| Probe line | Cause |
|------------|-------|
| `bwrap: setting up uid map: Permission denied` | The kernel created the user namespace without capabilities: with `kernel.apparmor_restrict_unprivileged_userns=1` it does so for a process without `CAP_SYS_ADMIN`. |
| `bwrap: Creating new namespace failed: ...` | `user.max_user_namespaces` is 0 or used up, or a seccomp or AppArmor profile denies `unshare`. |
| `bwrap: Can't mount proc on /newroot/proc: Operation not permitted` | The container's `/proc` has entries mounted over it (masked paths). |
| `bwrap: Can't bind mount ... Invalid argument` | The root mount is marked unbindable (some container runtimes); the probe retries every launch under `unshare --mount`: the `unshare` on `PATH` for the `bwrap` on `PATH`, and for a `bwrap` named by path (a hosted run's layer's) only an `unshare` beside it, never one from `PATH`. Without one the probe refuses, and the host must make the root mount private first (`mount --make-rprivate /`), as the layer's `start.sh` does. |

A native build's compile (`hyperloom.measure.runner.native.isolated`) has its own two tiers, also without a switch:
user 65534 through `setpriv` (in a network namespace of its own where `unshare --net` works) when the build service is
root and its user namespace maps 65534, else bubblewrap with the same probe and the same choice of binary, where it
also starts a sandbox with a network of its own (the probe's canary keeps the host's network, so a host that allows no
new network namespace is named here). Under bubblewrap the compile runs as the service's own user and keeps its
supplementary groups (as root where the user namespace maps only root), so it reads in the bound `/etc` and `/opt`
whatever that account reads, as an agent's sandbox does; only its work directory is writable, the toolchain, both
interpreters, the tree the child imports from, the image's packages and the upstream build directory are read-only (an
aiter module build lays its farm in the work directory, and the image's aiter directories the farm links into are
read-only), it sees only `/sys/devices/system/cpu` of `/sys`, the base `/dev` nodes and no GPU, it has a network of
its own (outside addresses give `ENETUNREACH`), and the signing key and trust file (registered as the app's private
paths) are absent, also where a bound tree holds them (that tree is bound entry by entry without them); its environment
is only `PATH`, `HOME`, `TMPDIR` and `LC_ALL`. The build record's `isolation` entry is `uid+netns`, `uid` or
`bwrap+netns`. Where neither tier works, the toolbox refuses at start (`build_preflight`) and each build fails as the
machine's, naming both reasons as its choice found them; a bubblewrap whose probe gives no answer in time is not
refused but logged, and each build asks again.

A multi-branch session records in its manifest and `MB-report.json` (`agent_isolation`) how the paths that launch its
agents and tests start them (`hyperloom_multibranch.scheduling.launch_isolation`): its specialists as the agent runner's
chooser gives them now (`account`, `bwrap`, else `none` with the reasons, or `refused` where they must be isolated and
no tier can); its agent and dev sandboxes and its tests on this host as their choosers give them now (`bwrap`, else
`none` with the reasons, or `refused` in a hosted run whose layer bubblewrap cannot sandbox them), or `hosted` when the
grant's sandboxes run them. A sandbox whose own paths bubblewrap cannot bind safely runs as before even where the record
says `bwrap`, which its own WARNING says. `MB-report.md` has the same as one line (`agent isolation: specialists <...>,
sandboxes <...>, tests <...>: <reasons>`), and the session logs one WARNING at its start naming what runs unconfined. A
multi-branch `hosted run` starts only when nothing it launches in the app container would run unconfined or be refused:
with the layer's bubblewrap its specialists, and with `placement.measurement: local` its sandboxes and tests too, run
under it; a spec measured there ends before anything starts while one of them would not.

The tests that launch real sandboxes (marker `agent_isolation`) skip on a host without bubblewrap or a user namespace
for it; with `HYPERLOOM_REQUIRE_ISOLATION_TESTS=1`, which CI sets after allowing unprivileged user namespaces, they
fail instead.
