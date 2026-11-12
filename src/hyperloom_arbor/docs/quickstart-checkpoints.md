# Quickstart: pause, continue, move and fork a campaign

A campaign writes a checkpoint at every pass boundary, so it can be stopped cleanly and continued later, carried to
another node, or branched to try something else from the same point. This page shows each, step by step. The reference
is [operations.md, "Checkpoints, pause and fork"](operations.md#checkpoints-pause-and-fork); every flag is in
[cli.md](cli.md).

| You want to | Command | Ends with |
| --- | --- | --- |
| See what you can go back to | `python3 -m arbor checkpoint list --config <cfg>` | the checkpoints, their names, the session's lineage |
| Stop cleanly, continue later | `python3 -m arbor pause --config <cfg>` | the server stopped, checkpoint `pause-NNNN`, exit 76 |
| Continue | `python3 -m arbor run --config <cfg>` | the next iteration |
| Continue on another node | `python3 -m arbor restore --config <cfg>`, then `run` | the exact tree of the checkpoint, then the next iteration |
| Try another prompt from here | `python3 -m arbor checkpoint fork --from <cfg> --config <fork cfg> --prompt-profile <dir>` | a new session, a copy of this one up to the checkpoint |
| Try another map from here | the same, with `--fork-policy exploratory --directions <map.json>` | a new session with the new map |

Names in angle brackets are yours: `<cfg>` is the campaign's config, `<session>` its `session_root`.

## 1. What you have already

- **Checkpoints are on by default.** After every pass (one iteration, or a round end) the session writes one into
  `<session>/checkpoints/`, as `NNNN.json` for the checkpoint after iteration NNNN. `"checkpoints": false` in the
  config turns them off. A checkpoint that fails to write is printed and never stops the run.
- **What one holds:** the session's state and the lengths of its logs; the served tree as patches against the root,
  with its exact git history; the config (credentials redacted), the pack and the prompts in use; the dossier; and a
  fingerprint of the model, framework, image and operating point. Each file is stored once, named by its digest.
- **Naming one.** Wherever a command takes a checkpoint (`REF`), give an iteration (`12`), a name (`pause-0012`,
  `before-routing-work`), or `latest`.

```bash
python3 -m arbor checkpoint list --config <cfg>              # every checkpoint and name, and where the session came from
python3 -m arbor checkpoint show --config <cfg> 12           # one manifest
python3 -m arbor checkpoint verify --config <cfg> --all      # every object, id and format rechecked
python3 -m arbor checkpoint create --config <cfg> --name before-routing-work   # a named one, now (no run may be going)
```

A named checkpoint is a frozen copy under `checkpoints/names/`. Name the points you may want to come back to.

## 2. Pause, then continue

Use it to free the GPUs, to change something between iterations, or to end the day without losing work.

```bash
python3 -m arbor pause --config <cfg> --reason "node needed for an A/B"
```

1. **`pause` returns at once.** It writes `<session>/PAUSE` (when, why, by whom) and nothing else.
2. **The run finishes the iteration in flight first**, verdict included. That can take as long as an iteration does.
   Changed your mind before then? `python3 -m arbor pause --config <cfg> --cancel`.
3. **At the boundary the run:**
   - stops its server;
   - checks that the tree is committed;
   - writes the checkpoint `pause-NNNN`;
   - records `<session>/paused.json`;
   - prints `[arbor] Paused (...); exit 76.` and exits with code **76**, not 0, so a script can tell a pause from an
     end.
4. **To continue,** run the same command that started it:

   ```bash
   python3 -m arbor run --config <cfg>
   ```

   It carries on at the next iteration, with the same bars, map and counters.

**Not a pause:**
- **`STOP`** in the session ends the campaign.
- **Ctrl-C, or a host's SIGTERM,** stops it at once: the iteration in flight is NOT MEASURED, and the previous
  boundary's checkpoint is what remains.

## 3. Continue on another node

The session directory holds everything the new node needs. The container and the served tree are rebuilt there.

1. **The session.** Make `<session>` available on the new node, `checkpoints/` included: shared storage, or a copy.
   Keep the config's `session_root` pointing at it.
2. **The container.** Start it from the **same image** as before: `restore` compares the image's tree with the
   checkpoint's and refuses another.
3. **The served tree.** Make it a git repository whose first commit is the image's code, exactly as at the first start
   ([quickstart-master-worker.md §1.6](quickstart-master-worker.md#16-the-workspace-the-served-tree-as-a-git-repository)).
   Leave it at that first commit.
4. **Restore, then run:**

   ```bash
   python3 -m arbor restore --config <cfg>              # the latest checkpoint; --checkpoint REF for another
   python3 -m arbor run --config <cfg>
   ```

`restore` verifies the checkpoint, rebuilds its `HEAD` commit for commit, then compares the tree it built with the
checkpoint's, byte for byte. Every restore is recorded in `<session>/checkpoints/restores.jsonl`. It also puts the
dossier into this Arbor checkout when the checkout has none.

## 4. Fork: try something else from the same point

A fork is a new session that starts as a copy of another one at a checkpoint. The parent is not touched, and the two
never update each other.

### 4.1 The fork's config

Copy the parent's config, then give the fork its own:
- **`session_root`:** an empty directory;
- **`docker_container` and `workspace_root`:** another container and served tree, so the two branches never edit
  each other's tree (another GPU set, or another node);
- **`dossier`:** another name (a directory under `docs/dossier/` in the Arbor checkout), so the two never write the
  same dossier files.

A fork refuses to share any of the four with its parent. With `--force`, the fork may use the parent's workspace and
dossier, for a parent that will not run again.

### 4.2 An exact fork: the same campaign, another prompt

An exact fork keeps the parent's bars, counters, map and iteration. Only the instructions may differ.

```bash
mkdir -p prompts/terse && cp <my claude_loop_prompt.md> prompts/terse/claude_loop_prompt.md
python3 -m arbor checkpoint fork --from <cfg> --checkpoint pause-0012 --config <fork cfg> \
    --branch terse-prompt --prompt-profile prompts/terse
python3 -m arbor run --config <fork cfg>
```

- **`--prompt-profile`:** a directory with `claude_loop_prompt.md`, `critic_prompt.md`, or both, under the built-ins'
  names. It is copied into the fork's session. A config can also name one itself (`prompt_profile`, or
  `prompt_overrides` for single files).
- **`--branch`:** the branch's id; by default, the fork session directory's name.

### 4.3 An exploratory fork: another map, or fresh counters

```bash
python3 -m arbor checkpoint fork --from <cfg> --checkpoint 12 --config <fork cfg> --branch moe-first \
    --fork-policy exploratory --directions <directions.json> --reset null_streak --reset barren_streak
```

- **`--directions`:** a map that replaces the parent's: an object with a `directions` list, as `directions.json` is.
- **`--reset`:** sets a scheduling counter to 0: `null_streak`, `probe_streak`, `tier_iters`, `barren_streak`,
  `secondary_only_streak`, `invalid_streak`, or `all`. Repeat the flag for several.
- **Recorded:** every change goes into the fork's `fork.json` and a `NOTE` row of its history.

### 4.4 What a fork writes, and how to read the result

- **The copy:** the parent's session as of the checkpoint: its state files, its logs cut back to their lengths, its
  iterations up to the checkpoint. It copies; it never links.
- **Left out:** the parent's stop and pause markers, its heartbeat, its exit records, its clock and its checkpoints.
- **Its lineage:** `fork.json` names the parent's session, checkpoint, id and iteration, and the policy.
  `checkpoint list --config <fork cfg>` shows it.
- **Its own first checkpoint,** named `fork`.
- **The tree:** restored into the fork's workspace, unless `--no-restore` (then run `arbor restore` before `run`).

Then `arbor run --config <fork cfg>` starts at the parent's next iteration.

- **Every row names its instructions.** Each lesson row records `provenance`: the prompts' digest, the pack's, the
  config's, and the Arbor commit. So when two branches differ, the rows say which instructions produced which verdict.
- **Comparing branches:** run a same-node A/B of their trees. Neither merges into the other by itself; to combine them,
  integrate one into the other and measure again.

## 5. Under a host

`pause` works the same in component mode. The exit record's reason is `paused` and the code is 76, so a host can tell
a pause from an end or a crash.
- **A host that does not know exit 76** may read it as a crash and restart Arbor in the same session. Arbor then
  carries on at the next iteration: the pause is only a checkpoint.
- **A host's own graceful stop is usually SIGTERM.** The iteration in flight is NOT MEASURED, and the previous
  boundary's checkpoint stays.
- **Restoring under a host** is the host's call: it decides which session, container and tree a run gets.

## 6. Which Arbor continues which session

`python3 -m arbor version --json` gives:
- **`interfaces.session`:** the session format this Arbor writes;
- **`continues.session`:** the range of formats it continues.

A session's format is its latest checkpoint's. `run`, `master` and `worker` refuse, with exit 2, a session outside the
range: continue it with an Arbor that covers it.

## 7. What goes wrong

| What you see | Why | What to do |
| --- | --- | --- |
| `pause` returned, but the run goes on | it stops only at the next boundary, after the iteration in flight has its verdict | wait for it; `--cancel` takes the pause back |
| `... a driver is running on <session> ...; pause or stop it first` | `checkpoint create`, `restore` and `fork` need the session to themselves | pause the run, or wait for it to end |
| `the workspace's root tree is ..., the checkpoint's ...: another image or base` | the container is not from the checkpoint's image, or the tree's first commit is not the image's code | start the container from the same image, and rebuild the tree's first commit from it; `--force` only if you mean to change the base |
| `the workspace is not at its root (HEAD ..., root ...)` | the tree already has commits or edits | `--force` resets it to its first commit before the restore |
| `checkpoint NNNN holds no tree` | the tree held a credential-shaped value when the checkpoint was taken, so it was not captured | pass credentials in the environment, never in `arbor/config.json`; restore an earlier checkpoint that holds a tree |
| `verify` fails, or `restore` and `fork` refuse a checkpoint | an object is missing or altered, or a captured text holds a credential | use another checkpoint (`checkpoint list`) |
| `the fork's session_root is the parent's`, or `... session ... is not empty` | a fork needs a new, empty session | give the fork's config its own empty `session_root` |
| `the fork's container and workspace are the parent's` | both branches would edit one tree | give the fork its own `docker_container` and `workspace_root`; `--force` if the parent will not run again |
| `the fork's dossier is the parent's (...)` | both branches would write one dossier | give the fork its own `dossier` name |
| `an exact fork keeps the map and the counters: --directions and --reset need ...` | `--directions` and `--reset` change what an exact fork keeps | add `--fork-policy exploratory` |
| `run` exits 2 on a session from another Arbor | its format is outside this Arbor's `continues.session` | run it with an Arbor whose `version --json` covers the format |
