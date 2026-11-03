# Using the dossier and profiler skills

A campaign plans from two skills. The `dossier` skill says where the work is and ranks it, as a map
of directions. The `profiler` skill measures where GPU time goes; the dossier's profile part and
every re-profile rest on it. The main agent follows each skill's own file
(`.claude/skills/dossier/SKILL.md`, `.claude/skills/profiler/SKILL.md`). This page is for the
person who sets a campaign up or uses the skills by hand: how to turn them on, what to give them,
what to read back, and whether a dossier's directions can go into a pack
([below](#can-the-dossiers-directions-go-into-the-pack)).

## Turning them on

- **List them in the pack:** `skills: [dossier, profiler]` in `pack.yaml`. The default-on list
  (`DEFAULT_ON_SKILLS`, `arbor/pack/loader.py`) is empty, so a pack that does not list a skill does
  not have it. The serving template (`packs/serving`) lists both.
- **The dossier skill switches on map rounds** (`arbor/directions.py`) when the workload is not
  `document` and `policy.map_rounds` is above 0 (the default is 1). With `map_rounds: 0` the agent
  keeps the skill, and the harness tracks no map.
- **The profiler skill replaces the pack's own profiling.** Every profile follows it, ahead of any
  procedure in the pack's knowledge (the PyTorch profiler, captures under `--enforce-eager`). A
  pack without it keeps its own way.
- **Writing a dossier is a run of LEARN iterations.** Raise `policy.max_consecutive_probes`, or the
  default cap of 2 in a row cuts it short (the serving template sets 6).
- **To confirm they loaded:** the program the agent reads lists them under "Skills". The serving
  template's knowledge asks the agent to say so in its first DONE line when they are missing.

## The dossier

A dossier is written once per model, workload and GPU type, then refreshed part by part. It has
twelve parts: model, system, code reading, target, workload, baseline, profile, kernels, reference
platform, earlier optimizations, directions and fast bench. Part 11, the directions, ends with the
map a campaign works.

### Where dossiers are, and how one is found

- **In a repository:** `docs/dossier/<model>-<workload>-<gpu>/`. `README.md` is the index, with one
  file per part (`NN-<part>.md`) and one per part per update round (`NN.uK-<part>.md`). Arbor has
  Model M on MI350X and on MI355X.
- **The lookup:** `bash .claude/skills/dossier/find-dossier.sh` lists every dossier the checkout can
  see, each with its header (project, pins, platform, workload, date):
  - this checkout's `docs/dossier/`, and the enclosing project's when Arbor is a submodule;
  - the dossiers campaigns have shared,
    `/shared/arbor/projects/<project>/<user>/<campaign>/knowledge/dossier/`. These are read in
    place where the share is mounted, and through `ssh arbor-share` elsewhere.
    `ARBOR_DOSSIER_NO_SHARE=1` skips them.
- **A match** is the same model, workload and GPU type. Different pins or another node class make
  some parts stale; they do not make the dossier missing. Another GPU type needs another dossier.

### In a campaign

What the agent does:

1. **It looks the dossier up first.**
   - **Found:** it reads the index and refreshes only the parts that are stale for the tree it
     serves.
   - **Missing:** it writes one as LEARN iterations, in the order
     `packs/serving/knowledge/dossier-and-profile.md` gives. Parts 6 and 7 use the node.
   - **What only a person can answer** (a reference platform's traces, AMD's earlier work on the
     model, the target) goes under "Open asks" in the index, and the campaign goes on.
2. **It states the map** with one `DIRECTIONS: [...]` line, taken from the newest part 11.
   - The harness keeps the map in `<session>/directions.json`, and the map starts at the next
     iteration.
   - The first two iterations of this phase may be LEARNs past the cap (`map_bootstrap_waiver`).
   - A campaign that has no map after four iterations runs on, with a reminder in every program.
3. **It works the map,** closing each direction with a `DIRECTION:` line. When every direction is
   closed, what follows is an update round, a fresh start, a rewrite or a stop
   ([how-it-works.md](how-it-works.md#planning-the-dossier-the-map-and-rounds)).

What you do:

- **Before the launch:** if a dossier exists, put it where the lookup will find it. Commit it to
  `docs/dossier/` on the branch the campaign runs from; on the clusters, that is the branch the
  mirror carries.
- **While it runs:**
  - The map is in `<session>/directions.json`; the dashboard shows it as a checklist.
  - A stop that waits for the operator writes `MAP_STOP.md`.
  - Never edit `directions.json`: only the harness writes it, and a hand edit is undone.
- **After it ends:** a dossier the campaign wrote or refreshed sits in the node's Arbor checkout,
  not in the repository. Bring it back either way:
  - commit it to `docs/dossier/<name>/`, as the Model M dossier and its rounds were;
  - or share it with the campaign's knowledge, where the lookup finds it:
    `python3 -m arbor.collect knowledge --session <session> --config <cfg> --dossier <dir>`.

### By hand

Run it from an Arbor checkout, in Claude Code or another agent that reads `.claude/skills/`. Ask
for the skill by name, for example "/dossier: look up or write the dossier for `<model>` on
`<workload>`, `<GPU>`".
- **What it does:** the same steps as in a campaign. It asks its questions together, at the start.
- **What it needs:**
  - the served tree at its pins;
  - the launch and environment the server runs with;
  - a node for parts 6 and 7.

**Check it before a campaign relies on it:**

```bash
python3 .claude/skills/dossier/check-dossier.py docs/dossier/<name>               # parts 01-11 and the index
python3 .claude/skills/dossier/check-dossier.py docs/dossier/<name> --round u2    # one update round
python3 -m arbor.directions check docs/dossier/<name>/11-directions.md --dossier docs/dossier/<name>
```

- **`check-dossier.py`** checks that each file exists and holds real content: at least 300 words
  and 10 numbers. It does not check that the content is right.
- **`arbor.directions check`** reads the part's `## Map` line and checks each direction and each
  kernel plan it names. It also prints the order `depends_on` allows.
  - Any line that begins with `Direction:` or `Directions:`, even bold or in a bullet, is read as a
    map line and fails the check. Reword it.

### For a host: the dossier pass

A host that runs several Arbors from one map gets that map from one Arbor run first:
1. **The pass:** `component.stop_at: "map"`, with the host's profile in `component.profile`.
2. **The split:** `python3 -m arbor.directions split` cuts the map into shares.
3. **The branches:** each Arbor gets its share as `component.plan`.

[hosts.md](hosts.md#the-dossier-pass) has the keys and what each run does.

## The profiler

The campaign finds where GPU time goes with three tools:
- **ruppell** traces the served program with graphs on, and attributes every kernel, graph replays
  included, to its module, op and step;
- **rocprofv3** collects the hardware counters;
- **TraceLens** reads a window of steps in detail.

### In a campaign

- **ruppell has to be where the workload runs.**
  - **The copy:** `.claude/skills/profiler/ruppell`, at the commit in `VENDOR.md`. It is built
    against the container's ROCm with `runtime/build.sh`.
  - **The Model M launchers** mount it read-only at `/ruppell`, with a writable work directory
    (`scripts/model-m-campaign/campaign_model_m.sbatch`). Do the same in another launcher.
  - **Without a mount,** the agent copies it in and builds it the first time it profiles. The
    skill's "Get ruppell where the workload runs" says how.
  - **A stale build** left in a container from an older copy traces without saying what it lacks.
    The skill has the agent compare the build's commit with the copy's before every trace.
- **The frameworks it traces** are the vendored copy's adapters (`ruppell/python/ruppell/adapters/`).
  - **At this commit:** vLLM and DLRM. SGLang's adapter has been on ruppell's `main` since
    2026-10-27 and is not vendored yet.
  - **A framework without an adapter:** the agent takes a kernel census with
    `rocprofv3 --kernel-trace` (names, counts and GPU time, without modules or steps), and says so
    beside every number.
- **Validation:** ruppell has passed its checks on two targets, both on MI350X: vLLM serving
  DeepSeek V4.1 Flash, and DLRMv4 training.
  - Any other model, load or GPU is a new target: the agent runs ruppell's new-target checks first.
  - Until they pass, it labels every number unvalidated.
- **What it leaves:** each profile in its own directory, `<session>/profiles/<name>/`.
  - **`PROFILE.md`:** the report, with a `## Candidates` section that the knowledge collector
    quotes.
  - **Beside it:** the small files that let a reader check the report.
  - **Raw traces** stay on the node.
- **The host's profile:** in component mode, `component.profile` gives the agent a profile the host
  has already taken. The agent profiles only what it does not cover.

### By hand

Ask for `/profiler` in Claude Code from an Arbor checkout. Or, on the machine or in the container
that runs the workload:

```bash
docker cp .claude/skills/profiler/ruppell <container>:/ruppell
bash /ruppell/runtime/build.sh          # libruppell_tool.so and libruppell_marker.so
export PYTHONPATH=/ruppell/python
python -m ruppell run --out RUN -- <the workload command>    # vLLM: RUPPELL_VLLM=1 in the server's environment
```

- **The settings** (detail sampling, which step kinds to trace in full, run-time tuning) come from
  the route file for your target. The profiler skill's table routes you to it, inside ruppell's
  own skill (`ruppell/.claude/skills/ruppell-profile/`).
- **On a cluster:**
  - analyse on the GPU node, never on a login node;
  - run nothing else on the node's GPUs while a traced run measures;
  - stop every server you start.

## Can the dossier's directions go into the pack?

**Not as a map. As knowledge, yes, but the map stays in the dossier.**

**A pack has no place for a map.**
- `pack.yaml` has no directions key.
- A pack may hold only `knowledge/`, `validator/` and `eval/` (`ALLOWED_PACK_DIRS`,
  `arbor/pack/loader.py`).
- Its knowledge is rendered into the agent's program as text. The harness never reads a map from it.

The harness takes a map from three places only:

| From | Mode | What the campaign then does |
|---|---|---|
| the agent's `DIRECTIONS:` line, written from the dossier's newest part 11 | standalone, and a host's dossier pass | works it as its own map: it may add up to 3 directions a round, and runs an update round when all are closed |
| `component.plan`: a list of directions, or the path of a file holding one (a `split` share) | component mode only; standalone ignores the key | works these as a lane and nothing else: it refuses a `DIRECTIONS` line, sends what it would add to the host as `proposals`, and stops (`done`) when all are closed |
| `python3 -m arbor.stages seed --config <cfg> --direction m-<x> --plan <plan.md>` | any | carries out one rewrite's kernel plan, stage by stage; parallel mode's coordinator seeds its lanes too |

So it depends on what you want.

**A campaign that starts from directions you choose:** put them in the dossier.
1. Write them as the `## Map` line of the newest part 11.
2. Check the line (above).
3. Commit the dossier where the lookup finds it.

The agent emits that list in its first iterations, and the map starts from it.
- **Carry over what is settled.** Give a direction that is already done, killed or below the floor
  that status and a `why`, instead of leaving it out, so the campaign does not add it back.
- **Neither dossier in Arbor has a `## Map` line yet**: they were written before map rounds. A
  campaign that starts from them writes its own map from part 11's tables. To decide the list
  yourself, add the line.

**Several campaigns, each on its own share:** use component mode, and give each campaign a
`component.plan` from
`python3 -m arbor.directions split <part 11 or directions.json> --shares N --out <dir>`
([hosts.md](hosts.md)).
- A standalone campaign has no equivalent: it ignores `component.plan`.
- A config `directives` line can ask a campaign to work a subset, but the harness does not enforce
  it.

**What the map does not hold** goes in a hand-off in the pack's knowledge: what the base already
has, each kill and whether it can be reopened, the bar, the lessons.
- **The source:** cut it from the dossier's 10.uK.
- **The example:** `packs/model-m-c8-tp4/knowledge/c4-handoff.md`.
- **Its rules:**
  - **Map IDs:** cite the dossier's IDs, and keep the ranked list in the dossier only. The agent is
    told to emit part 11's list, and a second list in the knowledge gives it two to choose from.
  - **Which pack:** a pack made for that model and workload, never `packs/serving`. The serving
    template is used by every host workload that has no pack of its own.
  - **Size:** keep it short. Knowledge is in the prompt, whole, at every iteration; the C4 hand-off
    is 32 KB.
  - **Edits:** an edit reaches a running campaign at its next iteration. Knowledge is read each
    time the program is rendered; it is not pinned like the validator.
