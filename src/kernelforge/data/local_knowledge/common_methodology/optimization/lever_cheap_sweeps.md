---
title: cheap sweeps — one command per data point, and keep the knobs
kind: technique
gens: [gfx942, gfx950]
dtypes: [any]
regimes: [prefill, decode, training, both]
updated: 2026-08-21
---

# cheap sweeps — measure the constant instead of arguing about it

## TL;DR
The acceptance benchmark answers one expensive question (the whole suite, scored as a single mean).
For "hold the source fixed, vary one dispatch constant, time one shape" use the **sweep primitive**:
expose the constant as `FORGE_SWEEP_<NAME>` with today's value as the default, echo
`sweep_const: <NAME> <value>` on every read, and call
`python3 -m kernelforge.mcp_server.tools.bench --driver <driver command> --case <CASE_ID> --set <NAME>=<value>`.
One sweep point is then one command instead of an edit plus a gate cycle, cheap enough to call
dozens of times inside one iteration. **Keep the knobs in the source, defaulted to the winning
literals, for the whole search** — collapsing them to bare literals mid-campaign destroys the sweep
surface every later session would have inherited. Strip them only at final submission, and only if
the task demands a knob-free deliverable.

## The primitive

    python3 -m kernelforge.mcp_server.tools.bench \
        --driver <driver command> --case <CASE_ID> \
        --set BLOCK_H=32 --set NUM_WARPS=4

Pass `--driver` exactly the command you were told to run the driver with. If your session names a
wrapper (a lock is interposed when other lanes share this GPU), name the **wrapper** here, never the
raw driver: the raw driver is denied by the in-session gate, and it would time against another
lane's benchmark.

## Making a constant sweepable

Read it from the host as `FORGE_SWEEP_<NAME>`, defaulting to the value in force today, echo every
read, and convert the string against the type of the default:

```python
_SWEEP_TRUE = {"1", "true", "yes", "on"}
_SWEEP_FALSE = {"0", "false", "no", "off"}

def _sweep_const(name, default):
    value = os.environ.get("FORGE_SWEEP_" + name)
    if value is None:
        return default
    print(f"sweep_const: {name} {value}", flush=True)
    if isinstance(default, bool):
        token = value.strip().lower()
        if token in _SWEEP_TRUE:
            return True
        if token in _SWEEP_FALSE:
            return False
        raise ValueError(f"FORGE_SWEEP_{name}: not a boolean: {value!r}")
    return type(default)(value)
```

**Do not drop the bool branch, and do not put it after the `int` case** — `bool` is a subclass of
`int`, and `bool("0")` and `bool("false")` are both `True`, so a bare `type(default)(value)` turns
every OFF point into a second measurement of the ON configuration. That is the one sweep bug the
echo cannot catch: the echo reports the string the *host sent*, never the value the *source
computed*, so `sweep_const: USE_FUSED_EPILOGUE 0` prints identically whether the kernel took the
fused path or not. Both ends of the axis then time the same code, agree inside the noise band, and
the sweep reports "this flag makes no difference" about a flag it never actually turned off — fully
confirmed, and wrong. Refuse an unrecognized token loudly rather than falling back to the default;
a point that silently times the default is the same wrong answer with a typo for a cause.

The echo is a **contract, not decoration**. A sweep whose knob is never actually read would
otherwise time the default twice and report "this constant does not matter" — the most expensive
kind of wrong answer, because it closes a live axis. A point under this prefix with no echo fails
and carries no time.

## A knob the source already reads under its own name

Instrumentation is only for constants that are currently bare literals. A constant the source
*already* reads from the environment under its own name needs none: pass `--verbatim-names` and
every `--set` name is exported exactly as written. That is the only way to reach a knob a
third-party or baseline module owns — a vendor library's `PKG_SMALL_BATCH_TILE`, a compiler's
`TRITON_*`, a framework's own dispatch bound — none of which will ever print forge's echo line.

Because no echo comes back, such a point returns marked **UNCONFIRMED**: nothing proves the source
read the value. The number means something only against a reference point taken with no `--set` at
all, in the same round. Take that reference first, or the sweep tells you nothing.

Names the measurement itself runs on are refused before any process starts — device selection,
toolchain paths, cache directories, `PATH`. Those are not knobs of the kernel, and a sweep that set
one would time something other than the configuration it claims to be timing.

## Keep the knobs through the search

A `FORGE_SWEEP_` knob is how the next session re-opens a question this one answered with two data
points. Rules:

1. **Default every knob to the current winning literal.** The default path must reproduce the
   committed number exactly; the knob changes what is *reachable*, never what is *shipped by
   default*.
2. **Do not collapse knobs back to literals mid-campaign.** A knob deleted in iteration 4 is an axis
   that iteration 12 has to re-author before it can even ask the question, which in practice means
   it never asks. Shipping a sweepable surface is the cheap half of an optimization; re-authoring it
   is the expensive half.
3. **Strip at submission only, and only if required.** If the deliverable must be knob-free, do that
   as the last edit, replacing each read with its measured winner and re-running the gate to prove
   the collapse changed nothing.
4. **A knob is not a result.** Leaving a knob in place does not make the axis "explored"; the number
   you measured through it does.

## Sweep discipline

- **Sweep coupled constants JOINTLY.** A tile geometry timed at a launch config tuned for the *old*
  geometry is not a measurement of that tile, and a negative from such a point closes nothing.
- **Sweep in BOTH directions.** Every literal on the host dispatch path is a search variable, not a
  given — a floor, a cap, a minimum count, a bucket boundary nobody has questioned is exactly where
  an untested default hides. That includes a constant whose default comes from `os.environ`: it is
  an ordinary module constant in an ordinary file (see `[[optimization/lever_edit_surface.md]]`).
- **Sweep numbers are exploratory; the gate refuses them as evidence.** They tell you which edit to
  make; the gate still decides whether it survives.
- **Respect the noise band.** Read the reported `wall_min`/`wall_max` spread before believing a small
  difference, and re-run the point rather than ranking inside the band
  (`[[profiling/measure_protocol.md]]`).
- **A whole-suite point has no spread of its own.** A driver that ignores or rejects `--bench-case`
  times its whole suite instead; the requested case's time still stands and the result says so, but
  there are no per-iteration lines to read a spread from.
- **A build/run failure is not a slow result.** A configuration that will not build or will not run
  reports that, with no time attached. Never rank it as a measurement.

## Pitfalls
- Collapsing the knobs "to keep the file clean" before the campaign ends — the single most common way
  a later session loses an axis it already paid for.
- A knob whose default does not equal the shipped literal: every un-swept run silently benchmarks a
  different kernel than the one under review.
- Sweeping one member of a coupled pair and reading the negative as a verdict on the axis.
- Treating a constant as out of reach because its default is read from the environment, or because it
  lives in a sibling module rather than the anchor file.
- Converting a swept value with `type(default)(value)` when the default is a bool: `bool("0")` is
  `True`, so the OFF point benchmarks the ON configuration and the echo confirms the point anyway.

## Verify
- Each accepted point echoed `sweep_const: <NAME> <value>` for every `--set` you passed.
- Each boolean point actually flipped. The echo proves the read, not the parse, so before you accept
  a flat boolean axis, show that the `0` and the `1` end reach different code — a differing log line,
  a differing build, a time outside the band.
- The knob-free default path reproduces the committed wall time inside the noise band.
- The winning literal survives the real gate, not only the sweep.

## See also
- `[[optimization/lever_edit_surface.md]]` — which constants and files are in reach at all.
- `[[optimization/lever_autotune.md]]` — searching a structured config space once the cheap
  one-constant questions are answered.
- `[[profiling/measure_protocol.md]]` — noise band, warmup, same-session A/B.
