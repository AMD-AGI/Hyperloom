# Contributing to Arbor

## Where things live

- **The repository:** `AMD-AGI/Arbor` is Arbor's home since 2026-10-29, when `e77185e` merged
  its `main`, the original Arbor's, into this tree.
  - `the internal Arbor repository`, the home from 2026-10-28, still gets every push, so nobody who
    reads it is cut off, until it is retired.
  - `the internal Arbor repository`, `<fork>/Arbor` (`hyperloom`) and `<old branch>` are retired,
    each with a notice; nothing lands there any more.
- **Branches:** work lands on `main` directly since 2026-10-29, one change per commit;
  `<old branch>`, where the Arbor core team worked before, stopped at `4045752`. Add commits on top
  of `main` and never rebase it: both copies must stay the same history, and
  `an internal organization`'s enterprise ruleset forbids force pushes there.
- **Issues:** this repository's tracker, and the Arbor developers' board
  (`boards/ArborDevelopment` on the the platform agent board), which has one issue per piece of work
  in progress. Work with the Hyperloom team happens on their board, `boards/hyperloom-multibranch`.
- **CODEOWNERS:** AMD-AGI's, `.github/CODEOWNERS` (`* <maintainer>`), which came with the merge
  (`the internal Arbor repository` issue #2 asked for one).

## Tests

```bash
python3 -m pytest -q arbor/tests evals/tests
```

Run it from the repository root with a Python that has `pyyaml` and `pytest` (the dashboard's
smoke test also needs `node`). There is no CI: run the suite before you push. Four tests fail on
`main` and are known: `evals/tests/test_pipeline.py::CommandIntegrationTest`, which reads slash
commands deleted in September ([evals/README.md](evals/README.md)). Any other failure is yours.

Tests live beside the code they cover (`arbor/tests/`, `arbor/tests/parallel/` with a fake cluster).
A change to the loop, the gate or a mode comes with a test that shows the behaviour, on fakes or on
`@host` repositories (see `arbor/tests/test_component_mode.py` and `test_outside.py`).

## Changes

- **One change per commit,** with a message that says what changed and why, and the evidence (the
  test, the run, the board topic).
- **Standalone stays as it was** unless the change is meant for it: modes are switched by config
  (`mode`, `policy`), and a new behaviour is off by default where it could change a running
  campaign.
- **Never edit a protected validator** under a running campaign; the gate pins `validator/` at the
  session's start.
- **Update the documentation in the same change:** [README.md](README.md) and `docs/` describe the
  code as it is. A change to a config key, a file in the session, a verdict or a mode updates
  [docs/reference.md](docs/reference.md) or [docs/how-it-works.md](docs/how-it-works.md).
- **Secrets:** never commit a key or token; scan for `ANTHROPIC_` and 32-hex strings before you push.
