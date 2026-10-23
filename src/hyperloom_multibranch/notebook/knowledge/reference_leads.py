# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""InferenceX's reference launches for this session's model, as leads PRELUDE gives every specialist (#109 item 10).

A run given an InferenceX checkout as the reference ``inferencex`` (``--reference inferencex=DIR``) has its copy read
here: the benchmark matrices ``configs/*-master.yaml``, for the single-node entries of this session's model, matched
by its name without the organization. Each entry is a lead naming the script InferenceX launches it with, resolved as
its runners resolve it, and what the entry was written for: GPU, framework, precision, image, the workload and the TP
and concurrency it sweeps, at the checkout's commit, with the dimensions where it differs from this session. Entries on
this session's GPU, framework and precision come first.

A lead says it was not measured here, and it is never ranked or replayed among the champions the Recipe KB measured: a
config from other hardware or another workload is worth exactly its provenance line (#109 c2000). For a model the KB
has no record of, it is the one validated config a specialist can diff the root's against.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from hyperloom_multibranch.contracts.notebook.records import SessionProfile

log = logging.getLogger(__name__)

#: The reference name a run gives its InferenceX checkout under.
INFERENCEX_REFERENCE = "inferencex"
MATRIX_GLOB = "configs/*-master.yaml"
SCRIPTS = "benchmarks/single_node"
FIXED = "fixed-seq-len"
AGENTIC = "agentic-coding"
#: The directory under :data:`SCRIPTS` that holds each scenario's scripts, as InferenceX's runners name it.
SCENARIO_DIRS = {FIXED: "fixed_seq_len", AGENTIC: "agentic"}
#: The workload a lead names for an agentic scenario, and for an AgentX session, which has no sequence lengths.
AGENTIC_WORKLOAD = "agentic"
MAX_REFERENCE_LEADS = 3
REVISION_CHARS = 9
SETUP = ("gpu", "framework", "precision")


@dataclass(frozen=True)
class ReferenceLead:
    """One InferenceX matrix entry for this session's model; ``script`` is relative to the checkout, ``""`` when none
    of the names InferenceX would launch it with is there, and ``differing`` maps a dimension to (theirs, ours)."""

    config: str
    script: str
    model: str
    gpu: str
    framework: str
    precision: str
    image: str
    workload: str
    revision: str = ""
    differing: Mapping[str, tuple[str, str]] = field(default_factory=dict)

    def describe(self) -> str:
        pin = f" @ {self.revision[:REVISION_CHARS]}" if self.revision else ""
        script = f"`{INFERENCEX_REFERENCE}/{self.script}`" if self.script else "no launch script in the copy"
        written = f"{self.framework} {self.precision} on {self.gpu}, image `{self.image}`, {self.workload}"
        where = "".join(f", {key} {theirs} (here {ours})" for key, (theirs, ours) in self.differing.items())
        return f"InferenceX {self.config}{pin}: {script}, written for {written}{where}; not measured here"

    def render(self) -> str:
        return f"- {self.describe()}"


def reference_leads(
    checkout: Path, profile: SessionProfile, revision: str = "", *, limit: int = MAX_REFERENCE_LEADS
) -> tuple[ReferenceLead, ...]:
    """The leads of ``checkout``'s matrices for ``profile``'s model, closest first; ``()`` when it holds none.

    A matrix that cannot be read is logged and skipped, since a reference only adds leads. Blocks on the files."""

    wanted = _name(profile.model)
    entries = [
        (config, entry)
        for config, entry in _entries(checkout)
        if _name(str(entry.get("model", ""))) == wanted and not entry.get("multinode")
    ]
    leads = (_lead(checkout, config, entry, profile, revision) for config, entry in entries)
    return tuple(sorted(leads, key=_distance))[:limit]


def _distance(lead: ReferenceLead) -> tuple[int, ...]:
    """How far ``lead`` is from this session: the setup it differs in first, then the workload, then the sweep."""

    return tuple(sum(1 for key in keys if key in lead.differing) for keys in (SETUP, ("workload",), ("tp", "conc")))


def _name(model: str) -> str:
    return model.rstrip("/").rsplit("/", 1)[-1].lower()


def _entries(checkout: Path) -> Iterable[tuple[str, Mapping[str, object]]]:
    for matrix in sorted(checkout.glob(MATRIX_GLOB)):
        try:
            document = yaml.safe_load(matrix.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            log.warning("reference leads: %s cannot be read, its entries are left out: %s", matrix, exc)
            continue
        if isinstance(document, Mapping):
            yield from ((str(key), entry) for key, entry in document.items() if isinstance(entry, Mapping))


def _lead(
    checkout: Path, config: str, entry: Mapping[str, object], profile: SessionProfile, revision: str
) -> ReferenceLead:
    scenarios = entry.get("scenarios") if isinstance(entry.get("scenarios"), Mapping) else {}
    scenario = _scenario(scenarios, profile)
    runs = _runs(scenarios.get(scenario))
    gpu = _gpu(str(entry.get("runner", "")))
    framework, precision = str(entry.get("framework", "")), str(entry.get("precision", ""))
    theirs = dict(zip(SETUP, (gpu, framework, precision)))
    ours = dict(zip(SETUP, (profile.hardware.lower(), profile.framework.lower(), profile.precision)))
    differing = {key: (theirs[key], ours[key]) for key in SETUP if theirs[key].lower() != ours[key].lower()}
    kind, sweep = _kind(scenario, runs) or scenario, _sweep(runs)
    differing.update(_workload_differing(kind, runs, profile))
    return ReferenceLead(
        config=config,
        script=_script(checkout, scenario, str(entry.get("model-prefix", "")), theirs, runs),
        model=str(entry.get("model", "")),
        gpu=gpu,
        framework=framework,
        precision=precision,
        image=str(entry.get("image", "")),
        workload=f"{kind} at {sweep}" if sweep else kind,
        revision=revision,
        differing=differing,
    )


def _scenario(scenarios: Mapping[str, object], profile: SessionProfile) -> str:
    """The entry's scenario for this session's workload when it has it, and else the first it has."""

    preferred = FIXED if profile.isl is not None else AGENTIC
    return preferred if preferred in scenarios else next(iter(scenarios), preferred)


def _runs(scenario: object) -> list[Mapping[str, object]]:
    """Each sequence-length setting of a scenario, its search space folded in."""

    items = scenario if isinstance(scenario, list) else []
    return [
        {**item, **space}
        for item in items
        if isinstance(item, Mapping)
        for space in (item.get("search-space") or [{}])
        if isinstance(space, Mapping)
    ]


def _gpu(runner: str) -> str:
    """``mi355x`` of ``mi355x`` and of ``cluster:mi355x-amds``, as InferenceX names a runner's GPU."""

    return runner.rsplit(":", 1)[-1].split("-", 1)[0].lower()


def _lengths(isl: object, osl: object) -> str:
    return f"ISL {isl}/OSL {osl}"


def _kind(scenario: str, runs: list[Mapping[str, object]]) -> str:
    """The workload a scenario was written for: ``agentic``, or each of its sequence lengths."""

    if scenario == AGENTIC:
        return AGENTIC_WORKLOAD
    return ", ".join(sorted({_lengths(run["isl"], run["osl"]) for run in runs if run.get("isl") and run.get("osl")}))


def _tps(runs: list[Mapping[str, object]]) -> list[int]:
    return sorted({run["tp"] for run in runs if isinstance(run.get("tp"), int)})


def _sweep(runs: list[Mapping[str, object]]) -> str:
    concs = [conc for run in runs for conc in _concs(run)]
    parts = ["TP " + ", ".join(map(str, _tps(runs)))] if _tps(runs) else []
    return ", ".join([*parts, f"conc {min(concs)}-{max(concs)}"] if concs else parts)


def _workload_differing(
    kind: str, runs: list[Mapping[str, object]], profile: SessionProfile
) -> dict[str, tuple[str, str]]:
    """Where the workload a scenario sweeps leaves out this session's: its kind or lengths, its TP, its concurrency."""

    here = AGENTIC_WORKLOAD if profile.isl is None else _lengths(profile.isl, profile.osl)
    differing = {} if here in kind.split(", ") else {"workload": (kind, here)}
    tps, concs = _tps(runs), [conc for run in runs for conc in _concs(run)]
    if tps and profile.tp not in tps:
        differing["tp"] = (", ".join(map(str, tps)), str(profile.tp))
    if concs and not min(concs) <= profile.conc <= max(concs):
        differing["conc"] = (f"{min(concs)}-{max(concs)}", str(profile.conc))
    return differing


def _concs(run: Mapping[str, object]) -> list[int]:
    listed = run.get("conc-list")
    if isinstance(listed, list):
        return [conc for conc in listed if isinstance(conc, int)]
    return [conc for conc in (run.get("conc-start"), run.get("conc-end")) if isinstance(conc, int)]


def _script(
    checkout: Path, scenario: str, prefix: str, theirs: Mapping[str, str], runs: list[Mapping[str, object]]
) -> str:
    """The script InferenceX's single-node runners launch the entry with: ``<base>_<framework>``, else ``<base>`` with
    ``_atom`` for atom alone, ``_mtp`` added for speculative decoding; ``""`` when neither is in the copy."""

    base = f"{prefix}_{theirs['precision']}_{theirs['gpu']}"
    spec = "_mtp" if any(run.get("spec-decoding") == "mtp" for run in runs) else ""
    atom = "_atom" if theirs["framework"] == "atom" else ""
    directory = f"{SCRIPTS}/{SCENARIO_DIRS.get(scenario, SCENARIO_DIRS[FIXED])}"
    for name in (f"{base}_{theirs['framework']}{spec}.sh", f"{base}{atom}{spec}.sh"):
        if (checkout / directory / name).is_file():
            return f"{directory}/{name}"
    return ""


__all__ = [
    "INFERENCEX_REFERENCE",
    "MAX_REFERENCE_LEADS",
    "ReferenceLead",
    "reference_leads",
]
