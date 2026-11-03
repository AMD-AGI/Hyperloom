"""A parallel run's configuration: one JSON file, checked once, then pinned in the store.

    {"run_id": "c6p", "store": "/data/campaigns/parallel",
     "arbor_ref": "main@<sha>", "arbor_root": "/data/campaigns/Arbor-par",
     "template_config": "configs/mi350x/model-m-c8-c6.json",
     "dossier_dir": "docs/dossier/model-m-mi350x/u3-c8",
     "gates": "configs/parallel/gates-c8.json", "lane_sessions": "/data/campaigns/sessions",
     "max_lanes": 4, "max_rounds": 0, "no_keep_rounds": 2,
     "seed": {"map": ".../11.u4-directions.md", "overlay": "configs/parallel/c6p-r01-lanes.json",
              "bar_record": "..."},
     "image_src": {"vllm": "<the image's stock vllm/, unpacked on the host>"},
     "base_vmu": "<prefix of the base's .diff/.blobs; default the template's base_vmu>",
     "base_aiter": "<a landed_aiter.py every lane and A/B arm applies>",
     "dossier_runner": "host", "dossier": {"pack": "parallel-dossier", "iterations": {"A": 8, "B": 8}},
     "cluster": {"partition": "<partition>", "time_limit": "04:00:00", "log_dir": "...",
                 "lane_script": "scripts/model-m-campaign/campaign_model_m.sbatch",
                 "profile_script": "scripts/parallel/profile_round.sbatch",
                 "ab_script": "scripts/parallel/ab_model_m.sbatch"},
     "kinds": {"kernel-plan": {"policy": {...}, "space": {...}}}}

Relative paths are the Arbor checkout's (`arbor_root`). `arbor_ref` names the
commit every lane's worktree checks out (`<branch>@<sha>`, or any ref); it is
resolved to a sha once, at `init`, and pinned in run.json with the gates'
sha256. A run's config is never edited in place: a different config is a new
run.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,23}$")
LANE_KINDS = ("increment", "kernel-plan", "architectural", "upstream")


class ConfigError(ValueError):
    pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def gates_sha(gates: dict) -> str:
    """The gates' canonical sha256: what ab_model_m.sbatch recomputes on the node."""
    return sha256_text(json.dumps(gates, sort_keys=True))


def validate_gates(gates: Any) -> dict:
    if not isinstance(gates, dict) or not isinstance(gates.get("objective"), dict):
        raise ConfigError("gates: an object with an objective block (metric, direction, measurement)")
    obj = gates["objective"]
    if not obj.get("metric") or obj.get("direction") not in ("max", "min"):
        raise ConfigError("gates.objective names its metric and a direction of max or min")
    arms = str(gates.get("order", "B C C B")).split()
    if not arms or set(arms) - {"B", "C"} or "B" not in arms or "C" not in arms:
        raise ConfigError(f"gates.order {gates.get('order')!r} is a sequence of B and C runs")
    if int(gates.get("min_runs_per_arm", 2)) < 1:
        raise ConfigError("gates.min_runs_per_arm is at least 1")
    return gates


def read_gates(path: str | Path) -> tuple[dict, str]:
    try:
        gates = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"gates: {path} cannot be read as JSON ({exc})") from None
    return validate_gates(gates), gates_sha(gates)


@dataclass(frozen=True)
class ClusterCfg:
    partition: str = ""
    account: str = ""
    time_limit: str = "04:00:00"
    log_dir: str = ""
    lane_script: str = "scripts/model-m-campaign/campaign_model_m.sbatch"
    profile_script: str = "scripts/parallel/profile_round.sbatch"
    ab_script: str = "scripts/parallel/ab_model_m.sbatch"
    dossier_script: str = "scripts/parallel/dossier_round.sh"
    host: str = ""                       # ssh to a login node when the controller runs elsewhere


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    store: Path
    arbor_root: Path
    arbor_ref: str
    template_config: Path
    dossier_dir: Path
    gates: Path
    lane_sessions: str
    max_lanes: int = 4
    max_rounds: int = 0
    no_keep_rounds: int = 2
    seed: dict = field(default_factory=dict)
    image_src: dict = field(default_factory=dict)
    base_vmu: str = ""
    base_aiter: str = ""
    base_repo: dict = field(default_factory=dict)
    dossier_runner: str = "host"
    dossier: dict = field(default_factory=dict)
    first_round: int = 1                 # the dossier's update round (uK) of round 1, when unseeded
    cluster: ClusterCfg = field(default_factory=ClusterCfg)
    kinds: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def root(self) -> Path:
        """The run's own directory in the store."""
        return self.store / self.run_id

    def path(self, value: str | Path) -> Path:
        """A path from the config: absolute as given, else in the Arbor checkout."""
        p = Path(value)
        return p if p.is_absolute() else self.arbor_root / p

    def template(self) -> dict:
        return json.loads(self.template_config.read_text())

    def kind(self, name: str) -> dict:
        return dict(self.kinds.get(name) or {})

    def dossier_iters(self, part: str) -> int:
        return int((self.dossier.get("iterations") or {}).get(part, 8))


def _need(cond: bool, problem: str) -> None:
    if not cond:
        raise ConfigError(problem)


def from_dict(raw: dict[str, Any]) -> RunConfig:
    _need(isinstance(raw, dict), "the run config is a JSON object")
    for key in ("run_id", "store", "arbor_root", "template_config", "dossier_dir", "gates", "lane_sessions"):
        _need(bool(str(raw.get(key) or "").strip()), f"the run config names {key}")
    run_id = str(raw["run_id"])
    _need(bool(RUN_ID_RE.match(run_id)), f"run_id {run_id!r}: lower-case letters, digits and -, at most 24")
    arbor_root = Path(raw["arbor_root"])
    rel = lambda v: Path(v) if Path(v).is_absolute() else arbor_root / v  # noqa: E731
    max_lanes = int(raw.get("max_lanes", 4))
    _need(max_lanes >= 1, "max_lanes counts nodes: at least 1")
    seed = dict(raw.get("seed") or {})
    _need(not seed or bool(seed.get("map")), "seed names the 11.uK map it starts from (seed.map)")
    kinds = dict(raw.get("kinds") or {})
    _need(not set(kinds) - set(LANE_KINDS), f"kinds: one of {', '.join(LANE_KINDS)}")
    cl = dict(raw.get("cluster") or {})
    unknown = set(cl) - set(ClusterCfg.__dataclass_fields__)
    _need(not unknown, f"cluster: unknown keys {sorted(unknown)}")
    _need(str(raw.get("dossier_runner", "host")) == "host", "dossier_runner: host (the only runner built)")
    return RunConfig(
        run_id=run_id, store=Path(raw["store"]), arbor_root=arbor_root,
        arbor_ref=str(raw.get("arbor_ref") or "HEAD"), template_config=rel(raw["template_config"]),
        dossier_dir=rel(raw["dossier_dir"]), gates=rel(raw["gates"]), lane_sessions=str(raw["lane_sessions"]),
        max_lanes=max_lanes, max_rounds=int(raw.get("max_rounds", 0)),
        no_keep_rounds=int(raw.get("no_keep_rounds", 2)), seed=seed,
        image_src=dict(raw.get("image_src") or {}), base_vmu=str(raw.get("base_vmu") or ""),
        base_aiter=str(raw.get("base_aiter") or ""), base_repo=dict(raw.get("base_repo") or {}),
        dossier_runner="host", dossier=dict(raw.get("dossier") or {}),
        first_round=int(raw.get("first_round", 1)), cluster=ClusterCfg(**cl), kinds=kinds, raw=dict(raw))


def load(path: str | Path) -> RunConfig:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{path}: not a readable JSON file ({exc})") from None
    return from_dict(raw)


def ref_sha(ref: str) -> str:
    """`<branch>@<sha>` pins the sha; anything else is a ref for git to resolve."""
    return ref.rsplit("@", 1)[1] if "@" in ref else ref
