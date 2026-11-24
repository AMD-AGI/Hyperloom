# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX client's tokenized dataset in the shared cache: which tests read it, which may add to it, and its key.

aiperf, the client an AgentX round runs, keeps each corpus it tokenized in a content-addressed directory, the one
``AIPERF_DATASET_MMAP_CACHE_DIR`` names (``aiperf/dataset/mmap_cache.py``). On a hit it restores ``dataset.dat`` and
``index.dat`` into the run's own directory byte for byte and skips the tokenizer and the composer; on a miss it tokenizes
and then adds the entry (``aiperf/dataset/dataset_manager.py``). Its key holds the corpus, the tokenizer's name, the seeds
and every prompt and trace setting that shapes the bytes, but neither aiperf's own code nor the tokenizer's files, so the
cache's key here adds both (:func:`~hyperloom.measure.jit.key.dataset_key`).

The test uses the cache as it uses the compiled kinds (:class:`~hyperloom.measure.jit.session.JitCacheSession`): the
verified snapshot is copied into a directory of the test's own, which ``AIPERF_DATASET_MMAP_CACHE_DIR`` names, and the
round's client inherits that variable. The shared store is never named to the client: aiperf takes a lock file and writes
a new entry in the directory it is given, so a miss tokenizes into the test's directory, as it does with no cache. Only the
host adds to the store, from a signed result, and only from a test that ran no patched code: the server runs under the same
account as the client and could write into the client's directory.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from hyperloom.common.benchmark_config import BenchmarkPoint, BenchmarkPointError
from hyperloom.measure.jit.key import dataset_key
from hyperloom.measure.jit.session import KindPlan
from hyperloom.measure.jit_kinds import AIPERF_DATASET
from hyperloom.measure.spec import MeasurementSpec

__all__ = ["CLIENT_PACKAGES", "MAX_MODEL_FILE_BYTES", "dataset_plan", "model_files"]

#: The packages of the AgentX client: a test that patches one runs a client whose code the key does not cover.
CLIENT_PACKAGES = frozenset({"aiperf", "agentx_cli"})
#: A model file other than the weights that is larger than this is not hashed: the kind is off for that model.
MAX_MODEL_FILE_BYTES = 256 * 1024**2
#: The variables aiperf reads its cache's place and switch from; a sheet that sets one decides for itself.
_CACHE_VARIABLES = "AIPERF_DATASET_MMAP_CACHE_"
#: Where an AgentX point reads its model, and so its tokenizer, when it does not use the point's model.
_MODEL_PATH = "MODEL_PATH"
#: Files no tokenizer reads: the weights, by suffix.
_WEIGHTS = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt", ".h5", ".msgpack", ".npz", ".onnx")
_CHUNK = 8 * 1024**2
_DIRECTORY = "dir"


def dataset_plan(spec: MeasurementSpec, configs: Sequence[Mapping], client: str | None) -> KindPlan | str | None:
    """The plan of the dataset kind for the test whose points are ``configs``, the reason it is off, or ``None`` when no
    point runs the AgentX client. ``client`` is the hash of the inference_suite staging the client runs from."""
    try:
        points = [BenchmarkPoint.from_document(config) for config in configs]
    except BenchmarkPointError as exc:
        return f"a point cannot be read: {exc}"
    agentx = [point for point in points if point.agentx]
    if not agentx:
        return None
    if client is None:
        return "the staging the AgentX client runs from is not known"
    refused = _client_changed(spec, points)
    if refused is not None:
        return refused
    models = {str(point.envs.get(_MODEL_PATH) or point.model) for point in agentx}
    if len(models) != 1 or not next(iter(models)):
        return "the AgentX points do not name one model to read their tokenizer from"
    model = models.pop()
    files = model_files(Path(model))
    if isinstance(files, str):
        return files
    return KindPlan(AIPERF_DATASET, dataset_key(client, model, files), read_only=_read_only(spec))


def _client_changed(spec: MeasurementSpec, points: Sequence[BenchmarkPoint]) -> str | None:
    """Why the sheet's client is not the one the key names: it patches the client's code, or picks its cache itself."""
    patched = sorted({root.import_name for root in spec.environment.roots if root.import_name in CLIENT_PACKAGES})
    if patched:
        return f"the test patches {', '.join(patched)}, the client's code, which the key does not cover"
    named = {name for point in points for name in point.envs if name.startswith(_CACHE_VARIABLES)}
    named |= {name for name in spec.process_env if name.startswith(_CACHE_VARIABLES)}
    if named:
        return f"the sheet sets {', '.join(sorted(named))} itself"
    return None


def model_files(model: Path) -> dict[str, str] | str:
    """The sha256 of each file at the top of the model directory but the weights: the tokenizer, its configuration and
    any code it runs (``trust_remote_code``) are among them. A reason instead when one cannot be read or is too large."""
    found: dict[str, str] = {}
    try:
        entries = sorted(model.iterdir())
    except OSError as exc:
        return f"the model directory cannot be listed ({type(exc).__name__})"
    for path in entries:
        if path.name.endswith(_WEIGHTS) or not path.is_file():
            continue
        try:
            if path.stat().st_size > MAX_MODEL_FILE_BYTES:
                return f"the model file {path.name} is larger than a tokenizer file is hashed"
            found[path.name] = _sha256(path)
        except OSError as exc:
            return f"the model file {path.name} cannot be read ({type(exc).__name__})"
    return found


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _read_only(spec: MeasurementSpec) -> str | None:
    """Why the test may read the dataset but not add to it: code it patched runs in the server, under the account the
    client runs as, and could write a dataset into the client's directory that every later test would replay."""
    patched = any(entry.op == "upsert" for root in spec.environment.roots for entry in root.entries)
    if patched or any(attachment.kind == _DIRECTORY for attachment in spec.attachments):
        return "the test carries patched code, which runs beside the client and could write into its dataset directory"
    return None
