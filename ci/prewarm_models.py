#!/usr/bin/env python3
"""ci/prewarm_models.py — pre-populate /wekafs/models/ from HuggingFace,
bypassing SaFE playground download (which is single-flight + slow).

Why exist:
  SaFE's POST /api/v1/playground/models triggers an internal K8s Job that
  downloads via huggingface_hub from inside a small worker pool. For a
  130-model weekly batch this serial-ish path becomes the gate. Pulling
  the files directly to /wekafs/models/<slug>/ from a node that mounts
  the volume removes that gate.

  When SaFE register API is later invoked for the same source URL, two
  possible benign outcomes:
    1) SaFE backend dedups on existing target files -> returns model_id
       in seconds, no re-download.
    2) SaFE backend re-invokes huggingface_hub anyway; in that case the
       hf-hub local cache (or the target dir itself) makes the second
       fetch a near-no-op due to ETag / sha256 matching.
  Either way, the register call ceases to be a bottleneck.

Layout (same convention as SaFE-registered models already in /wekafs/models):
  /wekafs/models/<owner>-<repo>/         # final destination
  /wekafs/models/.tmp/<slug>.part/       # in-flight, atomic rename on success

Inputs:
  --candidates ci/candidates/topN.json   (preferred, drives the same pool as
                                          generate_hf_matrix.py)
  --repos repo1 repo2 ...                (ad-hoc list)
  - (stdin)                              (one repo_id per line)
  --batch-index N --batch-size M         (optional, slice the candidates list
                                          the same way generate_hf_matrix does)

Env:
  HF_TOKEN                       required for gated repos
  HF_HUB_ENABLE_HF_TRANSFER=1    optional, enables the high-speed transfer client

Usage examples:
  # Whole candidates pool, 16 concurrent repos:
  HF_TOKEN=hf_xxx python3 prewarm_models.py \\
      --candidates ci/candidates/top600_2026-05-12.json \\
      --target-root /wekafs/models --concurrency 16

  # Same slice as a single batch dispatch (matches optimize-batch.yml):
  HF_TOKEN=hf_xxx python3 prewarm_models.py \\
      --candidates ci/candidates/top600_2026-05-12.json \\
      --batch-index 0 --batch-size 10 \\
      --target-root /wekafs/models

  # Smoke a single repo:
  HF_TOKEN=hf_xxx python3 prewarm_models.py --repos Qwen/Qwen3-14B-AWQ
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

# huggingface_hub is light + already a transitive dep of vllm/sglang images,
# but on a bare runner we may need to install it. Workflow installs it
# explicitly; here we just import.
try:
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
except ImportError as e:
    print(f"ERROR: huggingface_hub not installed: {e}\n"
          "Run: pip install --quiet 'huggingface_hub>=0.24'", file=sys.stderr)
    sys.exit(2)


log = logging.getLogger("prewarm")


# ── Slug + path helpers ─────────────────────────────────────────────────────


def slug(repo_id: str) -> str:
    """HF repo_id → /wekafs/models/<slug>/ folder name.

    Matches the SaFE backend convention already used by the 47 models on
    /wekafs/models: a single '/' separator becomes '-' and all other
    characters are preserved verbatim (including pre-existing '-').

      Qwen/Qwen2.5-7B-Instruct           → Qwen-Qwen2.5-7B-Instruct
      meta-llama/Llama-3.1-8B            → meta-llama-Llama-3.1-8B
      deepseek-ai/DeepSeek-R1            → deepseek-ai-DeepSeek-R1
      dphn/dolphin-2.9.1-yi-1.5-34b      → dphn-dolphin-2.9.1-yi-1.5-34b
    """
    return repo_id.replace("/", "-")


def dest_dir(target_root: Path, repo_id: str) -> Path:
    return target_root / slug(repo_id)


def tmp_dir(target_root: Path, repo_id: str) -> Path:
    return target_root / ".tmp" / f"{slug(repo_id)}.part"


# ── Completeness check (uses HF tree API) ───────────────────────────────────


def is_complete(dest: Path, repo_id: str, hf_api: HfApi, token: str) -> bool:
    """True iff dest/ has every file the HF repo claims, with non-zero size.

    Cheap: one ``list_repo_files`` call. Tolerant of interrupted prior runs
    because partial dest dirs miss at least one file or have a zero-sized
    file (.part suffix is gone after atomic rename, so we don't need to
    check it here).
    """
    if not dest.is_dir():
        return False
    try:
        files = hf_api.list_repo_files(repo_id, token=token)
    except Exception as e:
        log.warning("[%s] HF list_repo_files failed (%s) — re-downloading",
                    repo_id, e)
        return False
    for f in files:
        p = dest / f
        if not p.is_file() or p.stat().st_size == 0:
            return False
    return True


# ── Per-repo download ───────────────────────────────────────────────────────


def _dir_stats(p: Path) -> tuple[int, float]:
    """Return (n_files, total_GB) under p, ignoring .part suffixes."""
    n = 0
    total = 0
    for f in p.rglob("*"):
        if f.is_file() and not f.suffix.endswith(".part"):
            n += 1
            total += f.stat().st_size
    return n, total / 1e9


def download_one(repo_id: str, target_root: Path, hf_token: str,
                 inner_workers: int = 4) -> dict:
    """Download a single HF repo to <target_root>/<slug>/.

    Returns a dict: status (OK/SKIP/FAIL), size_gb, n_files, elapsed_s, reason.
    """
    start = time.time()
    dest = dest_dir(target_root, repo_id)
    tmp = tmp_dir(target_root, repo_id)
    hf_api = HfApi()

    if is_complete(dest, repo_id, hf_api, hf_token):
        n, gb = _dir_stats(dest)
        return {"status": "SKIP", "size_gb": gb, "n_files": n,
                "elapsed_s": 0, "reason": "already complete"}

    # Clean stale .part if previous run aborted mid-flight
    if tmp.exists():
        log.info("[%s] cleaning stale .tmp dir %s", repo_id, tmp)
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id,
            local_dir=str(tmp),
            local_dir_use_symlinks=False,
            token=hf_token or None,
            max_workers=inner_workers,
            # Skip the giant pytorch_model.bin if a safetensors index already
            # covers all weights; saves ~half the bytes on many older repos
            # that ship both. SaFE / vllm prefer safetensors anyway.
            allow_patterns=None,
            ignore_patterns=[
                "*.h5", "*.msgpack", "*.onnx", "*.tflite",
                "consolidated.*",  # legacy llama-cpp dumps
            ],
            tqdm_class=None,
        )
    except RepositoryNotFoundError:
        shutil.rmtree(tmp, ignore_errors=True)
        return {"status": "FAIL", "size_gb": 0, "n_files": 0,
                "elapsed_s": time.time() - start,
                "reason": "repo not found (gated or deleted)"}
    except HfHubHTTPError as e:
        shutil.rmtree(tmp, ignore_errors=True)
        return {"status": "FAIL", "size_gb": 0, "n_files": 0,
                "elapsed_s": time.time() - start,
                "reason": f"HF HTTP error: {e}"[:200]}
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        return {"status": "FAIL", "size_gb": 0, "n_files": 0,
                "elapsed_s": time.time() - start,
                "reason": f"{type(e).__name__}: {e}"[:200]}

    # Atomic-ish swap. We rmtree dest first if it exists (it shouldn't, since
    # is_complete returned False earlier, but a partial dest could be there).
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    try:
        tmp.rename(dest)
    except OSError:
        # Cross-device or other rename failure — fall back to copy+rmtree.
        log.warning("[%s] rename failed, fallback to copytree+rmtree", repo_id)
        shutil.copytree(tmp, dest)
        shutil.rmtree(tmp, ignore_errors=True)

    n, gb = _dir_stats(dest)
    return {"status": "OK", "size_gb": gb, "n_files": n,
            "elapsed_s": time.time() - start}


# ── Input loading ───────────────────────────────────────────────────────────


def _load_candidates(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cands = data.get("candidates", [])
    return [c["repo_id"] for c in cands if c.get("repo_id")]


def _slice_repos(repos: list[str], batch_index: int | None,
                 batch_size: int | None) -> list[str]:
    """Apply optional --batch-index / --batch-size slice (same semantics as
    generate_hf_matrix.py so a workflow's prewarm + optimize stages can target
    the exact same set of repos)."""
    if not batch_size:
        return repos
    bi = batch_index or 0
    start = bi * batch_size
    end = start + batch_size
    return repos[start:end]


def load_repos(args: argparse.Namespace) -> list[str]:
    if args.repos:
        return list(args.repos)
    if args.candidates:
        repos = _load_candidates(args.candidates)
        return _slice_repos(repos, args.batch_index, args.batch_size)
    # stdin (one repo per line)
    if not sys.stdin.isatty():
        repos = [ln.strip() for ln in sys.stdin if ln.strip()
                 and not ln.startswith("#")]
        if repos:
            return repos
    raise SystemExit(
        "provide --candidates FILE or --repos REPO [REPO ...] "
        "or pipe a list to stdin")


# ── Main loop ───────────────────────────────────────────────────────────────


def _format_extras(result: dict) -> str:
    parts = [f"{result.get('size_gb', 0):.1f}GB"]
    if result.get("n_files", 0) > 0:
        parts.append(f"{result['n_files']}f")
    if result.get("elapsed_s"):
        secs = int(result["elapsed_s"])
        if secs < 60:
            parts.append(f"{secs}s")
        else:
            parts.append(f"{secs // 60}m{secs % 60:02d}s")
    if result.get("reason"):
        parts.append(f"[{result['reason']}]")
    return "  ".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", type=Path,
                   help="candidates JSON from build_candidates.py "
                        "(preferred, drives the same pool as the batch matrix)")
    p.add_argument("--repos", nargs="+",
                   help="explicit HF repo IDs (alt to --candidates / stdin)")
    p.add_argument("--batch-index", type=int, default=None,
                   help="optional slice start (0-based, same semantics as "
                        "generate_hf_matrix.py)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="optional slice length (when set, takes "
                        "candidates[batch_index*batch_size : +batch_size])")
    p.add_argument("--target-root", type=Path, default=Path("/wekafs/models"),
                   help="root dir for <slug>/ subdirs "
                        "(default /wekafs/models; on c04u01 the real "
                        "underlying path is /mnt/weka/models via a symlink)")
    p.add_argument("--concurrency", type=int, default=16,
                   help="parallel repo downloads (default 16, HF token "
                        "usually tolerates this)")
    p.add_argument("--inner-workers", type=int, default=4,
                   help="parallel file workers inside a single snapshot_download "
                        "(default 4, multiplies with --concurrency)")
    p.add_argument("--exclude-done", action="store_true",
                   help="also skip repos listed in already_done.json")
    p.add_argument("--already-done", type=Path,
                   default=Path(__file__).parent / "candidates" / "already_done.json")
    p.add_argument("--log", type=Path,
                   help="also append progress to this file (in addition to stdout)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        log.warning("HF_TOKEN unset — gated repos will fail with 401/403")

    repos = load_repos(args)
    if args.exclude_done and args.already_done.exists():
        done = {m["repo_id"] for m in
                json.loads(args.already_done.read_text()).get("models", [])}
        before = len(repos)
        repos = [r for r in repos if r not in done]
        log.info("--exclude-done: %d → %d repos", before, len(repos))

    if not repos:
        log.error("no repos to prewarm — exiting")
        return 1

    args.target_root.mkdir(parents=True, exist_ok=True)
    (args.target_root / ".tmp").mkdir(parents=True, exist_ok=True)

    log.info("PREWARM start: %d repos → %s (concurrency=%d, inner=%d)",
             len(repos), args.target_root, args.concurrency, args.inner_workers)

    ok = skip = fail = 0
    total_gb = 0.0
    fail_repos: list[tuple[str, str]] = []
    started = time.time()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency,
        thread_name_prefix="prewarm",
    ) as ex:
        future_to_repo = {
            ex.submit(download_one, r, args.target_root, hf_token,
                      args.inner_workers): r
            for r in repos
        }
        for done_count, fut in enumerate(
                concurrent.futures.as_completed(future_to_repo), 1):
            repo = future_to_repo[fut]
            try:
                result = fut.result()
            except Exception as e:
                result = {"status": "FAIL", "reason": f"executor: {e}",
                          "size_gb": 0, "n_files": 0, "elapsed_s": 0}

            status = result["status"]
            if status == "OK":   ok += 1
            elif status == "SKIP": skip += 1
            else:
                fail += 1
                fail_repos.append((repo, result.get("reason", "?")))

            total_gb += result.get("size_gb", 0)
            log.info("[%d/%d] %s %s  %s",
                     done_count, len(repos), status, repo,
                     _format_extras(result))

    elapsed = time.time() - started
    log.info("PREWARM done: ok=%d skip=%d fail=%d  total=%.1fGB  elapsed=%dm%ds",
             ok, skip, fail, total_gb,
             int(elapsed // 60), int(elapsed % 60))

    if fail_repos:
        log.warning("Failures:")
        for repo, reason in fail_repos:
            log.warning("  - %s: %s", repo, reason)

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
