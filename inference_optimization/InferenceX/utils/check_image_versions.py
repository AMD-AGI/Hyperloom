"""
check_image_versions.py

Queries Docker Hub for the latest tags of all images referenced in amd-master.yaml
and reports whether any configs are using outdated versions.

Usage:
    python3 utils/check_image_versions.py --config-files .github/configs/amd-master.yaml
    python3 utils/check_image_versions.py --config-files .github/configs/amd-master.yaml --timeout 10

Exit code is always 0 — this check is informational and must not block CI.
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import requests
import yaml
from packaging.version import Version, InvalidVersion


DOCKERHUB_TAGS_URL = "https://hub.docker.com/v2/repositories/{repo}/tags"
DEFAULT_TIMEOUT = 10  # seconds per HTTP request
PAGE_SIZE = 100       # tags per page fetched from Docker Hub


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ImageInfo:
    """Represents one unique image tag found in amd-master.yaml."""
    full_tag: str          # e.g. "lmsysorg/sglang:v0.5.9-rocm700-mi35x"
    repo: str              # e.g. "lmsysorg/sglang"
    tag: str               # e.g. "v0.5.9-rocm700-mi35x"
    gpu_variant: str       # e.g. "mi35x", "mi30x", "" if not applicable
    config_keys: list      # which amd-master.yaml keys use this image


@dataclass
class CheckResult:
    image: ImageInfo
    latest_stable: Optional[str] = None   # newest non-rc tag matching same GPU variant
    latest_rc: Optional[str] = None       # newest rc tag matching same GPU variant
    is_outdated: bool = False
    has_rc_available: bool = False
    error: Optional[str] = None           # set if Docker Hub query failed


# ---------------------------------------------------------------------------
# Tag parsing helpers
# ---------------------------------------------------------------------------

GPU_VARIANT_RE = re.compile(r'(mi\d+x?)', re.IGNORECASE)
DATE_SUFFIX_RE = re.compile(r'-(\d{8})$')
RC_RE = re.compile(r'rc\d+', re.IGNORECASE)


def extract_gpu_variant(tag: str) -> str:
    """Extract GPU variant from tag, e.g. 'mi35x' from 'v0.5.9-rocm700-mi35x'."""
    m = GPU_VARIANT_RE.search(tag)
    return m.group(1).lower() if m else ""


def is_rc(tag: str) -> bool:
    return bool(RC_RE.search(tag))


def _parse_version_from_tag(tag: str, repo: str) -> Optional[tuple]:
    """
    Return a sortable tuple for comparison. Higher = newer.
    Returns None if tag cannot be parsed.

    Strategies by repo:
      lmsysorg/sglang : "v0.5.9-rocm700-mi35x"  → semver tuple from leading v*
      vllm/*          : "v0.18.0"                → semver tuple
      rocm/sgl-dev    : "*-20260215"             → date int from trailing YYYYMMDD
      rocm/atom       : "rocm7.2.0-*"            → semver from rocm* prefix
    """
    try:
        if repo.startswith("rocm/sgl-dev"):
            m = DATE_SUFFIX_RE.search(tag)
            if m:
                return (int(m.group(1)),)
            # fallback: try semver from leading v
            v_match = re.match(r'v([\d.]+(?:\.post\d+)?)', tag)
            if v_match:
                return (str(Version(v_match.group(1))),)
            return None

        if repo.startswith("rocm/atom"):
            m = re.match(r'rocm([\d.]+)', tag)
            if m:
                return (Version(m.group(1)),)
            return None

        # lmsysorg/sglang and vllm/*: strip leading 'v' and trailing suffixes
        v_match = re.match(r'v([\d.]+(?:\.post\d+)?(?:rc\d+)?)', tag)
        if v_match:
            return (Version(v_match.group(1)),)
        return None

    except (InvalidVersion, ValueError):
        return None


def _is_newer(candidate_tag: str, current_tag: str, repo: str) -> bool:
    """Return True if candidate_tag is strictly newer than current_tag."""
    c = _parse_version_from_tag(candidate_tag, repo)
    cur = _parse_version_from_tag(current_tag, repo)
    if c is None or cur is None:
        return False
    try:
        return c > cur
    except TypeError:
        return False


# ---------------------------------------------------------------------------
# Docker Hub queries
# ---------------------------------------------------------------------------

def _fetch_tags(repo: str, timeout: int) -> list[str]:
    """
    Fetch all tag names for a Docker Hub repo.
    Paginates until no next page.
    Returns empty list on any error (caller decides how to surface).
    """
    tags = []
    url = DOCKERHUB_TAGS_URL.format(repo=repo)
    params = {"page_size": PAGE_SIZE, "ordering": "last_updated"}
    pages_fetched = 0
    max_pages = 10  # safety cap — enough for any repo we care about

    while url and pages_fetched < max_pages:
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            break

        for result in data.get("results", []):
            name = result.get("name", "")
            if name:
                tags.append(name)

        url = data.get("next")  # None when last page
        params = {}             # next URL already includes params
        pages_fetched += 1

    return tags


def find_latest(repo: str, current_tag: str, gpu_variant: str, timeout: int) -> dict:
    """
    Returns {latest_stable, latest_rc, error} for a given repo + current tag.
    Filters to same GPU variant (if applicable).
    """
    all_tags = _fetch_tags(repo, timeout)
    if not all_tags:
        return {"latest_stable": None, "latest_rc": None,
                "error": f"Could not fetch tags from Docker Hub for {repo}"}

    # Filter to same GPU variant
    if gpu_variant:
        candidate_tags = [t for t in all_tags if gpu_variant.lower() in t.lower()]
    else:
        candidate_tags = all_tags

    latest_stable = None
    latest_rc = None

    for t in candidate_tags:
        if _parse_version_from_tag(t, repo) is None:
            continue  # skip unparseable tags (e.g. "latest", sha digests)
        if is_rc(t):
            if latest_rc is None or _is_newer(t, latest_rc, repo):
                latest_rc = t
        else:
            if latest_stable is None or _is_newer(t, latest_stable, repo):
                latest_stable = t

    return {"latest_stable": latest_stable, "latest_rc": latest_rc, "error": None}


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def load_images_from_configs(config_files: list[str]) -> list[ImageInfo]:
    """
    Parse amd-master.yaml (and any other config files) and collect unique images.
    Only AMD/ROCm repos are included (lmsysorg, rocm/*, vllm/vllm-openai-rocm).
    """
    AMD_REPOS = {"lmsysorg", "rocm", "vllm"}  # top-level Docker Hub orgs to check

    seen: dict[str, ImageInfo] = {}  # full_tag → ImageInfo

    for config_file in config_files:
        try:
            with open(config_file) as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"[check_image_versions] WARNING: could not read {config_file}: {e}",
                  file=sys.stderr)
            continue

        if not isinstance(data, dict):
            continue

        for config_key, config_val in data.items():
            if not isinstance(config_val, dict):
                continue
            full_tag = config_val.get("image", "")
            if not full_tag or ":" not in full_tag:
                continue

            repo, tag = full_tag.split(":", 1)
            org = repo.split("/")[0]
            if org not in AMD_REPOS:
                continue

            if full_tag not in seen:
                seen[full_tag] = ImageInfo(
                    full_tag=full_tag,
                    repo=repo,
                    tag=tag,
                    gpu_variant=extract_gpu_variant(tag),
                    config_keys=[config_key],
                )
            else:
                seen[full_tag].config_keys.append(config_key)

    return list(seen.values())


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _status_line(result: CheckResult) -> str:
    if result.error:
        return f"  Status:   ⚠  could not check ({result.error})"

    lines = []
    if result.is_outdated:
        lines.append(f"  Latest stable: {result.latest_stable}")
        lines.append(f"  Status:        ⚠  OUTDATED — newer stable version available")
    else:
        lines.append(f"  Status:        ✓  up to date")

    if result.has_rc_available and not result.is_outdated:
        lines.append(f"  Latest rc:     {result.latest_rc}")
        lines.append(f"                 ℹ  rc available — test at your discretion")
    elif result.has_rc_available and result.is_outdated:
        lines.append(f"  Latest rc:     {result.latest_rc}")

    return "\n".join(lines)


def print_report(results: list[CheckResult]) -> None:
    border = "=" * 54
    print(f"\n+{border}+")
    print(f"|{'IMAGE VERSION CHECK REPORT':^54}|")
    print(f"+{border}+\n")

    outdated_count = sum(1 for r in results if r.is_outdated)
    rc_count = sum(1 for r in results if r.has_rc_available and not r.is_outdated)

    for result in results:
        img = result.image
        gpu_label = f"  GPU: {img.gpu_variant}" if img.gpu_variant else ""
        print(f"Image: {img.repo}{gpu_label}")
        print(f"  Current tag:   {img.tag}")
        print(f"  Used in:       {', '.join(img.config_keys)}")
        print(_status_line(result))
        print()

    # Summary line
    if outdated_count == 0 and rc_count == 0:
        print("All images are up to date.")
    else:
        if outdated_count:
            print(f"⚠  {outdated_count} image(s) have newer stable versions on Docker Hub.")
            print("   Update the image tags in .github/configs/amd-master.yaml to pick up fixes and performance improvements.")
        if rc_count:
            print(f"ℹ  {rc_count} image(s) have newer rc versions available (opt-in testing).")
    print()

    # Write to GitHub Step Summary if available
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a") as f:
                f.write("\n## Image Version Check\n\n")
                if outdated_count == 0 and rc_count == 0:
                    f.write("All AMD/ROCm images are up to date. ✓\n")
                else:
                    f.write("| Image | Current | Latest Stable | Status |\n")
                    f.write("|-------|---------|---------------|--------|\n")
                    for r in results:
                        status = "✓" if not r.is_outdated else "⚠ OUTDATED"
                        latest = r.latest_stable or "(n/a)"
                        f.write(f"| `{r.image.repo}` ({r.image.gpu_variant or 'any'}) "
                                f"| `{r.image.tag}` | `{latest}` | {status} |\n")
                    if rc_count:
                        f.write("\n> **Note:** Some images have newer rc (release candidate) versions available.\n")
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Docker Hub for newer AMD/ROCm images than those in amd-master.yaml"
    )
    parser.add_argument(
        "--config-files",
        nargs="+",
        default=[".github/configs/amd-master.yaml"],
        help="YAML config files to parse (default: .github/configs/amd-master.yaml)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout per request in seconds (default: {DEFAULT_TIMEOUT})",
    )
    args = parser.parse_args()

    images = load_images_from_configs(args.config_files)
    if not images:
        print("[check_image_versions] No AMD/ROCm images found in config files.")
        return

    results = []
    for img in images:
        latest = find_latest(img.repo, img.tag, img.gpu_variant, args.timeout)
        result = CheckResult(image=img, error=latest["error"])

        if not latest["error"]:
            ls = latest["latest_stable"]
            lr = latest["latest_rc"]

            if ls and _is_newer(ls, img.tag, img.repo):
                result.latest_stable = ls
                result.is_outdated = True
            else:
                result.latest_stable = ls

            if lr and _is_newer(lr, img.tag, img.repo):
                result.latest_rc = lr
                result.has_rc_available = True

        results.append(result)

    print_report(results)
    sys.exit(0)  # always exit 0 — informational only


if __name__ == "__main__":
    main()
