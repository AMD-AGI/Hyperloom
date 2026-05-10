#!/usr/bin/env python3
"""Submit commands to a running RayJob cluster.

Usage:
    python ray_submit.py --ray-address ray://<head>:10001 --command "bash run_baseline.sh"
    python ray_submit.py --ray-address ray://<head>:10001 --command "curl -X POST http://localhost:8888/start_profile"

Used by executor.sh in Remote mode to dispatch operations to the RayJob cluster
instead of running them locally.
"""

import argparse
import sys


def submit_command(ray_address: str, command: str, timeout: int = 3600) -> dict:
    import ray
    import subprocess as sp

    ray.init(address=ray_address, ignore_reinit_error=True)

    @ray.remote(num_gpus=0, num_cpus=1)
    class CmdRunner:
        def run(self, cmd: str, timeout_s: int) -> dict:
            result = sp.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout_s)
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

    runner = CmdRunner.remote()
    ref = runner.run.remote(command, timeout)
    return ray.get(ref, timeout=timeout + 60)


def main():
    parser = argparse.ArgumentParser(description="Submit command to Ray cluster")
    parser.add_argument("--ray-address", required=True, help="Ray client address, e.g. ray://<head>:10001")
    parser.add_argument("--command", required=True, help="Shell command to execute on the cluster")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds (default: 3600)")
    args = parser.parse_args()

    try:
        result = submit_command(args.ray_address, args.command, args.timeout)
    except Exception as e:
        print(f"ERROR: Failed to submit command: {e}", file=sys.stderr)
        sys.exit(1)

    if result["stdout"]:
        print(result["stdout"], end="")
    if result["returncode"] != 0:
        if result["stderr"]:
            print(result["stderr"], end="", file=sys.stderr)
        sys.exit(result["returncode"])


if __name__ == "__main__":
    main()
