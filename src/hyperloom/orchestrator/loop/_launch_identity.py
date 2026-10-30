# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The launch configuration ``current_best`` was measured on, and how well it is evidenced."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path
from typing import Any, Mapping

from hyperloom.common.coerce import to_str_list
from hyperloom.common.launch_log_evidence import launch_argv_from_log, observed_server_identity_from_log
from hyperloom.inference_optimizer.grid_server_args import merge_cumulative_extra_server_args

from ..kernel.geak_config import _geak_overlay_digest
from ._keep_patch_column import _source_snapshot_layers


class LaunchIdentityMixin:
    """Pinning and reproducing the launch a measurement ran under."""

    @staticmethod
    def _handoff_launch_identity(env_spec: Mapping[str, Any]) -> str:
        """Return a stable identity for the complete GEAK baseline launch."""
        config = env_spec.get("config") if isinstance(env_spec.get("config"), Mapping) else {}
        # The recipe PATH is deliberately excluded: a baseline re-run repoints
        # ``baseline_config_path`` at a new timestamped YAML without re-stamping
        # current_best, so hashing the path drops the same-config reference for a
        # byte-identical recipe. The digest already carries the recipe content.
        payload = {
            "base_launch_recipe_digest": str(env_spec.get("base_launch_recipe_digest") or ""),
            "extra_server_args": str(config.get("extra_server_args") or ""),
            "extra_envs": dict(config.get("extra_envs") or {}),
            "server_launch_flags": str(config.get("server_launch_flags") or ""),
            "source_snapshots": list(env_spec.get("source_snapshots") or []),
            "overlay_pythonpath": str(env_spec.get("overlay_pythonpath") or ""),
            "overlay_digest": str(env_spec.get("overlay_digest") or ""),
        }
        # Keep legacy identities unchanged for launches with default controls.
        for key in ("remove_args", "unset_envs"):
            values = sorted(set(to_str_list(config.get(key))))
            if values:
                payload[key] = values
        if str(config.get("args_mode") or "").strip().lower() == "replace":
            payload["args_mode"] = "replace"
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    @staticmethod
    def _launch_recipe_digest(path: str) -> str:
        """Hash the recipe content so an in-place recipe edit invalidates tput."""
        try:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            return ""

    def _measurement_launch_flags(self, measurement: Mapping[str, Any]) -> str:
        """Read resolved launch flags only from the promoted measurement's evidence."""
        existing = str(measurement.get("resolved_server_launch_flags") or "").strip()
        if existing:
            return existing
        evidence = measurement.get("launch_evidence")
        if isinstance(evidence, Mapping):
            observed = str(evidence.get("observed_server_launch_flags") or "").strip()
            if observed:
                return observed
        framework = (
            str(
                (evidence.get("framework") if isinstance(evidence, Mapping) else "")
                or os.environ.get("FRAMEWORK", "")
                or "sglang"
            )
            .strip()
            .lower()
        )
        paths = [str(measurement.get("server_log_path") or "").strip()]
        if isinstance(evidence, Mapping):
            paths.append(str(evidence.get("actual_server_log_path") or "").strip())
        for key in ("benchmark_workspace", "workspace"):
            workspace = str(measurement.get(key) or "").strip()
            if workspace:
                root = Path(workspace)
                paths.append(str(root / "server.log"))
        for path in dict.fromkeys(path for path in paths if path):
            flags = launch_argv_from_log(path, framework)
            if flags:
                return flags
        return ""

    @staticmethod
    def _resolved_sglang_server_config(launch_evidence: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve selected SGLang ``ServerArgs`` fields from declared args.

        A captured command line is preferred evidence. This fallback makes the
        declaration inspectable when a reused ready server did not emit a fresh
        CLI line; it is intentionally not treated as observed launch evidence.
        """
        if str(launch_evidence.get("framework") or "").strip().lower() != "sglang":
            return {}
        raw_args = str(launch_evidence.get("requested_server_args") or "").strip()
        model_path = str(launch_evidence.get("model_path") or "").strip()
        try:
            from sglang.srt.server_args import ServerArgs

            parser = argparse.ArgumentParser(add_help=False)
            ServerArgs.add_cli_args(parser)
            tokens = shlex.split(raw_args)
            if model_path and not any(token == "--model-path" or token.startswith("--model-path=") for token in tokens):
                tokens = ["--model-path", model_path, *tokens]
            namespace, _unknown = parser.parse_known_args(tokens)
        except (ImportError, OSError, ValueError, SystemExit):  # optional declared-only evidence
            return {}
        fields = (
            "model_path",
            "tokenizer_path",
            "host",
            "port",
            "tp_size",
            "dp_size",
            "pp_size",
            "mem_fraction_static",
            "context_length",
            "max_total_tokens",
            "max_running_requests",
            "attention_backend",
            "quantization",
            "dtype",
            "disable_cuda_graph",
            "enable_dp_attention",
            "enable_ep_moe",
            "chunked_prefill_size",
            "schedule_policy",
        )
        out: dict[str, Any] = {}
        for field_name in fields:
            value = getattr(namespace, field_name, None)
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[field_name] = value
        return out

    @staticmethod
    def _measurement_observed_server_identity(measurement: Mapping[str, Any]) -> dict[str, Any]:
        """Return archived SGLang ``ServerArgs`` evidence from this measurement only."""
        evidence = measurement.get("launch_evidence")
        if isinstance(evidence, Mapping):
            identity = evidence.get("observed_server_identity")
            if isinstance(identity, Mapping) and identity:
                return {str(key): value for key, value in sorted(identity.items())}
        if str((evidence or {}).get("framework") if isinstance(evidence, Mapping) else "").lower() != "sglang":
            return {}
        for path in (
            str(measurement.get("server_log_path") or ""),
            str((evidence or {}).get("actual_server_log_path") if isinstance(evidence, Mapping) else ""),
        ):
            if path:
                identity = observed_server_identity_from_log(path, "sglang")
                if identity:
                    return identity
        return {}

    @staticmethod
    def _observed_launch_identity(
        declared_identity: str,
        observed_flags: str,
        observed_server_identity: Mapping[str, Any] | None = None,
    ) -> str:
        """Hash actual captured launch data under the declared identity."""
        if not declared_identity or (not observed_flags and not observed_server_identity):
            return ""
        payload = json.dumps(
            {
                "declared_launch_identity": declared_identity,
                "observed_server_launch_flags": observed_flags,
                "observed_server_identity": dict(observed_server_identity or {}),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    @staticmethod
    def _identity_verification_status(
        *,
        expected_identity: str,
        measurement: Mapping[str, Any],
    ) -> str:
        """Classify whether identity rests on observed or declared evidence."""
        declared_identity = str(measurement.get("declared_launch_identity") or "")
        if not expected_identity or declared_identity != expected_identity:
            return "unverified"
        evidence = measurement.get("launch_evidence")
        observed = str(measurement.get("resolved_server_launch_flags") or "").strip()
        if not observed and isinstance(evidence, Mapping):
            observed = str(evidence.get("observed_server_launch_flags") or "").strip()
        observed_server_identity = LaunchIdentityMixin._measurement_observed_server_identity(measurement)
        if observed or observed_server_identity:
            return "verified_observed"
        if isinstance(evidence, Mapping) and (
            str(evidence.get("requested_server_args") or "").strip()
            or bool(evidence.get("requested_server_env"))
            or str(evidence.get("recipe_digest") or "").strip()
        ):
            return "verified_declared_only"
        return "unverified"

    def _stamp_current_best_measurement(self, evidence: Mapping[str, Any] | None = None) -> None:
        """Attach the exact promoted config identity to ``current_best``."""
        cb = self.shared_state.current_best
        evidence = evidence if isinstance(evidence, Mapping) else {}
        launch_evidence = evidence.get("launch_evidence") or {}
        launch_evidence = dict(launch_evidence) if isinstance(launch_evidence, Mapping) else {}
        measurement = {
            "schema_version": 2,
            "tput": float(cb.get("tput") or 0.0),
            "benchmark_workspace": str(
                evidence.get("workspace") or evidence.get("single_workspace") or cb.get("workspace") or ""
            ),
            "server_log_path": str(
                evidence.get("server_log_path") or launch_evidence.get("actual_server_log_path") or ""
            ),
            "launch_evidence": launch_evidence,
            "launch_evidence_path": str(evidence.get("launch_evidence_path") or ""),
        }
        measurement["resolved_server_launch_flags"] = self._measurement_launch_flags(measurement)
        observed_server_identity = self._measurement_observed_server_identity(measurement)
        if observed_server_identity:
            launch_evidence["observed_server_identity"] = observed_server_identity
        measurement["observed_server_identity"] = observed_server_identity
        measurement["resolved_server_config"] = observed_server_identity or self._resolved_sglang_server_config(
            launch_evidence
        )
        declared_identity = str(
            self.build_env_spec(
                measurement=measurement,
                server_launch_flags=measurement["resolved_server_launch_flags"],
            )["launch_identity"]
        )
        measurement["declared_launch_identity"] = declared_identity
        measurement["observed_launch_identity"] = self._observed_launch_identity(
            declared_identity,
            measurement["resolved_server_launch_flags"],
            observed_server_identity,
        )
        measurement["identity_verification_status"] = self._identity_verification_status(
            expected_identity=declared_identity,
            measurement=measurement,
        )
        # Do not publish a partial measurement. All parsing and identity
        # preparation above is local; these paired assignments are the commit.
        cb["measurement"] = measurement
        self.shared_state.current_best_measurement = measurement

    def current_best_launch_config(self) -> dict[str, Any]:
        """The launch config ``current_best`` was measured on.

        Returns:
            Server args, envs, removal/replacement controls, and ``final_overlay``.
        """
        cb = self.shared_state.current_best
        args = str(cb.get("extra_server_args") or "").strip()
        envs: dict[str, str] = {}
        raw_envs = cb.get("extra_envs")
        if isinstance(raw_envs, Mapping):
            for key, value in raw_envs.items():
                name = str(key)
                # A ``-``-prefixed key is a server arg; exported as an env the
                # backend ignores it and the flag is silently lost.
                if name.startswith("-"):
                    token = name if value in ("", None) else f"{name}={value}"
                    args = merge_cumulative_extra_server_args(args, token, "")
                else:
                    envs[name] = str(value)
        return {
            "extra_server_args": args,
            "extra_envs": envs,
            "remove_args": to_str_list(cb.get("remove_args")),
            "unset_envs": to_str_list(cb.get("unset_envs")),
            "args_mode": "replace" if str(cb.get("args_mode") or "").strip().lower() == "replace" else "append",
            "final_overlay": str(cb.get("final_overlay") or "").strip(),
        }

    def build_env_spec(
        self,
        *,
        measurement: Mapping[str, Any] | None = None,
        server_launch_flags: str | None = None,
        server_launch_envs: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Fully-reproducible descriptor of ``current_best``'s launch environment.

        Layers, in the order a consumer must apply them to reconstruct the exact
        stack ``current_best`` was measured on:

          * ``config``  — cumulative server args + env vars (the reversible layer).
          * ``source_snapshots`` — ordered durable source-layer snapshots
            (``scope=source_patch`` entries), each a self-contained directory
            (see :mod:`source_snapshot`) that reconstructs the patched framework
            tree independent of the mutable live checkout.
          * ``overlay_pythonpath`` — the authored-kernel overlay prefix.
          * ``base_launch_recipe`` — the baseline Magpie recipe to launch from.

        This is the single source of truth the GEAK handoff forwards so the
        baseline ref is materialized from the SAME layers as ``current_best``
        (not just its flags/env), closing the cross-harness baseline gap.
        """
        cb = self.shared_state.current_best
        materialized = self.current_best_launch_config()
        # ``extra_server_args`` remains the current_best delta; ``extra_envs`` is the
        # current_best delta layered over any recipe launch environment (see below).
        # Full engine flags must come from the same promoted measurement, not a
        # search over historical benchmarks with a similar throughput.
        if not isinstance(measurement, Mapping):
            measurement = self.shared_state.current_best_measurement
        if server_launch_flags is None:
            server_launch_flags = self._measurement_launch_flags(measurement)
        recipe_path = str(self.shared_state.baseline_config_path or "")
        env_spec = {
            "schema_version": 1,
            "config": {
                "extra_server_args": materialized.get("extra_server_args") or "",
                # A launch-time environment the recipe applied itself sits under
                # the current_best delta, which wins on a shared name.
                "extra_envs": {**dict(server_launch_envs or {}), **dict(materialized.get("extra_envs") or {})},
                "remove_args": list(materialized["remove_args"]),
                "unset_envs": list(materialized["unset_envs"]),
                "args_mode": materialized["args_mode"],
                # Authoritative, COMPLETE engine flags (run-specific stripped);
                # when unavailable, args_mode governs recipe inheritance.
                "server_launch_flags": server_launch_flags,
            },
            # current_best's embedded stack was promoted with its tput. Reading the
            # mutable global stack here could attach an unrelated source patch after
            # a resume or partial state write.
            "source_snapshots": _source_snapshot_layers(cb.get("optimization_stack")),
            "overlay_pythonpath": materialized.get("final_overlay") or "",
            "overlay_digest": _geak_overlay_digest(str(materialized.get("final_overlay") or "")),
            "base_launch_recipe": recipe_path,
            "base_launch_recipe_digest": self._launch_recipe_digest(recipe_path),
            # Backward-compatible alias for existing GEAK v2 consumers.
            "launch_recipe": recipe_path,
            # Additive evidence record. Consumers can distinguish a captured
            # launch from a declaration resolved without a new CLI line.
            "measurement_evidence": dict(measurement.get("launch_evidence") or {}),
            "measurement_identity": {
                "declared_launch_identity": str(measurement.get("declared_launch_identity") or ""),
                "observed_launch_identity": str(measurement.get("observed_launch_identity") or ""),
                "verification_status": str(measurement.get("identity_verification_status") or "unverified"),
                "observed_server_identity": dict(measurement.get("observed_server_identity") or {}),
                "resolved_server_config": dict(measurement.get("resolved_server_config") or {}),
            },
        }
        env_spec["launch_identity"] = self._handoff_launch_identity(env_spec)
        return env_spec
