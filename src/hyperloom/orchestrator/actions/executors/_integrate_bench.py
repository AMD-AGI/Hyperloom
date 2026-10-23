# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Benchmarking one applied patch, and the artifacts the benchmark runs against."""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_str_list
from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.common.perf_metric import constraint_fields
from hyperloom.inference_optimizer.grid_server_args import compose_server_args
from hyperloom.orchestrator.actions.executors._workload_envs import prepare_grid_benchmark_config

from ...bringup import write_boot_observation
from ...delivery import file_digest
from ...delivery.ledger import append_record, load_records, mark_prepared
from ._accuracy_gate import (
    framework_run_eval_envs,
    grade_accuracy,
    is_eval_origin,
    parse_eval_results,
    read_eval_probe,
)
from hyperloom.orchestrator.backends import run_grid
from ._grid_runner import GridVariant, _num_gpus_for_config, sanitize_result_dir, sanitize_script_name
from ._grid_variant_filter import (
    apply_aiter_moe_pin_filter,
    apply_multi_node_invalid_variants,
    apply_user_skip_list,
    resolve_skip_spec,
)
from ._integrate_results import _PatchBench
from ._patch_artifacts import _ArtifactSpec
from ._workload_envs import default_baseline_config


class BenchStageMixin:
    """Put the applied patch and its artifacts in front of the benchmark, and read what the boot reported back."""

    def _backup_artifacts(
        self,
        specs: list[_ArtifactSpec],
        *,
        backup_root: Path,
    ) -> list[dict[str, str]]:
        """Commit every artifact target's preimage without touching the tree.

        Runs before the attempt's first mutation, so each record describes the
        tree as the attempt found it. The checkpoint that closes the batch is
        what :func:`restore_pending_integrate` later reads: until it lands, no
        artifact may be installed.

        Args:
            specs: The resolved artifact specs whose targets will be installed.
            backup_root: Directory under which clobbered targets are saved.

        Returns:
            Per-artifact error records; empty when the whole plan is committed.
        """
        errors: list[dict[str, str]] = []
        backup_root.mkdir(parents=True, exist_ok=True)
        for idx, spec in enumerate(specs):
            try:
                existed = spec.target.exists()
                backup_path: str | None = None
                mode = spec.target.stat().st_mode & 0o7777 if existed else None
                if existed:
                    backup_path = str(backup_root / f"{idx:03d}_{spec.target.name}.bak")
                    shutil.copy2(spec.target, backup_path)
                record = {
                    "target": str(spec.target),
                    "rel_target": spec.rel_target,
                    "root": str(spec.root),
                    "kind": spec.kind,
                    "existed": existed,
                    "backup": backup_path,
                    "source": str(spec.source),
                    "pre_image_sha256": file_digest(Path(backup_path)) if backup_path else "",
                    "mode": mode,
                }
                if existed and not record["pre_image_sha256"]:
                    raise OSError(f"artifact backup unreadable: {spec.target}")
                if not append_record(backup_root, record):
                    raise OSError(f"artifact backup ledger write failed: {spec.target}")
            except OSError as exc:
                errors.append({"artifact": spec.rel_target, "error": repr(exc)})
        if not errors and not mark_prepared(backup_root):
            errors.append({"artifact": "", "error": "artifact backup checkpoint could not be persisted"})
        return errors

    def _apply_artifacts(
        self,
        specs: list[_ArtifactSpec],
        *,
        backup_root: Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Install non-diff tuned artifacts over their committed preimages.

        Args:
            specs: The resolved artifact specs to install.
            backup_root: Directory :meth:`_backup_artifacts` committed under.

        Returns:
            A ``(applied, errors)`` tuple: the committed record of each artifact
            now installed, in order, and per-artifact error records.
        """
        committed = {str(row["target"]): row for row in load_records(backup_root)}
        applied: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for spec in specs:
            record = committed.get(str(spec.target))
            if record is None:
                errors.append({"artifact": spec.rel_target, "error": "no committed preimage for this target"})
                continue
            try:
                spec.target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(spec.source, spec.target)
            except OSError as exc:
                errors.append({"artifact": spec.rel_target, "error": repr(exc)})
                continue
            applied.append(record)
        return applied, errors

    async def _bench_patch(
        self,
        bench_leg: _PatchBench,
        *,
        extra_server_args_applied: str,
        extra_envs_applied: dict[str, str],
        unset_envs: "list[str] | None" = None,
        variant_suffix: str = "",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run a 1-variant Magpie bench under the patched server + accuracy gate.

        Args:
            bench_leg: The task params (config / model / bench knobs), the
                per-task workspace root, the originating specialist task id
                (names the variant), ``SharedState.model_path`` (the last rung
                of the model-path precedence) and the session bounds. The
                caller owns the session context, so it resolves the bounds:
                ``session_deadline_sec`` is the monotonic-clock session budget
                deadline, or ``None`` when unbounded, and
                ``variant_expected_sec`` the expected bench runtime used to
                decide whether the remaining budget can fit this bench at all.
            extra_server_args_applied: Server CLI arguments for the variant.
            extra_envs_applied: Environment overrides layered onto the variant.
            unset_envs: Extra env names to remove for this leg, on top of the
                task's ``base_unset_envs``. Used by the switch-off parity leg,
                which has to guarantee the rewrite switches are absent even when
                an earlier KEEP put them into the base configuration.
            variant_suffix: Appended to the variant name so a second leg does not
                collide with the first one's grid slot.

        Returns:
            A ``(bench_result_dict, gate_evidence)`` tuple where
            ``bench_result_dict`` carries ``effective_config`` (the env/arg layers
            the variant launched with, for a faithful replay), and
            ``gate_evidence`` carries ``accuracy_pass`` (True / False / None)
            and ``eval_probe`` (the generation-pathology record, or ``None``).
        """
        from ._ray_serving import maybe_serving_lease

        params = bench_leg.params
        gate_evidence: dict[str, Any] = {
            "accuracy_pass": None,
            "accuracy": None,
            "enablement_accuracy": None,
            "enablement_accuracy_task": "",
            "enablement_accuracy_metric": "",
            "eval_probe": None,
        }
        base_config = Path(params.get("config_path") or self.default_config_path or default_baseline_config())
        if not base_config.exists():
            raise RuntimeError(f"integrate_patch bench: config not found at {base_config}")
        model_path = resolve_session_model_path(
            params=params,
            state_model_path=bench_leg.state_model_path,
            for_serving=True,
        )
        gpu_type = str(params.get("gpu_type") or "").strip().lower() or os.environ.get("GPU_TYPE", "").strip().lower()
        benchmark_script = sanitize_script_name(params.get("benchmark_script"))
        result_dir = sanitize_result_dir(params.get("result_dir"))
        base_extra_args = str(params.get("base_extra_args") or "").strip()
        args_mode = str(params.get("base_args_mode") or "append")
        base_remove = to_str_list(params.get("base_remove_args"))
        config_path = prepare_grid_benchmark_config(
            base_config,
            bench_leg.output_root,
            model_path=model_path or None,
            gpu_type=gpu_type or None,
            benchmark_script=benchmark_script,
            extra_envs=framework_run_eval_envs(params),
            remove_args=params.get("base_remove_args"),
            unset_envs=params.get("base_unset_envs"),
            args_mode=args_mode,
            out_name="integrate_patch.with_envs.yaml",
        )

        envs = dict(extra_envs_applied)
        # Eval-origin enablement needs a raw accuracy for its runnable gate, so
        # RUN_EVAL=true must survive any variant overlay.
        if params.get("enablement") and is_eval_origin(params):
            envs["RUN_EVAL"] = "true"
        variant = GridVariant(
            name=f"integrate-patch-{bench_leg.specialist_task_id[:8]}{variant_suffix}",
            extra_server_args=extra_server_args_applied,
            extra_envs=envs,
            unset_envs=to_str_list(unset_envs),
            note=f"integrate_patch:{bench_leg.specialist_task_id}{variant_suffix}",
        )
        variant.overlay_pythonpath = params.get("overlay_pythonpath", "")
        variant.source_patches = list(params.get("source_patches", []))
        runtime_override = params.get("runtime_override")
        if isinstance(runtime_override, dict) and runtime_override:
            # Preserve list/dict values; apply_runtime_override expects them.
            variant.runtime_override = dict(runtime_override)

        # The compatibility filter is deliberately absent: it shells out to build
        # the framework's arg parser, and a single authored variant must not pay
        # for that probe.
        grid, mn_dropped = apply_multi_node_invalid_variants([variant])
        grid, pin_dropped = apply_aiter_moe_pin_filter(grid)
        grid, skip_dropped = apply_user_skip_list(grid, skip_spec=resolve_skip_spec(params))
        if not grid:
            rejection = (mn_dropped + pin_dropped + skip_dropped)[0]
            return (
                {
                    "status": "skipped",
                    "error_class": "filtered_before_bench",
                    "error": str(rejection.get("reason") or rejection.get("source") or ""),
                    "workspace": "",
                    "materialized_config": str(config_path),
                },
                gate_evidence,
            )

        # Ray-managed GPU execution holds the lease for the whole run_grid so the
        # patch benchmark serializes against other serving instead of colliding
        # with a concurrent GPU specialist server on the same card. ``None`` keeps
        # the local path (multi-node / RAY_EXEC off / pytest default).
        serving_lease = maybe_serving_lease(num_gpus=_num_gpus_for_config(config_path))
        try:
            results = await run_grid(
                base_yaml_path=config_path,
                base_extra_args=base_extra_args,
                grid=grid,
                output_root=bench_leg.output_root,
                magpie_python=params.get("magpie_python") or None,
                keep_going_on_failure=False,
                model_path=model_path or None,
                gpu_type=gpu_type or None,
                benchmark_script=benchmark_script,
                result_dir=result_dir,
                base_args_mode=args_mode,
                base_extra_envs=dict(params.get("base_extra_envs") or {}),
                base_remove_args=base_remove,
                base_unset_envs=to_str_list(params.get("base_unset_envs")),
                serving_lease=serving_lease,
                session_deadline_sec=bench_leg.session_deadline_sec,
                variant_expected_sec=bench_leg.variant_expected_sec,
            )
        finally:
            if serving_lease is not None:
                serving_lease.close()

        bench: dict[str, Any] = {}
        if results:
            result = results[0]
            effective_unset = list(dict.fromkeys(to_str_list(params.get("base_unset_envs")) + list(variant.unset_envs)))
            base_envs = dict(params.get("base_extra_envs") or {})
            bench = {
                **result.to_dict(),
                "status": result.status,
                "output_throughput": result.output_throughput,
                # ``VariantResult`` names these ``ttft_mean_ms`` / ``tpot_mean_ms``;
                # the emitted keys stay ``ttft_ms`` / ``itl_ms`` for the collectors.
                "ttft_ms": result.ttft_mean_ms,
                "itl_ms": result.tpot_mean_ms,
                # The session's constraints fail closed: a lane that drops these refuses every KEEP.
                **constraint_fields(result),
                # Benchmark dir; ``grade_accuracy`` locates accuracy artifacts here.
                "workspace": result.workspace or "",
                "error": result.error or "",
                "error_class": result.error_class,
                "nonfatal_warnings": list(result.nonfatal_warnings),
                # Launch evidence is the immutable proof of the server that produced this
                # measurement. It must survive the patch result and current-best
                # promotion so GEAK can verify the handoff.
                "launch_evidence": dict(result.launch_evidence),
                "launch_evidence_path": result.launch_evidence_path or "",
                "server_log_path": result.server_log_path or "",
                # Materialized config used for this bench; needed by revalidation.
                "materialized_config": str(config_path),
                # The composed stack, not the variant alone: revalidation replays this
                # and would otherwise boot without the base layer. RUN_EVAL is dropped --
                # the replay owns its own eval contract.
                "effective_config": {
                    "extra_envs": {
                        key: value
                        for key, value in {**base_envs, **variant.extra_envs}.items()
                        if key != "RUN_EVAL" and key not in effective_unset
                    },
                    "extra_server_args": compose_server_args(
                        inherited_args="",
                        base_extra_args=base_extra_args,
                        variant_extra_args=variant.extra_server_args,
                        remove_args=base_remove,
                        args_mode=args_mode,
                    ),
                    "remove_args": list(base_remove),
                    "unset_envs": list(effective_unset),
                    "args_mode": args_mode,
                },
            }
        # Classified here so the gate compares two observations produced by the same reader.
        bench["boot_observation_path"] = await asyncio.to_thread(self._observe_bench_boot, bench, bench_leg.output_root)
        # lm-eval writes to ``$EVAL_RESULT_DIR`` under the grid slot, not inside
        # the ``benchmark_*`` workspace. Grade from the slot so the recursive
        # search finds eval output while honoring an explicit ``result_dir``
        # override the same way the grid subprocess does.
        eval_search_root = result_dir or (str(Path(bench["workspace"]).parent) if bench.get("workspace") else "")
        # Guarded: an empty root would send the recursive scan over the cwd.
        if eval_search_root:
            gate_evidence["eval_probe"] = read_eval_probe(eval_search_root)
        if bench.get("status") != "succeeded":
            return bench, gate_evidence
        framework = params.get("framework") or os.environ.get("FRAMEWORK") or None
        results_payload = parse_eval_results(
            eval_search_root, framework=framework, benchmark_mode=bench_leg.benchmark_mode
        )
        accuracy = results_payload.get("accuracy")
        # Raw accuracy for the KB record; ``accuracy_pass`` only carries a verdict.
        measured = float(accuracy) if isinstance(accuracy, (int, float)) else None
        gate_evidence["accuracy_pass"] = grade_accuracy(
            eval_search_root,
            params.get("accuracy_baseline"),
            framework=framework,
            benchmark_mode=bench_leg.benchmark_mode,
        )
        gate_evidence["accuracy"] = measured
        # The enablement branch applies a floor of its own, so it needs the raw score.
        if params.get("enablement"):
            gate_evidence["enablement_accuracy"] = measured
            gate_evidence["enablement_accuracy_task"] = str(results_payload.get("task") or "")
            gate_evidence["enablement_accuracy_metric"] = str(results_payload.get("metric") or "")
        return bench, gate_evidence

    def _observe_bench_boot(self, bench: dict[str, Any], output_root: Path) -> str:
        """Classify the server this bench ran against and persist the observation.

        Args:
            bench: The variant result, naming the server log it produced.
            output_root: The task workspace, used to key the artifact when the
                variant reports no slot of its own.

        Returns:
            str: The observation artifact's path, empty when there was no server
            log to classify or no session to write under.
        """
        from ...bringup import observe_bringup
        from .baseline import open_bringup_attempt, read_bringup_log, server_child_elapsed_sec

        server_log = str(bench.get("server_log_path") or "").strip()
        if not server_log or self.session_dir is None:
            return ""
        slot = Path(bench["workspace"]) if bench.get("workspace") else output_root
        read = read_bringup_log(Path(server_log))
        verdict = observe_bringup(
            server_log=read.text,
            server_elapsed_sec=server_child_elapsed_sec(read.text),
            wrapper_stderr=str(bench.get("error") or ""),
            session_dir=Path(self.session_dir),
        )
        return write_boot_observation(
            verdict.observation,
            session_dir=Path(self.session_dir),
            output_dir=slot,
            attempt=open_bringup_attempt(slot),
        )
