# Ray-managed GPU Execution — P2 收尾计划(把最后两个非-Ray GPU 路径接入 Ray)

## 0. 目标(一句话)

把 **`integrate_patch` 和 `framework_agent`** 这两个仍走**本地 `run_with_session_kill`** 的
GPU/serving benchmark 路径,接入与 baseline / explore / sweep / conc_sweep 一致的
**Ray-managed serving lease**(`maybe_serving_lease → ServingLease`),使"所有会实际占用
GPU 的 serving/benchmark 进程都活在 Ray task/actor 的 GPU 租约内"这一 §4.2 硬不变量
在单节点上**无一例外**。

> 这是 `ray_modify.plan.md` 的补丁级收尾(P2),不改状态机、不改契约,只补两个漏迁的
> 执行器。多节点仍走既有路径(`maybe_serving_lease` 在多节点返回 `None`)。

---

## 1. 现状(为什么要做)

### 1.1 已在 Ray 上的 GPU 执行器(参照物)

| 执行器 | 文件 | 是否传 `serving_lease` |
|---|---|---|
| baseline / profile / roofline | `actions/executors/baseline.py`（profile 继承之） | ✅ `bench_lease = maybe_serving_lease(...)` |
| explore | `actions/executors/explore.py` | ✅ 每轮一个 `round_serving_lease`（P2-fix 已改） |
| sweep | `actions/executors/sweep.py` | ✅ `sweep_lease` |
| conc_sweep | `kernel/conc_sweep.py` | ✅ 每 arm 一个 `arm_lease` |
| GPU specialist | `loop/dispatcher.py` + `_ray_serving.py` | ✅ `GpuSpecialistLease`（整进程在 actor 内） |
| kernel-agent（trace_analyze / GEAK / forge） | `kernel/request_handlers.py` | ✅ 本地 wrapper，但设 `RAY_ADDRESS`，GPU 工作提交到 Ray |

### 1.2 仍非-Ray 的两处(本计划要修)

| # | 位置 | 调用 | 后果 |
|---|---|---|---|
| A | `actions/executors/integrate_patch.py:~2265` | `run_grid(...)` **不传 `serving_lease`** | patch E2E 验证 benchmark 起 vLLM 走**本地** `run_with_session_kill`,**不在 Ray actor 内、不持 `serving_slot`** |
| B | `actions/executors/framework_agent.py:~1100` | `run_grid(...)` **不传 `serving_lease`** | FRAMEWORK 阶段 candidate benchmark 同样走**本地路径** |

`_grid_runner.run_grid` 的分叉点在 `_grid_runner.py`(约 838 行):

```python
if serving_lease is not None:
    ... # Ray: serving_lease.run_session_kill(...) —— 进程活在 ServingActor 内
else:
    proc = run_with_session_kill(cmd, ...)   # 本地子进程(宿主),非 Ray
```

即只要不传 `serving_lease`,就落到本地分支。

### 1.3 影响评估(为什么值得做,但不是 P0 紧急)

- **正确性**:这两个任务由 dispatcher 派发时仍会拿 SQLite `benchmark_lane`(与 serving /
  profile / gpu_research 互斥),所以**不会真的和别的 GPU 工作抢卡** —— 双层保护里 SQLite
  那层还在。因此**不是数据竞争 bug**。
- **一致性 / 统一执行层**:它们**不在 Ray 自定义资源层**(不持 `serving_slot`、不受
  Ray `num_gpus` 调度/排队/放置),与 §0 "统一 GPU 执行层"的目标不符;也无法享受
  P2-fix 的"per-round actor 复用 + actor 自愈"稳定性收益。
- **可观测性**:这两处的 GPU 进程不出现在 `ray status` / `gpu_leases` / ServingActor 树里,
  排查"谁在用 GPU"时是盲区。

---

## 2. 改造方案(与 sweep.py 完全同构)

参照 `sweep.py:352-369` 的既有写法。**接口零改动**,只在 `run_grid` 外面裹一个 lease。

### 2.1 通用 patch 形状

```python
# 1) 顶部 import(若尚未导入)
from ._ray_serving import maybe_serving_lease
from ._grid_runner import _num_gpus_for_config   # run_grid 同模块已导出

# 2) 调用处:创建 lease → try/finally 关闭
#    §12 T1:一个 lease 覆盖这次 benchmark 的 warmup+measure 全部 round,
#    server 进程活在 actor 内,不脱离 GPU 租约(§4.2)。None(多节点 / RAY_EXEC off /
#    pytest 默认)透明退回本地路径,行为不变。
_lease = maybe_serving_lease(num_gpus=_num_gpus_for_config(config_path))
try:
    results = await run_grid(
        ...,                         # 原有参数一字不动
        serving_lease=_lease,        # 新增这一行
    )
finally:
    if _lease is not None:
        _lease.close()               # 释放 GPU 租约(幂等,不抛)
```

### 2.2 A — `integrate_patch.py`

- 位置:`_bench(...)` 内、当前 `results = await run_grid(...)`(约 2265-2280 行)。
- `config_path` 已在同函数上文 materialize 好(约 2241 行),可直接用于
  `_num_gpus_for_config(config_path)`。
- 改动:

```python
        from ._ray_serving import maybe_serving_lease  # 顶部 import 更佳
        from ._grid_runner import _num_gpus_for_config

        ip_lease = maybe_serving_lease(num_gpus=_num_gpus_for_config(config_path))
        try:
            results: list[VariantResult] = await run_grid(
                base_yaml_path=config_path,
                base_extra_args=str(params.get("base_extra_args") or "").strip(),
                grid=[variant],
                output_root=output_root,
                magpie_python=params.get("magpie_python") or None,
                variant_timeout_sec=int(
                    params.get("variant_timeout_sec", self.variant_timeout_sec),
                ),
                keep_going_on_failure=False,
                model_path=resolved_model or None,
                gpu_type=resolved_gpu or None,
                benchmark_script=override_script,
                result_dir=override_result_dir,
                base_args_mode=str(params.get("base_args_mode") or "append"),
                serving_lease=ip_lease,          # ← 新增
            )
        finally:
            if ip_lease is not None:
                ip_lease.close()
```

### 2.3 B — `framework_agent.py`

- 位置:candidate benchmark 处、当前 `results = await run_grid(...)`(约 1100-1114 行)。
- `config_path` 已在同函数上文 materialize(约 1085-1091 行)。
- 改动同构:

```python
        from ._ray_serving import maybe_serving_lease  # 顶部 import 更佳
        from ._grid_runner import _num_gpus_for_config

        fa_lease = maybe_serving_lease(num_gpus=_num_gpus_for_config(config_path))
        try:
            results: list[VariantResult] = await run_grid(
                base_yaml_path=config_path,
                base_extra_args=str(params.get("base_extra_args") or "").strip(),
                grid=[variant],
                output_root=output_root,
                magpie_python=params.get("magpie_python") or None,
                variant_timeout_sec=int(
                    params.get("variant_timeout_sec", self.variant_timeout_sec),
                ),
                keep_going_on_failure=False,
                model_path=resolved_model or None,
                gpu_type=resolved_gpu or None,
                benchmark_script=override_script,
                result_dir=override_result_dir,
                serving_lease=fa_lease,          # ← 新增
            )
        finally:
            if fa_lease is not None:
                fa_lease.close()
```

---

## 3. 关键不变量与注意点

1. **`serving_slot` 串行**:接入后,`integrate_patch` / `framework_agent` 的 benchmark 会经
   `ServingLease` 持有 Ray `serving_slot`,自动与 serving / explore / conc_sweep / GPU
   specialist(gpu_research_lane)在 **Ray 层**互斥 —— 补上了之前只有 SQLite `benchmark_lane`
   单层的缺口(现在双层齐全)。
2. **不要重复持锁死锁**:这两个执行器由 dispatcher 派发时**是否已经拿了 `benchmark_lane`?**
   —— 需确认(见 §4 步骤 0)。SQLite `benchmark_lane` 与 Ray `serving_slot` 是**两套独立资源**,
   同时持有不会互相死锁(一个是进程内 SQLite 记账,一个是 Ray 调度)。但要保证
   **lease 在任务结束前 `close()`**,否则 `serving_slot` 泄漏会让后续 serving 永久 PENDING
   (与最初 install.sh 漏声明 `serving_slot` 的死锁同源)。`try/finally` 是硬要求。
3. **多节点**:`maybe_serving_lease` 在 `is_multi_node()` 或 `INFERENCE_OPTIMIZER_RAY_EXEC=off`
   时返回 `None` → 自动退回本地路径,**多节点行为零改动**(多节点收尾另计,见 §4.2 P4 skeleton)。
4. **server teardown**:`ServingLease` 路径下 server 由 actor 内 lifecycle 收尾;本地路径由
   `run_with_session_kill` 收尾。两者都已实现,不需要额外 teardown 代码。
5. **`_num_gpus_for_config`**:从 materialize 后的 YAML 读 `TP` 作为 `num_gpus`。integrate_patch /
   framework_agent 都是整机 serving benchmark(非 disjoint),`num_gpus=TP` 正确(单卡 TP=1)。

---

## 4. 落地步骤

0. **前置确认**:grep dispatcher / 这两个 action 的派发路径,确认它们是否已持
   `benchmark_lane`(`locks.try_acquire_many([... 'benchmark_lane' ...])`)。
   - 若已持 `benchmark_lane`:Ray `serving_slot` 是**额外一层**,OK。
   - 若未持:接 Ray lease 后由 `serving_slot` 兜底互斥,也 OK(甚至更好)。
1. 按 §2.2 / §2.3 打两处 patch(加 import + `serving_lease=` + `try/finally close`)。
2. `python -c "import ast; ast.parse(open(f).read())"` 两文件语法自检 + `ReadLints`。
3. 单测:
   - 现有 `integrate_patch` / `framework_agent` 相关测试全过(默认走本地路径,因为
     pytest 下 `_should_use_ray_backend()` 返回 False —— 保证 hermetic)。
   - 新增(可选)一个 `INFERENCE_OPTIMIZER_RAY_EXEC=1` 的用例,断言 `run_grid` 收到了
     非 `None` 的 `serving_lease`(monkeypatch `maybe_serving_lease` / `run_grid` 探针,
     参照 `test_ray_backend_unit.py` 里 ServingLease 的既有测法)。
4. 单节点真机验证:跑一轮让 FRAMEWORK 阶段真的 benchmark candidate、且触发一次
   `integrate_patch`(需有 Critic verdict 的 specialist patch),确认:
   - `ray status` 在这两个 benchmark 期间显示 `serving_slot 1.0/1.0`;
   - 进程树里 vLLM 是 `ray::ServingActor` 的子进程(不是 optimizer 直接子进程);
   - 结束后 `serving_slot` 释放回 `0.0/1.0`,无泄漏。
5. 覆盖率:CI 90% 线;两处改动很小,补上 §3 的 import 与分支即可。

---

## 5. 验收标准

- [ ] `integrate_patch` / `framework_agent` 的 benchmark 在单节点走 Ray,vLLM 活在
      `ServingActor` 内、持 `serving_slot`。
- [ ] `grep -rn "run_grid(" src/hyperloom/orchestrator` 后,**所有** GPU/serving benchmark
      调用点都传 `serving_lease`(specialists/rebench.py 例外——它本就跑在 GPU specialist
      的 actor 内,无需再套 lease)。
- [ ] 多节点 / `RAY_EXEC=off` / pytest 行为不变(`maybe_serving_lease → None`)。
- [ ] 无 `serving_slot` 泄漏(`try/finally close` 到位)。
- [ ] `ray_modify.plan.md` §5 的"所有 GPU 执行归 Ray"在单节点上无例外。

---

## 6. 备注:与临时调试改动的关系

本计划是**永久性**收尾,与 `[TEMP-DEBUG]` 的两个 orchestration prompt commit
(`e0a7b25d` 偏向 GPU specialist、`2805540a` 强制 bench=true)**无关**,后者验证完应回退。
本计划改的是执行器代码,不碰 prompt。
