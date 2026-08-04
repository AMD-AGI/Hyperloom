# EXPLORE decision 轮 overtime-kill 锚点量纲不对称（精度门耗时被计入被测量、却不在锚点内）

- **状态**：待修复
- **严重级别**：高 —— EXPLORE 阶段在特定配置下 100% 无法产出 KEEP，且失败表现为"变体太慢"，掩盖了真实收益
- **类型**：回归（regression）。由两个各自正确的改动叠加产生，无任何代码/注释/测试把它们联系起来
- **首次实证**：session `Qwen3-14B-FP8_20260730T085022Z_63134a89`
  （`/wekafs/zgong/Hyperloom-Sessions/atom/Qwen3-14B-FP8/20260730T085022Z`）

---

## 1. 一句话描述

EXPLORE 的 decision 轮软超时闸门，拿 **`吞吐 benchmark + 全量 1319 题 gsm8k`** 的耗时，去比 **`2.0 × 仅含吞吐 benchmark 的 baseline 锚点`**。分母被抽掉了一个数分钟量级的项，分子里还留着，导致该闸门在数学上不可能通过——它筛掉的不是"慢变体"，而是"任何开着精度门的变体"。

---

## 2. 设计原意（这正是要恢复的目标）

`explore.py:955-957` 的注释把不变式写得很清楚：

> The soft deadline is anchored on the warm client-only measure time and enforced
> from the server-ready marker, so **both the measured runtime and this anchor
> exclude cold boot / warmup**.

`explore.py:441-442` 说明该软闸门与硬帽的分层关系，软闸门代表：

> the designed **"slower than baseline"** bound

CLI 帮助（`cli/parser.py:1181-1193`）对用户的承诺：

> kill fires when **POST-READY (pure hot client) wall-clock** exceeds
> `decision_anchor_sec * RATIO`；默认 2.0 = "kill at +100% over the warm client anchor"

`state/shared_state.py:469-471` 对锚点字段的定义：

> Baseline WARM measure-round wall-clock (client-only, no boot); anchors the
> explore overtime kill **apples-to-apples**.

**综合起来，该机制的设计契约是三条：**

1. **它只用来判断"吞吐是否比 baseline 慢"** —— 语义是纯粹的 throughput 快慢比较；
2. **锚点与被测量必须同量纲**（apples-to-apples）—— 注释把不变式表述为"两边排除相同的东西"；
3. 超过 `kill_ratio`（默认 2.0）意味着"慢了一倍以上，不值得继续等"。

设计上，**精度评测（accuracy gate / gsm8k）根本不应进入这个比较**。它不是吞吐信号，耗时也与吞吐快慢无关。

---

## 3. Bug 的具体表现

### 3.1 代码层：量纲被单方面破坏

deadline 计算（`orchestrator/actions/executors/explore.py:951-958`）：

```python
decision_anchor_sec = (
    baseline_warm_runtime_sec if (use_warm_decision and baseline_warm_runtime_sec > 0)
    else baseline_runtime_sec
)
decision_deadline_sec = decision_anchor_sec * overtime_kill_ratio
```

两轮的传参差异（`explore.py`）：

| 轮次 | 行号 | `soft_deadline_sec` |
|---|---|---|
| warmup 轮 | `explore.py:1052` | `None`（**无 deadline**） |
| decision 轮 | `explore.py:1132` | `decision_deadline_sec`（**受约束**） |

锚点来源链：

- `baseline.py:1871` —— baseline round-2 强制 `run_eval=False`
- `baseline.py:1896` —— round-2 的 wall-clock 写入 `result["measure_round_runtime_sec"]`
- `writeback.py:2229-2231` —— 该值被提升为 `shared_state.baseline_warm_runtime_sec`

被测量一侧：

- `_workload_envs.py:985-987` —— `RUN_EVAL` 默认 `"true"`，explore 变体继承之
- `explore.py` / `_grid_runner.py` 中 **grep `run_eval` 无任何命中** —— explore 从不改写 `RUN_EVAL`

**净结果**：

| | 内容 | `RUN_EVAL` | 本次实测 |
|---|---|---|---|
| **锚点（分母）** | baseline round-2，**仅 benchmark** | `'false'` | **198.38 s** |
| **被测（分子）** | 变体 decision 轮，**benchmark + 全量 1319 题 gsm8k** | `'true'` | 需 ~574 s |

> 两侧的 `RUN_EVAL` 取值均已从 session 落盘的 materialized `config.yaml` 直接读取确认。

### 3.2 运行层：一个干净的正收益变体被冤杀

变体 `fp8_kv_only_low_risk`，参数仅 `--kv_cache_dtype fp8`。

**warmup 轮（12:14:35 → 12:24:21，无 deadline）—— 完整成功：**

| 项 | 实测 |
|---|---|
| 吞吐 benchmark | 192/192 完成，**122.1 s** |
| output_throughput | **1446.4 tok/s = +19.39%**（baseline 1211.5） |
| total_output_tokens | 176,609（与 baseline 逐位相同，`ignore_eos` 钉死输出长度） |
| mean_tpot | 38.56 ms（baseline 46.98 ms，更快） |
| **全量 gsm8k** | **跑完**，`n-samples: {original: 1319, effective: 1319}` |
| **精度** | **strict 0.9492 / flexible 0.9492**（baseline 0.9507，基本持平） |
| chat POST | 2377 条 |
| **轮总耗时** | **586 s** |

**decision 轮（12:24:21 → 12:31:03，deadline = 396.8 s）—— 401.8 s 被杀：**

```
_grid_runner: variant fp8_kv_only_low_risk killed_overtime
    (runtime=401.8s deadline=396.8s est_output_tput=n/a tok/s)
explore: variant fp8_kv_only_low_risk KILLED_OVERTIME
    (runtime=401.8s vs warm anchor=198.4s, ratio=2.02x, kill_ratio=2.00x);
    skipping KEEP/REVERT ladder.
```

**这个变体既不慢也没坏**：它在无约束那一轮把全套流程（含全量精度评测）在 586 s 内干净跑完，精度正常，输出 token 数与 baseline 完全一致（排除了生成退化）。

### 3.3 算账：闸门在数学上不可能通过

以 warmup 轮实测数拆解（server.log 首行 12:14:47 − 轮启动 12:14:35 ⇒ boot ≈ 12 s）：

```
warmup 轮总计            586 s
  ├─ server boot        ~12 s
  ├─ 吞吐 benchmark      122 s
  └─ 全量 gsm8k         ~452 s

decision 轮（复用热 server，无 boot）需要
    = 122 s (bench) + ~452 s (gsm8k) ≈ 574 s
decision 轮预算
    = 198.38 × 2.0                    = 396.8 s
缺口                                   ≈ 177 s
实测                                   401.8 s 被杀（刚过线即死）
```

**即使变体的 benchmark 快到 0 秒，单是全量 gsm8k 的 ~452 s 也已超过 396.8 s 的预算。**

**按设计契约（同量纲）本应如何判定：**

```
锚点（纯 bench）      198.38 s
该变体（纯 bench）     122.1 s
比值                  0.62x   ← 不但没慢，还快 38%
0.62 < kill_ratio 2.0  ⇒  应当通过，进入 KEEP/REVERT 阶梯
```

### 3.4 影响面：本次 session 的量化后果

- **21 次 `killed_overtime`**，每一次的 `runtime` 都精确是 **401.8 s**、`deadline` 都是 **396.8 s** —— 确定性失效，非负载波动。
- EXPLORE 累计 **accepted 0 / rejected 30 / tested 30**，三个宏观周期全部 0 KEEP。
- 最终 `cumulative_gain = 0.00%`，`cumulative_gain_validated = 0.00%`（报告标注 *"never validated — no full-stack rebench ran in this session"*），`current_best = baseline`。
- 被误杀的确凿收益：**`--kv_cache_dtype fp8` 的 +19.4%**，经 3 次独立测量互证、精度正常。
- ledger 中 30 条 rejected 的 `tput` 全为 `None`，导致 LLM 无法区分"方向错"与"方向对但被误杀"，后续把可用的下划线参数改成不可用的连字符别名（4 次），并使搜索空间在后期几乎完全枯竭（多轮 `payload=N → runnable=0`）。

---

## 4. 引入路径（git 已核实，两个 commit 互不知情）

1. **`2d992adde`（2026-06-23）** 建立 warm anchor 机制：
   > "the per-variant overtime kill now anchors on the baseline WARM measure time
   > (new `SharedState.baseline_warm_runtime_sec`, promoted from the baseline
   > measure round) so a one-time cold boot / aiter recompile no longer trips it."

   **此时 round-2 仍跑 eval**，分子分母都含 eval，比值成立，设计自洽。

2. **`07b7178e7`（2026-07-27）** `fix(baseline): stop measuring accuracy twice...`
   给 round-2 加了 `run_eval=False`。其理由充分且正确（`baseline.py:1856-1863` 注释）：
   > "Running it twice **cost minutes per baseline** and, worse, doubled the window
   > in which a server death could take the eval down with it -- **which is exactly
   > how a run was lost**."

   但该 commit **只改了 `baseline.py` + 1 个测试**，提交信息中
   `anchor` / `overtime` / `explore` / `measure_round_runtime` 的出现次数为 **0**。

3. 而 round-2 的 wall-clock 正是被 `writeback.py:2229-2231` 提升为
   `baseline_warm_runtime_sec` 的那个值。**三天后即本次 session。**

**两个改动各自正确，组合未被审视。** 作者知道 eval 是"数分钟"量级，却恰好把它从"日后会成为超时分母"的那一轮里删掉了。

### 为什么测试没拦住

- 唯一设置 `baseline_warm_runtime_sec` 的测试是
  `tests/test_explore_executor.py:847`（值 5.0，kill_ratio 1.20）。
- 其 fixture `_write_baseline_yaml` 只写 `envs: {TP, CONC, ISL, OSL}`，**完全不含 `RUN_EVAL`**。
- 且 benchmark subprocess 被 mock 成瞬时返回 —— **eval 耗时在结构上无法被表达**。
- `grep RUN_EVAL tests/test_explore_executor.py` 无任何命中。

所以该 bug 对 CI 完全不可见。

### 同类问题曾被当作 bug 修过（先例）

`a36edb9b7` —— *"fix(baseline): use round-1 full-run wall-clock as overtime anchor
— anchoring on the client-only time would soft-kill normal variants as
KILLED_OVERTIME."*

**完全相同的失效模式**（锚点缺少了变体必须付出的成本项），当时即按 bug 处理。

### 旁证：SWEEP 已经做了正确的事

`sweep.py:132-134`：

```python
# Accuracy eval is concurrency-invariant, so it is always skipped per
# sweep point (the accuracy gate still runs on explore / baseline).
variant_envs["RUN_EVAL"] = "false"
```

**SWEEP 显式把 eval 从其变体中归一化掉了，EXPLORE 没有。** 说明"该量纲需要对齐"在本仓库内已是既有认知。

---

## 5. 修复目的

**核心目标：让这个软上限重新回到「只衡量 throughput」，不再包含 accuracy eval 的耗时。**

这是一次**正确性修复**，不是调参、不是放宽阈值：

1. **恢复设计原意** —— 该闸门自设计之初就只用于识别"吞吐比 baseline 慢一倍以上"的变体
   （`explore.py:441-442` 的 *"slower than baseline" bound*、CLI 的 *"+100% over the warm
   client anchor"*）。精度评测既不是吞吐信号，其耗时也与吞吐快慢无关，本就不该参与这个比较。

2. **恢复 apples-to-apples 不变式** —— `state/shared_state.py:469-471` 与
   `explore.py:955-957` 都明文要求锚点与被测量同量纲。当前锚点排除了 eval，被测量却包含 eval，
   不变式已被破坏。

3. **消除确定性误杀** —— 修复后，判据回归为「变体纯 bench 时间 vs baseline 纯 bench 时间」。
   以本例校验：`122.1 / 198.38 = 0.62x < 2.0` ⇒ 正确放行，进入 KEEP/REVERT 阶梯。

4. **不放宽把关强度** —— 真正吞吐慢一倍以上的变体仍应被杀。修复只移除量纲错误，
   `kill_ratio` 语义与默认值 2.0 保持不变。

**验收标准：**

- 对任一变体，`runtime` 与 `anchor` 两侧必须包含**相同的阶段构成**；
- 精度评测的耗时**不得**计入 overtime 判据的分子；
- `fp8_kv_only_low_risk`（纯 bench 122.1 s，锚点 198.38 s）这类变体必须能通过软闸门；
- 一个纯 bench 耗时 > `2.0 × 198.38 s` 的变体仍必须被杀。

---

## 6. 候选修复方向

> 两个方案都直接服务于 §5 的目标：让软上限只衡量 throughput。
> 具体实现待评审后确定。

### 方案 A —— overtime 判据只计 bench 阶段（治本）

**做法**：把 overtime 判据的分子从"整轮墙钟"改为"精度评测开始前的墙钟"，即从总耗时中
排除 eval 段。判据回归为纯粹的
`variant_bench_time / baseline_warm_runtime_sec`。

**优点**

- **最贴合设计原意**。不改变任何一轮跑什么，只修正"拿什么去比"，
  精确实现 `explore.py:441-442` 的 *"slower than baseline" bound* 与
  CLI 承诺的 *"POST-READY (pure hot client) wall-clock"*。
- decision 轮**保留**精度门，`explore.py:1279-1298` 的 fail-closed 语义完全不动。
- 对 `stack_rebench` 轮（`explore.py:1432`，共用同一 `decision_deadline_sec`）自动同样生效。

**代价 / 风险**

- 需要一个可靠的"eval 起止时刻"信号。当前 `_subprocess_kill.py` 的软时钟是从
  server-ready 标记起算的单调墙钟，**并不知道 benchmark 与 eval 的分界**，
  因此需要新增一个阶段边界信号（例如让 `atom_mi300x.sh` 在进入 `run_eval` 前写一个
  标记行 / sentinel 文件，由 `_communicate_with_soft_deadline` 的 server.log 增量扫描识别，
  见 `_subprocess_kill.py:808-818` 现有的 `saw_ready` 扫描机制）。
- 改动面比 B 大，跨 Hyperloom 与 Magpie/InferenceX 脚本两侧。

### 方案 B —— decision 轮关闭 `RUN_EVAL`（治标，改动小）

**做法**：仿 `sweep.py:132-134` 的既有做法，在 decision 轮（及 `stack_rebench` 轮）
强制 `RUN_EVAL=false`，使被测量与锚点在构成上直接对齐。

**优点**

- 改动集中在 `explore.py`，不触碰 Magpie/InferenceX 脚本。
- 仓库内已有先例（SWEEP 正是这么做的），风格一致。
- 顺带缩短每个变体的 decision 轮耗时（本例可省约 452 s/变体）。

**代价 / 风险（必须同步处理，否则会从"误杀"变成"全部卡死"）**

- `explore.py:1279-1298` 是 **fail-closed** 的：拿不到 eval 结果时
  `accuracy_ok = False` → `outcome = "REVERT"`。若只是关掉 `RUN_EVAL` 而不动这里，
  所有变体会因"无精度结论"被判 REVERT——问题形态改变但结果照旧为 0 KEEP。
- 因此必须**同步把精度门移到 KEEP 之后**（例如挪至 stack_rebench 阶段，或 KEEP 后单独一次
  精度确认），并明确"decision 轮不做精度判定"的新语义。
- 期间存在一个窗口：变体在通过精度确认前已被计为 KEEP，需确认
  `optimization_stack` 的回滚路径可覆盖"精度事后不合格"的情形。

### 方案 A + B —— 推荐组合

**A 与 B 不冲突，职责互补：**

- **A 保证判据正确**：即使某一轮仍开着 eval（例如 `stack_rebench` 出于精度确认需要
  必须跑 eval），overtime 判据也不会把它算进去。这是不变式层面的修复。
- **B 减少无谓开销**：decision 轮的目的本就是测吞吐，跑全量 1319 题精度评测是重复劳动
  （warmup 轮或 KEEP 后确认一次即可），关掉它能显著压缩每变体耗时。

**组合后的形态**：decision 轮只跑 benchmark（B），且 overtime 判据在任何轮次都只
统计 bench 段（A）。这样即便未来某轮重新打开 eval，也不会再次踩中同一个坑——
**A 是防回归的护栏，B 是当下的效率优化。**

**若只能做一个**：选 **A**。它是正确性修复，且不需要动精度门的位置，风险面更小；
B 单独实施必须连带改造精度门，改动链更长。

---

**无论采用哪个方案，都应同时补一条测试**：fixture 显式设置两侧 `RUN_EVAL`，并让 mock 的
benchmark 能表达"eval 阶段耗时"，断言 deadline 判据不受 eval 耗时影响。当前测试结构无法覆盖此类
回归（见 §4）。

---

## 7. 相关但独立的问题（不要与本 issue 混为一谈）

同一 session 中另有 **9 次 2400 s 硬帽超时**，其根因**完全不同**，不应一并修改：

- 硬帽锚点是 `baseline_runtime_sec = 625.63 s`，来自 baseline **round-1**，
  而 round-1 是 `RUN_EVAL: 'true'` —— **锚点包含 eval，量纲是对称的**。
- 公式（`explore.py:429-459`）：`max(2400, min(14400, 625.63 × 2.5 = 1564)) = 2400 s`，
  相对含 eval 的完整 baseline 有 **3.8 倍余量**。**硬帽机制本身无 bug。**
- 那 9 次全部（9/9，无反例）含 `--block-size 32`；已定位为 **ATOM 侧的正确性 bug**
  （`aiter_attention.py:137-141` 把物理块大小硬编码回 16，`backends.py:218-219` 算出
  `block_ratio=2` 却从不使用，`backends.py:264-267` 原样拷贝逻辑块号导致注意力读错页）。
  该问题应单独立 issue，归属 ATOM 而非 Hyperloom。

---

## 8. 证据边界（诚实标注）

**已实证：**

- 两轮 materialized `config.yaml` 的 `RUN_EVAL` 取值（`false` / `true`）
- warmup 轮 586 s 内跑完全量 1319 题 gsm8k，精度 strict 0.9492
- 变体纯 bench 122.1 s、锚点 198.38 s、deadline 396.8 s、401.8 s 被杀
- 21 次 overtime kill 的 `runtime`/`deadline` 数值完全一致
- 两个 commit 的改动文件范围与提交信息措辞
- 测试 fixture 不含 `RUN_EVAL`、subprocess 被 mock

**推算（非直接观测）：**

- decision 轮「bench 122 s + gsm8k ~452 s ≈ 574 s」的拆解，是用 warmup 轮实测数
  （586 − 122 − 12）反推的。decision 轮本身在 401.8 s 即被杀、未跑完，故 574 s 是外推值。
  **但结论不依赖该精度**——单 gsm8k 一项 ~452 s 就已超出 396.8 s 预算。

**未验证：**

- 该变体若真进入 KEEP/REVERT 阶梯，能否通过后续 stack_rebench 稳定性门
  （`DEFAULT_STACK_STABLE_PCT = 0.5`）。它有 3 次一致的 +19.4% 测量，通过概率高，但未实跑。
