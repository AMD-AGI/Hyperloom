# P2 — Session 目录 + Seed kit 装配

> 第二阶段的对外可见产物：每一条 dynamic action 派发瞬间，所有
> "orchestration 看到的世界"以 artifact 形式落盘并可独立复检；seed kit
> 由 Coordinator 统一装配，sub-agent 不能改写它的输入边界。
>
> 对应 dynamic_action.MD §3.3。

---

## 1. 目标

把 P1 的 stub dispatch 升级成**完整的派发现场**：所有未来阶段需要
的输入文件齐备、可独立审计；本阶段仍**不**真正运行 sub-agent，stub
executor 的行为暂保留，但 artifact 与 seed kit 必须真实写入。

具体可观测的产物：

1. dispatch 后 `$SESSION_DIR/agents/orchestration/dynamic_actions/<dyn_id>/`
   下出现 `spec.json` + `seed_kit.json` 两个文件，内容可读、可静态
   review。
2. seed kit 装配的 helper 在 Coordinator 侧实现，输入为派发时的
   SharedState 快照 + roofline + profile，输出为闭合字段集的 JSON。
3. seed kit 的体积、字段都满足 cap（DEFAULT token ≤ 8K）。

---

## 2. 触及的架构平面

| 平面 | 改动性质 | 中心思想 |
|---|---|---|
| Session 目录骨架 | 加 1 个目录 | 与既有 `agents/<role>/` 子目录平级新增 |
| Coordinator dispatch 钩子 | P1 dispatch 路径上追加 seed kit 装配步骤 | 装配在 dispatch 瞬间完成，落盘后再发 task |
| Seed kit 装配 helper | 新建 1 个独立 helper module | 专属逻辑独立，便于 review 和测试 |
| dyn_id 生成器 | 新建 | round + seq 可预测命名 |

---

## 3. 目录结构与 dyn_id

### 3.1 目录布局

dynamic action 的所有 artifact 落在：

```
$SESSION_DIR/
└── agents/
    └── orchestration/
        └── dynamic_actions/
            └── <dyn_id>/
                ├── spec.json               # 派发元描述（P2 落盘）
                ├── seed_kit.json           # 输入 snapshot（P2 落盘）
                ├── sub_agent_journal.md    # ReAct 历史（P3 落盘）
                ├── proposal_set.json       # 输出 patch（P3 落盘）
                ├── critic_verdict.json     # Critic 裁决（P5 落盘）
                ├── dispatch_history.jsonl  # 每次 grid run 结果（P5 累计）
                └── telemetry.json          # 累计 gain / outcome（P6 聚合源）
```

**P2 阶段只负责前两个文件**；其余文件由后续阶段写入，但目录骨架在
dispatch 瞬间就预创建（mkdir -p），让后续阶段的写入不需要额外目录创建
逻辑。

### 3.2 路径归属为何是 `agents/orchestration/`

dynamic action 是**由 orchestration 派发**的——artifact 路径归属体现
"谁是派发方"。这与 `agents/critic/` / `agents/orchestration/inbox.jsonl`
的命名口径一致：`agents/<role>/` 表示"由该 role 创建/拥有"。

不放 `agents/dynamic_actions/`（与 specialist 平级）的理由：dynamic
action 不是一个独立 role 的 agent，它是 orchestration 的一种 dispatch
方式；放在 orchestration 名下让"谁是 owner、谁可以审计"一目了然。

不放 `runs/specialist/` 旁边（如 `runs/dynamic/`）的理由：
`runs/specialist/` 是 specialist sub-agent 的 worktree 工作目录，归
sub-agent 自己；dynamic action 的 artifact 是 orchestration 的 dispatch
现场，二者归属语义不同。具体的 sub-agent worktree 仍走 `runs/dynamic/<dyn_id>/`
（详见 P3）——"派发现场"与"执行工作目录"分两个根。

### 3.3 dyn_id 命名（DEFAULT）

格式：`dyn-<round>-<seq>`

- `<round>`：派发时的 EXPLORE round 编号；
- `<seq>`：本 round 内 dynamic action 的序号（从 1 开始）。

例：第 3 个 EXPLORE round 内派发的第 1 个 dynamic action → `dyn-3-1`。

### 3.4 命名为何如此

- **可读性**：目视即可知道这条 dynamic action 在哪个 round；
- **与 specialist task_id 风格保持区分**：specialist 用更长的 task_id；
  dynamic 用 `dyn-` 前缀，目录和日志中一眼可分辨；
- **可预测**：同一 round 内 seq 单调递增，artifact 排序自然按时间序；
- **碰撞约束**：同 round 同 seq 重复（如重启后重派）→ 视为冲突，由
  Coordinator 在创建时拒绝重复 dyn_id（错误信号 = 重启逻辑出问题，
  fail-fast 比静默覆盖好）。

注：由 round-cap = 1 决定，正常情况下每个 round 只会有 `dyn-<round>-1`，
`-2` 极少出现（只在第一条派发被立即拒绝且不计入 cap、随后立即重派的
极端场景出现）。

---

## 4. spec.json 的字段集

`spec.json` 是 orchestration 派发时**对自己派发意图的快照**——它必须
满足"这条派发可以**单独被审计**，不依赖 SharedState 的当前状态"。

### 4.1 字段集（封闭枚举）

- **dyn_id**：本 dynamic action 的 ID。
- **dispatched_at**：派发时间戳（ISO 8601）。
- **round_index**：派发时的 EXPLORE round 编号。
- **payload**：P1 §3 中的完整 payload（含 motivation_gap_text /
  scope_domains / side_effects_declared / budget_hint）。
- **policy_gate_decision**：本次 dispatch 通过的 PolicyGate 校验结果
  快照（哪些 rule 被检查、最终 verdict）。
- **resource_lane**：派发占用的 lane 名（DEFAULT `research_lane`）。

### 4.2 字段集为何封闭

- **任何"自由备注"字段都不允许**——orchestration 想加任何额外信息，
  必须先在 P1 payload 字段集中正式扩展，走设计变更流程；
- **policy_gate_decision 字段为审计而存在**：未来如果某条 dynamic
  action 失败、需要复盘"派发时是否本应被拒"，spec.json 上的 PolicyGate
  快照就是事实依据。

---

## 5. seed_kit.json 的字段集

seed kit 是 sub-agent 的**全部输入**——sub-agent 在 worktree 内能看到
的"orchestration 给我的世界"完全由这个文件决定。

### 5.1 字段集（封闭枚举，DEFAULT）

| 字段 | 含义 | 量级上限 |
|---|---|---|
| `motivation_gap_text` | 复制自 spec.json，方便 sub-agent 直接 prompt 用 | ≤ 1K tokens |
| `roofline_summary` | 当前 baseline 的 roofline 摘要：bound type / 各 op 占比 / 主瓶颈 / 关键 ratio | ≤ 1K tokens |
| `profile_keyslices` | 关键 op 的 timing / call count / memory 摘要 | ≤ 6 条，总 ≤ 1.5K tokens |
| `kept_patches` | 已采纳的 patches 摘要：domain / patch_id / 简要 rationale / 落地后 gain | ≤ 20 条，总 ≤ 1.5K tokens |
| `reverted_patches` | 已 revert 的 patches 摘要：domain / patch_id / revert reason | ≤ 10 条，总 ≤ 1K tokens |
| `kb_pitfalls` | KB 中跟 scope_domains × motivation 相关的 pitfall 命中 | ≤ 10 条，总 ≤ 1.5K tokens |
| `source_root_hints` | 源码路径白名单（受 `INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS` 约束的目录指针） | ≤ 0.5K tokens |

总 token 预算：DEFAULT ≤ 8K（约 6K 输入 + 2K 浮动 buffer）。

### 5.2 装配逻辑的中心思想

seed kit 装配 helper 的设计原则：

- **输入**：派发时的 SharedState 快照（不可变副本）+ 最近一次 roofline
  artifact + 最近一次 profile artifact + payload（需要 scope_domains
  来过滤 KB pitfalls）。
- **输出**：闭合字段集的 dict（直接序列化为 seed_kit.json）。
- **副作用**：仅写文件，不修改任何 state。

#### a. 信息选择是任务调度问题，不是探索问题

哪些 KB pitfalls 算 relevant、哪些 profile slice 是关键的——这是基于
`scope_domains` 的**确定性筛选**，不需要 LLM 介入。把这一层放在
Coordinator 侧让规则透明、可静态 review。

如果放到 sub-agent 侧（让 sub-agent 自己去 KB 检索"我应该看哪些
pitfalls"），sub-agent 就能定义自己的输入边界——seed kit 退化为一个
"hint"而非"承诺"，§1.2 红线中的"输入语料受控"被绕过。

#### b. 装配规则尽量是 deterministic

- KB pitfalls：用 `scope_domains` 与 `motivation_gap_text` 提取的关键
  词做检索（top-K = 10）；分数低于阈值的不入；
- profile_keyslices：选 timing 占比最高的前 N 个 op，与 `scope_domains`
  做交集；
- kept_patches：取**最近 20 条** + **scope_domains 命中**的并集；
- reverted_patches：取**最近 10 条**（最近的更可能避免重复犯错）。

避免引入"基于 LLM 的相关性打分"——任何 LLM 介入意味着可争议、不可
复现。

#### c. 字段缺失的降级行为

- 缺 roofline → `roofline_summary` 留空字符串 + spec.json 标注降级；
  不阻断 dispatch；
- 缺 profile → 同上；
- 缺已采纳/revert patches → 字段为 `[]`；
- KB pitfalls 命中为 0 → 字段为 `[]`；不视为错误；
- 整个 seed kit 是空（即所有可选字段都缺）→ 仍允许 dispatch，但 spec.json
  打 `degraded_dispatch=true` 标志，下游（P3 runner / P4 critic）可见
  此标志并相应调整 prompt 或审查严格度。

#### d. 字段集为何封闭（再次强调）

字段集封闭是 P3 工具白名单收敛的**前提**——sub-agent 的 prompt 在
runner 侧装配时只 reference 这些字段，不会出现"如果 seed kit 里有 X
就用 X"的分支。任何新增 seed kit 字段都必须走设计变更流程。

### 5.3 不应进入 seed kit 的字段（明确排除）

- **完整 SharedState dump**：seed kit ≠ SharedState，sub-agent 看不到
  其他 dynamic action 的 summary（避免互相影响 / 信号污染）；
- **未审查的 LLM 历史 inbox**：sub-agent 不应该读 orchestration 的
  inbox.jsonl（信息隔离）；
- **kernel patch 历史**：scope_domains 不允许全为 kernel（P1 §4.1 组
  C），所以 kernel patch 历史天然不应进入 seed kit；
- **previous dynamic actions 的 journal/proposal_set**：避免 dynamic
  action 之间互相学习——这是 §1.8 "跨 session 学习暂不处理"的同源约束，
  在 v1 也不允许"同 session 内 dynamic action 之间学习"。

---

## 6. Coordinator dispatch 钩子的扩展

P1 的 stub dispatch 流程：
`PolicyGate → 生成 dyn_id → mkdir → stub executor → 写空记录`。

P2 在中间插入 seed kit 装配：

```
PolicyGate
  → 生成 dyn_id
  → mkdir agents/orchestration/dynamic_actions/<dyn_id>/
  → 写 spec.json
  → 调装配 helper 写 seed_kit.json
  → 派发 task（任务 payload 携带 spec_path + seed_kit_path）
  → stub executor（仍为空 proposal_set，本阶段不变）
```

### 中心思想

- **写盘在派发前完成**——若 spec.json / seed_kit.json 任一写盘失败，
  **不**派发 task，回滚 round-cap 计数，artifact 目录清理；
- **task payload 中只携带路径，不携带内容**——避免 task queue 里的
  payload 体积爆炸；下游 runner 通过路径读 artifact；
- **fail-fast**：seed kit 装配过程中如果检测到内部 invariant violation
  （如 token 总量超 cap），直接抛错、撤销 dispatch；不允许"装配失败但
  是用 best-effort 输出"——seed kit 的精度是 §1.2 红线的输入边界。

---

## 7. 与既有 Session 目录骨架的关系

既有 `_SESSION_SKELETON` 列出 `agents/orchestration/` / `agents/critic/`
等基础目录。P2 在这个骨架中**新增** `agents/orchestration/dynamic_actions/`
作为 dynamic action 派发现场的根。

注意：`agents/orchestration/dynamic_actions/<dyn_id>/` 子目录是**按
需创建**（每次 dispatch 时 mkdir），不是 session 启动时预创建。

---

## 8. 依赖与前置条件

P2 必须在 P1 之后实施，且依赖 P1 的：

- PolicyGate 已能拒/通过 dispatch；
- dyn_id 生成器存在；
- artifact 根目录骨架已扩展。

P2 的产物会被 P3 / P4 / P5 / P6 / P8 直接复用：

- P3 runner 启动时读 `seed_kit.json` 装配 sub-agent prompt；
- P4 critic 在审查时可读 `spec.json` 验证 motivation 是否成立；
- P5 端到端连线时把 `proposal_set.json` 和 `critic_verdict.json` 落到
  同一目录；
- P6 SharedState summary 引用 `agents/orchestration/dynamic_actions/<dyn_id>/`
  作为完整 artifact 路径指针；
- P8 重启时扫此目录识别未完成的 dynamic action。

---

## 9. 验收信号

| # | 测试场景 | 期望结果 |
|---|---|---|
| 1 | 完整有效 dispatch | 目录创建；spec.json + seed_kit.json 写入；token 总量在 cap 内；stub executor 仍返回空 proposal_set |
| 2 | seed kit 装配失败（如 token 超 cap） | dispatch 整体回滚；目录清理；round-cap 不递增 |
| 3 | 缺 roofline | 装配通过；`roofline_summary=""`；spec.json 标 `degraded_dispatch=true` |
| 4 | 缺 profile + 缺 KB 命中 | 装配通过；相应字段空；标 degraded |
| 5 | scope_domains 中某 domain 在 KB 中无 pitfall | 该字段为空数组，不报错 |
| 6 | dyn_id 重复（强制构造冲突） | 抛错，dispatch 拒绝 |
| 7 | seed_kit.json 内容静态校验 | 字段集与 §5.1 一致，无未声明字段，无未声明嵌套结构 |

---

## 10. DEFAULT / 待 review

| # | 条目 | DEFAULT | 备注 |
|---|---|---|---|
| 1 | dyn_id 格式 | `dyn-<round>-<seq>` | README §3 #1 |
| 2 | seed kit 总 token 预算 | ≤ 8K | README §3 #7 |
| 3 | 各类条数上限 | KB pitfalls 10 / kept 20 / reverted 10 / profile 6 | README §3 #8 |
| 4 | KB 检索 top-K | 10 | 待 review |
| 5 | 降级 dispatch 是否允许 | 是（标 degraded 但不阻断） | 待 review |
| 6 | dyn_id 冲突时是抛错还是覆盖 | 抛错 | 强烈建议保留 fail-fast |
| 7 | spec.json policy_gate_decision 快照粒度 | rule 名 + verdict | 待 review |

---

## 11. 与 §1.2 红线的对应关系

| 红线 | 在 P2 的落点 |
|---|---|
| 输入语料受控（specialist 是 own-domain；dynamic 是跨 domain + 全 profile + 全 roofline + 已尝试 patches 摘要） | 由 §5.1 字段集封闭 + §5.2 装配规则保证；sub-agent 看到的输入完全由 Coordinator 决定 |
| 不能让 sub-agent 自己定义自己的输入边界 | seed kit 装配在 Coordinator 侧 |
| 不能"prompt 拼好直接喂模型" | seed kit 字段集封闭，禁止自由备注字段 |
| 不能动 kernel-owned actions | scope_domains 全 kernel 在 P1 已被拒；P2 装配时 KB pitfalls 自然不会引入 kernel-only patch 摘要 |
