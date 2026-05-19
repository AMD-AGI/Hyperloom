# Dead-F — 静态 prompt `.md` 仍含 v0.6 vocab

> 风险等级: **HIGH-misleading** (system prompt 是 LLM 角色"宪法")
> 体检报告: `../KB_design_gaps.MD` §12.7

## 1. 问题描述

`orchestrator/system_prompts/*.md` 是 system prompt fragment, 每个角色
每 tick 注入到 LLM 上下文. **比 prompt_builder.py 还危险**, 因为
reviewer 读 Python 代码时容易忽略静态 md 文件.

当前 4 个静态 md 都含 v0.6 vocab 残留:

| 文件 | 主要问题 |
|---|---|
| `orchestration.md` | EXPLORE allowlist 列 legacy actions + 强制 validate_stack + 示例 backends |
| `critic.md` | EXPLORE 列 backends/params/validate_stack |
| `kernel.md` | 头部自称 "v0.6", 引用 DESIGN §7.2 (DESIGN.md 不存在) |
| `robustness.md` | 自称 "v0.7", prune_branch families 含 backends/validate_stack |

后果: LLM 角色"宪法"未升级, 角色按 v0.6 行事.

## 2. 详细位置清单

### F.1 orchestration.md

| 行 | 内容 | 问题 |
|---|---|---|
| 32-42 | EXPLORE 允许集列 `backends, params, validate_stack` | M3 应改为 `explore, specialist` |
| 59, 100-103 | 强制 validate_stack 段 | Dead-C |
| 95-96 | 示例 `delegate{action_name='backends', ...}` | 应改为 `explore` |
| 53-54, 113-114 | 通告 scoreboard retired (✅) | 保留, 这是正确内容 |

### F.2 critic.md

| 行 | 内容 | 问题 |
|---|---|---|
| 15-18 | EXPLORE 列 backends/params/validate_stack | M3 改为 `explore` |
| 17 | "Specialist as propose_action='explore'" (✅) | 保留 |

### F.3 kernel.md

| 行 | 内容 | 问题 |
|---|---|---|
| 1, 5-6 | 头部声明 "v0.6", 引用 "DESIGN §7.2" | DESIGN.md 不存在 (用 KB_design §3.x) |
| 36-42 | "v0.6 invariant unchanged" | 与 v0.8 phase 不变量混淆 |

### F.4 robustness.md

| 行 | 内容 | 问题 |
|---|---|---|
| 1, 5 | 自称 "v0.7", legacy ClaudeBackend 引用 | 版本号漂移 |
| 86 | `prune_branch` families 含 `backends`, `validate_stack` | 应改为 `explore` |

## 3. 设计意图

§3.15 §2.3 速查表 + §3.2 §5 phase allowlist + §3.5 specialist 命名都是
v0.8 单源真相. 静态 md 应当与 prompt_builder.py 一致, 共同体现 v0.8
vocab.

## 4. 根本原因

静态 md 不在 `prompt_builder.py` 测试覆盖里. M2/M3/M5 PR 都改 prompt_builder.py,
但**没人 grep 静态 md** 同步改. CI 也没 lint 静态 md 内容.

## 5. 修复路径

### PR 5.1 — orchestration.md 重写 EXPLORE 段

```markdown
### EXPLORE phase

You may propose:
- `explore` — primary action; the `payload.grid` carries up to K variants
  selected from the latest specialist proposal_set (or default seeds).
- `specialist` — `delegate{action='specialist', params={domain, gap, ...}}`
  to dispatch an LLM sub-agent for a specific domain.
- `recover` — phase-orthogonal escape hatch.

You may NOT propose:
- `backends` / `params` / `validate_stack` — these were merged into
  `explore` in v0.8 M3 (KB_design §3.4). PolicyGate will deny with
  `rule='action_deprecated'`.
- `profile` / `kernel_opt` / `integrate` / ... — these are KERNEL-phase
  actions; PolicyGate R1 phase_incompatible.
```

### PR 5.2 — orchestration.md 删除 mandatory validate_stack

Lines 59, 100-103: 删除. KB_design §3.4 / Dead-C 详述.

### PR 5.3 — orchestration.md 删除 backends 示例

Line 95-96: 改为 `delegate{action_name='specialist', params={domain: 'framework_specialist', gap: '<canonical_id>'}}`.

### PR 5.4 — critic.md 更新 EXPLORE 段

```markdown
EXPLORE phase actions (review scope):
- `explore` — review the grid; emit `verdict_map` per variant
  (KB_design §3.5 §5 / Gap-11)
- `specialist` — review the dispatch (gap match + domain fit)
```

### PR 5.5 — kernel.md 头部 + 不变量段

- 头部 `"v0.6"` → `"v0.8"`
- 删除 `DESIGN §7.2` 引用, 改 `KB_design/3.2 §5.3 (KERNEL phase)`
- `v0.6 invariant unchanged` 段改为 "v0.6→v0.8 unchanged: KERNEL phase
  actions are still kernel-owned (Inv-3 source); profile-before-kernel_opt
  sequence preserved."

### PR 5.6 — robustness.md 头部 + prune_branch families

- 头部 `"v0.7"` → `"v0.8"`
- Line 86 `families` 列表把 `backends`, `validate_stack` 替换为 `explore`

### PR 5.7 — CI lint

加 `tests/test_static_prompts_no_legacy_vocab.py`:

```text
def test_no_legacy_action_names_in_static_prompts():
    forbidden = {"backends", "params", "validate_stack",
                 "Action scores", "scoreboard", "MARATHON_PRIORS"}
    md_files = [
        "orchestration.md", "critic.md", "kernel.md", "robustness.md",
    ]
    base = Path(__file__).parents[1] / "orchestrator" / "system_prompts"
    for md in md_files:
        content = (base / md).read_text()
        for tok in forbidden:
            if tok.lower() in content.lower():
                # Allow historical mentions ("v0.6 ... retired") with
                # explicit context — fail only on usage.
                if "retired" not in content.lower() and "deprecated" not in content.lower():
                    pytest.fail(f"{md} contains legacy vocab {tok!r}")

def test_static_prompt_headers_say_v08():
    md_files = ["orchestration.md", "critic.md", "kernel.md", "robustness.md"]
    base = Path(...)
    for md in md_files:
        header = (base / md).read_text().splitlines()[0:5]
        assert any("v0.8" in line for line in header), \
            f"{md} header doesn't mention v0.8"
```

## 6. 验收口径

- [ ] 4 个 md 文件不含 legacy action 名 (除明确标 "retired")
- [ ] 4 个 md 头部都标 v0.8
- [ ] CI lint test 全绿
- [ ] LLM fresh session 中, system prompt 注入后, `backends` /
      `validate_stack` 词汇仅作"已废"语境出现

## 7. 风险 / 回退

- 纯静态文本改动. 风险: LLM 角色行为变化 (按新 prompt 行事). 灰度建议:
  跑 1-2 个 fresh session 对比 LLM 决策差异.
- **回退**: revert md 改动即退到 v0.6/v0.7 vocab.

## 8. 关联

- `Dead-A` / `Gap-10` (legacy actions allowed) — 同步关闭
- `Dead-C` (validate_stack 死路径) — 删除 mandatory
- `Dead-D` (KERNEL_OPT_PIPELINE_BODY) — 同类: prompt 文本中的 scoreboard
