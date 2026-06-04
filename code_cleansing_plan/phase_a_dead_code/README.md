# Phase A — 死代码 / 退役代码清除

## 目的

把主计划 §8 退役登记表逐项删干净,同时删除"合并途中产生的废弃物"。
这是**纯减法**相位:几乎每个 commit 都应是大幅净删除。

> 顺序在逻辑合并(Phase B)之前:**先扔垃圾,再搬家**,避免把死代码一起合并进新结构。

## 步骤文件

- [`01_retired_actions.md`](01_retired_actions.md) — 退役 action 名(setup/classify/backends/params/validate_stack/select_kernels)。
- [`02_cortex_kb_remnants.md`](02_cortex_kb_remnants.md) — Cortex/T2-T3 KB / NDJSON flusher 残留。
- [`03_resume_migration_removal.md`](03_resume_migration_removal.md) — 跨版本 resume 迁移读取器(breakdown 兼容除外)。
- [`04_stubs_noops_shims.md`](04_stubs_noops_shims.md) — stub/no-op 执行器、proxy、payload 别名 shim。
- [`05_env_install_cleanup.md`](05_env_install_cleanup.md) — 退役 env / install flag / auth proxy。
- [`conduct.md`](conduct.md) — 本相位行为准则。

## 入口标准

- Phase 0 出口标准全满足(护栏全绿 + 基线已记录)。

## 出口标准

- [ ] §8 登记表每项:已删 或 已标注"保留+原因"。
- [ ] 退役 action 引用计数(Phase 0 第 5 项)归零(仅迁移/拒绝测试保留)。
- [ ] 护栏全绿;CLI flag 金标准 diff 仅显示"已登记的退役 flag"消失。
- [ ] 每个删除项一个独立 commit。

## 通用删除流程(每一项都走一遍)

1. **定位**:`rg` 全仓搜符号/字符串,列出所有 call-site。
2. **判活死**:确认无活引用(测试桩、迁移读取器、拒绝测试不算活引用)。
3. **删除**:删定义 + 所有死引用 + 相关注释 + 相关配置/yaml。
4. **跑护栏**:`pytest <keep-list>`;若有测试专测该退役项,一并删(记在 Phase E 也可,但既然现在删了功能,这里直接删更干净)。
5. **提交**:`Remove retired <X>`,确认 `git diff --stat` 净减。

## 进度记录

| 步骤 | commit(s) | 说明 |
|---|---|---|
| 01 退役 action | `c8b17d80` `01e154d7` | 删 `params_no_promote_streak` plateau proxy + 退役 action 重命名历史注释。 |
| 02 Cortex KB 残留 | `1d6c3fe8` `00359eee` | 删退役 Cortex KB flusher 守护机制、`cortex_kb_constants` 死模块、过期 NDJSON 注释。 |
| 03 resume 迁移读取器 | `1c4e195e` | 删跨版本 resume phase 推断(保留 breakdown 兼容 + scoreboard drop-list 日志)。 |
| 04 stub/no-op/shim | `a1170bd1` | 删未接线 roofline stub executor + 空的 kernel-only 执行器表(保留活别名 apply_patch / params.domain)。 |
| 05 env/install/auth proxy | `5de0be25` `e6a3cd94` | 删未接线 framework-agent `phase-fetch`/`phase-emit-proposal`;删 install.sh `--with-*`/`--all-backends`/`--backend` no-op flag 解析 + 未用 WITH_* 变量;删 robustness `IntentEmitter` 退役 DB writer 路径;清 framework-agent 未建成的 `fa agent`/`agent/` 文档,使 SKILL/README 与实现一致。 |

净效果:本相位 commit 全部为净删除。

### 保留项(标注原因)

- `tracelens_analysis.py::_default_workspace_path` 的 `$WORKSPACE_PATH` 二级
  fallback:现役 backward-compat,4 个专测断言其行为,删除会改外部可观测行为
  (违背总纲领"外部行为不变")。
- robustness `auth_proxy_unhealthy` 复活守护测试:信号本体已退役,该测试只是
  防止其复活,低成本保留。
- scoreboard drop-list 日志 + `--legacy-action-scores` flag:仍是现役日志路径
  且有测试覆盖。

## 出口验证

护栏 keep-list 全绿(17 文件):main 313 · critic 21 · robustness 54 ·
kernel 5 · framework 4(framework 数较 Phase 0 的 8 减少,因 step 05 删除了
`phase-fetch`/`phase-emit-proposal` 专测,属预期)。
