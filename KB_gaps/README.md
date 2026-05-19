# KB_gaps — v0.8 设计差距实施手册

> 父文档: `../KB_design_gaps.MD` (体检报告 + 总目录)
> 本目录: 把体检报告中每条 gap / 死代码项展开为 *单 PR 级别* 的实施稿,
> 给出根因 / 修复路径 / 验收口径 / 风险.
>
> 本目录 = "拆 PR 用". `KB_design_gaps.MD` = "review 用".

## 阅读次序

```
功能 gap (要新增接线)             死代码 (要清理)
─────────────────────             ──────────────
P0 阻断 (3 条)                    HIGH-misleading (4 类)
  Gap-01  SpecialistRunner          Dead-A  legacy action 链
  Gap-02  KnowledgePlane            Dead-B  scoreboard 残骸
  Gap-03  specialist_done           Dead-C  validate_stack 死路径
                                    Dead-D  KERNEL_OPT_PIPELINE_BODY

P1 主要 (8 条)                    MEDIUM-noise (3 类)
  Gap-04  KERNEL auto profile        Dead-E  cortex_kb_flusher 未拉
  Gap-05  SWEEP auto dispatch        Dead-F  静态 prompt v0.6 vocab
  Gap-06  CLOSE 5 步顺序器           Dead-G  测试反向锁死废止
  Gap-07  T2 per-variant
  Gap-08  T3 per-variant
  Gap-09  gaps[] 字段
  Gap-10  legacy phase allowlist
  Gap-11  Critic verdict_map

P2 次要 (5 条)
  Gap-12  T0 落 CLI
  Gap-13  legacy stub yamls
  Gap-14  explore_search lock
  Gap-15  plateau proxy 双轨
  Gap-16  KnowledgePlane help 文字
```

## 文件命名

- `Gap-NN_<slug>.md` — 功能 gap, 来自 `KB_design_gaps.MD` 第 4–6 章
- `Dead-X_<slug>.md` — 死代码项, 来自 `KB_design_gaps.MD` 第 12 章

## 文档模板 (每份 gap MD 至少含)

1. **问题描述** — 一句话; 引体检报告对应章节锚点
2. **现状代码 trace** — 文件 + 行号; 调用图; "实际跑起来发生了什么"
3. **设计意图** — 引 KB_design §3.x / Mn
4. **根本原因** — 为什么这个 gap 存在 (历史遗留 / 接线漏 / 协议不齐)
5. **修复路径** — PR 拆分 + 每条 PR 具体改动
6. **验收口径** — 客观可测
7. **风险 / 回退** — 修复后可能引入的新风险, 回退路径
8. **关联 gap** — 哪些其他 gap 必须一起修

## 优先级地图

```
Sprint-A (1-2 周): Gap-02 → Gap-01 → Gap-03  (P0 specialist 链)
                   Dead-B                     (scoreboard 残骸顺手清)

Sprint-B (1-2 周): Gap-04, Gap-05, Gap-06   (phase 入口 hook 三连)
                   Gap-07, Gap-08            (per-variant T2/T3)
                   Gap-09                    (gaps[] 字段)
                   Dead-C, Dead-D            (validate_stack + KERNEL_OPT_PIPELINE_BODY)
                   Dead-A                    (legacy action 链关闭, 配合 Gap-10)

Sprint-C (清理):   Gap-10, Gap-11            (Critic 协作大改动)
                   Gap-12, Gap-13, Gap-14    (P2 收尾)
                   Dead-E, Dead-F, Dead-G    (flusher / md / tests)
```

## 关联文档

- 体检报告: `../KB_design_gaps.MD`
- 设计源: `../KB_design.MD`, `../KB_design/3.1`–`3.15`
- 里程碑: `../KB_design/3.13_milestones/M1`–`M7.md`
- 风险: `../KB_design/3.14_risks/README.md`
- v0.6→v0.8 速查: `../KB_design/3.15_v06_v08_cheatsheet/README.md`

## 不写代码的原则 (同 KB_design)

本目录的 MD 只回答 *如果我是另一个工程师, 看完这份文档能不能直接动手*.
具体代码 diff 不写, 留到 PR.
