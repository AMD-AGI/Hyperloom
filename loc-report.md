# Hyperloom `src/` 代码体量报告

> 基线：`src/hyperloom/**/*.py`，commit `66e7c63102b6`，worktree 干净。
> 范围按确认口径限定为 `src/hyperloom`，不含 `scripts/`、`examples/`、`docs/` 与非 Python 资产。
> 全部数字可复算：`loc-census.py`、`loc-redundancy.py`、`loc-scenarios.py`
> （产物 `loc-census.json`、`loc-redundancy.json`、`loc-scenarios.json`）。
> 逻辑行以 `radon raw` 的 LLOC 为准，另给 SLOC 与 AST 语句数作交叉校验。

---

## 1. 当前总体情况

| 口径 | 全树 | 生产代码 | 测试代码 |
| --- | ---: | ---: | ---: |
| 文件数 | 1,104 | 487 | 617 |
| 物理行 physical | 561,407 | 278,845 | 282,562 |
| 源码行 SLOC（去空行/注释/docstring） | 385,997 | 176,423 | 209,574 |
| **逻辑行 LLOC（radon）** | **262,393** | **118,143** | **144,250** |
| AST 语句数 | 218,595 | 97,049 | 121,546 |

三个口径互为交叉验证：我自己按 `tokenize` 实现的 SLOC 为 385,997，`radon` 独立算出 387,359，
相差 1,362 行（0.35%），差异来自两者对「行内注释 + 跨行字符串」归属的判定不同。
AST 语句数（218,595）比 LLOC（262,393）少 43,798，两者本就不该相等：radon 以「逻辑行」为单位，
AST 以「语句节点」为单位，复合语句的子句头部、分号分隔的多语句在两种口径下计数方式不同。
**引用时须指明口径**——同一棵树在这三种定义下分别是 56.1 万、38.6 万、26.2 万行。

三件事值得单独指出：

1. **测试比生产代码更大。** 测试 144,250 逻辑行 vs 生产 118,143，比例 **1.22 : 1**。
   按文件数是 617 : 487。整树一半以上的体量在测试树里。
2. **31.2% 的物理行不是代码。** 561,407 − 385,997 = **175,410 行**是空行、注释与 docstring。
   谈「代码总量」时用物理行会系统性高估三成。
3. **`orchestrator` 几乎没有自己的测试目录。** 它 60,940 生产逻辑行，同目录测试仅 735 行；
   覆盖它的测试实际上住在 `inference_optimizer/tests/`（443 文件 / 112,090 逻辑行的单体测试目录）。

### 1.1 按子系统（生产代码，逻辑行降序）

| 特征 | 文件 | 物理行 | 逻辑行 | 归属 |
| --- | ---: | ---: | ---: | --- |
| `agents/kernel` | 49 | 38,736 | 18,183 | 核心 |
| `orchestrator/actions` | 54 | 43,895 | 16,728 | 核心 |
| `inference_optimizer/breakdown` | 49 | 25,236 | 11,337 | 核心 |
| `orchestrator/phases` | 16 | 21,831 | 10,122 | 核心 |
| `orchestrator/kernel` | 12 | 17,225 | 7,438 | 核心 |
| `orchestrator/loop` | 11 | 15,634 | 6,925 | 核心 |
| `agents/robustness` | 40 | 13,234 | 5,209 | 可选 |
| `inference_optimizer/multi_node` | 26 | 11,560 | 5,011 | 可选 |
| `orchestrator/knowledge` | 22 | 9,634 | 4,589 | 核心 |
| `inference_optimizer/cli` | 13 | 11,894 | 4,451 | 核心 |
| `orchestrator/state` | 12 | 8,294 | 3,993 | 核心 |
| `agents/framework` | 28 | 9,068 | 3,868 | 可选 |
| `common` | 23 | 6,120 | 2,495 | 核心 |
| `agents/critic` | 18 | 5,507 | 2,260 | 可选 |
| 其余 21 个特征 | 114 | 40,977 | 15,534 | 混合 |

---

## 2. 不删功能时最多能保留多少代码

口径：对外行为与功能集合完全不变，只做等价清理。结论分两档，因为「功能」是否包含测试覆盖率
会让答案差一个数量级。

### 2.1 严格等价（行为与覆盖率都不变）：只能删 1.0%

| 可删项 | 规模 | 判据 |
| --- | ---: | --- |
| 逐 token 完全相同的整份文件（去掉冗余副本） | 16 文件 / 807 物理行 | 归一化 token 流 sha256 相同 |
| 上述文件之外，函数体逐 token 相同的冗余副本 | 475 份 / 4,937 物理行 | 函数级 token 流 sha256 相同（≥25 token） |
| **合计（已去重叠计数）** | **5,744 物理行 = 全树 1.02%** | |

**因此严格等价前提下必须保留 555,663 物理行，即 99.0%。**

整份文件重复共 9 组 / 13 份冗余文件，全部在测试树。其中 **7 组是实质重复的测试文件对**
（另 2 组是 2 行与 0 行的 `__init__.py`，无意义）：

| 冗余对 | 每份行数 |
| --- | ---: |
| `test_framework_gap_composer.py` / `..._units.py` | 157 |
| `test_runner_collector.py` / `..._branches_unit.py` | 144 |
| `test_coverage_boost_unit.py` / `test_critic_utils.py` | 139 |
| `test_kb_request.py` / `..._branches_unit.py` | 121 |
| `test_assessment.py` / `..._branches_unit.py` | 103 |
| `test_retry.py` / `..._branches_unit.py` | 85 |
| `test_inbox_parser_branches.py` / `..._branches_unit.py` | 48 |

`method-evaluation-phase3.info.md` §5.3 记的是「三对逐字节完全重复的测试文件」，
**实测为 7 对**，该条断言少了 4 对——正好印证该文档自己 §5.3 的说明：测试树的断言从未过 Verify 门。

生产代码里的复制粘贴几乎不存在：5,744 行里生产侧只占约 126 行（14 组）。这是个正面结果。

### 2.2 死代码：可机械证明的部分接近于零

`vulture --min-confidence 60` 在生产代码上报出 461 项、共 4,938 行。**这个数字不能直接采信**：
按体量排序的前 15 项**全部**是动态派发，静态引用计数必然看不到调用点：

| 被误报的符号 | 实际派发方式 |
| --- | --- |
| 10 个 `breakdown/reporters/_renderers/*.py` 的 `render()` | 各 renderer 模块自注册到中央注册表 |
| `writeback.py` 的 `_promote_baseline` / `_promote_explore` / `_promote_integrate_patch` / `_promote_framework_agent` | `writeback.py:2868` 的字符串派发表 `{"baseline": "_promote_baseline", ...}` |
| `intent_router.py` 的 `_handle_request` | `intent_router.py:67` 的 `IntentType.REQUEST: "_handle_request"` |

逐项复核后剩下的真实死代码不足百行量级。

同样地，模块级可达性分析（从 46 个 `__main__` 入口 + 7 个 console script + 字符串里出现的模块路径出发）
报出 136 个不可达模块 / 34,878 SLOC，但逐个回查引用后**没有一个能判定为死模块**——
kernel agent 的工具全部由 orchestrator 以 `python -m` 子进程方式调用，导入图看不见这条边。
**结论：可达性分析在这个代码库上不能作为删除依据。**

### 2.3 若「功能」只指产品功能、不含测试覆盖率：可再删 9.0%

刷覆盖率而生的测试文件（文件名带 `_units`/`_branches`/`_coverage`/`_boost`）：
**106 文件 / 50,535 物理行 / 27,824 逻辑行**，占测试树逻辑行的 19.3%。
删掉它们不会移除任何产品功能，只降低覆盖率。

**这一档下保留 505,128 物理行，即 90.0%。**

---

## 3. 只保留核心功能需要多少代码

「核心」按 README 自己的产品定义划定：TraceLens 轨迹分析 → Arbor 优化环
（Think → Decide → Implement → Benchmark，含 Dynamic Specialist Agent 与 Knowledge Base）
→ GEAK 内核优化 → session 报告 → CLI。

### 3.1 第一层：整块砍掉可选特征

| 砍掉的特征 | 文件 | 物理行 | 逻辑行 |
| --- | ---: | ---: | ---: |
| `agents/robustness` | 40 | 13,234 | 5,209 |
| `inference_optimizer/multi_node` | 26 | 11,560 | 5,011 |
| `agents/framework` + `orchestrator/framework` | 38 | 13,351 | 5,698 |
| `agents/critic` | 18 | 5,507 | 2,260 |
| `agents/quantization` | 9 | 2,153 | 782 |
| `baseline_comparison` / `tools` / `agentx` / `assets` 等 | 21 | 3,943 | 1,580 |
| **合计** | **152** | **49,748** | **20,540** |

剩余核心：**335 文件 / 229,097 物理行 / 97,603 逻辑行**（占生产逻辑行 82.6%）。

### 3.2 第二层：核心内部的可选表面

保留一类工作负载（推理服务）、一个内核后端（GEAK）、一条内核语言路线、
以及 session 报告真正需要的章节：

| 裁掉的表面 | 文件 | 物理行 | 逻辑行 |
| --- | ---: | ---: | ---: |
| Forge 后端（保留 GEAK） | 3 | 5,556 | 2,825 |
| collective 代码生成 | 4 | 2,281 | 1,122 |
| diffusion 工作负载路径 | 3 | 1,369 | 684 |
| 13 个非核心报告章节（21 个 renderer 保 8） | 13 | 1,551 | 641 |
| FlyDSL 重写路线 | 1 | 633 | 300 |
| **合计** | **24** | **11,390** | **5,572** |

### 3.3 核心所需代码量

| | 文件 | 物理行 | 逻辑行 | 占今日生产代码 |
| --- | ---: | ---: | ---: | ---: |
| 今日生产代码 | 487 | 278,845 | 118,143 | 100% |
| 第一层后 | 335 | 229,097 | 97,603 | 82.6% |
| **第二层后（核心所需）** | **311** | **217,707** | **92,031** | **77.9%** |

**核心功能需要约 92,000 逻辑行生产代码（217,707 物理行）。可砍掉的是 26,112 逻辑行，占 22.1%。**

配套测试：随第一层一起消失的可选 agent 测试为 95 文件 / 24,394 物理行 / 12,338 逻辑行。

### 3.4 代价：21 对特征间导入边需要切断

核心集合对被砍特征仍有 21 对特征级导入边，最重的几条：

| 边 | 导入次数 |
| --- | ---: |
| `orchestrator/actions` → `orchestrator/framework` | 10 |
| `orchestrator/actions` → `inference_optimizer/multi_node` | 9 |
| `orchestrator/phases` → `orchestrator/framework` | 9 |
| `orchestrator/phases` → `agents/framework` | 6 |
| `inference_optimizer/cli` → `inference_optimizer/multi_node` | 4 |

framework agent 的耦合最深（4 个核心特征都依赖它），是这次裁剪中唯一需要真正重构而非直接删除的部分。

---

## 4. 结论

1. **这个代码库不是「大部分是可选功能」。** 生产代码 77.9% 是核心环本身，
   最激进的功能裁剪也只能减掉 22.1%。
2. **不删功能时几乎删不动。** 严格等价的可删量是 1.0%；可机械证明的死代码不足百行量级，
   `vulture` 与可达性分析在这个代码库上都产生大比例假阳性。
3. **体量真正的去处是测试树与非代码行。** 测试逻辑行是生产的 1.22 倍，
   其中 19.3% 是刷覆盖率产生的；物理行里 31.2% 是空行、注释与 docstring。
   任何以「减少代码量」为目标的动作，收益都在这两处，而不在功能裁剪。

---

## 5. 方法与局限

**方法**：`radon raw` 取 LLOC/SLOC；`ast` 取语句数；重复检测用 `tokenize` 归一化 token 流的
sha256（忽略注释、docstring、空白与缩进），文件级与函数级各做一轮，并扣除重叠计数；
死代码用 `vulture --min-confidence 60` 并逐项人工复核；可达性用模块级导入图，
种子为 7 个 console script + 46 个 `__main__` 入口 + 字符串字面量里出现的 `hyperloom.*` 模块路径。

**局限**（按确认的「快速估算」强度）：

- 核心/可选的归属是产品判断，不是测量结果。第 3 节的划分依据是 README 的功能主张，换一种产品定义数字就会变。
- 第二层裁剪（3.2）假定每个可选表面能干净摘除，未逐个验证其调用点；这是估算而非可执行的删除清单。
- 未做等价重构（合并同类实现）的估算，因此第 2 节的 1.0% 是「删除」口径的下界，不是「重写后最小实现」的答案。

**与既有基线的一致性**：本文的文件数与物理行数与 `code-clean-ref/.run-manifest.json`
逐项相同（1,104 / 561,407；生产 487 / 278,845；测试 617 / 282,562），可直接与该审计基线对照。
