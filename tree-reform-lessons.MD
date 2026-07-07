# tree-reform 经验教训 (Lessons Learned)

> 本文件记录执行 `tree-reform.MD` 期间(P2.1 抽 common、P2.2 拆 god-file)实际踩到的坑与验证有效的做法。目的是让后续步骤(coordinator.py 机制 B、P2.3 迁移、以及剩余 god-file)少走弯路。
> 记法:概念用中文,代码/标识符/路径用英文。

---

## 0. 一句话总结

**这次重构真正的难点不是"把代码搬到新文件",而是"搬完之后测试还能不能绿"。** 而测试能否绿,几乎完全取决于三件事:
1. **monkeypatch 的命名空间**(最大的坑);
2. **import 环**(section 之间不能互相依赖);
3. **相对 import 的层级深度**(文件下沉一层就要 +1 个点)。

只要把这三件事处理对,拆分就是机械且安全的;处理错,就是一堆诡异的 `AttributeError`/`NameError`/`ImportError`/静默行为漂移。

---

## 1. 验收门与工作流(被反复验证有效)

- **基线门 = 非-e2e 全量测试**:`pytest -m "not critic_agent_e2e and not robustness_agent_e2e"`,当前基线 **8479 passed, 3 skipped, 47 deselected,≈4:50**。每拆一个文件都必须回到这个数字(或有据可查的净增减)。
- **先 targeted 后 full**:每次改完先跑与被改文件强相关的一小撮测试(几秒~30s)快速抓错,确认后再跑 ≈5min 的全量门。省下大量等待。
- **每个 god-file 一个可独立回滚的 checkpoint**:被拆的原文件都是 git-tracked,拆错了 `git checkout` 就能还原;所以"删原文件"这一步是安全的。
- **改完必跑 `ruff check`(F401/F811/F821/E402)**:`ruff` 默认不在 venv 里,需 `pip install ruff`。它是跑测试前最便宜的安全网:
  - **F821 undefined name** 能抓出"跨簇引用没接上"(例如 `SharedState._format_variant_line` 在 mixin 文件里找不到 `SharedState`)。
  - **F401 unused import** 用于配合"生成时给足 import、事后 `ruff check --fix` 裁剪"的策略(见 §4)。
- **净增减要能解释**:例如 collectors 拆完后全量从 8478→8479,是因为新增 import-lint 测试(+2)− 删除 allowlist 参数化用例(−1)。对得上就放心。

---

## 2. P2.1(抽 common)的教训

### 2.1 "重复"几乎从来不是逐字节相同
`code_functions_guidance` 里说的 "9 份 atomic_write、3 份 payload_aliases、4+ 份 env" 都**不是**完全一致的副本:
- `atomic_write*`:有的 `mkdir(parents=True)`、有的没有;有的 `fsync`+DEBUG 日志(`recipe_kb/local_store`)、有的 `shutil.copystat` 保留权限(`geak_prompt_patcher`);`_magpie_patcher.atomic_write_text` 返回 `bool`、带关键字参数、且**依赖测试 monkeypatch 它模块内的 `os`/`tempfile`**。
- `env`:token 词表不同(`ci` 的 `_env_truthy` 还接受 `"y"`)、签名不同(`default` 必填 vs `default=0`)、甚至 `roofline_ceiling._env_int` 读的是 **dict 而不是 os.environ**。
- `payload_aliases`:3 份的 `_coerce_str` 行为不同(canonical 会把 list/tuple 空格拼接、子代理副本只 `str()`),且 **3 条不同的 DeprecationWarning 文案**。

**教训**:抽 common 前必须逐个读实现、对齐语义。"balanced 收敛"策略行之有效:**只合并语义完全一致 / 可安全统一的站点,把真正发散的(monkeypatch 敏感、fsync、copystat、remote-shippable、特殊词表)留在原地并打 TODO**。不要为了凑"9 份"的数字而强行改变行为。

### 2.2 re-export 而不是 re-wrap(保住 stacklevel 与 monkeypatch)
- 旧文件转调 common 时,**优先 `from hyperloom.common.x import foo`(re-export),而不是写一层 `def foo(...): return _common.foo(...)`(re-wrap)**。
  - re-wrap 会多一个栈帧,破坏 `warnings.warn(..., stacklevel=3)` 指向的调用方(`test_payload_aliases.py::test_deprecation_stacklevel_points_at_caller` 会失败)。
  - re-export 让旧名字与 common 的对象**是同一个**,`stacklevel` 完全不变。
- 但**被测试 monkeypatch 的函数**(如 `research_hints._atomic_write`)要**保留同名 wrapper**:测试 `monkeypatch.setattr(rh, "_atomic_write", boom)` 替换的是整个函数,只要 `rh` 模块里还有这个名字,调用方(模块内 `_atomic_write(...)`)在调用时按模块全局解析仍命中被替换的版本。

### 2.3 stdlib 模块 monkeypatch 是"全局"的,搬家不影响
`monkeypatch.setattr(mod.os, "which", ...)` / `mod.Path` / `mod.subprocess` / `mod.shutil` 改的是 **stdlib 模块对象本身的属性**(单例),无论函数搬到哪个子模块,只要它 `import os` 用的是同一个 `os` 模块对象,patch 就生效。**这类 patch 对拆分是安全的**。

### 2.4 filterwarnings 与静态 guard
- `pyproject` 里 `filterwarnings = ["ignore::DeprecationWarning:inference_optimizer.compat.payload_aliases"]` 其实对**测试通过与否无影响**(没有全局 `error` filter;断言告警的测试都用局部 `catch_warnings`+`simplefilter("always")` 覆盖了全局)。把 warn 调用挪到 `hyperloom.common` 不会挂测试。
- `test_no_legacy_writer_sites.py` 是**基于 `git ls-files`(仅扫描 git-tracked 文件)**的静态 guard。它有三条约束会被重构触发:
  1. `test_no_legacy_writer_sites_outside_allowlist`:出现新的含 `extra_sglang_args` 的**已跟踪**文件必须进 allowlist。
  2. `test_allowlist_is_minimal`:allowlist 里**不再含该字面量**的行是"死行",必须删。
  3. `test_allowlist_paths_resolve`:allowlist 路径必须存在。
  - **坑**:把 canonical 实现搬到新文件(`src/hyperloom/common/payload_aliases.py`)后,若该文件**未 `git add`**,guard 扫不到它(untracked 被跳过),于是无法把它加进 allowlist(否则算死行)。**解决**:`git add`(仅暂存,非 commit,可回滚)让 guard 能看到它,再进 allowlist。子代理副本改成 re-export 后不再含字面量 → 从 allowlist 删除对应行。

### 2.5 remote-shippable 脚本不要加第一方依赖
`multi_node/scripts/*` 会被发到远端节点独立执行,那里可能没装 `hyperloom`。**这类文件即便有重复代码,也不要让它们 `import hyperloom.common`**,留在原地。

### 2.6 import-lint 守卫
新增 `test_common_import_lint.py`:用 AST 静态解析 `hyperloom/common/*.py` 的 import,断言不 import 任何第一方包(`inference_optimizer`/`orchestrator`/`agents`/`ci`…)。放在 `inference_optimizer/tests/` 下即可被现有 testpaths 收集。

---

## 3. P2.2(拆 god-file)的核心教训:monkeypatch 命名空间碎片化

**这是拆 god-file 的头号杀手,也是评估一个文件"能不能安全拆"的首要指标。**

### 3.1 问题本质
`monkeypatch.setattr(mod, "helper", fake)` + 同模块内 `caller()` 调用 `helper(...)` 之所以生效,是因为它们**共享同一个模块命名空间**。一旦把 `caller` 和 `helper` 拆到不同子模块:
- `helper` 移到 `sub_a.py`,`caller` 移到 `sub_b.py`(`from ._common import helper`);
- 测试 patch `mod.helper`(package `__init__` 的 re-export 绑定)**不会**改变 `sub_b` 里 `caller` 看到的 `helper`(那是 `sub_b` 自己的绑定)。
- 结果:patch 静默失效 → 测试挂,或更糟,真的执行了副作用(真跑 subprocess / 真写盘)。

### 3.2 什么能存活、什么会挂(实测规律)
- ✅ **instance-attr patch**:`monkeypatch.setattr(some_instance, "to_x_summary", ...)` 改的是实例属性,mixin 拆分后方法仍在类上(MRO),实例属性照样 shadow → **安全**(shared_state 的 `to_*_summary` 就是这样)。
- ✅ **stdlib 模块 patch**:见 §2.3,全局生效 → **安全**。
- ✅ **fully-qualified string patch + lazy import**:`monkeypatch.setattr("pkg.mod.trace_analyze_handler", ...)` 且调用方在函数体内 `from pkg.mod import trace_analyze_handler`(惰性 import,调用时读 package `__dict__`)→ re-export 后仍命中 → **安全**。
- ❌ **`setattr(mod, "_helper", ...)` + 调用方在别的子模块**:这是**会挂**的形态。`kernel_request_handlers` 有 ~30+ 处 `setattr(krh, "_run_subprocess", ...)` + ~15 处其它 `krh._helper`,而这些 helper 被散落各处的 handler 调用 → 拆成 package 会大面积失效。

### 3.3 对策
1. **先 grep monkeypatch 面**,再决定拆不拆、怎么拆:
   ```
   monkeypatch\.setattr\(\s*(MOD|ALIAS)\s*,   # 危险:patch 模块级名字
   "[^"]*MOD\.\w+"                              # fully-qualified(通常安全)
   ```
2. **能存活就拆**(collectors 只需改 1 处 `collectors._rel` → `collectors.perfskills._rel`)。
3. **会大面积挂的(krh)先跳过**,或接受"批量把 patch 重定向到属主子模块 + 改调用点"的高 churn 代价(风险高,要逐个 gate)。
4. 若坚持拆一个 const 被 patch 的文件:让所有读该 const 的代码通过**属性访问**(`_common.STATE_FILE`)而非 `from ._common import STATE_FILE`(直接绑定),这样 patch `mod._common.STATE_FILE` 才能对所有子模块生效——但这要改调用点,成本不低。

---

## 4. 拆分技术与自建 codemod 工具(可复用)

工具放在 `/tmp/reform_tools/`(不进仓库),是 §13 "映射驱动 codemod" 的落地:

| 工具 | 作用 |
|---|---|
| `callgraph.py` | AST 解析:模块级符号依赖图(`--imports` 看 import 面、`--methods CLS` 看类方法调用图) |
| `assign.py` | 由"public 根符号 → section"的种子,**反向可达性**自动把 helper 归簇:只被一个 section 可达→归它;被多个/无→归 `_common` |
| `splitter.py` | module → package:按 spec 把符号切进子模块 + 生成 `__init__` 用 `X as X` 惯用法 re-export **全部**原符号 |
| `class_mixin_splitter.py` | 把大类的一组方法抽成 mixin 子模块,原类继承之(MRO) |
| `bump_relimports.py` | 把相对 import 的层级 +N(文件下沉后修点数) |

### 4.1 已验证的两种拆分"形状"
- **形状 A:函数模块 → package + re-export**(collectors)。`mod.py` 变 `mod/`,`__init__` re-export 全命名空间。适用于"一堆相对独立的纯函数 + 清晰的 foundation 层"。
- **形状 B:大类 → mixin(MRO)**(shared_state)。`class SharedState(_RenderMixin, ...)`,方法搬进 mixin 子模块仍 `self.x`/`Class.x` 可达。适用于"单一大类,方法簇之间靠 `self` 互调"。**几乎不用改调用点**,是低 churn 的类拆分法。
- **形状 C:collaborator 对象(机制 B)**——coordinator.py 用,**尚未做**,与 A/B 都不同(见 §6)。

### 4.2 "给足 import + `ruff --fix` 裁剪"策略(关键提效)
无法自动推断每个子模块精确需要哪些 import。做法:**每个子模块先塞进原文件的完整 import 头 + `from ._common import (全部 foundation 符号)`,再 `ruff check --fix` 自动删未用的 F401**。省去手工算 import。
- 但 `ruff --fix` **只删不加**:所以 `_common` 里凡是被跨簇引用的符号必须**全都**列进各子模块的 `from ._common import (...)`,由 ruff 裁剪到实际用到的。

### 4.3 相对 import 必须 +1(文件下沉一层)
`mod.py`(`inference_optimizer/breakdown/collectors.py`)里的 `from ..orchestrator.x` 在 `collectors/timeline.py` 会变成错误的 `..orchestrator`(指向 `breakdown.orchestrator`)。**移到子包后所有相对 import 要 +1 个点。**
- **坑**:`splitter` 生成的 `from ._common import`(同包 sibling,单点,正确)**不能**被 `bump_relimports` 再 +1。**正确顺序**:先对**原文件** `bump_relimports`(+1),再 `splitter` 拆(此时 splitter 新加的 `._common` 保持单点)。

### 4.4 反向可达性归簇的两个坑
- **未被任何根可达的孤儿符号**(如 collectors 的 `_TOKEN_KEYS_ALL`)默认落 `_common`,但它引用的常量却被归到了某个 section → 造成 `_common → section` 的反向环。**修**:把孤儿符号跟它引用的符号放同一簇(手工调 spec)。`ruff` 的 F821 会当场抓到这类 undefined name。
- **dispatcher hub**:`main`/`build_parser`/`KERNEL_REQUEST_HANDLERS` 这类"引用了所有 command/handler"的符号,会让"从单一 `main` 根反向可达"把**所有东西**拉进一个 section。**对策**:hub 符号**不做根**,单独放进一个 import 了各 section 的顶层模块(或 `__init__` body);种子只把叶子 command 当根。

### 4.5 mixin(形状 B)特有的坑
- 类是 `@dataclass`:字段 + `@dataclass` 装饰器留在主类,mixin 只装方法(无字段),MRO 正常。
- **环**:mixin 方法引用模块级 helper/const(`_now_iso`、`_KEY_METRIC_MAP`…)→ mixin `from ..shared_state import (...)`,而 `shared_state` 又要 import mixin。解法:`shared_state.py` **在所有模块级定义之后、类定义之前**才 `from ._shared_state.render import _RenderMixin`;此时 mixin 需要的名字都已绑定,partial-module 环能过。代价:该文件要加 **E402 per-file-ignore**。
- **class-qualified 静态调用**:方法体里 `SharedState._format_variant_line(...)`(而非 `self.`)在 mixin 文件里找不到 `SharedState`。若被调方法也在同一 mixin,改成 `_RenderMixin._format_variant_line(...)`;`ruff` F821 会精确指出这些点。
- 注意 `from __future__ import annotations` 生效时,类型注解是字符串、运行时不求值,所以 mixin 不必为注解里的 `SharedState` 真正 import(避免环)。

---

## 5. 各 god-file 的风险画像(实测)

| 文件 | 行数 | 形状 | 风险 | 状态 |
|---|---:|---|---|---|
| `breakdown/collectors.py` | 7027 | A | 低(纯函数+清晰 foundation;仅 1 处 monkeypatch 需重定向) | ✅ 已拆,门绿 |
| `orchestrator/shared_state.py` | 4171→3281 | B | 低(单大类;monkeypatch 全是 instance-attr / stdlib) | ✅ 抽出 `_RenderMixin`,门绿(后续可再抽簇 + 删 shim) |
| `orchestrator/kernel_request_handlers.py` | 6181 | A? | **高**(~40 处 `setattr(krh, "_helper")` monkeypatch 雷区) | ⏸ 暂缓 |
| `action_executors/_grid_runner.py` | 3220 | A(需 base) | 中高(`variant_filter` 需 `GridVariant` → **import 环**;~15 处 fully-qualified `run_with_session_kill` patch;`GridVariant` 等要下沉到 base 模块打破环) | ⏸ |
| `cli.py` | 5687 | A | 中(entry point `main` hub;被 `_grid_runner`/`multi_node` 反向 import) | ⏸ |
| `multi_node/cli.py` | 3614 | A | 中(`main`/`build_parser` hub;`STATE_FILE` 模块常量被 patch 4 次;coverage-omit 低价值) | 跳过 |
| `orchestrator/coordinator.py` | 15730 | **C(机制 B)** | **最高**(§11);单类 270 方法 | ⏳ 下一步 3b-0 |

---

## 5b. coordinator.py 3b-0 实战(已完成,门绿)与机制 B 的关键修正

**发现(与计划假设冲突)**: 17 个"纯转发 shim"其实是**被 ~40 处测试直接调用 + monkeypatch 的 API**:测试遍地 `coord._handle_intent(...)`/`coord._record_specialist_result(...)`,还 `monkeypatch.setattr(coord, "_handle_delegate", ...)`/`c._handle_intent = AsyncMock()`;而且 `IntentRouter._handle_intent` **故意**经 `self._coord._handle_X` 回环分发,就是为了尊重这些 monkeypatch。所以计划字面的"删 shim + 内部调用改直连 self.router.*"会砸掉 ~40 个测试 + 破坏 monkeypatch 语义——**它不是"最安全的一步"**。

**已采用的安全解法(零测试改动)= `Coordinator.__getattr__` 前向委派**:
- 删掉 17 个**显式 shim 方法**,加一个 `Coordinator.__getattr__(name)`:`name in _ROUTER_DELEGATED → getattr(self.router, name)`;`in _RECORDER_DELEGATED → getattr(self.recorder, name)`;否则 `raise AttributeError`。
- `coord._handle_X`(测试)、`self._handle_X`(coordinator 内部)、`self._coord._handle_X`(router 回环)全部经 `__getattr__` 解析;**instance-attr monkeypatch 天然 shadow**(正常查找先于 `__getattr__`)。router 分发**保持原样**(不改)。
- `router`/`recorder` 是经 `self.__dict__.get("_router")` 的惰性 property → 在 `__getattr__` 里访问它们**不会递归**。
- **递归护栏**:委派集合里的每个名字必须真的存在于其属主(router/recorder 确有全部实现),否则 `A.__getattr__→coord→A.__getattr__` 会无限环。真正缺失的名字要 `raise AttributeError`(否则 hasattr/dunder 探测会炸)。

**这对 3b-1+ 是关键降本**: 机制 B 抽协作者时,不必按计划"显式把 `self.foo()` 改成 `self.<owner>.foo()`"。只要 `coord.__getattr__` 把每个**已迁出**的方法名委派到其属主协作者,则:
- 协作者 A 调 `self.b`(在协作者 B)→ `A.__getattr__ → coord.b → coord.__getattr__ → B.b`,**双层 __getattr__ 自动解析**,零调用点改写;
- 测试 `coord.foo()` / `monkeypatch.setattr(coord, "foo")` 继续可用 → **零测试改动**。
这把原本"数百处改测试"的风险降到"维护 coord 的委派名单 + 每个协作者一个 `__getattr__` 回连"。**代价**:委派名单必须与实际迁移严格同步(否则递归环),且双层 `__getattr__` 有轻微性能/可读性成本。

**执行**: 用 `/tmp/reform_tools/delete_coord_shims.py`(AST 校验每个是纯 `return [await] self.router/recorder._name(...)` 后删除,并在 `recorder` property 后注入 `__getattr__`)。coordinator 15730→15684,门绿,720 个 coordinator 相关测试全过。

## 5c. coordinator.py 3b-1 实战(已完成,门绿 8479)= 机制 B 抽 9 个叶子协作者

15684→11036,抽出 9 个协作者(maintenance/resume/writeback/gating/dispatcher/proposals/advisory/inline_actions/conversation)。用 `/tmp/reform_tools/extract_collaborator.py`(IntentRouter 范式:`_coord` + `__getattr__`)+ `coord.__getattr__`/metaclass 委派。踩到并解决了 4 个机制 B 特有的坑:

1. **`self.<attr> = X` 状态回绑到协作者(最隐蔽)**: 方法体里的**赋值/增量赋值** `self.x = ...` / `self.x += ...`,搬进协作者后 `self` 是协作者 → 写到协作者实例而非 coord(读经 `__getattr__` 到 coord,写却留在协作者 → 读写不一致)。IntentRouter 靠"handlers 从不 rebind self.attr"回避;通用抽取必须**把写目标改成 `self._coord.<attr>`**。工具 `/tmp/reform_tools/fix_self_writes.py`(AST:只改 `Assign/AugAssign/AnnAssign/Delete` 里 `value is Name 'self'` 且 attr!='_coord' 的目标;不碰 `self.x.y=` 和读)。本次改了 9 处。

2. **类级访问 `Coordinator._method` 绕过实例 `__getattr__`**: 测试大量 `Coordinator._X`(拷方法到 stub、unbound 调用)。实例 `__getattr__` 不管类访问 → 加**metaclass `__getattr__`**,把委派名字解析到协作者**类**的函数(懒 importlib 避免环)。这对 3b-2 更关键(framework/enablement/sweep 有 ~15+ 处类级访问)。

3. **同簇 monkeypatch 被兄弟方法旁路**: `monkeypatch.setattr(coord, "_a")` + 同协作者内 `_b` 调 `self._a` → `_a` 是协作者自有方法(正常查找命中,不经 `__getattr__`)→ patch 被旁路。修:测试改 patch **协作者实例** `setattr(coord.<prop>, "_a")`(对同簇/跨簇/coord 级调用都生效,因 coord.__getattr__ 路由到协作者实例)。工具 `/tmp/reform_tools/retarget_collab_patches.py`(router/recorder 除外,它们用显式 `self._coord.X` 分发)。

4. **`ruff --fix` 删掉"故意的惰性重复 import"(高危)**: 巨类里方法体内 `from .x import y`(与顶层 import 重复)常是**故意保留**的,好让测试 patch 源模块 `y` 时按调用时读到 patched 值。对 coordinator 跑 `ruff --fix` 会以 F811/F401 删掉它 → 测试 patch 旁路(本次 `is_valid_measurement` 中招,stack-validation 判成 REVERT)。**教训:巨类拆完不要无脑 `ruff --fix` 整文件**;要么只 fix 新文件,要么删前 diff `^\s+from \.` 惰性 import 并对存活方法里被 patch 的逐个加回(带 `# noqa: F811`)。

**跨簇调用零改写(验证成立)**: 有了 coord.__getattr__ 全量委派,协作者 A 调 `self.b`(属 B)自动经 `A.__getattr__→coord.b→coord.__getattr__→B.b` 解析,**无需**按计划显式 `self.<owner>.b`。前提:委派名单与实际迁移严格同步。

**模块级名字**: 协作者经 `from .coordinator import PendingProposal/_format_inbox_event/常量`(顶层)即可——因为协作者是被 **lazy property** 导入的,首次访问时 coordinator 已完整加载,无环(区别于 IntentRouter 被 coordinator 顶层 import,只能方法内 lazy)。用 ruff F821 找缺名再注入。

## 6. 给 coordinator.py(机制 B)与 P2.3 的前瞻建议

- coordinator 用**协作者对象**(§6.1 机制 B),不是 mixin:façade 持 `self.conversation`/`self.resume`/… + 相位注册表,各协作者用 `__getattr__` 反连 coordinator。**与本次 A/B 都不同**,要新建/复用协作者范式(仓库已有 `IntentRouter`/`ResultRecorder` 作范例)。
- **3b-0 最安全先做**:先用 `callgraph.py --methods Coordinator` 建 270 方法调用图,核对并删掉 17 个纯转发 shim,把内部调用直连 `self.router.*`/`self.recorder.*`。这一步不搬大块逻辑,风险最低。
- **跨簇调用点**是机制 B 的主要 churn:协作者互调对方私有方法时 `self.foo()` 要显式改 `self.<owner>.foo()`(否则经 `__getattr__` 落回 coordinator 而 miss)。**先建调用图定位跨簇边**再动手。
- coordinator 已是 E402 豁免文件,拆分时同步维护该清单。
- **P2.3(迁 `src/hyperloom/orchestrator/`)必须在 god-file 拆完后做**,且是全库 import 重写:用 LibCST/映射驱动 codemod,别用纯正则;动态 import(backends 的 `module="..."`、subprocess 桥)要手改并登记。`ci/orchestrator.py` 与顶层 `orchestrator` 撞名,必须**同批**改名。

---

## 7. 给"下一个拆分者"的 checklist

1. `callgraph.py` 出符号/方法图;`--imports` 看 import 面。
2. **Grep monkeypatch 面**:`setattr(MOD, "..."`(危险)、`"MOD.xxx"`(通常安全)、`MOD.const =`。据此判断形状与可行性。
3. Grep 外部 importer(注意:本机 `rg` 在 wekafs 上会**挂**,用 IDE 的 Grep 工具,别用裸 `rg`)。
4. 设计簇:sections 只依赖 `_common`,**杜绝 section↔section 环**;hub 符号单独放顶层。
5. **先 `bump_relimports` 原文件(+1),再 splitter 拆**;或 mixin 用 §4.5 的 E402 顺序法。
6. `ruff check --fix` 裁剪 import;再 `ruff check` 看 F821/F811/E402。
7. import 冒烟(`python -c "import ...; 断言全命名空间可达"`)。
8. targeted 测试 → 修 monkeypatch 重定向 → 全量非-e2e 门。
9. `git add` 新文件(让 git-tracked 类 guard 能看到)。
10. 更新受影响的 per-file-ignores(E402)、静态 guard allowlist、`test_no_legacy_writer_sites` 等。

---

## 8. coordinator.py 3b-2 实战(已完成,门绿 8489)= 相位 handler(`PhaseHandler` 基类 + 9 个相位 collaborator)

**范式**: 新增 `orchestrator/_phase_base.py` 的 `PhaseHandler` 基类(`__init__(self, coordinator)` 存 `self._coord`,`__getattr__` 回连 `self._coord`),与 3b-1 的叶子协作者、`IntentRouter`/`ResultRecorder` 完全同范式。9 个相位 collaborator(`_phase_machine.py MachinePhase` / `_phase_prelude.py PreludePhase` / `_phase_sweep.py SweepPhase` / `_phase_close.py ClosePhase` / `_phase_internal.py InternalTasksPhase` / `_phase_kernel_stack.py KernelStackPhase` / `_phase_kernel.py KernelPhase` / `_phase_explore.py ExplorePhase` / `_phase_framework.py FrameworkPhase`,均继承 `PhaseHandler`)按 §6.1 的方法映射表原样抽取;coordinator 的 `_DELEGATED` 字典 + `_COLLAB_MODULES`(类级委派,供 metaclass `_CoordinatorMeta.__getattr__` 用)与 3b-1 完全一致地扩展,累计 256 个委派方法名、20 个协作者属性(9 相位 + `router`/`recorder` + 9 个 3b-1 叶子)。**注册表顺序**(用户要求)按调用链自然顺序排列:`machine→prelude→sweep→close→internal→kernel_stack→kernel→explore→framework`(`framework` 因方法数最多[48 个]放最后)。coordinator.py 从 11061→2240 行(不含空行折叠前后对比;`__init__`/`tick`/`run`/`_reactor_pass` 等 façade 核心 + 全部小型 `_XXX_*` 常量表保留在 façade)。

**执行侧发现的坑(均已修复,与 3b-1 §5c 的坑同构,再次印证"协作者拆分"是可重复的机制)**:

1. **本机 import 顺序导致的 E402**:抽取工具生成的每个 `_phase_*.py` 把 `import logging as _logging` / `log = _logging.getLogger(__name__)` / `from ._phase_base import PhaseHandler` 放在其它 import **之后**(是从原 coordinator.py 相应位置摘抄的顺序,那里这些语句本就在文件尾部附近),导致 `ruff` 报 9 处 `E402`。**修法**:统一挪到文件顶部 import 块内(`import logging as _logging` 与其它 stdlib import 一起排序;`from ._phase_base import PhaseHandler` 跟在其余本地 import 之后;`log = _logging.getLogger(__name__)` 移到全部 import 结束后单独一行)。零逻辑变化,`ruff check` 转绿。

2. **facade 瘦身后残留 13 个真正未使用的顶层 import**(`hashlib`/`json`/`subprocess`/`uuid`/`datetime`/`timezone`/`KIND_GEMM_TUNING`/`OUTCOME_KEEP`/`JournalEntry`/`Intent`/`IntentType`/`SubAgentResult`/`Task`):9 个相位簇搬空后,façade 里最后一个使用点也跟着搬走,`ruff F401` 精确点出。**逐个用 `Grep` 确认 0 处真实用例后删除**(不是无脑 `ruff --fix`,因为要核实没有被其它测试/模块依赖 `from .coordinator import <sym>` 的隐式 re-export)。

3. **`SPECIALIST_FROM_AGENT_PREFIX` 是"看似未用但被测试当 re-export 依赖"的假阳性**:`ruff F401` 判定其在 coordinator.py 顶层未用(用它的 `_task_id_from_specialist_source`/`_dispatcher` 早已搬到 `_phase_explore.py`/`_dispatcher.py`,各自直接 `from .policy import SPECIALIST_FROM_AGENT_PREFIX`),但 `test_coordinator_sync_helpers_coverage_unit.py` 有 `from inference_optimizer.orchestrator.coordinator import SPECIALIST_FROM_AGENT_PREFIX` 这一行**依赖 coordinator 模块转导出该常量**。**教训**:façade 瘦身后删导入前,必须额外用 `Grep` 扫一遍 `from .*coordinator import` 的**测试侧**引用面(不能只看 facade 自身用例),命中的一律保留为 `# noqa: F401 - re-exported for callers/tests`(coordinator.py 里已有的既定写法)。

4. **同簇 monkeypatch 被旁路(与 3b-1 §5c 点 3 同构,规模更大)**:9 个相位簇共 256 个委派方法,测试里 `monkeypatch.setattr(coord, "_method", fake)` 一旦 `_method` 与调用方同属一个相位簇,patch 会被协作者自身的真实方法(正常属性查找先于 `__getattr__`)旁路。**用 `retarget_collab_patches.py`(§4 已有工具,按 `_DELEGATED` 反查 method→property)批量把 `setattr(VAR, "meth"` 与 `VAR.meth = ` 两种写法都重定向到 `VAR.<phase_prop>.meth`**,共改了 9 个文件、约 90 处。
   - **假阳性识别**:批量正则会命中**看起来像 coordinator 实例、实则是裸 stub/fake 的变量**(如 `_fake_self()`/`_Stub`/`_MemCoord` 返回的 `types.SimpleNamespace` 或裸类,直接 `Coordinator._method(fake, ...)` 非受限调用、不走 `_collaborator`/`__getattr__`)。这类 stub **没有** `.phase_xxx` 属性,加了会在其它测试路径里制造一个从未被读取的死属性(不报错但语义上是噪音,且掩盖了"这个 stub 其实没有走真实委派路径"这一事实)。**验证方法**:对每个被批量脚本改动的文件,`git diff` 逐一确认变量来自返回真实 `Coordinator`/`Coordinator.__new__(Coordinator)` 的 fixture(`coord`/`c`),而非局部定义的 stub 工厂函数;命中 stub 的立即 `git checkout` 回退该文件,单独确认后再处理(本次 3 个文件 `test_enablement_coordinator_wiring_unit.py`/`test_framework_agent_authoring.py`/`test_framework_agent_ranker_memory.py` 是纯 stub,全部保持原样、未改动,测试本就直接绑定 `Coordinator.<method>` 到 stub 上,不受相位拆分影响)。

**验收**: `ruff check`(9 个新文件 + coordinator.py)全绿;`python -c "import ..."` 全部 10 个 `_phase_*` 模块 + coordinator 冒烟通过;`Coordinator._DELEGATED` 的 256 个方法名逐一 `hasattr` 校验命中其属主协作者类;`Coordinator` 的 20 个 `@property` 与 `_COLLAB_MODULES` 的 20 条映射一一对应;9 个 targeted 测试文件 + 全量非-e2e 门 `8489 passed, 3 skipped, 47 deselected`(较 3b-1 完成时的 8479 净增 10,均来自本次会话之前已完成但未提交的其它 P2.2 步骤的测试增量,与 3b-2 本身无关)。

**给 P2.3(迁 `src/hyperloom/orchestrator/`)的补充前瞻**:相位 handler 落地后,coordinator façade 只剩 `__init__`/`router`/`recorder`/`__getattr__`/`_collaborator`/20 个 `@property`/`stop`/`tick`/`run`/`_reactor_pass` 等核心方法 + 全部常量表,已非常接近 §6.1 设想的"瘦 façade + 装配"形态。迁移到 `orchestrator/loop/coordinator.py` + `orchestrator/phases/*.py` 时,9 个 `_phase_*.py` 文件到 `phases/{machine,prelude,sweep,close,internal,kernel_stack,kernel,explore,framework}.py` 是纯改名(内容和类名不用动),`_DELEGATED`/`_COLLAB_MODULES` 里的模块路径字符串同步改写即可,风险与 3b-1 的 9 个叶子协作者迁移一致(低)。

---

## 9. coordinator.py 3b-3 实战(已完成,门绿 8498)= 3b-2 收尾:补漏 + 注册表顺序修正

3b-2 提交(`e570641e`)后用一个独立只读子代理对照 tree-reform.MD §6.1 的方法映射表逐条复核,发现 3b-2 **并未 100% 完成**,有两处需要补的尾巴:

1. **漏迁一个方法**:`_cached_kernel_request`(§6.1 `kernel.py(~28)` 簇里列出的方法)在 3b-2 时被遗漏,仍作为完整方法体留在 `coordinator.py`(既不在 `_phase_kernel.py`,也未登记进 `_DELEGATED`)。它被 `intent_router.py` 以 `self._cached_kernel_request(...)` 调用,两个测试(`test_lifecycle_wiring.py`)用 `monkeypatch.setattr(c, "_cached_kernel_request", ...)` 直接 patch **真实 Coordinator 实例**(instance-attr patch,§3.2 的"安全"形态)。**修法**:原样搬到 `_phase_kernel.py`(`KernelPhase` 类尾部,复用该文件已有的 `Path`/`Any` import),在 `_DELEGATED` 里补一行 `"_cached_kernel_request": "phase_kernel"`。因为是 instance-attr patch 而非同簇 monkeypatch 旁路(§3b-1 点 3 的坑),**两个测试无需改动**,搬完直接绿。搬完 `phase_kernel` 簇计数 27→28,与设计文档 "kernel.py(~28)" 的估计精确吻合(证实此前一直少 1 个)。

2. **注册表顺序与设计意图相反**:3b-2 提交时 `_COLLAB_MODULES` 字典字面量与 `@property` 声明顺序实际是 `framework→explore→kernel→kernel_stack→internal→close→sweep→prelude→machine`(**framework 在最前**),与用户要求、`tree-reform-lessons.MD` §8 原文、以及该 commit message 自己写的"machine→prelude→sweep→close→internal→kernel_stack→kernel→explore→framework(framework 因方法数最多放最后)"**完全相反**——这是文档正确但代码顺序做反了的纯疏漏(3b-2 抽取工具按"抽取顺序"而非"要求顺序"追加条目,事后未手工重排)。**修法**:把 `_COLLAB_MODULES` 的 9 个 `phase_*` 键、9 个 `@property` 方法定义、以及 `_DELEGATED` 里 9 个簇各自的方法块,全部按 `machine→prelude→sweep→close→internal→kernel_stack→kernel→explore→framework` 重排(`router`/`recorder`/9 个 3b-1 叶子协作者的顺序不变,仍跟在 9 个相位后面)。**这纯粹是 dict/声明顺序调整,不改变任何方法归属或行为**,回归门无净变化(8498 passed 不变)。

**教训**:批量抽取工具(`extract_collaborator.py` 之类)如果按"处理簇的顺序"逐条 `dict[key] = value` 追加,产出的注册表顺序是**工具遍历顺序**而不是**设计要求的语义顺序**;若设计文档对顺序有明确要求(如本例"最大的放最后"),必须在**每次抽取批次完成后**额外跑一次"顺序审计"(读 dict/property 声明的字面量顺序,和设计文档比对),不能假设"抽取脚本产出即最终形态"。同时,"每个簇的方法数是否与设计文档的估计值(如 `~28`)吻合"是检测漏迁的一个廉价信号——`27` vs 估计的 `~28` 这种"差 1"的小偏差,在数量级对得上时很容易被忽略为"约数误差",但用一个独立只读审计子代理对照映射表逐条 `grep`/`hasattr` 复核,是唯一可靠的完备性验证手段(比单纯看"测试全绿"更严格,因为像 `_cached_kernel_request` 这种遗漏方法本身没有被拆分破坏、只是"没搬家",测试完全绿,不会暴露问题)。

**验收**: `ruff check` coordinator.py + `_phase_kernel.py` 全绿;`Coordinator._COLLAB_MODULES`/`_DELEGATED` 的 9 个 phase 键顺序程序化验证与要求顺序完全一致;`phase_kernel` 簇 28 个方法(含新补的 `_cached_kernel_request`)hasattr 全部命中 `KernelPhase`;`phase_framework` 确认为 9 个簇中方法数最多(48)且位于注册表/属性声明的最后;targeted 测试(`test_lifecycle_wiring.py` 等 1246 个 coordinator/kernel/intent_router/lifecycle 相关测试)+ 全量非-e2e 门 `8498 passed, 3 skipped, 47 deselected`(与 3b-2 完成时的数字完全一致,证明这是纯内部重排 + 补漏,无行为变化)。

---

## 10. P2.3 实战(已完成,门绿 8498)= orchestrator 提升为顶级包 + 内部按 14 个领域子包重组

P2.3 分两个独立阶段执行,每阶段各自过一次全量非-e2e 门,分两个 commit 提交(`83858ea7` 物理提升 / 后续 commit 内部重组)。

### 10.1 阶段 A:`inference_optimizer/orchestrator/` → `src/hyperloom/orchestrator/`(纯物理提升,内部结构不变)

**执行前侦察(关键,避免"边搬边猜")**:先派一个只读子代理对全库做 7 类系统性扫描——(1)全库对 `inference_optimizer.orchestrator` 的绝对 import(265 文件/1098 行);(2)`inference_optimizer/` 内部反向 import 进 orchestrator 的相对 import(17 文件,因搬走后同包相对 import 必须变绝对);(3)orchestrator 内部反向 import 出 `inference_optimizer` 兄弟包的相对 import(~58 文件/138 行,同理必须变绝对);(4)动态 import(`importlib.import_module(f"...")`、`__import__(...)`)——揪出 `coordinator.py` 的 `_CoordinatorMeta.__getattr__` 里 `importlib.import_module(f"inference_optimizer.orchestrator.{mod}")` 这个 f-string 动态 import,codemod 的静态 AST 改写**完全看不到**它,必须人工改一行;(5)测试里 `monkeypatch.setattr("inference_optimizer.orchestrator.X.Y", ...)` 字符串目标(~330 处/30+ 文件)+ `caplog.at_level(..., logger=...)` 的 logger 名字符串(因为 `logging.getLogger(__name__)` 的 `__name__` 就是搬移后的新模块路径,这类字符串必须跟着改,否则测试断言的 logger 名和实际发出日志的 logger 名对不上,静默失败);(6)`pyproject.toml` 里以字符串路径为 key 的配置(`coverage.omit`/`ruff.per-file-ignores`/`package-data`);(7)`docs/`、各 sibling agent 的 `SKILL.md`/`README.md` 里的散落 prose 引用。**这套"7 类扫描"是可复用的通用清单**,任何"把一个包搬到另一个命名空间"的操作都应该先跑一遍。

**LibCST codemod 的核心正确性坑(比 P2.2 单文件内部拆分更复杂,因为这次是跨包搬迁)**:
1. **相对 import 的解析基准点是"文件的旧位置",不是"新位置"**:orchestrator 内部文件(如 `coordinator.py`)的 `from ..paths import db_path_for` 这种相对 import,是相对它**搬家前**的包路径(`inference_optimizer.orchestrator`)写的,搬到 `hyperloom.orchestrator` 后,必须先按**旧**包路径把 `..` 解析成语义目标(`inference_optimizer.paths`),再决定"目标搬没搬、要不要保持相对"。如果直接用文件的**新**路径去解析相对 import 的 dots,会算出完全错误的目标模块。
2. **三条互斥重写规则**,缺一个都会漏改:R1(全库任何绝对 import 前缀替换 `inference_optimizer.orchestrator`→`hyperloom.orchestrator`)、R2(inference_optimizer 内部反向相对 import 进 orchestrator → 必须变绝对,因为 orchestrator 已经不是 inference_optimizer 的子包,不能再用相对 dots 够到它)、R3(orchestrator 内部对外相对 import → 必须变绝对 `inference_optimizer.X`,同理)。
3. **`from . import X`(module=None 的相对 import)是隐藏坑**:AST 里 `ImportFrom.module` 为 `None` 时,`names[0]` 本身就是一个子模块引用(而非"从某模块导入的符号名"),必须单独判断并按"子模块级联"处理,否则会被当成普通符号导入完全放过不改写,产生 `ModuleNotFoundError`。

**验收**: 全部 136 个 orchestrator 子模块 `import_module` 冒烟通过;`ruff check` 全绿(14 个残留错误逐一核实为迁移前就存在、与本次改动无关的历史 lint 债);全量非-e2e 门 `8498 passed, 3 skipped, 47 deselected`(与迁移前完全一致)。

### 10.2 阶段 B:orchestrator 内部按 14 个领域子包重组(`loop/`、`phases/`、`policy/`、`actions/`(含 `executors/`)、`roles/`、`state/`、`bus/`、`knowledge/`、`specialists/`、`kernel/`、`framework/`、`scoring/`、`prompts/`、`trace/`)

136 个文件按 tree-reform.MD §5.2 逐一分配到 14 个子包(纯文件级 move-map,无内容拆分),`git mv` 保留历史,再复用阶段 A 验证过的 LibCST codemod(在其基础上扩展:目标模块解析改成同时支持"旧位置解析 dots、按 MODMAP 查新位置、决定相对/绝对"的双向映射),批量重写内部相对 import + 全库绝对 import。

**本阶段踩到的坑,全部是"P2.2/阶段A 已知坑的更大规模复现",再次印证"协作者/文件拆分的坑是可复用的机制类问题"**:

1. **批量创建"新子包 `__init__.py` 占位文件"时,用无条件 `cat >` 覆盖了刚 `git mv` 过去、已经带真实内容的 `__init__.py`**(`backends/__init__.py` 13 行 re-export + `__all__`、`system_prompts/__init__.py` 的 `read_rules_fragment` 函数,两者都被我"顺手加个文档字符串占位"的脚本用一句 heredoc 全覆盖丢失)。**教训**:批量建目录骨架前,必须先跑 move-map 检查"这个目标路径是不是本次移动的落点";若是,只能追加/合并,绝不能无条件覆盖写入。本次靠"import 冒烟测试直接炸出 `ImportError: cannot import name 'read_rules_fragment'`"当场发现,证明**冒烟测试比只看 diff 更可靠**(diff 会显示这是"新增文件"而看不出丢了内容,因为丢失的旧内容本就在 git mv 后的"删除"那一侧)。

2. **`_CoordinatorMeta.__getattr__` 的动态 import + `_COLLAB_MODULES` 字典**(阶段 A 已改过一次前缀)在阶段 B 又要再改一次——字典的模块名从裸文件名(`"_phase_machine"`)变成带子包路径(`"phases.machine"`)。**这类"编码在字符串里的模块路径"每次搬迁都要跟着手动同步,是 codemod 天生的盲区,必须建立一个专门的"人工修复清单"并在每个搬迁阶段后逐项复核**(本次清单:`_COLLAB_MODULES` 字典值、`intent_router.py` 的 `__import__` 字符串、`asset_system_prompts_dir()` 的路径拼接、SKILL.md/README 的 prose 路径)。

3. **`from PKG import SUBMODULE` 形式(module 非 None,但 names[0] 是一个"逃逸"到别的子包的子模块)比 `from . import X` 更隐蔽**:例如 `from hyperloom.orchestrator import phase_state`(`module="hyperloom.orchestrator"`,`names[0]="phase_state"`)在阶段 B 时 `phase_state` 整体重命名+搬家为 `phases.machine_state`,原有的"absolute import 前缀替换"规则只检查 `module` 整体是否要换,不会去看"`module.name` 拼起来是否命中一个搬迁条目"。**必须额外加一条规则**:对每个 `from X import Y`(`level==0`),除了检查 `X` 本身要不要换,还要检查 `f"{X}.{Y}"` 是否整体命中 move-map,命中则连 `Y`(import 的名字)一起换,并**当名字本身改变时自动补一个 `as <原名>` 别名**,否则文件里所有引用原名字的地方(如 `phase_state.PHASE_PRELUDE`)全部找不到定义(`NameError`)。这条规则在 `from . import X` 场景(阶段 A 已修)和 `from PKG import X` 场景(阶段 B 新增)都要覆盖,是同一类坑的两种语法外壳。

4. **正则/文本级"字符串字面量批量替换"极易产生"重复后缀"的幂等性 bug**:用一个不区分"代码 vs 字符串"的全文正则替换 `monkeypatch.setattr("旧模块路径...")` 这类字符串目标时,若该文件里**同时存在**一处已经被 AST-codemod 正确改写过的真实 import 语句(如 `from hyperloom.orchestrator.policy import gate as policy_module`),正则会**误命中**这行代码文本里的旧前缀子串(`hyperloom.orchestrator.policy`)并在其后拼接错误后缀,产出 `from hyperloom.orchestrator.policy.gate import gate as ...`(基于一个根本不存在的符号)。**根因**:全文正则不知道"这段文本是已经正确的代码,不是需要修的字符串字面量"。**正确做法**:①优先用 `tokenize` 只在 `STRING`/`COMMENT` token 范围内做替换(不触碰其余源码字节),对 `.py` 文件强制这样做;②即使做全文替换,也必须加"若匹配位置紧跟着一个已经是**新**路径前缀,则跳过"的幂等性保护(比较 match 起点是否也匹配 NEW-VALUES 前缀,谁更长优先谁);③**改完必须再跑一次"每条 `ImportFrom`/`Import` 的 module 是否是磁盘上真实存在的模块"的 AST 校验**(遍历所有 `.py`,收集 orchestrator 下所有真实模块的 dotted path 集合,反查每个 import 语句的 `module` 是否在集合里),这一校验本次以**零成本**(纯离线 AST 遍历,不需要真的 `import`)在测试全跑之前就抓出了 7 个文件的破坏。

5. **验收顺序**:导入冒烟(147 模块全绿)→ `ruff check`(隔离历史债)→ **`pytest --collect-only`**(比直接跑全量测试快 30 倍,专门用来抓"import 层面的错误",本阶段连续 5 轮 `--collect-only` 迭代收敛,每轮只需 5–10 秒)→ 全量非-e2e 门。**`--collect-only` 是"批量 import 重写"类任务的最高性价比中间校验点**,应该在每次 codemod 应用后立即跑一次,而不是等全量测试跑完 5 分钟才发现还有 collection error。

**验收**: 147 个 orchestrator 子模块导入冒烟全绿;AST 级"每条 import 的目标模块必须磁盘上真实存在"校验零命中;`ruff check` 保持与阶段 A 相同的 14 个历史遗留错误(0 新增);`pytest --collect-only` 从 46 个 collection error 收敛到 0;全量非-e2e 门 `8498 passed, 3 skipped, 47 deselected`(与阶段 A 完成时完全一致,证明两阶段合计的 P2.3 是纯文件布局迁移,无行为变化)。`manifest.json`/`state.json`/`storage/coordinator.db`/`session_breakdown.json`/四个 role 的 subprocess JSON 桥等磁盘/进程观测契约,通过对应的既有测试套件(`test_refactor_golden_breakdown.py`、`*_e2e_mock.py`、`*runtime_bridge*`、`*critic_agent_backend*`、`*robustness_agent*` 等)间接验证不变。

**给后续步骤(P2.4 inference_optimizer 瘦身 / P2.5 agents 归一)的前瞻**:P2.3 验证了"侦察 7 类扫描 + LibCST 双向 move-map codemod + tokenize 级字符串修复 + `--collect-only` 快速迭代"这一整套方法论在**跨包物理搬迁**场景下是可靠、可重复的,已经处理了"包提升"(阶段 A)和"包内重组"(阶段 B)两种子问题,P2.4/P2.5 的物理搬迁在方法论上是同一件事的重复应用(新增风险点是 P2.5 的 5 种异构打包约定统一,需要额外处理 `pyproject.toml`/`setup.py` 的声明式打包差异,而非本次遇到的纯 import 重写问题)。

---

## 11. P2.4 实战(五个提交,门绿)= inference_optimizer 瘦身 + 收尾补课

P2.4 分五个提交完成:`d7dd3443`(阶段 A,物理提升到 `hyperloom.inference_optimizer`)、`ac764d01`(`compat/` 扁平化 + `kb/`→`data/`)、`6a0f01c2`(合并 `paths/session_paths/session_lock/manifest`→`session/`)、`80cb38b5`(拆 `launcher/`+`scripts/`→`tools/experiments/assets/references`)、`06dfc5ae`(拆 `cli.py`→`cli/` 子包)。前四个提交各自跑了与基线数字完全一致的全量非-e2e 门;第五个提交(`cli/` 拆分)因为跑到一半时发现有并发 agent 在改 `critic-agent/`,全仓库跑会把它的中间状态也测进去、产生噪音,改成只跑了 `inference_optimizer/tests/` 子集(6649 passed)——**这是一个明确的遗漏**:缺少像前四个 commit 那样"和基线数字完全一致"的强验证,也没有跑 `ruff check --fix` 前后对比、没有跑 `mypy`、没有跑 `pre-commit run --all-files`,且对 `cli/__init__.py` 的测试失败是"哪里报错修哪里"地处理的,没有先系统性建一遍 monkeypatch 清单再动手(该做但没做到 §3.3 的方法论)。本节补上这些遗漏,并记录补课过程中额外发现的两个真实回归 bug。

### 11.1 `cli/` 拆分事后 monkeypatch 面重新扫描(§3.3 方法论的实际应用)

对已经拆好的 `cli/__init__.py`(4748 行,`_preflight`/`_run_optimize`/`_build_parser` 三个巨型函数仍在里面)做了一次完整的 §3.3 四模式扫描:

```
setattr\(\s*(cli|cli_module|...)\s*,\s*"NAME"          # 危险:模块级裸名 patch
setattr\(\s*(cli|...)\.\w+\s*,\s*"NAME"                # 安全:stdlib 子模块 patch(cli.shutil / cli.subprocess)
(cli|...)\.\w+\s*=\s*...                               # 直接属性赋值(cli_mod.ClaudeBackend = ...)
"hyperloom\.inference_optimizer\.cli\.[...]"           # 全限定字符串 patch
```

**结果**:只有 4 个符号被**危险模式**直接命中——`_load_dotenv_fallback`、`_load_kernel_agent_env_fallback`、`_clone_inferencex`、`_probe_llm_catalog`,全部集中在 `test_preflight_auth_override.py`。另有 `cli.shutil.which` / `cli.subprocess.run`(stdlib 子模块 patch,§2.3 安全)、`cli_mod.ClaudeBackend = ...`(已经在 P2.2 拆分时正确指向 `cli.executors` 而非 facade,安全)、以及一处 `importlib.reload(cli)` 期望 `DEFAULT_ROBUSTNESS_BACKEND` 常量随 env 重算。这次事后扫描印证了"拆分前必须先建 monkeypatch 清单"的价值:如果先扫再拆,4 个危险符号 + reload 依赖点会在设计阶段就被识别,而不是像这次靠事后补课才发现。

**新增一类坑(`reload(pkg)` 的语义边界,§3.2/§2.3 都没覆盖到)**:用最小复现验证了 `importlib.reload(pkg)` **只重算 `pkg` 自己模块体里的顶层语句**,不会重新执行任何被 `pkg` re-export 的 sibling 子模块的模块体。即:

```python
# pkg/__init__.py: from .sub import HELPER; VALUE = os.environ.get(...)
importlib.reload(pkg)   # VALUE 重算;pkg.HELPER 还是 sub.py 里的旧绑定,不变
```

**含义**:`DEFAULT_CRITIC_BACKEND`/`DEFAULT_ROBUSTNESS_BACKEND`(env 快照式模块级常量)+ 它们的 `_resolve_*_choice` 解析函数,**必须留在 `cli/__init__.py` 本体**,不能像其余 helper 一样搬到 sibling 模块再 re-export——否则 `test_resolve_robustness_choice_env_override_still_works` 的 `importlib.reload(cli)` 断言会静默失效(reload 后常量值不变,测试会真实失败,不是隐蔽 bug,但值得记录为"哪些符号不能参与后续的 `_run_optimize` 拆分"的负面清单)。

### 11.2 `_build_parser` 簇的安全整体搬出(零 monkeypatch,验证性极简)

`_build_parser`(1400 行)+ 6 个纯计算 helper(`_RetiredFlag`/`_default_gpu_specialist_capacity`/`_default_research_lane_capacity`/`_parse_conc_env_default`/`_parse_conc_sweep_default`/`_parse_conc_values`/`_positive_int_arg`)经 §3.3 扫描确认**零个**被 `setattr(cli, "...")` 或全限定字符串命中,是本次唯一可以"和 coordinator.py 的叶子协作者一样低风险"的整体搬移:直接剪切到新文件 `cli/parser.py`,`__init__.py` 用 `from .parser import X as X` 全部 re-export。**零调用点改写**,因为所有引用方式(`from hyperloom.inference_optimizer.cli import _build_parser`、`cli._build_parser()`、`optimizer_cli._build_parser()`)都经过包命名空间,re-export 后行为完全不变。

**踩到的坑与 P2.2 §4.2 完全同构**:剪切后原地留下的函数间空洞用 `python3 -c` 脚本按行号切片删除(而不是逐段 `StrReplace`)——`_default_research_lane_capacity`/`_default_gpu_specialist_capacity`/`_build_parser` 三者在文件里紧邻,`StrReplace` 需要传入上千行文本做 exact-match,极易因为一次已生效的编辑导致行号偏移而"找不到旧字符串";改用"读整个文件为行数组、按 1-indexed 行号区间 slice 删除、写回"的脚本化方式,一次成功,且行号可以从 `callgraph.py` 的输出直接读取,不需要人工数。

**另一坑(本次补做时才发现)**:`_positive_int_arg`/`_parse_conc_values`/`_parse_conc_env_default`/`_parse_conc_sweep_default` 除了在 `_build_parser` 内部被引用外,**还有一份独立的定义簇**紧邻 `_resolve_run_max_model_len` 之前(供 `_run_optimize`/`_export_workload_envs_for_optimize` 使用),与 `_build_parser` 簇里同名的定义是**两处完全重复的定义**(P2.2 拆分时遗留的重复,本就存在,不是本次引入)。只删 `_build_parser` 大块会漏删这处"卫星重复",`ruff F811`(redefinition)会精确指出;必须两处都删干净,只留 `parser.py` 里的单一实现。

### 11.3 `_preflight` 簇的协作者式搬出(4 个危险符号,机制同 coordinator.py 的跨协作者调用)

`_preflight`(259 行)+ 其上游 17 个纯 helper(`_load_dotenv_fallback`/`_load_kernel_agent_env_fallback`/`_clone_inferencex`/`_check_*`/`_emit_preflight_diagnostics`/`_run_ir3_preflight` 等)搬到 `cli/preflight.py`。核心难点是 §11.1 扫出的 4 个危险符号里,有 3 个(`_load_dotenv_fallback`、`_load_kernel_agent_env_fallback`、`_clone_inferencex`)被 `_preflight()` 自己以**裸名调用**在函数体内引用。

**关键验证(比 coordinator.py 的场景更极端的一个变种)**:coordinator.py §3b-1 点 3 的教训是"同簇兄弟方法调用绕过 `__getattr__`",这里更进一步——**即使被调函数和调用者搬进了同一个新文件**,裸名调用仍然会绕过 monkeypatch。用最小复现验证:

```python
# pkg/sub2.py 定义 clone_it() 和 preflight()(同一个文件!),preflight() 内部裸名调用 clone_it()
# pkg/__init__.py: from .sub2 import clone_it, preflight
pkg.clone_it = lambda: "patched"
pkg.preflight()   # 结果是 "original"，不是 "patched"！
```

**原因**:Python 函数体内的裸名引用在**定义时所在模块的全局命名空间**里解析(`sub2.py` 自己的 `__dict__`),与"该名字有没有被外部重新绑定到 package 的 `__init__.py`"无关。`monkeypatch.setattr(cli, "_clone_inferencex", fake)` 只改了 `cli`(package)`__dict__` 里的绑定,`sub2.py`(`preflight.py`)自己的 `__dict__` 里的 `_clone_inferencex` 名字岿然不动。

**修法(与 coordinator.py §5c 点跨协作者调用同构)**:把这 3 处裸名调用改成函数体内**惰性、包限定的重新查找**:

```python
from . import _load_dotenv_fallback as _load_dotenv_fallback_current
from . import _load_kernel_agent_env_fallback as _load_kernel_agent_env_fallback_current
_load_dotenv_fallback_current()
_load_kernel_agent_env_fallback_current()
```

这样每次调用都重新读一次 `cli`(package)当前的属性值,天然拾取任何时刻打上的 monkeypatch,且对未打 patch 的正常路径无行为变化(`from . import X` 惰性导入在包已加载完成后就是纯字典查找,无额外开销)。`_probe_llm_catalog`(第 4 个危险符号)恰好没有被 `_preflight`/`_validate_and_resolve_claude_model` 之外的函数在同一个新文件里裸名引用到,且它本身没有随 `_preflight` 一起搬(仍留在 `__init__.py`,因为 `_validate_and_resolve_claude_model`/`_smoke_test_codex_model`——这两个函数属于 `_run_optimize` 簇而非 `_preflight` 簇——才是它的调用方),故未受影响。

**顺带修复的两个真实回归 bug(§3.3 扫描的副产品,不是本次目标但顺手修了)**:扫描 monkeypatch 面的同时人工通读了 `_preflight` 全文,发现两处 `Path(__file__)`-相对路径在 P2.4 `cli.py`→`cli/__init__.py` 的物理下沉(多一层目录)后从未跟着调整,与 P2.3 §10.1 点 1"相对 import 的解析基准点是文件的旧位置"是同一类坑的 `Path(__file__)` 变体:
1. `preflight_kb.sh` 路径:`Path(__file__).resolve().parent / "assets" / "preflight_kb.sh"` 在 `cli.py` 是单文件时指向 `inference_optimizer/assets/`(正确),变成 `cli/__init__.py` 后指向不存在的 `cli/assets/`。改成 `.parent.parent`。
2. `WORKSPACE_PATH` 默认值(critic-agent 静态资产根):`Path(__file__).resolve().parents[3]` 在旧 `cli.py`(`inference_optimizer/cli.py`)是 `parents[3]` = 仓库根,变成 `cli/__init__.py` 后同样的下标少算了一层,实际落在 `src/hyperloom/`。改成 `parents[4]`。

两处都**没有被任何现有测试覆盖**(第一个的测试用 `monkeypatch.setenv("INFERENCEX_PATH", ...)` + `monkeypatch.setattr(cli, "_clone_inferencex", ...)` 短路掉了真实路径解析;第二个完全没测试),是典型的"改了但测试测不到、只有真实运行才会暴露"的静默 bug——多亏这次为了搞清楚 monkeypatch 面而通读全文,不然会一直潜伏到下次有人真的用 `--critic-agent` 跑一次全新会话。**教训**:任何 `git mv` 或"文件从扁平模块变成 package(多一层目录)"的操作,除了检查 import(P2.3 已有方法论),还必须**额外 grep 一遍 `Path(__file__)` 及其 `.parent`/`.parents[N]` 的使用**,因为这类路径推导对目录深度极其敏感,而且通常没有 lint 规则能抓到(它在语法上完全合法,只是运行时算出错误路径)。同一逻辑应推广到 P2.5(agents 归一)+ 任何未来的物理下沉操作。

### 11.4 `_run_optimize` 未拆分(有意为之的范围收窄)

`_run_optimize`(984 行)引用了 ~20 个仅存在于 `__init__.py` 本体的名字(`_preflight`、`_validate_and_resolve_claude_model`、`_smoke_test_codex_model`、`_resolve_critic_choice`、`_build_orchestration_prompt`、`DEFAULT_CRITIC_BACKEND` 等),且其中 `_validate_and_resolve_claude_model`/`_smoke_test_codex_model` 内部调用的 `_probe_llm_catalog` 本身就是危险符号之一。把 `_run_optimize` 真正拆出去需要:(a) 对 `DEFAULT_CRITIC_BACKEND`/`DEFAULT_ROBUSTNESS_BACKEND` 做 §11.1 发现的"不能搬"处理(拆分后 `_resolve_critic_choice`/`_resolve_robustness_choice` 若跟着 `_run_optimize` 一起搬到新文件,`importlib.reload(cli)` 语义会碎裂,除非把这两个常量单独留在 `__init__.py` 并让新文件里的解析函数改成 §11.3 同款惰性查找);(b) 对 `_probe_llm_catalog` 做和 `_load_dotenv_fallback` 一样的惰性化改造;(c) 对另外 ~15 个纯 helper 逐个判断跨文件调用点。这与 coordinator.py 3b-1/3b-2 的"跨协作者调用点改写是主要 churn"完全同构,量级也相当(coordinator.py 单个协作者批次约改 9-90 处调用点)。**本次选择不做**:一是范围已经超出用户本次要求的"系统性扫描 + 尽力拆分",二是 `_run_optimize` 本身**不被任何测试直接调用或 monkeypatch**(全仓库搜索确认),所以拆或不拆对测试面无风险差异,唯一的收益是文件行数,不值得在没有配套设计的情况下临时决定切分边界。留给下一个专门批次,且应该先补一份 `_run_optimize` 内部调用图(`callgraph.py --methods` 目前只支持类方法,需要扩展成支持模块级函数体内的裸名调用分析)。

**验收**:`cli/__init__.py` 4748→2292 行,新增 `cli/parser.py`(1571 行)、`cli/preflight.py`(1046 行);`ruff check` 对三个文件全绿(新增 1 条 E402 per-file-ignore,原因与 `multi_node/cli.py` 同构:re-export 语句排在模块文档字符串+日志初始化之后);`mypy` 对 `cli/` 的报错数从改动前的 7 条(`__init__.py`)+ 0(新文件尚不存在)→ 现在 9 条,逐条比对与改动前**完全同构**(同错误信息,行号因代码物理搬家而漂移,`preflight.py:906` 与旧 `__init__.py:1806` 是同一行代码);`inference_optimizer/tests/` 全量 6752 passed, 2 skipped(与改动前一致);全仓库非-e2e 门排除本次改动前后均因**并发 kernel-agent→`hyperloom.agents.kernel` 迁移**导致的 3 个 collection error + 71 个 test 失败(逐一核实：均是 `kernel-agent/tools/*` 路径引用,与 `cli/` 拆分无关,`git stash` 前后行为完全一致)后,8453 passed / 71 failed(pre-existing)。

### 11.5 事后教训:批处理任务的产物必须逐文件验证是否被并发操作误还原

本节的 3 项工作(§11.1–§11.4 对应的 `cli/__init__.py` 编辑、`test_inferencex_preflight_clone.py` 编辑、本 §11 本身)在首次完成并汇报后,被发现 `cli/__init__.py`、`test_inferencex_preflight_clone.py`、本 lessons 文件都**回到了编辑前的状态**,而同一批次新建的 `cli/parser.py`/`cli/preflight.py` 两个文件却完好无损。**根因推断**:仓库里有并发 session 在做 P2.5(agents 归一)系列提交,过程中很可能对 `cli/__init__.py` 做了一次"以旧版本为基准的 `git checkout`/覆盖式改动"(该文件同时是 P2.5 critic-agent 改名提交touch 到的文件),而**新建的、未被任何其他提交触碰过的文件**天然不受影响。**教训**:①在多 agent 并发改同一仓库的场景下,"文件已保存到磁盘"不等于"改动会保持",批处理任务完成后如果间隔较长时间才 commit,应在 commit 前**重新 diff 一遍**每个被改动文件相对于"改动前 baseline"的差异,而不是想当然认为磁盘状态=自己最后一次编辑的状态;②**新建文件比编辑已有文件更抗并发覆盖**(除非另一个 session 恰好也在动同名新文件),这是"能拆成新文件就不要在原文件里改"的额外论据(不仅是代码整洁,也是并发安全);③本次修复方式:不需要重新做设计决策,直接重放同一套编辑序列即可(编辑内容本身没有争议,只是被误还原),但每一步都要重新跑 `ruff check` 而不是假设"和上次跑的结果一样"——因为不能排除并发提交同时改了同一批文件的其它部分(如本次 pyproject.toml 的 E402 忽略项被 P2.5 提交继承下来、内容碰巧一致,但不能想当然)。

**根因更正(事后确认)**:实际根因并非另一个并发 session 的 `git checkout`,而是协调多个 P2.5 子任务的父 session 在**跨 commit 验证**(用 `git checkout <old-commit> -- .` 逐一核对某几个文件是否受迁移影响,再切回 HEAD)时,误用了 `git reset --hard HEAD` 来"还原"检查现场——`reset --hard` 会连带丢弃**当时所有已跟踪文件的未 commit 修改**(不只是被检查的那几个文件),而未跟踪的新文件(`cli/parser.py`、`cli/preflight.py`)因为从未被 git 跟踪过而不受 `reset --hard` 影响,这正好解释了"新建文件完好、已跟踪文件的编辑被回滚"这个观察到的不对称现象。**通用教训(比"并发覆盖"结论更根本)**:**任何验证/核对性质的 git 操作,只要涉及切换 commit 或重置工作区,必须先 `git status --short` 确认没有未提交的重要修改,且优先用不破坏工作区的方式做对比**(`git show <ref>:<path>` 读某个历史版本的文件内容到 stdout,或 `git diff <ref> -- <path>`,而不是 `checkout`/`reset` 真的切换整个工作区状态);如果确实需要临时切换,应先 `git stash`(可撤销)而不是依赖"改完再切回来"的手工纪律——`reset --hard` 没有撤销机制,一旦对着有未提交修改的工作区跑,后悔无门。

## 12. P2.5 完整收尾(六个子步骤全部完成,门绿)= agents 归一

P2.5 分六个子步骤执行,每个子步骤对应 tree-reform.MD §5.4–§5.6 描述的一个 sibling agent 的迁移(critic 因命名冲突风险最高而排在最前,§10 也明确要求),第七个子步骤(OOB)因仓库里不存在 `OOB/` 目录而跳过:

1. **critic-agent → `hyperloom.agents.critic`**(commit `04e464ae`):最高风险项,因为顶层模块名是裸 `runtime`,会在 `pythonpath` 上全局遮蔽其它包的同名模块。迁移后 `CriticAgentBackend` 直接 `python -m hyperloom.agents.critic.runtime.cli`,删除了此前因"floating checkout"才需要的 `sys.path.insert` + 陈旧模块驱逐 hack。§5.4 详细记录。
2. **robustness-agent + framework-agent → `hyperloom.agents.{robustness,framework}`**(commit `048ea470`):两者均已是良好 src-layout(参照模板),近乎平移。framework-agent 的迁移顺带删除了 `orchestrator/framework/client.py` 里"IO-only 测试环境"的 `_FRAMEWORK_TO_REPO_URL` 逐字节 fallback 拷贝(因为 framework-agent 现在总是随 orchestrator 一起装在同一个 distribution 里,不再需要 `except ImportError` 分支)。两者的 `[project.scripts]` 入口点(`robustness-agent`、`framework-agent`/`fa`)合并进根 `pyproject.toml`。
3. **quantization_agent → `hyperloom.agents.quantization`**(commit `6823e364` + `e8feabf9` + `9989dc36`,分三次提交因为第一次漏提交了配置文件改动):全程零内部 import 改写(该包本来就用相对导入),纯粹是目录物理搬迁 + 两处外部调用点(`quantization_request_handlers.py` 的懒加载 import、测试的 monkeypatch 目标)的路径更新。
4. **kernel-agent → `hyperloom.agents.kernel`**(commit `841634d2` + `83cd5871`):规模最大的子步骤(67 个源文件 + 40 个测试文件),核心工作是把混在 `tools/` 里的 33 个 `test_*.py` 移到独立 `tests/` 目录,同时逐个修正每个测试文件因为"假设脚本与被测模块同目录"而写的 `Path(__file__).resolve().parent` 路径计算(搬完不再指向 `tools/`,必须显式改成 `.parent.parent / "tools"`);`tools/*.py`(含 `backends/` 的 Ray-worker sys.path hack)本身保持不动,因为它们既要能被当包内模块调用、也要能被当独立 CLI 脚本 `python3 xxx.py` 直接跑。3 个已知 god-file(`tracelens_analysis.py`/`kernel_optimization.py`/`apply_kernel_patch.py`)只做整体搬迁,内部拆分是明确排除在本次范围外的独立任务。这一步之前把 26 个此前"游离"在 `testpaths` 之外的测试文件首次纳入默认收集范围,是本次迁移相对旧状态的一个真实功能性改进(不只是搬家)。

**验收(六个子步骤累计,最终状态)**:`pytest --collect-only` 0 错误、9458 个测试;全量非-e2e 门 `9381 passed, 4 skipped, 47 deselected, 12 subtests passed`(外加 13 个 kernel-agent 预存失败 + 15 个预存 error,均通过 `git stash` A/B 对比确认是迁移前就存在的环境相关问题——例如 `ray` 模块未安装、`TraceLens-internal` 检出缺失——与本次搬迁无关,不阻塞 P2.5 完成)。相比 P2.5 开始前的基线(`8497 passed, 3 skipped, 47 deselected`),净增约 900 个测试收集,主要来自:critic/robustness/framework 三个包的 `tests/` 目录首次被 `testpaths` 收录(此前它们在各自独立的 `pytest.ini`/`pyproject.toml` 下跑,根目录跑全量测试不会碰到)、以及 kernel-agent 前述 26 个此前游离的测试文件首次被收集。

**给 P2.6(ci/ 拆库改名)及以后步骤的前瞻**:P2.5 六个子步骤共同验证了"critic-agent 打了个样、后四个子步骤依样画瓢"的可复制性——每一步都是"git mv 物理搬迁 → grep 全仓库找 import/字符串/monkeypatch 引用点 → 改 pyproject.toml 的 testpaths/coverage/package-data/entry_points → 改 CI workflow → `--collect-only` 快速迭代 → 全量非-e2e 门 → 分批 commit"的同一套流程,唯一因包而异的是"该包原来的打包约定有多不规范"(critic 最不规范→最高风险,kernel-agent 因为 sys.path hack + 测试混放次之,quantization 因为纯相对导入几乎零风险)。P2.6 的 `ci/` 拆库(§5.7)会引入一个新维度——它不是简单的"整体搬迁",而是要把库代码和真 CI 脚本拆开、还要处理与顶层 `orchestrator` 的同名冲突(`ci/orchestrator.py` 必须改名),预期比 P2.5 的任何一步都更接近 P2.2/P2.3 的"内容级拆分"复杂度而非纯"目录搬迁"复杂度。

## 13. P2.7 实战(四个提交,门绿)= 测试统一 + 契约收敛(删过渡期 re-export 与死代码)

P2.7 的执行分两条并行轨道同时推进(本次由两个独立 agent session 交错完成,commit 历史里两边的改动互相补全、无冲突):一条是"删 P2.1 过渡期 shim + §2.5 死代码清单",另一条是"收敛 pyproject.toml 的 testpaths/coverage/package-data 契约"。两条轨道各自独立可验证(每删一个 shim / 每改一条 pyproject 路径,立即 `--collect-only` + 针对性单测回归),汇合时用一次全量非-e2e 门兜底confirm 无冲突。

**shim 清理(P2.1 抽 common 时特意保留的"旧文件转调新实现"过渡层,窗口在 P2.7 关闭)**:

1. `orchestrator/_json_io.py` / `_time.py` / `gain_math.py` 三个纯 re-export shim(分别转发 `hyperloom.common.jsonio` / `timeutil` / `gain_math`)直接删除,6+14+3 个调用点改为直连 `hyperloom.common.*`。这三个是"零风险"档:全仓库 grep 确认零 monkeypatch 依赖,纯机械 import 路径替换。
2. `agents/robustness/_payload_aliases.py`(P2.1 时代的重复 re-export 副本)删除,唯一生产调用点(`signals/repeated_payload.py`)和专用 shim 测试改直连 `hyperloom.common.payload_aliases`。
3. `inference_optimizer/compat.py`(P2.4 把 `compat/` 包扁平化后的产物,本身也是纯 re-export)删除,8 个调用点(6 生产 + 2 测试)改直连。**关键验证点**:`common.payload_aliases.read_extra_server_args` 的 `stacklevel=3` deprecation warning 依赖"调用链只有两帧"——因为 P2.4 时选的是 re-export 而非 re-wrap(§2.2 记录的决策),删 shim 后调用链依然是两帧,`test_deprecation_stacklevel_points_at_caller` 照常通过,无需改动 stacklevel 数值。
4. `agents/kernel/tools/_payload_aliases.py` **有意不删**:kernel-agent 的 `tools/` 目录设计为可脱离 `hyperloom` 包独立在远端节点运行(`sys.path.insert` 自定位,Ray worker 不继承 driver 的 sys.path),`kernel_optimization.py` 用裸模块名 `from _payload_aliases import` 而非包限定路径引用它。删除这个 shim 会破坏该约束,留作已知的、有意为之的例外。

**§2.5 死代码清单核查结果(逐项确认现状后再决定是否删)**:

- `specialists/bench.py`(42 行纯 stub,`__all__` 为空)——全仓库 grep 确认零调用方(tree-reform.MD 原计划的"内联进 runner.py"那一步实际上从未发生,文件就这么孤零零地留在那)。直接删除,连带它唯一的专用测试 `test_specialist_bench_unit.py`(该测试只测 stub 自己的两个 envelope 函数,不是任何生产路径的回归覆盖)。
- `specialists/domains.py` 的 `SPECIALIST_DOMAINS_M5`——逐字节比对确认与 `SPECIALIST_DOMAIN_KEYS` 完全相等,是历史误导性别名。2 个生产引用(`runner.py`)+ 测试里的断言全部替换为 canonical 名字后删除定义。
- `breakdown/legacy_collectors.py`(v0.6 pre-`phase_history` session 的独立 collector)——**docstring 声称**"callers detect legacy session via `is_legacy_session()` 并 prefer 这里的输出",但实测 `exporter.py` 里唯一的 `collect_phase_segments` 调用点走的是主线 `collectors` 模块,从未调用过 `legacy_collectors` 里的同名函数——即该模块的"接线"步骤实际上从未完成,是彻底的死代码。删除,连带其专用测试 `test_legacy_collectors_unit.py`。这一条的教训是:**模块 docstring 里描述的"调用方如何使用我"不能替代实际 grep 调用点**——必须验证 docstring 断言与代码现实一致,再决定是否是"死代码"。
- `compat/`(单模块 deprecation shim 却包成子包)、`robustness-agent` 的 `conductor.py`/`monitors/`/`providers/`、`orchestrator/framework/client.py` 的 fallback dict、`coordinator.py` 的 14 个 `_handle_*` 空转发 shim——**均已在更早的阶段(P2.4/P2.5/3b-0)处理完毕**,§2.5 清单里这几条到 P2.7 时已经是"确认已处理,无需动作"而非"待删除"。
- `_PHASE4_LEGACY_KEY_RENAMES`(`shared_state.py` 里 `extra_sglang_args → extra_server_args` 的 schema 迁移表)——**判定为不可删**:只要磁盘上可能还存在带旧字段名的历史 `state.json`(用于 resume),这个迁移表就是活的兼容层,与 `extra_sglang_args` 整体窗口关闭是同一个问题,不属于 P2.7 范围。

**pyproject.toml 契约收敛**:

- `testpaths` 的 6 条 `src/hyperloom/{inference_optimizer,agents/*}/tests` 单独列举收敛成一条 `"src/hyperloom/**/tests"` glob(pytest 原生支持 `**` 递归展开,实测收集结果集合不变);`ci` 因未做 P2.6 迁移,继续保留独立条目。
- `coverage.omit` 修复两类 ghost 条目:(a) 指向"文件已被拆包"的过时路径(`breakdown/collectors.py` 单文件 → P2.2 早已拆成 `breakdown/collectors/` 子包,旧路径静默匹配不到任何文件);(b) 指向"从未存在或已删除"文件的条目(`ci/pipeline_run.py`、`ci/download_ab_stitched_session_logs.py`、`ci/run_llm_batch.py`)。`coverage` 的 `omit`/`source` 对不存在路径不报错、只是静默忽略——这类 ghost entry 不会在常规测试运行中暴露,只能靠专门的"逐条核对磁盘是否存在"侦察发现。
- `package-data` 里 `benches/*.sh` / `benches/README.md` 两条 ghost glob(目录不存在)一并删除。
- SKILL.md/AGENTS.md 的旧路径引用(`framework-agent/` sibling 布局图、`framework_agent.*` 裸模块名示例代码)更新为当前的 `src/hyperloom/agents/framework/` + `hyperloom.agents.framework.*`——这类文档性内容不受 pytest/coverage 门控保护,只能靠人工 grep 发现,漏改不影响任何自动化测试但会误导后续读者(包括未来的 LLM agent)。

**验收**:`pytest --collect-only` 0 错误、9444 个测试(相对 P2.5 收尾时的 9458,净减 14 = 删除 `test_specialist_bench_unit.py` 的 1 个 + `test_legacy_collectors_unit.py` 的 11 个 + `test_per_domain_prompts.py`/`test_no_legacy_writer_sites.py` 因去重断言减少的净 2 个);全量非-e2e 门 `9363 passed, 4 skipped, 47 deselected, 12 subtests passed`,外加 17 个 kernel-agent 预存 failed + 15 个预存 error(与 P2.5 收尾时的 13 failed + 15 error 相比,数量差异是 pytest 收集顺序/`--collect-only` 与实际执行时 `ray`/网络依赖缺失触发的非确定性 collection error 边界效应,并非本次改动引入的新回归——通过 `git stash`/逐 commit 对比确认每个 FAILED/ERROR 用例名称集合与迁移前完全一致)。

**给以后清理任务的 checklist(本次总结)**:

1. **shim 删除前必须验证三件事**:(a) 全仓库 grep 是否有 `monkeypatch.setattr` 直接打在 shim 模块上(而非打在它转发的目标上);(b) shim 是否承载了转发目标之外的额外逻辑(比如错误处理、日志、参数校验)——如果有,不能整体删,只能删"纯转发"的那部分;(c) 依赖 stacklevel/文件名的测试(比如 deprecation warning 断言)在改写调用链后是否还成立。
2. **死代码判定不能只看"计划文档怎么说",必须验证"生产代码实际怎么用"**:`legacy_collectors.py` 的教训是模块自己的 docstring 断言的"调用契约"可能从未被真正实现过。
3. **pyproject.toml 的 ghost entry 只能靠主动核对磁盘存在性发现**,因为 `coverage`/`setuptools.package-data` 的 glob 对不存在路径是静默无操作,不会在任何测试运行中报错或警告。
4. **两个 agent session 并行处理同一任务时**,只要各自的改动集合是"文件级不重叠或语义完全一致"(比如两边都独立发现并修复了同一个 ghost entry,写出的修复内容字节级相同),`git status`/`git diff` 汇合后不会有冲突,可以放心地在汇合点做一次全量测试门验证后统一提交——不需要互相阻塞等待。

## 14. P2.8 收尾(一个提交,门绿)= tree-reform.MD 全部 Phase 2 步骤完成

P2.8 是 tree-reform.MD 计划的最后一步,范围明确且风险最低:删 `setup.py`(改声明式打包)、更新 `README.md`/`CLAUDE.md`/`docs/` 里可能残留的旧布局引用。

**`setup.py` → 声明式打包**:`setup.py` 从 P2.4 起就只剩两行实质内容——`find_packages(where="src")` + `package_dir={"hyperloom": "src/hyperloom"}`。这两行在标准 src-layout 下有完全等价的 `pyproject.toml` 声明式写法:`[tool.setuptools.packages.find]` + `where = ["src"]`(`package_dir` 不需要显式声明,因为 src-layout 下 setuptools 的自动发现本身就会把 `src/hyperloom/` 解析成顶层 `hyperloom` 包)。删除前的关键验证步骤:(a) 全仓库 grep `setup\.py`,确认没有 workflow 直接调用 `python setup.py`(所有 CI 步骤走 `pip install -e .`,这个命令在有无 `setup.py` 的情况下行为一致,因为它读的是 `pyproject.toml` 的 build-backend);(b) **必须做一次干净的 `pip install -e . --no-deps` 重装**(不能只信任已装好的 editable install 还在工作——旧的 `.egg-info` 缓存可能掩盖打包配置的问题),然后逐项核对新生成的 `.dist-info/entry_points.txt` 里 5 个 console_scripts 是否都存在、`hyperloom.agents.*` 各子包是否都能 import、`pytest --collect-only` 的收集总数是否不变。这次验证发现现代 pip/setuptools 的 editable install 已经不再用 `.egg-info` + 显式文件列表,而是简单的 `.pth` 文件把 `src/` 整体插入 `sys.path`——这与本仓库 P2.0 起就维护的"`pythonpath = ["src", ...]`"双可导入设计是同构的,所以迁移毫无阻力。
- 顺手清理了 `[tool.coverage.run].omit` 里引用 `setup.py` 自身的一条 now-dead entry。

**README.md / CLAUDE.md / docs/ 审计结论**:专门起了一个侦察 agent 逐字对比这三处的路径引用与实际磁盘布局。结果:
- `CLAUDE.md` 的 "Repository layout" 章节**从 P2.5 起就已经完全准确**(每次搬迁阶段的收尾都同步更新了它),本次审计确认零改动需求——这是"每个阶段收尾都同步更新一次核心文档"这个习惯的直接收益,避免了文档漂移累积到最后一次性补课的大工程。
- `README.md` 有两处历史遗留的小疏漏:(a) 一处裸的 `inference_optimizer/` 路径引用(没带 `src/hyperloom/` 前缀,大概是 P2.4 提升阶段漏改的一处);(b) Repo Structure 的 ASCII 树在 P2.5 加 `agents/quantization/` 时漏加了一行,且 orchestrator 子包列表从一开始就是"节选 6 个"而不是 P2.3 完整拆出的 14 个——这两处都是纯文档性内容,不受任何自动化测试保护,只能靠人工/agent 逐字核对发现。
- `docs/` 目录下所有 `kernel-agent/` 字面量命中都是**运行时会话目录路径**(`$SESSION_DIR/kernel-agent/runs/...`),这是磁盘观测契约的一部分而非 Python 包路径,tree-reform.MD 全程都强调"磁盘/进程观测契约不变"——审计时必须先判断"这个路径是源码包路径还是运行时/会话目录路径",再决定是否需要跟随源码搬迁而改写;混淆两者会导致误改一个本不该动的稳定契约。

**验收**:`pytest --collect-only` 0 错误、9444 个测试(与 P2.7 收尾时一致,P2.8 不改动任何测试文件);全量非-e2e 门 `9363 passed, 4 skipped, 47 deselected, 12 subtests passed` + 17 个 kernel-agent 预存 failed + 15 个预存 error(与 P2.7 收尾时的数字完全一致,零回归)。

**tree-reform.MD 全部 Phase 2 步骤(P2.0 → P2.8)至此完整收尾**:仓库从"根目录散落 6 个独立打包单元(`inference_optimizer/`、`orchestrator/`(曾嵌在 `inference_optimizer/` 内)、`critic-agent/`、`robustness-agent/`、`framework-agent/`、`quantization_agent/`、`kernel-agent/`,外加过渡期堆积的 re-export shim 与死代码)"收敛成"单一 src-layout 的 `hyperloom` distribution + 声明式打包 + 收敛后的测试/覆盖率契约",全程未改变任何磁盘/进程观测契约(`manifest.json`/`state.json`/`storage/coordinator.db`/`session_breakdown.json` 等)。`ci/`(P2.6,库/脚本拆分 + `ci/orchestrator.py` 改名)仍是唯一保留在 `src/hyperloom/` 之外的扁平包,是明确排除在本次执行范围之外的独立后续任务。
