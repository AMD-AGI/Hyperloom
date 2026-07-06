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
