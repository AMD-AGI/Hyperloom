# Phase B · 步骤 01 — 拆瘦 coordinator.py(12334 行)

## 现状

整个控制平面挤在一个文件:相位机驱动、intent 路由、任务物化、四角色反应堆、kernel 程序化 handler、resume、KB/journal、CLOSE 序列。

## 策略(先删后拆,不为拆而拆)

### A. 先在文件内删(Phase A 已删大半,这里补)

- 已移除行为的残留分支:no-more-leverage 安全网(~4599–4605)、N33 idle early-close(~4618–4624)、`consecutive_silent_ticks` 触发逻辑、mid-run report 停止(~10486–10492)。
- 确认这些分支删后,相位走向仍由 `phase_state.compute_next_phase` 决定(护栏验证)。

### B. 抽出"纯函数/弱耦合"段为协作模块(单向依赖)

候选抽出对象(每个独立 commit,抽完原处仅留薄调用):

| 抽出内容 | 目标模块(建议) | 依赖方向 |
|---|---|---|
| 模块级 helper(model class 推断、roofline watermark、baseline YAML 解析,1–583) | `orchestrator/coordinator_helpers.py` | 被 coordinator 单向 import |
| resume 检测/重放(`_detect_resume_state`/`replay_for_resume`,878–1125) | `orchestrator/coordinator_resume.py` | 单向 |
| CLOSE 序列(`_on_enter_close` 等,3707+) | `orchestrator/coordinator_close.py` | 单向 |
| intent handlers 分组(propose/review/delegate/request,5865–10011) | 按角色拆 `coordinator_intents_*.py` 或保留但分区 | 谨慎,耦合 self 多 |

> intent handlers 大量用 `self.*`,**强耦合**,拆出收益低、风险高。优先只做 A(删)+ helper/resume/close 抽出;intent handlers 段宁可**文件内分区 + 注释分隔**,不强拆。

### C. 依赖方向检查(防循环引用)

抽出后运行:
```bash
python -c "import inference_optimizer.orchestrator.coordinator"
pip install pydeps 2>/dev/null
pydeps inference_optimizer/orchestrator/coordinator.py --max-bacon=2 --show-deps 2>/dev/null | head
# 或简单 grep 反向 import
rg -n 'import.*coordinator' inference_optimizer/orchestrator/coordinator_*.py
```
拆出的模块**不得** import 回 `coordinator`。

## 目标

- `coordinator.py` 从 ~12k 降到明显更小(理想 < 6–7k,视 intent handlers 是否强拆而定)。
- 净行数不增(抽出 = 移动,非复制)。

## 验收

- [ ] A 段死分支删净,护栏绿。
- [ ] 抽出模块单向依赖,无循环。
- [ ] `coordinator.py` 行数显著下降,总 LOC 不增。
- [ ] commit 序列:`Remove dead coordinator branches` → `Extract coordinator helpers/resume/close`。

## ⚠️ 注意

- 拆 intent handlers 风险最高,**默认不拆**,除非能干净下沉。拿不准就停在"文件内分区"。
- 每抽出一块**立即跑护栏**,不要攒着拆完再测。
