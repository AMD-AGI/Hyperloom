# Phase D · 步骤 02 — protocol/ 与 intent_parser 职责厘清

## 现状(职责重叠/名实不符)

- `inference_optimizer/protocol/`:只有 `action_surfaces.py`(action 分类集合)+ 一行 `__init__`。
- 真正的 intent/envelope schema 在 `orchestrator/intent_parser.py`(`IntentType`、`Intent`、`INTENT_ENVELOPE_SCHEMA`、`validate_envelope`)。
- 直觉上"protocol"应放协议 schema,实际不在那里 → 名实不符,增加认知负担。

## 方案 A(已确认采用):把 intent schema 归到 `protocol/`

> 决策已定:**选方案 A,彻底简化**。允许改变所有调用该包的文件、并按需调整文件结构来达成内聚。不保留过渡性 re-export shim(那会留下又一层间接)。

- 将 `intent_parser.py` 的 schema 部分(`IntentType`/`Intent`/`INTENT_ENVELOPE_SCHEMA`/`validate_envelope`/`_PAYLOAD_REQUIRED`)移入 `protocol/`(如 `protocol/intent.py`)。
- `action_surfaces.py` 已在 `protocol/`,二者同属"协议层" → `protocol/` 名实相符、内聚。
- **直接改所有调用方** import 指向 `protocol`,**不留** `orchestrator/intent_parser.py` 的 re-export shim:
  - 若 `intent_parser.py` 移空 → 删除该文件;
  - 若它还含**解析逻辑**(非纯 schema)→ 解析逻辑可留在 `orchestrator/`(它依赖上层),但 schema/校验下沉到 `protocol/`,解析层单向 import `protocol`。
- 不为兼容旧 import 路径保留任何过渡层(本次允许破坏内部 import 路径)。

## robustness-agent envelope 平价

`robustness-agent/.../role/envelope.py` 镜像 intent schema。方案 A 后:
- 该镜像**仍保留**(独立包,见 Phase B 步骤 04 决策)。
- 确保平价测试对照的是新的 `protocol/intent.py`(更新测试 import 路径)。

## 操作

1. 移动 schema → `protocol/intent.py`。
2. 全仓改 import:`rg -n 'intent_parser' inference_optimizer` 逐个改指向 `protocol`(**不保留 re-export shim**);`intent_parser.py` 移空则删,残留解析逻辑则瘦身为单向依赖 `protocol`。
3. 同步更新 robustness-agent 平价测试的 import 路径。
4. 跑 envelope 契约/平价护栏。

## 验收

- [ ] 协议 schema 与 action_surfaces 同属 `protocol/`(方案 A)。
- [ ] envelope 校验/平价护栏绿。
- [ ] 无循环引用(protocol 是底层,不得 import orchestrator)。
- [ ] commit:`Consolidate intent schema into protocol layer`。

## ⚠️ 注意

- `protocol/` 是**最底层**,移入后**严禁** import `orchestrator`/`shared_state` 等上层,否则制造循环。
- envelope 形状是 §1 子进程 JSON 桥契约 —— 移动位置可以,**字段/形状不可变**。
