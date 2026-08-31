# Warm replay 的 section 命名：现状与请求

## 一句话

Hyperloom 发布到 KB Store 的 recipe record 里，section 名 `explore` / `framework` 是历史遗留的**阶段名**，语义已经变成**杠杆（lever）名**，名实不符。我们想改名，但 section 名同时是已发布记录里的物理路径前缀，改名会让存量记录无法回放。需要确认存量记录的处置方式。

---

## 背景：为什么名实不符了

2026-08-28 的 PR #1301 把 `EXPLORE` 阶段并进 `FRAMEWORK_AGENT`，`EXPLORE` 作为阶段已不存在。此后 record 的分流键从"哪个阶段产出的"改成"改动了哪个杠杆"：

| 杠杆 | 落到哪个 section |
|---|---|
| 配置（server args / envs） | `explore` |
| 源码补丁 / 上游 PR / enablement | `framework` |
| kernel | `kernel` |

所以现在 `explore` 指的是**配置杠杆**，`framework` 指的是**源码杠杆**。两个名字都在描述一个已经不存在的阶段划分。

保留这个分法本身是必要的：如果全部合并进 `framework`，配置调优赢的和打补丁赢的在 KB 里就再也分不开。问题只出在名字上。

---

## 记录当前长什么样

JSON 结构：

```
{
  "record_kind": "hyperloom_recipe",
  "knowledge_schema_version": 1,
  "value": {
    "config":    { "extra_server_args": ..., "extra_envs": ... },   // 扁平 replay config
    "explore":   { "patches": ["explore/overlays/000000/00-x.patch"], ... },
    "framework": { "patches": ["framework/overlays/000001/00-y.patch"], ... },
    "kernel":    { ... },
    "patch_timeline": ["explore/overlays/000000/00-x.patch", "framework/overlays/..."]
  }
}
```

随附的文件树：

```
files/explore/overlays/000000/00-x.patch
files/framework/overlays/000001/00-y.patch
```

**section 名同时充当三种角色**：`value` 下的 JSON 键、patch ref 的路径前缀、归档文件树里的物理目录名。三者必须一致。

---

## 服务端不关心，约束全部来自存量记录

vendored SDK 的注释写得很明确：

> The service treats `knowledge` as opaque, so this is a producer-side convention rather than part of the record schema; it is fixed because documents already in the store are written under this key.

SDK 里**没有任何硬编码的 section 名**，`_checked_section()` 只拒绝路径逃逸（点号开头/结尾、路径分隔符），任何名字都接受。

SDK 对缺失 section 的契约也是宽松的——`read()` 返回 `None`，docstring 说 *"Callers should treat it as a cold start rather than an error."*

**也就是说：严格性是 Hyperloom 这边加的，不是 store 要求的。** 约束纯粹来自"已经写进 store 的文档"。

---

## 改名会破在哪

读侧硬编码 5 处，全部在 Hyperloom 侧（我们自己能改）：

1. patch ref 正则只认两个前缀：`^(explore|framework)/overlays/...`
2. `value` 下三个 section 缺一即抛：`for section in ("explore", "framework", "kernel")`
3. ref 前缀查不到对应 KB 就抛 `unsupported owner`
4. `prior_file(ref)` 校验 ref 前缀必须等于本 section 名
5. `patch_timeline` 与各 owner 的 ref 集合必须完全相等

**失败形态是 fail closed**：任一处抛异常，整个 warm replay 被跳过，run 冷启动，只在 session state 里记一条 `warm_replay_outcome: {"status": "skipped", "reason": ...}`。不崩溃，但也不降级——不会"能读多少读多少"。

**双向都破**：

- 新 build 写新名字 → 老 build 读不了
- 新 build 读老记录里的 `explore/...` → 一样读不了（除非加别名层）

---

## 我们想做的

把配置杠杆的 section 从 `explore` 改成名副其实的名字。

**注意不能叫 `config`**：`value.config` 已经被扁平 replay config 占用。撞名不会报错，而是**静默出错**——`read_patches()` 在扁平 config 里找不到 `patches` 键，返回空列表，overlay 会无声消失。候选名字：`config_lever`。

---

## 请 KB 组确认三件事

**1. store 里现存的 `hyperloom_recipe` record 能否作废？**
如果这些记录都可以清掉、或者不再要求可回放，我们直接改名，不需要任何兼容层。这是最干净的路径。

**2. 如果不能作废，store 侧能否批量迁移？**
需要对现有 record 同时做三件事，缺一不可：
- 文件树 `files/explore/**` → `files/<新名>/**`
- JSON 键 `value.explore` → `value.<新名>`
- `value.patch_timeline` 和 `value.<新名>.patches` 里每条 ref 的前缀改写

**3. Hyperloom 之外还有谁读这些 record？**
有没有 dashboard、报表、其他 agent 在直接读 `value.explore` / `value.framework`？如果有，改名需要一起排期。这一条我们从代码里看不到，只能问。

---

## 兜底方案

如果 1 和 2 都不可行，我们在读侧加别名层（同时接受新旧两种前缀），上述 5 处各加一条映射。代价是长期背着两套名字，且新写入仍要决定用哪个前缀——建议只在确认无法迁移时才走这条。
