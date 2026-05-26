# 目标
- 帮我优化下 hyerloom，具体是我希望 hyperloom 提交任务给到 geak 前，在 hyperloom 上调用一个 unittest 的 agent 来生成 unittest，然后将 prompt + unitest 传给 geak 去优化
- /wekafs/zihao/2026/0518_v2/Hyperloom/inference_optimize/SKILL.md 这个是你要优化的 skill 的地址，这个目录下我已经做了一些优化，但是我觉得他优化的不够好，你要完成俩面几个方向的重构：
    - geak 中本身就有生成 unitest 的功能，/workspace/hyperloom/runtime/source-mirrors/geak。 它有很多的约束，我希望你完全借鉴它的模式，然后把他整理成一个 skills 放到 hyperloom ，然后在 hyperloom 中调用这个 skillsi。 在 hyperloom 中生成 unittest 能够更准确，生成 unittest 过程中可能需要一些信息，这些信息你可以再 hyperloom 中自主去获取；整理好的 unittest skills 在主 agent 中调用，运行时不要再依赖 geak; 你可以完全弃用现在 hyperloom 中的 unittest 模式，完全按照 geak 中的来； hyperloom 调用 unitest agent，他是作为主 agent 的一 skills 来执行，用的 llm 还有 agent 和主 agent是同一个(比如 claude/course）。
    - 我希望让 unittest 这个功能插入到 hyperloom 中尽可能地简洁, 更好的去做模块化，现在的已有的实现方式改动了大量的文件，我觉得很乱，你来做下更好的模块化，对现有地 hyperloom 流程上主要只有两处如下需要改动：
        - kernel-agent/tools/kernel_optimization.py 中在发起 geak 请求前，加上一个生成 unittest 的模块, 这个 unittest 模块指向一个 skills 文件，你的所有和 unittest 相关的实现以及后续的维护优化统一在这个 hyperloom 下的子文件进行
        - kernel-agent/tools/kernel_optimization.py 启动 geak 时多传一个 unittest path 的参数 --test-command


# 相关信息
- 生成的 unittest 格式你要完全复用 geak 中的格式
- 注意 unittest 调用的 kernel 的运行环境要和 vllm/sglang 或者其它 e2e 场景中运行 kernel 的 runingtime 一致；正确的 kernel 定义需要的 args，正确的运行时的 shape，正确的环境变量等等；总之希望生成的 unittest 能反应 e2e 优化时的运行状态
- 生成 unittest 依赖的信息比如 shape ，dtype 还有环境变量，你要从下面几个 context 中依次去找，如果上一个没找到就用退而求其次的下一个方法
    - tracelense 的结果
    - profile 的 trace 文件
    - 模型结构以及启动脚本
    - repo 中的默认 unittest shape
- 生成的 unittest 你要做下验证，能编译，能跑同，结果正确
- hyperloom 的代码地址在： /wekafs/zihao/2026/0518_v2/Hyperloom ; 你的所有修改直接在这个目录上进行就可以
- hyperloom 的 skills 在： /wekafs/zihao/2026/0518_v2/Hyperloom/inference_optimize/SKILL.md ; 这个是总的接口 skills，你的 unittest 是其中的一环
- unittest 要兼容 hip 和 triton 两种语言
- unittest 要做一些容错处理，如果生成地有问题那么退回到初始地模式；加上一个开关来控制是否在 hyperloom 中生成 unittest 还是走原始的模式
- 整个过程要考虑下， geak 优化后 apply 到 e2e 这个模块。要保证能走通整个流程

你做完集成之后，hyperloom 的任务提交示例：
```
@/wekafs/zihao/2026/0518_v2/Hyperloom/inference_optimize/SKILL.md
你先进入到这个目录，/workspace/hyperloom，结果也都放到这里边,优化这个模型 Qwen3-8B， TP=2, geak 至少提交 4 个任务
```

你先看下这个方案有什么不清楚的地方，确认之后再来执行和验证效果
