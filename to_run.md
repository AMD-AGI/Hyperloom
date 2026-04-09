# Hyperloom 本地运行指南

> 前提：已安装 Python（你的环境：Python 3.14.4），操作系统为 Windows。
>
> 你的 Python 路径：`C:\Users\zgong\AppData\Local\Microsoft\WindowsApps\python.exe`
>
> **重要**：你的环境中 `pip` 不在 PATH 里，需要用 `python -m pip` 代替 `pip`。

---

## 准备工作：安装 uv（推荐的包管理器）

```
python -m pip install uv
```

如果上面报错说找不到 pip 模块，先安装 pip：

```
python -m ensurepip --upgrade
python -m pip install uv
```

验证安装：

```
uv --version
```

---

## 场景一：查看静态优化 Dashboard

Dashboard 是纯静态 HTML 页面，只需要 Python 内置的 HTTP 服务器，**不需要安装任何第三方依赖，不需要虚拟环境**。

### 步骤

```
cd c:\gitclones\Hyperloom-main\dashboards
python -m http.server 8765
```

### 打开浏览器访问

| 页面 | 地址 |
|------|------|
| 综合仪表盘（所有模型 + 搜索树） | http://localhost:8765/prism-optimization-dashboard.html |
| GLM-5 优化时间线 | http://localhost:8765/glm5-optimization-timeline.html |
| Qwen3.5 优化时间线 | http://localhost:8765/qwen35-optimization-timeline.html |
| DFS 搜索树可视化 | http://localhost:8765/optimization-search-tree.html |

### 停止服务

在终端中按 `Ctrl + C` 即可。

---

## 场景二：运行 TurboQuant 量化评估

TurboQuant 是 KV Cache 量化工具，会加载 **Qwen/Qwen2.5-3B-Instruct** 模型进行评估。

> **注意**：evaluate.py 文档说明"在单 GPU 上约 3 分钟"。如果没有 GPU，PyTorch 会自动回退到 CPU 运行，但速度会**非常慢**（可能需要数十分钟甚至更久），且需要足够的内存（建议 16GB+ RAM）。

### 方式 A：使用 venv（Python 内置）

在 **cmd（命令提示符）** 中逐条执行：

```
cd c:\gitclones\Hyperloom-main

python -m venv .venv

.venv\Scripts\activate.bat

python -m pip install torch transformers accelerate scipy

cd training_optimization\turboquant

python evaluate.py
```

如果在 **PowerShell** 中，激活命令不同：

```
.venv\Scripts\Activate.ps1
```

### 方式 B：使用 uv（更快）

在 **cmd（命令提示符）** 中逐条执行：

```
cd c:\gitclones\Hyperloom-main

uv venv .venv

.venv\Scripts\activate.bat

python -m uv pip install torch transformers accelerate scipy

cd training_optimization\turboquant

python evaluate.py
```

### 评估输出说明

脚本会依次运行三项测试：

1. **注意力保真度** — 逐层计算量化前后注意力分数的余弦相似度
2. **大海捞针检索** — 在 1K–4K tokens 长度上下文中检索隐藏句子
3. **困惑度 (Perplexity)** — 衡量语言模型生成质量

预期结果（3.5-bit 量化）：检索准确率 1.000，困惑度与全精度基本一致（~1.12）。

### 退出虚拟环境

```
deactivate
```

---

## 常见问题

### `python -m pip` 也报错 "No module named pip"

运行以下命令让 Python 自带的 ensurepip 模块安装 pip：

```
python -m ensurepip --upgrade
```

### PowerShell 无法激活虚拟环境

如果报错"无法加载脚本，因为在此系统上禁止运行脚本"，先运行：

```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### torch 安装很慢或失败

PyTorch 包体积较大（~2GB）。如果下载慢，可以使用国内镜像：

```
python -m pip install torch transformers accelerate scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

uv 方式：

```
uv pip install torch transformers accelerate scipy --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 模型下载问题

evaluate.py 会从 HuggingFace 下载 **Qwen/Qwen2.5-3B-Instruct** 模型（约 6GB）。如果网络不通，可以设置 HuggingFace 镜像：

cmd 中：

```
set HF_ENDPOINT=https://hf-mirror.com
python evaluate.py
```

PowerShell 中：

```
$env:HF_ENDPOINT = "https://hf-mirror.com"
python evaluate.py
```
