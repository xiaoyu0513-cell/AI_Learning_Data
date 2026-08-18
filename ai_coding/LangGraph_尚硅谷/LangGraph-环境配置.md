# LangGraph-环境配置

> Refs: 
>
> 1. https://www.bilibili.com/video/BV1z3NY66EY1/?spm_id_from=333.337.search-card.all.click&vd_source=5d5cb2062ab059c137f6aa8f9809b93c
> 2. https://github.com/xbsheng/atguigu-note/tree/main/langgraph

![image-20260811115832282](https://i.postimg.cc/rVhHvc39/image-20260811115832282.png?dl=1)

# 00. 环境配置

> 从零开始，一步一步完成本课程所需的所有环境配置。

---

## 1. 安装 Miniconda

Conda 是一个包管理和环境管理工具。**Miniconda** 是 Conda 的精简版，只包含最基本的组件，体积小、安装快，推荐使用。

### 1.1 下载 Miniconda

打开 Miniconda 官方下载页面：  
https://docs.anaconda.com/miniconda/

根据操作系统选择对应版本：

| 操作系统 | 推荐安装包                                                   |
| -------- | ------------------------------------------------------------ |
| Windows  | `Miniconda3 Windows 64-bit`（.exe 安装包）                   |
| macOS    | `Miniconda3 macOS Apple M1 64-bit`（Apple 芯片）或 `macOS Intel x86 64-bit`（Intel 芯片） |
| Linux    | `Miniconda3 Linux 64-bit`（.sh 脚本）                        |

> **注意**：Miniconda 自带 Python 3.13，本课程也使用 Python 3.13，无需额外调整。

### 1.2 Windows 安装步骤

1. 双击下载的 `Miniconda3-windows-*.exe`，启动安装向导
2. 点击 **Next** → 选择 **Just Me (recommended)** → Next
3. 安装路径建议保持默认（如 `C:\Users\你的用户名\miniconda3`），点击 Next
4. **关键步骤**：勾选以下两项：
   - ✅ **Add Miniconda3 to my PATH environment variable**（将 Conda 加入系统 PATH）
   - ✅ **Register Miniconda3 as my default Python 3.x**（设为默认 Python）
5. 点击 Install，等待安装完成
6. 安装完成后，打开 **Anaconda Prompt (miniconda3)**（从开始菜单搜索）

验证安装是否成功：

```bash
conda --version
# 输出类似：conda 26.x.x
```

### 1.3 macOS 安装步骤

```bash
# 在终端中执行（以 Apple M1 芯片为例）
bash ~/Downloads/Miniconda3-latest-MacOSX-arm64.sh

# 按提示操作：
#  - 按 Enter 阅读协议
#  - 输入 yes 接受协议
#  - 确认安装路径（默认 ~/miniconda3 即可）
#  - 输入 yes 执行 conda init
```

安装完成后，关闭并重新打开终端，验证：

```bash
conda --version
```

### 1.4 Linux 安装步骤

```bash
# 在终端中执行
bash ~/Downloads/Miniconda3-latest-Linux-x86_64.sh

# 按提示操作，步骤同 macOS
```

---

## 2. Conda 基础操作

以下是本课程会用到的 Conda 命令，有个印象即可：

| 命令                                 | 作用                   |
| ------------------------------------ | ---------------------- |
| `conda --version`                    | 查看 Conda 版本        |
| `conda info --envs`                  | 列出所有虚拟环境       |
| `conda create -n 环境名 python=3.10` | 创建新环境             |
| `conda activate 环境名`              | 激活（进入）某个环境   |
| `conda deactivate`                   | 退出当前环境           |
| `conda remove -n 环境名 --all`       | 删除某个环境           |
| `conda list`                         | 列出当前环境已安装的包 |

> **核心概念**：虚拟环境就是一个隔离的 Python 运行空间，不同环境之间互不影响。比如你可以同时拥有一个用 Python 3.13 的 `langgraph` 环境和一个用 Python 3.12 的 `other_project` 环境。

---

## 3. 创建课程环境

### 3.1 创建并激活环境

```bash
# 创建名为 langgraph 的环境，指定 Python 3.13
conda create -n langgraph python=3.13

# 出现 Proceed ([y]/n)? 时输入 y 回车确认
# 等待下载和安装完成（约 1-2 分钟）

# 激活环境
conda activate langgraph
```

激活成功后，终端提示符前面会出现 `(langgraph)`，表示当前已进入该环境：

```
(langgraph) C:\Users\你的用户名>
```

验证 Python 版本：

```bash
python --version
# 输出：Python 3.13.x
```

### 3.2 配置 pip 镜像源（中国大陆用户建议）

由于课程依赖包较多，配置国内镜像可以大幅提升下载速度：

```bash
# 设置清华大学 pip 镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

验证配置：

```bash
pip config list
# 应看到：global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'
```

如果后续遇到镜像源不稳定，可随时恢复默认：

```bash
pip config unset global.index-url
```

---

## 4. 安装 PyCharm 并创建项目

PyCharm 是 JetBrains 推出的 Python IDE，**Community 版免费且功能齐全**，本课程全程使用 PyCharm 进行开发和调试。

### 4.1 下载 PyCharm

打开 PyCharm 官方下载页面：  
https://www.jetbrains.com/pycharm/download/

选择 **PyCharm Community Edition**（免费版），根据操作系统下载对应安装包：

| 操作系统 | 安装包格式       |
| -------- | ---------------- |
| Windows  | `.exe` 安装包    |
| macOS    | `.dmg` 磁盘映像  |
| Linux    | `.tar.gz` 压缩包 |

### 4.2 安装步骤

**Windows：**

1. 双击下载的 `.exe` 安装包，启动安装向导
2. 安装路径建议保持默认（如 `C:\Program Files\JetBrains\PyCharm Community Edition`）
3. **关键步骤**：在安装选项中勾选：
   - ✅ **Add "Open Folder as Project"**（右键菜单快捷打开项目）
   - ✅ **Add launchers dir to the PATH**（方便命令行启动）
4. 其余选项默认即可，点击 Install，等待安装完成
5. 安装完成后，勾选 **Run PyCharm** → Finish，启动 PyCharm

**macOS：**

1. 双击下载的 `.dmg` 文件
2. 将 PyCharm 图标拖入 **Applications** 文件夹
3. 从启动台打开 PyCharm

### 4.3 创建课程项目

1. PyCharm 启动后，点击 **New Project**（新建项目）

2. 在新建项目窗口中配置：
   
   | 设置项               | 值                                                           |
   | -------------------- | ------------------------------------------------------------ |
   | **Location**         | 选择一个合适的路径，末尾项目名填写 `langgraph`，如 `E:\code\langgraph` |
   | **Create Git repo**  | ❌ 不勾选（课程不涉及 Git）                                   |
   | **Interpreter type** | 选择 **Conda**（或 **Custom environment** → **Conda**）      |

3. Python 版本选 **Python 3.13**（Conda 会自动创建对应环境）

4. 点击 **Create**，等待 PyCharm 完成项目初始化

创建完成后，PyCharm 生成的空项目结构如下：

```
langgraph/          ← 项目根目录（PyCharm 自动创建）
├── .idea/           ← PyCharm 配置文件（无需手动修改）
└── main.py          ← 空入口文件（可删除或保留练手）
```

> 💡 **这就是本课程的项目根目录**。后续所有 `pip install`、`.env` 配置、代码文件都在这个目录下进行。`.idea/` 是 PyCharm 自动生成的配置目录，**不要手动修改或提交到 Git**。

### 4.4 PyCharm 终端

PyCharm 底部的 **Terminal** 面板（快捷键：`Alt + F12`）会自动激活当前项目配置的 Conda 环境，终端的提示符前面会显示 `(langgraph)`：

```
(langgraph) PS E:\code\langgraph>
```

> 👉 **后续所有命令（`pip install`、`python`、`jupyter` 等）都建议在 PyCharm 的 Terminal 中执行**，无需手动 `conda activate`。

---

## 5. 安装课程依赖

课程提供了两个依赖文件，均位于 `2.资料/` 目录下：

| 文件                    | 说明                                                         | 适用阶段    |
| ----------------------- | ------------------------------------------------------------ | ----------- |
| `requirements.txt`      | 核心依赖，约 180 个包                                        | 全部课程    |
| `requirements_full.txt` | 完整依赖，额外包含 Jupyter Notebook 和 `langgraph-cli` 部署工具 | Ch 9 及之后 |

### 5.1 安装核心依赖

```bash
# 确保已激活 langgraph 环境
conda activate langgraph

# 进入项目根目录
cd 课程目录/langgraph

# 安装核心依赖
pip install -r 2.资料/requirements.txt
```

安装过程约 5-10 分钟，会依次下载和安装约 180 个 Python 包。

### 5.2 安装完整依赖（推荐）

```bash
pip install -r 2.资料/requirements_full.txt
```

相比核心依赖，额外安装：

| 额外包                 | 用途                                         |
| ---------------------- | -------------------------------------------- |
| `jupyter`              | 在浏览器中运行 `.ipynb` 笔记本               |
| `langgraph-cli[inmem]` | LangGraph 本地开发服务器，用于 Ch 9 项目部署 |

### 5.3 核心包说明

安装完成后，环境中包含以下关键包（无需手动逐个安装）：

**LangGraph 全家桶：**

| 包名                            | 用途                         |
| ------------------------------- | ---------------------------- |
| `langgraph`                     | LangGraph 核心库，构建状态图 |
| `langgraph-prebuilt`            | 预置的 Agent 组件            |
| `langgraph-checkpoint`          | 检查点（Checkpoint）机制     |
| `langgraph-checkpoint-postgres` | PostgreSQL 持久化检查点      |
| `langgraph-sdk`                 | LangGraph Python SDK         |

**LangChain 全家桶：**

| 包名                  | 用途                  |
| --------------------- | --------------------- |
| `langchain`           | LangChain 核心        |
| `langchain-core`      | LangChain 基础抽象    |
| `langchain-community` | 社区集成              |
| `langchain-openai`    | OpenAI 集成           |
| `langchain-deepseek`  | DeepSeek 集成         |
| `langchain-anthropic` | Anthropic Claude 集成 |

**模型服务商 SDK：**

| 包名                      | 用途                    |
| ------------------------- | ----------------------- |
| `openai`                  | OpenAI 官方 SDK         |
| `anthropic`               | Anthropic 官方 SDK      |
| `dashscope`               | 阿里百炼（Qwen 等模型） |
| `tencentcloud-sdk-python` | 腾讯云 SDK              |

**其他重要依赖：**

| 包名                     | 用途                         |
| ------------------------ | ---------------------------- |
| `fastapi` / `uvicorn`    | Web 服务框架，部署时使用     |
| `pydantic`               | 数据校验，状态定义时大量使用 |
| `httpx` / `httpx-sse`    | HTTP 客户端 + SSE 流式支持   |
| `loguru`                 | 日志库                       |
| `python-dotenv`          | 读取 `.env` 环境变量文件     |
| `tiktoken`               | OpenAI 的 token 计数工具     |
| `SQLAlchemy` / `psycopg` | 数据库操作，持久化时使用     |
| `pymilvus`               | Milvus 向量数据库客户端      |

---

## 6. 配置 API Key

课程示例默认使用 **DeepSeek** 大模型，至少需要配置 DeepSeek 的 API Key。按实际需要也可以配置其他模型服务商。

### 6.1 获取 DeepSeek API Key

1. 打开 DeepSeek 开放平台：https://platform.deepseek.com/
2. 注册 / 登录账号
3. 进入 **API Keys** 页面，点击 **创建 API Key**
4. 复制生成的 key（形如 `sk-xxxxxxxx`），注意 key **只显示一次**，请立即保存

> DeepSeek 是新用户注册即送 500 万 tokens 额度，足够学完整套课程。

### 6.2 创建 .env 文件

在项目根目录（`langgraph/`）下新建 `.env` 文件，填入以下内容：

```properties
# ============================================
# 必填：课程示例默认使用 DeepSeek 模型
# ============================================
DEEPSEEK_API_KEY=sk-你的DeepSeek_API_Key

# ============================================
# 可选：其他模型服务商（如需切换模型时使用）
# ============================================
# OPENAI_API_KEY=sk-你的OpenAI_API_Key
# ANTHROPIC_API_KEY=sk-ant-你的Anthropic_API_Key

# ============================================
# 部署时需要（Ch 9 项目部署章节）
# ============================================
# LANGSMITH_API_KEY=lsv2-pt-你的LangSmith_API_Key
# LANGSMITH_TRACING=true
# LANGSMITH_ENDPOINT=https://api.smith.langchain.com
# LANGSMITH_PROJECT="langgraph-tutorial"
```

> `.env` 文件已在 `.gitignore` 中排除，**不会被提交到 Git**，可以放心保存密钥。

### 6.3 代码中如何读取 .env

课程每个 notebook 或示例代码开头都会包含以下两行，自动加载 `.env` 中的密钥：

```python
from dotenv import load_dotenv
load_dotenv(override=True)
```

之后创建模型客户端时，SDK 会自动从环境变量中读取对应的 API Key，无需在代码中硬编码：

```python
from langchain_deepseek import ChatDeepSeek

# API Key 自动从 DEEPSEEK_API_KEY 环境变量读取
model = ChatDeepSeek(model="deepseek-v4-flash")
```

---

## 7. Jupyter Notebook 配置

课程中的练习代码以 `.ipynb`（Jupyter Notebook）格式提供，位于 `3.代码/` 目录下。

### 7.1 启动 Jupyter

```bash
# 确保已激活 langgraph 环境
conda activate langgraph

# 进入项目根目录
cd 课程目录/langgraph

# 启动 Jupyter
jupyter notebook
```

执行后会：

1. 终端显示一个带 token 的本地 URL（如 `http://localhost:8888/?token=xxx`）
2. 自动打开默认浏览器，进入 Jupyter 文件管理界面

### 7.2 验证 Kernel

在 Jupyter 界面右上角或菜单栏中确认当前 Kernel 为 `Python (langgraph)`。如果不是：

1. 点击菜单 **Kernel** → **Change kernel** → 选择 `Python (langgraph)`

> 如果列表中没有 `Python (langgraph)`，说明没有创建对应的名称。在 `langgraph` 环境中重新执行：
>
> ```bash
> conda activate langgraph
> pip install ipykernel
> python -m ipykernel install --user --name langgraph --display-name "Python (langgraph)"
> ```

---

## 8. 环境验证

全部配置完成后，执行以下验证脚本，确保一切就绪。

### 8.1 基础验证

在终端中：

```bash
# 1. 确认 Conda 版本
conda --version

# 2. 确认 Python 版本和当前环境
python --version
# 输出：Python 3.13.x

# 3. 确认当前环境
conda info --envs
# langgraph 前面应有 * 号表示当前激活

# 4. 确认关键包已安装
pip show langgraph
pip show langchain-core
pip show langchain-deepseek
pip show jupyter
```

### 8.2 代码验证

在 Jupyter 中新建一个 Notebook，或创建一个测试脚本 `test_env.py`：

```python
# test_env.py
 
print("环境验证开始")
print("=" * 40)

# 1. Python 版本
import sys
print(f"Python 版本: {sys.version}")

# 2. 关键包
from importlib.metadata import version as pkg_version
print(f"LangGraph 版本: {pkg_version('langgraph')}")


import langchain_core
print(f"LangChain Core 版本: {langchain_core.__version__}")

import pydantic
print(f"Pydantic 版本: {pydantic.__version__}")

# 3. .env 加载
from dotenv import load_dotenv
import os
load_dotenv(override=True)

deepseek_key = os.getenv("DEEPSEEK_API_KEY")
if deepseek_key:
    print(f"DeepSeek API Key 已配置: {deepseek_key[:8]}...{deepseek_key[-4:]}")
else:
    print("⚠️  未检测到 DEEPSEEK_API_KEY，请在 .env 文件中配置")

# 4. 模型调用测试
if deepseek_key:
    from langchain_deepseek import ChatDeepSeek
    model = ChatDeepSeek(model="deepseek-v4-flash")
    response = model.invoke("回复'一切正常'四个字，不要其他内容")
    print(f"模型调用测试: {response.content}")
else:
    print("⚠️  跳过模型调用测试（缺少 API Key）")

print("=" * 40)
print("环境验证完成")
print("=" * 40)
```

运行脚本：

```bash
python test_env.py
```

如果一切正常，输出类似：

```
========================================
环境验证开始
========================================
Python 版本: 3.13.14
LangGraph 版本: 1.1.2
LangChain Core 版本: 1.2.18
Pydantic 版本: 2.12.5
DeepSeek API Key 已配置: sk-xxxx...xxxx
模型调用测试: 一切正常
========================================
环境验证完成
========================================
```

---

## 9. 常见问题

### Q1: `conda` 命令找不到？

**Windows**：确保安装时勾选了"Add Miniconda3 to my PATH"。如果还是找不到，从开始菜单打开 **Anaconda Prompt (miniconda3)** 即可。

**macOS / Linux**：执行 `~/miniconda3/bin/conda init` 后重启终端。

### Q2: PyCharm 中找不到 Conda 环境？

1. 打开 **File** → **Settings** → **Project: langgraph** → **Python Interpreter**
2. 点击齿轮 → **Add Interpreter** → **Add Local Interpreter**
3. 选择 **Conda Environment** → **Use existing environment**
4. 点击文件夹图标，浏览到 Conda 环境目录：
   - Windows: `C:\Users\你的用户名\miniconda3\envs\langgraph`
   - macOS/Linux: `~/miniconda3/envs/langgraph`
5. 选择 `python.exe`（或 `python`），点击 OK

如果列表中没有 `langgraph` 环境，先在 Terminal 中手动创建：

```bash
conda create -n langgraph python=3.13
```

创建后再回到上述步骤手动添加。

### Q3: `pip install` 速度很慢？

配置国内镜像源（见 3.2 节）：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q4: 安装依赖时报错 `Microsoft Visual C++ 14.0 is required`？

部分包（如 `unstructured`）需要 C++ 编译环境。解决方案：

1. 下载 Visual Studio Build Tools：https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
2. 安装时勾选 **"Desktop development with C++"** 工作负载
3. 重新执行 `pip install`

如果不想装 C++ 工具链，也可以跳过该包的学习章节，不影响核心课程。

### Q5: Jupyter 中找不到 `Python (langgraph)` Kernel？

```bash
conda activate langgraph
pip install ipykernel
python -m ipykernel install --user --name langgraph --display-name "Python (langgraph)"
```

重启 Jupyter 即可看到。

### Q6: `import langgraph` 报 `ModuleNotFoundError`？

说明包没有正确安装到当前环境：

```bash
# 确认当前在 langgraph 环境
conda activate langgraph

# 确认包已安装
pip show langgraph

# 如果未安装，重新安装
pip install -r 2.资料/requirements.txt
```

### Q7: 模型调用时报 `AuthenticationError`？

检查以下三点：

1. `.env` 文件中是否配置了正确的 API Key
2. `.env` 文件是否在项目根目录（`langgraph/` 下）
3. 代码开头是否执行了 `load_dotenv(override=True)`

### Q8: 想把环境恢复到初始状态？

```bash
# 退出当前环境
conda deactivate

# 删除并重建
conda remove -n langgraph --all
conda create -n langgraph python=3.13
conda activate langgraph
pip install -r 2.资料/requirements_full.txt
```

---

## 10. 环境配置检查清单

在开始第一章之前，逐项确认：

- [ ] Miniconda 已安装，`conda --version` 正常输出
- [ ] PyCharm Community Edition 已安装
- [ ] `langgraph` 项目已通过 PyCharm 创建
- [ ] PyCharm 解释器已配置为 Conda `langgraph` 环境（Python 3.13）
- [ ] 核心依赖已安装（`requirements.txt`）
- [ ] Jupyter 已安装并能正常启动
- [ ] `.env` 文件已创建，`DEEPSEEK_API_KEY` 已填写
- [ ] DeepSeek API Key 有可用额度
- [ ] 验证脚本全部通过

全部打勾后，就可以进入 [01-LangGraph基础入门](./01-LangGraph基础入门.md) 开始学习了。