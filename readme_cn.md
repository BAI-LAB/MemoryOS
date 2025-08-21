# MemoryOS

<div align="center">
  <img src="https://github.com/user-attachments/assets/eb3b167b-1ace-476e-89dc-1a7891356e0b" alt="logo" width="400"/>
</div>
<p align="center">
  <a href="https://arxiv.org/abs/2506.06326">
    <img src="https://img.shields.io/badge/Arxiv-paper-red" alt="Mem0 Discord">
  </a>
  <a href="#contact-us">
    <img src="https://img.shields.io/badge/Wechat-群二维码-green" alt="Mem0 PyPI - Downloads">
  </a>
  <a href="https://www.youtube.com/watch?v=WHQu8fpEOaU" target="blank">
    <img src="https://img.shields.io/badge/Youtube-Video-red" alt="Npm package">
  </a>
  <a href="https://discord.gg/SqVj7QvZ" target="_blank">
    <img src="https://img.shields.io/badge/Discord-Join_us-yellow" alt="Discord">
  </a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0" target="_blank">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="License: Apache 2.0">
  </a>
</p>

<h5 align="center"> 🎉 如果您喜欢我们的项目，请在 GitHub 上给我们一个星 ⭐ 以获取最新更新。</h5>

**MemoryOS** 旨在为个性化 AI 智能体提供记忆操作系统，以实现更连贯、个性化和情境感知的交互。它从操作系统的内存管理原理中汲取灵感，采用分层存储架构，包含四个核心模块：存储、更新、检索和生成，以实现全面高效的记忆管理。在 LoCoMo 基准测试中，该模型在 F1 和 BLEU-1 分数上平均提升了 **49.11%** 和 **46.18%**。

  - **论文**: <a href="[https://arxiv.org/abs/2506.06326](https://arxiv.org/abs/2506.06326)" target="_blank">[https://arxiv.org/abs/2506.06326](https://arxiv.org/abs/2506.06326)</a>
  - **网站**: <a href="[https://baijia.online/memoryos/](https://baijia.online/memoryos/)" target="_blank">[https://baijia.online/memoryos/](https://baijia.online/memoryos/)</a>
  - **文档**: <a href="[https://bai-lab.github.io/MemoryOS/docs](https://bai-lab.github.io/MemoryOS/docs)" target="_blank">[https://bai-lab.github.io/MemoryOS/docs](https://bai-lab.github.io/MemoryOS/docs)</a>
  - **YouTube 视频**: **MemoryOS MCP + RAG Agent That Can Remember Anything**
  - <a href="[https://www.youtube.com/watch?v=WHQu8fpEOaU](https://www.youtube.com/watch?v=WHQu8fpEOaU) ">[https://www.youtube.com/watch?v=WHQu8fpEOaU](https://www.youtube.com/watch?v=WHQu8fpEOaU) </a>

<span id='features'/>

## ✨ 主要特性

  * 🏆 **顶级的记忆管理性能**
    </br> 在长期记忆基准测试中取得 SOTA 成果，在 LoCoMo 基准上 F1 分数提升 49.11%，BLEU-1 分数提升 46.18%。

  * 🧠 **即插即用的记忆管理架构**
    </br> 支持无缝集成的可插拔记忆模块——包括存储引擎、更新策略和检索算法。

  * ✨ **轻松创建智能体工作流** (**MemoryOS-MCP**)
    </br> 通过调用 MCP 服务器提供的模块化工具，为各种 AI 应用注入长期记忆能力。

  * 🌐 **通用大语言模型支持**
    </br> MemoryOS 无缝集成各种大语言模型 (例如 OpenAI, Deepseek, Qwen ...)。

<span id='news'/>

## 📣 最新消息

  * *<mark>[new]</mark>* 🔥🔥🔥 **[2025-07-15]**: **🔌 支持** 向量数据库 [Chromadb](https://www.google.com/search?q=%23memoryos_chromadb-getting-started)
  * *<mark>[new]</mark>* 🔥🔥🔥 **[2025-07-15]**: **🔌 集成** [Docker](https://www.google.com/search?q=%23docker-getting-started) 到部署流程
  * *<mark>[new]</mark>* 🔥🔥 **[2025-07-14]**: **⚡ 加速** MCP 并行化
  * *<mark>[new]</mark>* 🔥🔥 **[2025-07-14]**: **🔌 支持** BGE-M3 & Qwen3 嵌入模型 (PyPI 和 MCP)
  * *<mark>[new]</mark>* 🔥 **[2025-07-09]**: **📊 评估** MemoryOS 在 LoCoMo 数据集上的表现: 公开可用 [👉复现](https://www.google.com/search?q=%23reproduce)
  * *<mark>[new]</mark>* 🔥 **[2025-07-08]**: **🏆 新增配置参数**
  * 新增参数配置: **similarity_threshold**。配置文件请参见 📖 [文档](https://bai-lab.github.io/MemoryOS/docs) 页面。
  * *<mark>[new]</mark>* **[2025-07-07]**: **🚀 5倍速度提升**
  * MemoryOS (PYPI) 实现已升级：通过并行化优化，**速度提升5倍** (延迟降低)。
  * *<mark>[new]</mark>* **[2025-07-07]**: **✨ 现已支持 R1 模型**
  * MemoryOS 支持配置和使用 **Deepseek-r1 和 Qwen3** 等推理模型。
  * *<mark>[new]</mark>* **[2025-07-07]**: **✨MemoryOS Playground 上线**
  * **MemoryOS Platform** 的 Playground 已经上线！[👉MemoryOS Platform](https://baijia.online/memoryos/)。如果您需要**邀请码**，请随时 [联系我们](https://www.google.com/search?q=%23community)。
  * *<mark>[new]</mark>* **[2025-06-15]**:🛠️ 开源 **MemoryOS-MCP** 发布！现在可在智能体客户端上配置，以实现无缝集成和定制。[👉 MemoryOS-MCP](https://www.google.com/search?q=%23memoryos-mcp-getting-started)。
  * **[2025-05-30]**: 📄 论文-**Memory OS of AI Agent** 已在 arXiv 上发布: [https://arxiv.org/abs/2506.06326](https://arxiv.org/abs/2506.06326)。
  * **[2025-05-30]**: **MemoryOS** 初始版本发布！具有短期、中期和长期角色记忆，以及自动用户画像和知识更新功能。

<span id='list'/>

## 🔥 MemoryOS 支持列表

<table>
<thead>
<tr>
<th>类型</th>
<th>名称</th>
<th>开源</th>
<th>支持</th>
<th>配置</th>
<th>描述</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">智能体客户端</td>
<td><strong>Claude Desktop</strong></td>
<td>❌</td>
<td>✅</td>
<td>claude_desktop_config.json</td>
<td>Anthropic 官方客户端</td>
</tr>
<tr>
<td><strong>Cline</strong></td>
<td>✅</td>
<td>✅</td>
<td>VS Code settings</td>
<td>VS Code 扩展</td>
</tr>
<tr>
<td><strong>Cursor</strong></td>
<td>❌</td>
<td>✅</td>
<td>Settings panel</td>
<td>AI 代码编辑器</td>
</tr>
<tr>
<td rowspan="6">模型提供商</td>
<td><strong>OpenAI</strong></td>
<td>❌</td>
<td>✅</td>
<td>OPENAI_API_KEY</td>
<td>GPT-4, GPT-3.5, 等.</td>
</tr>
<tr>
<td><strong>Anthropic</strong></td>
<td>❌</td>
<td>✅</td>
<td>ANTHROPIC_API_KEY</td>
<td>Claude 系列</td>
</tr>
<tr>
<td><strong>Deepseek-R1</strong></td>
<td>✅</td>
<td>✅</td>
<td>DEEPSEEK_API_KEY</td>
<td>中文大模型</td>
</tr>
<tr>
<td><strong>Qwen/Qwen3</strong></td>
<td>✅</td>
<td>✅</td>
<td>QWEN_API_KEY</td>
<td>阿里巴巴通义千问</td>
</tr>
<tr>
<td><strong>vLLM</strong></td>
<td>✅</td>
<td>✅</td>
<td>本地部署</td>
<td>本地模型推理</td>
</tr>
<tr>
<td><strong>Llama_factory</strong></td>
<td>✅</td>
<td>✅</td>
<td>本地部署</td>
<td>本地微调部署</td>
</tr>
</tbody>
</table>
所有模型调用均使用 OpenAI API 接口；您需要提供 API 密钥和基础 URL。

## 📑 目录

  * <a href='#features'>✨ 特性</a>
  * <a href='#news'>🔥 最新消息</a>
  * <a href='#list'>🔍 支持列表</a>
  * <a href='#structure'> 📁 项目结构</a>
  * <a href='#pypi-mode'>🎯 快速入门</a>
      * <a href='pypi-mode'>通过 PYPI 安装 MemoryOS</a>
      * <a href='#MCP-mode'>MemoryOS-MCP</a>
      * <a href='https://www.google.com/search?q=%23memoryos_chromadb-getting-started'>MemoryOS-chromadb</a>
      * <a href='https://www.google.com/search?q=%23docker-getting-started'>Docker</a>
  * <a href='#todo'>☑️ 待办事项</a>
  * <a href='https://www.google.com/search?q=%23reproduce'>🔬 如何复现论文中的结果</a>
  * <a href='#doc'>📖 文档</a>
  * <a href='#cite'>🌟 引用</a>
  * <a href='https://www.google.com/search?q=%23community'>🤝 加入社区</a>

<span id='vedio'/>

<span id='structure'/>

## 🏗️ 系统架构

<img src="https://github.com/user-attachments/assets/09200494-03a9-4b7d-9ffa-ef646d9d51f0" width="80%" alt="image">

## 🏗️ 项目结构

```
memoryos/
├── __init__.py         # 初始化 MemoryOS 包
├── __pycache__/        # Python 缓存目录 (自动生成)
├── long_term.py        # 管理长期角色记忆 (用户画像, 知识)
├── memoryos.py         # MemoryOS 主类，协调所有组件
├── mid_term.py         # 管理中期记忆，整合短期交互
├── prompts.py          # 包含用于 LLM 交互的提示 (例如，摘要，分析)
├── retriever.py        # 从所有记忆层检索相关信息
├── short_term.py       # 管理近期交互的短期记忆
└── updater.py          # 处理记忆更新，包括在不同层之间传递信息
└── utils.py            # 库中使用的工具函数
```

<span id='pypi-mode'/>

## 📖 MemoryOS_PyPi 上手指南

### 环境要求

  * Python >= 3.10
  * conda create -n MemoryOS python=3.10
  * conda activate MemoryOS

### 安装

#### 从 PyPi 下载

```bash
pip install memoryos-pro -i https://pypi.org/simple
```

#### 从 GitHub 下载 (最新版本)

```bash
git clone https://github.com/BAI-LAB/MemoryOS.git
cd MemoryOS/memoryos-pypi
pip install -r requirements.txt
```

### 基本用法

```python
import os
from memoryos import Memoryos

# --- 基本配置 ---
USER_ID = "demo_user"
ASSISTANT_ID = "demo_assistant"
API_KEY = "YOUR_OPENAI_API_KEY"  # 替换为您的密钥
BASE_URL = ""  # 可选：如果使用自定义 OpenAI 端点
DATA_STORAGE_PATH = "./simple_demo_data"
LLM_MODEL = "gpt-4o-mini"

def simple_demo():
    print("MemoryOS 简单演示")

    # 1. 初始化 MemoryOS
    print("正在初始化 MemoryOS...")
    try:
        memo = Memoryos(
            user_id=USER_ID,
            openai_api_key=API_KEY,
            openai_base_url=BASE_URL,
            data_storage_path=DATA_STORAGE_PATH,
            llm_model=LLM_MODEL,
            assistant_id=ASSISTANT_ID,
            short_term_capacity=7,
            mid_term_heat_threshold=5,
            retrieval_queue_capacity=7,
            long_term_knowledge_capacity=100,
            # 支持 Qwen/Qwen3-Embedding-0.6B, BAAI/bge-m3, all-MiniLM-L6-v2
            embedding_model_name="BAAI/bge-m3"
        )
        print("MemoryOS 初始化成功！n")
    except Exception as e:
        print(f"错误: {e}")
        return

    # 2. 添加一些基本记忆
    print("正在添加一些记忆...")

    memo.add_memory(
        user_input="你好！我是汤姆，我在旧金山做数据科学家。",
        agent_response="你好汤姆！很高兴认识你。数据科学是一个非常令人兴奋的领域。你主要处理什么样的数据？"
    )

    test_query = "你记得我的工作是什么吗？"
    print(f"用户: {test_query}")

    response = memo.get_response(
        query=test_query,
    )

    print(f"助手: {response}")

if __name__ == "__main__":
    simple_demo()
```

<span id='MCP-mode'/>

## 📖 MemoryOS-MCP 上手指南

### 🔧 核心工具

#### 1. `add_memory`

将用户与 AI 助手之间的对话内容保存到记忆系统中，以建立持久的对话历史和上下文记录。

#### 2. `retrieve_memory`

根据查询从记忆系统中检索相关的历史对话、用户偏好和知识信息，帮助 AI 助手理解用户的需求和背景。

#### 3. `get_user_profile`

获取从历史对话分析中生成的用户画像，包括用户的个性特征、兴趣偏好和相关知识背景。

### 1. 安装依赖

```bash
cd memoryos-mcp
pip install -r requirements.txt
```

### 2. 配置

编辑 `config.json`：

```json
{
  "user_id": "用户ID",
  "openai_api_key": "OpenAI API 密钥",
  "openai_base_url": "https://api.openai.com/v1",
  "data_storage_path": "./memoryos_data",
  "assistant_id": "助手ID",
  "llm_model": "gpt-4o-mini",
  "embedding_model_name": "BAAI/bge-m3"
}
```

### 3. 启动服务器

```bash
python server_new.py --config config.json
```

### 4. 测试

```bash
python test_comprehensive.py
```

### 5. 在 Cline 和其他客户端上配置

复制 `mcp.json` 文件，并确保文件路径正确。

```bash
"command": "/root/miniconda3/envs/memos/bin/python"
# 这应该更改为您虚拟环境的 Python 解释器
```

## 📖MemoryOS_Chromadb 上手指南

### 1. 安装依赖

```bash
cd memoryos-chromadb
pip install -r requirements.txt
```

### 2. 测试

```bash
# 编辑信息在 comprehensive_test.py 中
    memoryos = Memoryos(
        user_id='travel_user_test',
        openai_api_key='',
        openai_base_url='',
        data_storage_path='./comprehensive_test_data',
        assistant_id='travel_assistant',
        embedding_model_name='BAAI/bge-m3',
        mid_term_capacity=1000,
        mid_term_heat_threshold=13.0,
        mid_term_similarity_threshold=0.7,
        short_term_capacity=2
    )
python3 comprehensive_test.py
# 切换嵌入模型时，请确保使用不同的数据存储路径。
```

## 📖Docker 上手指南

您可以通过两种方式使用 Docker 运行 MemoryOS：拉取官方镜像或从 Dockerfile 构建您自己的镜像。这两种方法都适用于快速设置、测试和生产部署。

### 选项 1: 拉取官方镜像

```bash
# 拉取最新的官方镜像
docker pull ghcr.io/bai-lab/memoryos:latest

docker run -it --gpus=all ghcr.io/bai-lab/memoryos /bin/bash
```

### 选项 2: 从 Dockerfile 构建

```bash
# 克隆仓库
git clone https://github.com/BAI-LAB/MemoryOS.git

cd MemoryOS

# 构建 Docker 镜像 (确保 Dockerfile 存在)
docker build -t memoryos .

docker run -it --gpus=all memoryos /bin/bash
```

## 🎯 复现

```bash
cd eval
# 在代码中配置 API 密钥和其他设置
python3 main_loco_parse.py
python3 evalution_loco.py
```

<span id='todo'/>

## ☑️ 待办事项

MemoryOS 正在不断发展！以下是即将推出的功能：

  - **进行中🚀**: **集成基准测试**: 标准化的基准测试套件，用于 Mem0、Zep 和 OpenAI 的跨模型比较
  - 🏗️ 实现跨不同系统的无缝记忆交换和集成。

有想法或建议吗？欢迎贡献！请随时提交 issues 或 pull requests！🚀

<span id='doc'/>

## 📖 文档

更详细的文档即将推出 🚀，我们将在 [文档](https://bai-lab.github.io/MemoryOS/docs) 页面更新。

<span id='cite'/>

## 📣 引用

**如果您觉得这个项目有用，请考虑引用我们的论文：**

```bibtex
@misc{kang2025memoryosaiagent,
      title={Memory OS of AI Agent}, 
      author={Jiazheng Kang and Mingming Ji and Zhe Zhao and Ting Bai},
      year={2025},
      eprint={2506.06326},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.06326}, 
}
```

<span id='related'/>

<span id='community'/>

## 🎯 联系我们

百家 AI 是由北京邮电大学白婷副教授指导的研究团队，致力于为 AI 智能体打造情感丰富、记忆超群的大脑。

🤝 合作与建议: baiting@bupt.edu.cn

📣 关注我们的**微信公众号**，加入**微信群**或 <img src="[https://img.shields.io/badge/Discord-yellow](https://img.shields.io/badge/Discord-yellow)" alt="Discord"> [https://discord.gg/SqVj7QvZ](https://discord.gg/SqVj7QvZ) 获取最新动态。

<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="https://github.com/user-attachments/assets/42651f49-f1f7-444d-9455-718e13ed75e9" alt="百家Agent公众号" width="250"/> 
  <img src="https://github.com/user-attachments/assets/b1b53846-cfc3-4814-bfa5-8e88b3621b9e" alt="微信群二维码" width="250"/>

</div>

## 🌟 Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=BAI-LAB/MemoryOS&type=Timeline)](https://www.star-history.com/#BAI-LAB/MemoryOS&Timeline)

## 免责声明

本项目 MemoryOS (Memory Operation System) 由百家 AI 团队开发，与 memoryOS ([https://memoryos.com](https://memoryos.com)) 无任何关联。此处使用“MemoryOS”名称仅为学术讨论目的。
