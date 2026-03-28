# Travel Agent 项目架构文档

## 一、项目概述

Travel Agent 是一个基于 LangGraph 和 DeepSeek 构建的智能旅行规划系统，采用 Multi-Agent 架构，通过 MCP (Model Context Protocol) 集成外部工具（12306 火车票查询、高德地图导航等），为用户提供完整的旅行计划服务。

---

## 二、技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| Agent 框架 | LangGraph |
| LLM | DeepSeek (deepseek-chat) |
| RAG | Chroma 向量数据库 + LangChain |
| MCP 集成 | langchain-mcp-adapters + mcp |
| 前端 | Vite + React + TypeScript |
| 配置管理 | Pydantic + python-dotenv |

---

## 三、项目结构

```
Travel_Agent/
├── backend/                      # 后端代码
│   ├── agents/                   # Agent 实现
│   │   ├── coordinator.py       # LangGraph 多 Agent 编排器
│   │   ├── search_agent.py      # 搜索与 RAG Agent
│   │   ├── planner_agent.py     # 旅行计划规划 Agent
│   │   ├── writer_agent.py      # 攻略写作 Agent
│   │   ├── visualization_agent.py  # 路线图生成 Agent
│   │   └── api_routes.py        # Agent API 路由
│   ├── rag/                     # RAG 模块
│   │   ├── retriever.py         # 向量检索器
│   │   ├── embedding_service.py # 嵌入服务
│   │   ├── vector_store.py      # 向量存储
│   │   ├── document_processor.py # 文档处理
│   │   └── rag_tool.py          # RAG 工具
│   ├── config.py                 # 配置管理
│   ├── agent_manager.py          # Agent 管理器（单例）
│   ├── main.py                   # FastAPI 主入口
│   └── logger.py                 # 日志模块
│
├── frontend/                     # 前端代码 (Vite + React)
│   └── src/components/
│       └── ChatBox.tsx           # 聊天组件
│
├── src/                          # 共享模块
│   ├── mcp_tools.py              # MCP 工具管理器
│   ├── skill_loader.py          # 技能加载器
│   ├── prompts.py                # 提示词模板
│   └── schemas.py                # 数据模型
│
├── skills/                       # 技能定义
│   └── travel_plan.md            # 旅行计划技能模板
│
├── data/                         # 数据目录
│   ├── knowledge_base/           # 知识库（原始 + 处理后）
│   └── chroma_db/               # Chroma 向量数据库
│
├── logs/                         # 日志目录
│
└── pyproject.toml                # 项目配置
```

---

## 四、Multi-Agent 架构

### 4.1 Agent 组件

项目基于 LangGraph 构建了 4 个核心 Agent：

| Agent | 职责 | 依赖 |
|-------|------|------|
| **SearchAgent** | 信息检索与 RAG 管理 | DeepSeek LLM + Chroma |
| **PlannerAgent** | 旅行计划生成与验证 | MCP 工具 (12306, 高德地图) |
| **WriterAgent** | 攻略文本生成 | DeepSeek LLM |
| **VisualizationAgent** | 路线图可视化 | 豆包 Seedream API |

### 4.2 LangGraph 状态流

```
┌─────────────────┐
│   输入请求      │
│ (user_request) │
└────────┬────────┘
         ▼
┌─────────────────┐
│   search_node   │──── RAG 检索 + 外部搜索
│   (SearchAgent) │
└────────┬────────┘
         ▼
┌─────────────────┐
│   planner_node  │──── 生成旅行计划
│  (PlannerAgent)│     调用 MCP 工具
└────────┬────────┘
         ▼
┌─────────────────┐
│   writer_node   │──── 生成攻略文本
│  (WriterAgent)  │
└────────┬────────┘
         ▼
┌─────────────────────┐
│  visualize_node      │──── 生成路线图
│(VisualizationAgent) │     (可选)
└────────┬────────────┘
         ▼
    [最终输出]
```

### 4.3 核心状态定义 (AgentState)

```python
class AgentState(BaseModel):
    # 输入
    user_request: str = ""
    enable_search: bool = True
    enable_visualization: bool = True
    style: str = "friendly"  # friendly / professional / fun
    
    # 中间状态
    search_context: str = ""      # RAG 检索结果
    travel_plan: Optional[Dict] = None  # 旅行计划
    guide_content: str = ""       # 攻略内容
    route_map: Optional[Dict] = None    # 路线图
    
    # 执行追踪
    current_node: str = ""
    error: Optional[str] = None
```

---

## 五、MCP 集成

### 5.1 MCP 工具配置

项目通过 `MCPToolsManager` 类管理 MCP 工具连接，使用 `langchain_mcp_adapters` 的 `MultiServerMCPClient` 自动发现和加载工具。

**默认 MCP 服务器配置** (`src/mcp_tools.py`):

| 服务器 | 用途 | 命令 |
|--------|------|------|
| **12306-mcp** | 火车票查询 | `npx -y 12306-mcp` |
| **amap-maps** | 高德地图导航 | `npx -y @amap/amap-maps-mcp-server` |

### 5.2 Skill 定义

`skills/travel_plan.md` 定义了旅行计划技能模板：

```yaml
mcp_servers:
  - 12306-mcp      # 票务查询
  - amap-maps      # 地图导航
  - china-weather-mcp  # 天气查询
```

**输出格式包含**:
- 🚄 交通信息（车次、余票）
- 🌤️ 天气预报
- 🗺️ 目的地导航
- ✅ 行程验证
- 📅 推荐行程
- 💰 预算估算

### 5.3 MCP 工具在 PlannerAgent 中的使用

PlannerAgent 接收 MCP 工具列表，在生成旅行计划时自动调用：

```python
class PlannerAgent:
    def __init__(self, api_key: str, tools: List[Any] = None):
        self.tools = tools or []
        # tools 包含 12306、高德地图等 MCP 工具
```

---

## 六、RAG 模块

### 6.1 架构

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│ 文档处理     │───▶│ 嵌入服务        │───▶│ 向量存储     │
│(document_    │    │(embedding_      │    │ (Chroma DB)  │
│ processor)   │    │ service)        │    │              │
└──────────────┘    └─────────────────┘    └──────────────┘
                                                  │
                                                  ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│ 用户查询      │───▶│ 向量检索        │◀───│ 相似度匹配   │
│              │    │ (retriever)     │    │              │
└──────────────┘    └─────────────────┘    └──────────────┘
```

### 6.2 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `RAG_EMBEDDING_MODEL` | text-embedding-v4 | 嵌入模型 |
| `RAG_CHUNK_SIZE` | 500 | 文档分块大小 |
| `RAG_CHUNK_OVERLAP` | 50 | 分块重叠 |
| `RAG_TOP_K` | 3 | 检索返回数量 |
| `RAG_COLLECTION_NAME` | travel_knowledge | 集合名称 |

### 6.3 数据来源

RAG 模块支持两种数据源：
1. **小红书 (XHS)** - 用户分享的旅行攻略
2. **全网** - 携程、马蜂窝等平台信息

> **注意**: 当前项目外部搜索为模拟实现，未接入真实小红书搜索 API。

---

## 七、配置管理

### 7.1 环境变量 (.env)

```bash
# API 密钥
DEEPSEEK_API_KEY=sk-xxxxx
DASHSCOPE_API_KEY=sk-xxxxx
SEEDREAM_API_KEY=sk-xxxxx
AMAP_MAPS_API_KEY=xxxxx

# 服务器
HOST=0.0.0.0
PORT=8000

# RAG 配置
RAG_CHROMA_PERSIST_DIR=./data/chroma_db
RAG_EMBEDDING_MODEL=text-embedding-v4
```

### 7.2 配置类 (Settings)

```python
class Settings(BaseModel):
    # 应用
    APP_NAME: str = "Travel Agent API"
    APP_VERSION: str = "1.0.0"
    
    # API 密钥
    DEEPSEEK_API_KEY: Optional[str]
    DASHSCOPE_API_KEY: Optional[str]
    AMAP_MAPS_API_KEY: Optional[str]
    SEEDREAM_API_KEY: Optional[str]
    
    # Agent 配置
    RECURSION_LIMIT: int = 25
    MAX_CONVERSATION_HISTORY: int = 20
    
    # RAG 配置
    RAG_CHUNK_SIZE: int = 500
    RAG_TOP_K: int = 3
```

---

## 八、API 接口

### 8.1 WebSocket 聊天接口

```
WS /ws
```

**客户端发送**:
```json
{
  "message": "帮我规划一个北京到苏州的3天旅行"
}
```

**服务端事件**:
- `received` - 消息已接收
- `rag_context` - RAG 检索上下文
- `node_start` - Agent 节点开始
- `node_complete` - Agent 节点完成
- `complete` - 执行完成
- `final` - 最终结果

### 8.2 REST API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 根路径健康检查 |
| GET | `/api/health` | 服务健康检查 |
| POST | `/api/rag/add` | 添加文档到知识库 |
| POST | `/api/rag/search` | 搜索知识库 |

---

## 九、依赖项 (pyproject.toml)

```toml
dependencies = [
    "langchain>=0.3.0",           # Agent 框架
    "langchain-core>=0.3.0",
    "langchain-deepseek>=0.1.0",   # DeepSeek LLM
    "langgraph>=0.2.0",           # LangGraph
    "langchain-mcp-adapters>=0.2.0",  # MCP 集成
    "mcp>=1.0.0",                 # MCP 协议
    "pydantic>=2.0.0",            # 数据验证
    "python-dotenv>=1.0.0",       # 环境变量
    "pyyaml>=6.0.0",              # YAML 解析
    "markdown>=3.5.0",            # Markdown 处理
]

[project.optional-dependencies]
backend = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "websockets>=12.0",
]
```

---

## 十、总结

| 特性 | 说明 |
|------|------|
| **架构模式** | LangGraph Multi-Agent 状态流 |
| **LLM** | DeepSeek Chat |
| **工具集成** | MCP (12306, 高德地图) |
| **知识增强** | Chroma RAG + 本地知识库 |
| **前端交互** | WebSocket 实时流式输出 |
| **输出格式** | 技能模板定义 (travel_plan.md) |

本项目采用模块化设计，各 Agent 职责清晰，通过 LangGraph 的状态图机制实现灵活的 Agent 编排，同时利用 MCP 协议扩展外部能力，为用户提供智能化的旅行规划服务。
