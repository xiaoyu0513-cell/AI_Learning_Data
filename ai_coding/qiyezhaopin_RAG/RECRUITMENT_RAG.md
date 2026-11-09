# 招聘知识图谱 RAG

基于招聘岗位、技能、公司和候选人数据的问答与人岗技能差距分析原型。
项目位于 `AI_Learning_Data/ai_coding/qiyezhaopin_RAG/`，与 LangChain、LangGraph 学习目录平级。
下面的启动命令均从本项目目录执行。

## 功能与技术

- FastAPI 提供 SSE 流式问答、知识库构建、图谱搜索及人岗分析接口。
- Neo4j 存储岗位、技能、公司、候选人及技能前置依赖。
- BGE 中文嵌入模型和 Milvus 提供向量检索，结合 BM25 与 RRF 融合。
- DeepSeek 通过 OpenAI 兼容接口完成查询路由、图查询意图解析及回答生成。
- Next.js、React、TypeScript 提供问答、图谱、人岗分析、数据源和评估页面。

## 本地启动

需要 Python 3.10+、Node.js/npm 和已启动的 Docker Desktop（Linux 容器）。

在项目根目录的 PowerShell 中执行：

```powershell
python -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r backend/requirements.txt
Copy-Item backend/.env.example backend/.env
docker compose up -d --wait
```

在 `backend/.env` 填入自己的 `DEEPSEEK_API_KEY`。首次使用需下载
`BAAI/bge-small-zh-v1.5`；代码默认离线加载。模型尚未缓存时，在启动后端的终端设置：

```powershell
$env:HF_HUB_OFFLINE = "0"
$env:TRANSFORMERS_OFFLINE = "0"
```

启动后端：

```powershell
cd backend
& ../.venv/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

另开终端启动前端：

```powershell
cd frontend
npm.cmd ci
npm.cmd run dev -- --hostname 127.0.0.1
```

访问 http://127.0.0.1:3000 ，在数据源页面点击构建知识库后使用问答。
当前每次后端重启后需要重新执行构建，以初始化内存中的检索器。
后端 API 文档：http://127.0.0.1:8000/docs 。

VS Code 的“运行任务”中也提供数据库、后端、前端及启动全部任务；需先安装依赖并配置环境。
问答会将问题和检索上下文发送至配置的模型服务。

## 当前范围

- 种子数据为 20 个岗位、37 项技能、12 家公司和 6 名候选人。
- 数据入口为 `backend/data/seed/jobs.json`，文件上传界面尚未接通。
- 评估页指标为静态演示数据，不能作为真实评测结果。
- 会话管理及 Agentic 参数尚未实现；人岗匹配率目前按技能覆盖率计算。
- 当前为本地开发原型；公开部署前需配置认证、访问控制及独立的数据库凭据。

## Git 更新

从 `AI_Learning_Data` 仓库根目录执行：

```powershell
git status
git diff -- ai_coding/qiyezhaopin_RAG
git add ai_coding/qiyezhaopin_RAG
git diff --cached
git commit -m "feat: 描述本次功能变更"
git push origin main
```

只提交源码、配置模板和可公开的数据。真实 `.env`、Python 虚拟环境、
前端依赖、数据库卷和运行日志均由项目 `.gitignore` 排除。
