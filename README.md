# 简历智能问答系统 (SmartResume-RAG)

基于 LangChain + Chroma + DeepSeek-API 开发的 RAG 私有知识库问答系统，用于解决海量非结构化文档（如 PDF 简历、企业规章制度）的快速检索与精准问答。

## 💡 核心功能
- **PDF 解析与向量化**：使用 `PyPDFLoader` 解析简历，结合 BGE 中文嵌入模型构建本地 Chroma 向量库。
- **RAG 检索增强生成**：实现了从“文档切分 -> 向量化 -> 相似度检索 -> 大模型生成”的完整闭环。
- **Agent 智能问答**：引入 Agent 架构，使其能够自主调用知识库搜索工具，严格基于检索内容回答问题，有效降低大模型的幻觉。

## 🚀 如何运行
1. 安装依赖：`pip install -r requirements.txt`
2. 配置环境变量：在项目根目录新建 `.env` 文件，填入 `OPENAI_API_KEY`（DeepSeek Key）、`LLM_BASE_URL` 和 `LLM_MODEL`。
3. 启动项目：在 CMD 中输入 `python main.py`
4. 交互提问：根据提示输入问题（如“何家欢毕业于哪所学校？”），即可体验私有知识库问答。

## 🛠️ 技术栈
Python / LangChain / Chroma / DeepSeek-API / BGE Embedding
