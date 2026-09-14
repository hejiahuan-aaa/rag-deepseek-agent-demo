import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
load_dotenv()

MY_API_KEY = os.getenv("OPENAI_API_KEY")
MY_BASE_URL = os.getenv("LLM_BASE_URL")
MY_MODEL = os.getenv("LLM_MODEL")

print(f"API Key 读取状态: {'成功' if MY_API_KEY else '失败'}")
print(f"Base URL: {MY_BASE_URL}")
print(f"Model: {MY_MODEL}")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# ==================== 1. 加载PDF ====================
print("\n[1/4] 正在加载PDF文档...")
pdf_path = "./何家欢简历.pdf"
if not os.path.exists(pdf_path):
    print(f"❌ 错误：找不到文件 {pdf_path}，请确认PDF文件已放入 rag_project 文件夹！")
    exit()

loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ 成功加载文档，共 {len(documents)} 页")

# ==================== 2. 切分与向量化 ====================
print("\n[2/4] 正在切分文本并构建向量库...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)
print(f"✅ 文本切分完成，共生成 {len(split_docs)} 个片段")

model_name = "BAAI/bge-small-zh-v1.5"
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory="./chroma_db"
)
print("✅ 向量库构建完成")

# ==================== 3. 初始化LLM ====================
print("\n[3/4] 初始化 DeepSeek 模型...")
llm = ChatOpenAI(
    api_key=MY_API_KEY,
    base_url=MY_BASE_URL,
    model=MY_MODEL,
    temperature=0
)

# ==================== 4. RAG检索链与工具 ====================
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    return_source_documents=True
)

def rag_search_tool(query: str) -> str:
    """知识库查询工具"""
    res = qa_chain.invoke({"query": query})
    print(f"\n🔍 [调试] 检索到 {len(res['source_documents'])} 个相关片段")
    for i, doc in enumerate(res['source_documents']):
        print(f"  - 片段{i+1}: {doc.page_content[:80]}...")
    return res["result"]

tools = [
    Tool(
        name="resume_knowledge_search",
        func=rag_search_tool,
        description="用于查询何家欢的个人简历知识库，如毕业院校、项目经历、技能、联系方式等"
    )
]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的简历信息提取助手。必须优先调用工具获取简历信息，严格根据工具返回的内容回答。如果工具返回的信息中没有答案，请明确回复“简历中未提及”。"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ==================== 5. 入口运行（全部写在 main 里面，确保作用域正确） ====================
if __name__ == "__main__":
    print("\n[4/4] 系统就绪！")
    print("=" * 40)
    print("🤖 简历智能问答系统 (输入 q 退出)")
    print("=" * 40)

    while True:
        user_input = input("\n👤 请输入问题: ")

        if user_input.lower() in ["q", "quit", "exit"]:
            print("👋 再见！")
            break

        if not user_input.strip():
            continue

        try:
            output = agent_executor.invoke({"input": user_input})
            print("\n🤖 【回答】")
            print(output["output"])
            print("-" * 40)
        except Exception as e:
            print(f"❌ 发生错误: {e}")