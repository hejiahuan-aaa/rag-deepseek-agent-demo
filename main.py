import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ==========必须放在所有import最顶部==========
from dotenv import load_dotenv
import os
load_dotenv()

# ⚠️引号里面写.env的变量名字
MY_API_KEY = os.getenv("OPENAI_API_KEY")
MY_BASE_URL = os.getenv("LLM_BASE_URL")
MY_MODEL = os.getenv("LLM_MODEL")

print(f"读取key: {MY_API_KEY}")
print(f"读取base_url: {MY_BASE_URL}")
print(f"读取model: {MY_MODEL}")

# 下面才是库导入
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 使用本地开源embedding BGE
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# ---------------------- 1.加载文档、切分 ----------------------
documents = [
    Document(page_content="""
某健康管理公司业务资料
公司主要业务：面向客户提供健康AI管理服务，对接客户业务流程，搭建AI应用。
客户痛点：
1.客户内部资料多，员工查找资料效率低。
2.客户需要AI自动梳理业务问题，输出解决方案。
3.希望配置多个AI智能体分工完成工作，实现工作流自动化。
业务需求：将大模型AI工具接入客户业务，搭建RAG私有知识库，配置多Agent协作，实现文档问答、业务方案输出。
""")
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
split_docs = text_splitter.split_documents(documents)

# ----------------------2.本地BGE嵌入（不需要API）----------------------
model_name = "BAAI/bge-small-zh-v1.5"
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={"device":"cpu"},
    encode_kwargs={"normalize_embeddings":True}
)

vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory="./chroma_db"
)

# ----------------------3.初始化DeepSeek LLM ----------------------
llm = ChatOpenAI(
    api_key=MY_API_KEY,
    base_url=MY_BASE_URL,
    model=MY_MODEL,
    temperature=0
)

# ----------------------4.RAG检索链 ----------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k":3})
)

def rag_search_tool(query:str)->str:
    """知识库查询工具，查询业务文档资料"""
    res = qa_chain.invoke({"query":query})
    return res["result"]

tools = [
    Tool(
        name="business_knowledge_search",
        func=rag_search_tool,
        description="用于查询健康管理公司业务知识库，业务相关问题必须调用此工具获取文档内容"
    )
]

# ----------------------5.Agent提示词【修复关键点】 ----------------------
prompt = ChatPromptTemplate.from_messages([
    ("system","你是业务助手，回答问题优先调用知识库工具获取资料，严格基于文档内容回答。"),
    ("user","{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ----------------------入口运行 ----------------------
if __name__ == "__main__":
    print("\n=====运行Agent=====")
    output = agent_executor.invoke({"input":"客户业务存在哪些痛点，请给出解决方案"})
    print("\n【最终输出】")
    print(output["output"])
