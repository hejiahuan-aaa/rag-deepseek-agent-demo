from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

load_dotenv()

# 模拟业务文档
documents = [
    Document(page_content="""
某健康管理公司业务资料
公司主要业务：面向客户提供健康AI管理服务，对接客户业务流程，搭建AI应用。

客户痛点：
1.客户内部资料多，员工查找资料效率低。
2.客户需要AI自动梳理业务问题，输出解决方案。
3.希望配置多个AI智能体分工完成工作，实现工作流自动化。

业务需求：将大模型AI工具接入客户业务，搭建RAG私有知识库，配置多Agent协作工作流，实现文档问答、业务方案输出。
""")
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)
split_docs = text_splitter.split_documents(documents)

embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory="./chroma_db_web"
)
vectordb.persist()

llm = ChatOpenAI(temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k":3})
)

def rag_search_tool(query:str)->str:
    return qa_chain.invoke({"query":query})["result"]

tools = [
    Tool(
        name="业务知识库查询",
        func=rag_search_tool,
        description="用于查询企业私有业务文档，所有业务相关问题必须调用此工具"
    )
]

prompt = ChatPromptTemplate.from_messages([
    ("system","你是业务处理智能体。第一步先解析用户业务痛点，再调用工具查询知识库，最后输出完整AI解决方案。"),
    ("user","{input}"),
    ("agent_scratchpad","{agent_scratchpad}")
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# -------- streamlit网页UI --------
st.set_page_config(page_title="业务AI助手", page_icon="🤖")
st.title("🤖业务文档AI智能助手｜RAG+多Agent")
st.markdown("模拟AI应用工程师业务落地场景，输入业务问题获取AI解决方案")

user_input = st.text_area("请输入你的业务问题：", placeholder="例如：客户有哪些业务痛点，给出解决方案")

if st.button("提交查询"):
    if user_input.strip() == "":
        st.warning("请输入问题")
    else:
        with st.spinner("AI智能体正在处理中..."):
            res = agent_executor.invoke({"input": user_input})
            st.success("AI输出结果：")
            st.write(res["output"])

