# ===================== RAG 检索问答 最终代码 =====================
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
# from langchain_community.chains import RetrievalQA         //RetrievalQA已被移除出langchain_community.chains
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.prompts import PromptTemplate

# ===================== 1. 加载向量库（不用管txt文件！） =====================
# 加载嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 加载本地 FAISS 向量库
vector_db = FAISS.load_local(
    "faiss_index", 
    embeddings,
    allow_dangerous_deserialization=True  # 必须加，否则报错
)

# ===================== 2. 创建检索器 =====================
retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}  # 找最相关的3段内容      //对应检索top-k
)

# ===================== 3. 初始化本地大模型 =====================
llm = OllamaLLM(model="qwen2.5:1.5b")

# ===================== 补充：定义prompt =====================
template = """基于以下已知信息，简洁和专业地回答用户的问题。如果无法从中得到答案，请说“根据已知信息无法回答该问题”，不要编造答案。
已知信息：
{context}

问题：{question}
回答："""
prompt = PromptTemplate.from_template(template)

# ===================== 4. 构建 RAG 问答链-LCEL =====================
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ===================== 5. 开始提问！ =====================
print("\n✅ RAG 系统就绪！输入 'quit' 退出。\n")
while True:
    query = input("用户: ")
    if query.lower() in ['quit', 'exit', 'q']:
        break
    
    try:
        #  invoke 直接返回字符串结果
        response = rag_chain.invoke(query)
        print(f"AI: {response}\n")
    except Exception as e:
        print(f"发生错误: {e}")