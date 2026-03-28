# ===================== RAG 检索问答 最终代码 =====================
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
# from langchain_community.chains import RetrievalQA         //RetrievalQA已被移除出langchain_community.chains
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Tongyi
import os

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
os.environ["DASHSCOPE_API_KEY"] = "sk-3f8bbe84902140118ca0748ac123bdbc"  # 替换为你的DashScope API Key
llm = Tongyi(
    model="qwen-turbo",
    temperature=0.2,
    max_tokens=1024
)

# ===================== 补充：定义prompt =====================
template = """基于以下已知信息，按步骤回答用户问题：
步骤1：分析问题核心需求，列出需要从已知信息中验证的关键要点（仅列要点，无需展开）；
步骤2：从已知信息中提取与关键要点匹配的内容，验证是否能覆盖所有要点；
       - 若能覆盖：基于提取的内容组织答案，确保简洁、专业；
       - 若无法覆盖：直接输出“根据已知信息无法回答该问题”，禁止编造；
步骤3：按照上述分析，给出最终回答。

已知信息：
{context}

问题：{question}

请严格按照以下格式输出：
【思考过程】
步骤1：[你的核心需求分析]
步骤2：[你的信息提取与验证]
步骤3：[你的答案组织逻辑]

【最终回答】
[你的最终答案]"""
prompt = PromptTemplate.from_template(template)

# ===================== 4. 构建 RAG 问答链-LCEL =====================
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def extract_final_answer(response):
    """从思维链输出中提取最终回答（屏蔽中间思考过程）"""
    if "【最终回答】" in response:
        final_answer = response.split("【最终回答】")[-1].strip()
        return final_answer
    # 兜底：若模型未按格式输出，返回原始内容并提示
    return f"回答格式异常：{response}"

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    | extract_final_answer
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