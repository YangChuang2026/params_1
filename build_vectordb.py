# 导入依赖（使用你能跑通的正确导入格式）
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ===================== 已经写好的分块函数 =====================
def split_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " "]
    )
    return text_splitter.create_documents([text])


# ===================== 向量化与存储 =====================
# 1. 初始化嵌入模型
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# 2. 你的两个TXT文件路径
file1 = r"C:\Users\kym\Desktop\rare\intelligent lab\params_1\新秀系列微机控制电子万能试验机.txt"
file2 = r"C:\Users\kym\Desktop\rare\intelligent lab\params_1\新秀系列微机控制电子万能试验机软件管理者指南.txt"

# 3. 分别分块（调用你已有的函数）
doc1 = split_text(file1)
doc2 = split_text(file2)

# 4. 合并所有文档
all_docs = doc1 + doc2

# 5. 创建 FAISS 向量库
vector_db = FAISS.from_documents(all_docs, embeddings)

# 6. 保存到本地（下次直接加载，不用重新处理）
vector_db.save_local('faiss_index')

print("✅ 向量化完成！")
print("✅ 向量库已保存到：faiss_index 文件夹")