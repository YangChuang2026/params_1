# 导入分块工具
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 定义分块函数（你给的代码，我帮你完善好了）
def split_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 文本分块：500字一块，重叠50字，防止语义断裂
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,    # 每块最大500字
        chunk_overlap=50, # 块之间重叠50字
        separators=["\n\n", "\n", "。", "！", "？", " "] # 按中文句子分割
    )

    # 返回分块后的文档
    return text_splitter.create_documents([text])


# ===================== 你只需要运行下面代码 =====================
# 你的两个TXT文件路径（我已经帮你填好了！）
file1 = r"C:\Users\kym\Desktop\rare\intelligent lab\params_1\新秀系列微机控制电子万能试验机.txt"

# 如果你有第二个文件，把名字填进去，我先给你留好位置
file2 = r"C:\Users\kym\Desktop\rare\intelligent lab\params_1\新秀系列微机控制电子万能试验机软件管理者指南.txt"

# 开始分块
chunks1 = split_text(file1)
chunks2 = split_text(file2)
all_chunks = chunks1 + chunks2  # 合并两个文件的分块

# 输出结果看看
print("✅ 分块完成！")
print(f"总共分成了 {len(all_chunks)} 块")

# 打印前2块内容，让你看到效果
print("\n==== 第1块内容 ====")
print(chunks1[0].page_content)
print("\n==== 第2块内容 ====")
print(chunks2[0].page_content)