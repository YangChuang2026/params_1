# RAG 文档问答系统

基于 LangChain 和 FAISS 的检索增强生成（RAG）文档问答系统，支持对技术文档进行智能检索和问答。

## 项目简介

本项目实现了一个完整的 RAG 流程，包括：
- PDF 文档提取
- 文本分块处理
- 向量化存储
- 智能检索问答

## 项目结构

```
params_1/
├── tansform_txt.py          # PDF 转 TXT 工具
├── split_docs.py            # 文本分块工具
├── build_vectordb.py        # 构建向量数据库
├── rag_qa.py                # RAG 问答系统主程序
├── faiss_index/             # FAISS 向量索引存储目录
├── .gitignore               # Git 忽略配置
└── README.md                # 项目说明文档
```

## 功能模块

### 1. PDF 转 TXT ([tansform_txt.py](file:///c:/Users/kym/Desktop/rare/intelligent%20lab/params_1/tansform_txt.py))
- 使用 pypdf 库提取 PDF 文本内容
- 保存为 UTF-8 编码的 TXT 文件

### 2. 文本分块 ([split_docs.py](file:///c:/Users/kym/Desktop/rare/intelligent%20lab/params_1/split_docs.py))
- 使用 LangChain 的 RecursiveCharacterTextSplitter
- 分块大小：500 字符
- 重叠大小：50 字符
- 支持中文句子分割（按 `\n\n`, `\n`, `。`, `！`, `？`, ` ` 分割）

### 3. 构建向量库 ([build_vectordb.py](file:///c:/Users/kym/Desktop/rare/intelligent%20lab/params_1/build_vectordb.py))
- 使用 HuggingFace Embeddings（sentence-transformers/all-MiniLM-L6-v2）
- 创建 FAISS 向量索引
- 保存到本地 `faiss_index` 目录

### 4. RAG 问答 ([rag_qa.py](file:///c:/Users/kym/Desktop/rare/intelligent%20lab/params_1/rag_qa.py))
- 加载预构建的 FAISS 向量库
- 使用 Ollama 本地大模型（qwen2.5:1.5b）
- 基于 LCEL（LangChain Expression Language）构建问答链
- 支持交互式问答，输入 `quit`/`exit`/`q` 退出

## 环境要求

- Python 3.8+
- 依赖库：
  - langchain-text-splitters
  - langchain-huggingface
  - langchain-community
  - langchain-ollama
  - langchain-core
  - pypdf
  - faiss-cpu

## 使用流程

### 步骤 1：PDF 转 TXT（如需要）
```bash
python tansform_txt.py
```

### 步骤 2：文本分块（可选，用于预览）
```bash
python split_docs.py
```

### 步骤 3：构建向量数据库
```bash
python build_vectordb.py
```

### 步骤 4：启动 RAG 问答系统
```bash
python rag_qa.py
```

## 使用说明

1. 启动问答系统后，输入问题即可获取基于文档的智能回答
2. 如果问题无法从文档中找到答案，系统会提示"根据已知信息无法回答该问题"
3. 输入 `quit`、`exit` 或 `q` 退出系统

## 示例

```
✅ RAG 系统就绪！输入 'quit' 退出。

用户：试验机如何校准？
AI: 根据文档内容，试验机校准需要...（基于检索到的文档内容回答）

用户：quit
```

## 注意事项

1. 首次运行前需要确保已安装 Ollama 并下载 qwen2.5:1.5b 模型
2. FAISS 向量库构建完成后会保存到 `faiss_index` 目录
3. 如果修改了源文档，需要重新运行 `build_vectordb.py` 重建向量库
4. 确保所有文件路径配置正确（当前使用绝对路径）

## 技术栈

- **LangChain**: LLM 应用开发框架
- **FAISS**: Facebook AI 相似性搜索库
- **HuggingFace Transformers**: 嵌入模型
- **Ollama**: 本地大模型运行平台
- **pypdf**: PDF 处理库
