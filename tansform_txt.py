from pypdf import PdfReader

# ========== 我已经帮你填好路径了！直接运行 ==========
pdf_path = r"C:\Users\kym\Desktop\rare\intelligent lab\params_1\新秀系列微机控制电子万能试验机软件管理者指南.pdf"
txt_path = r"C:\Users\kym\Desktop\rare\intelligent lab\params_1\新秀系列微机控制电子万能试验机软件管理者指南.txt"

# 读取PDF
reader = PdfReader(pdf_path)
text = ""

# 提取所有页面文字
for page in reader.pages:
    text += page.extract_text() + "\n\n"

# 保存为 TXT
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(text)

print("✅ PDF 转 TXT 完成！")
print("📄 文本文件位置：", txt_path)