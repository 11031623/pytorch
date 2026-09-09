import os
knowledge_base_path = r"D:\Learning\machine-learning\pytorch\knowledge_base\knowledge_base"

files = ["01_overview.md", "02_tasks.md", "03_submission.md", "04_faq.md"]
all_content={}
for file_name in files:
    file_path = os.path.join(knowledge_base_path, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        all_content[file_name] = content
    print(f"已读取文件：{file_name}，字符数：{len(content)}")
print('\n')
print("第二阶段-划分文本")
chunk_size = 500
overlap = 50

all_chunks = []

for file_name, content in all_content.items():
    start = 0
    while start < len(content):
        end = min(start + chunk_size, len(content))
        chunk = content[start:end]
        all_chunks.append({
            "source": file_name,
            "text": chunk
        })
        next_start = end - overlap
        if next_start <= start:
            break
        start = next_start

print(f"共切分出 {len(all_chunks)} 个文本块\n")

# 打印前 10 个文本块
for i, chunk in enumerate(all_chunks[:10]):
    print(f"{i+1}. 来源：{chunk['source']}")
    # 只显示前 150 个字符，后面加 ...
    if len(chunk['text']) > 150:
        print(f"   文本块：{chunk['text'][:150]}...")
    else:
        print(f"   文本块：{chunk['text']}")