import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 
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
print("\n")
print("第三阶段")
from sentence_transformers import SentenceTransformer
import numpy as np
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#文本转成向量
chunk_texts = [chunk["text"] for chunk in all_chunks]
chunk_embeddings = model.encode(chunk_texts)

query = "招新题发布日期是什么？"
query_embedding = model.encode([query])

#  计算相似度
similarities = np.dot(chunk_embeddings, query_embedding.T).flatten() #点积越大，相似度越高
top_k_indices = np.argsort(similarities)[-3:][::-1]
print(f"问题：{query}\n")
print("最相关的资料块：")
for idx in top_k_indices:
    print(f"来源：{all_chunks[idx]['source']}")
    print(f"内容：{all_chunks[idx]['text'][:150]}...")