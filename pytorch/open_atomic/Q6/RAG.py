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

#第三阶段
from sentence_transformers import SentenceTransformer
import numpy as np
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#文本转成向量
chunk_texts = [chunk["text"] for chunk in all_chunks]
chunk_embeddings = model.encode(chunk_texts)

query =input('请输入问题')
query_embedding = model.encode([query])

#  计算相似度
similarities = np.dot(chunk_embeddings, query_embedding.T).flatten() #点积越大，相似度越高
top_k_indices = np.argsort(similarities)[-3:][::-1]
print(f"问题：{query}\n")
print("最相关的资料块：")
for idx in top_k_indices:
    print(f"来源：{all_chunks[idx]['source']}")
    print(f"相似度：{similarities[idx]:.4f}")
    print(f"内容：{all_chunks[idx]['text'][:150]}...")

#引入大模型-第四阶段
from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv(r"D:\Learning\machine-learning\pytorch\.env")
try:
    client = OpenAI(
       
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://llm-9tvuwidf1m142kyd.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )
    context= ""
    for idx in top_k_indices:
     context += f"[来源：{all_chunks[idx]['source']}]\n{all_chunks[idx]['text']}\n\n"
    prompt = f"""
    请根据以下资料回答问题：
    {context}

    问题如下：
    {query}
    要求：
    如果资料中有相关答案，请严格遵循资料内容给出答案
    当资料中没有足够信息回答问题时，必须明确输出“资料中未提及”或语义相同的表述，不得编造答案
    在最终回答中的末尾列出使用的来源文件名并且如果参考了多个来源文件 需要全部列出
    """
    completion = client.chat.completions.create(
        model="qwen3.8-max",  
        messages=[
            {'role': 'system', 'content': '你是校园信息助手。'},
            {'role': 'user', 'content': prompt}
        ]
    )
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"错误信息：{e}")