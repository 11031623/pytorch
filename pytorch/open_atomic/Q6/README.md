# 构建一个基于本地资料的 RAG 问答程序
## 项目内容
- 本项目实现RAG问答程序，使用者提出问题，程序通过在本地知识库检索信息，然后调用模型api回答
## 环境配置

### 1. 创建虚拟环境

```bash
python -m venv .venv
```

### 2. 激活虚拟环境

**Windows:**
```bash
.venv\Scripts\activate
```
### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 为 `.env`，填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env`，填入真实 Key：

```text
DASHSCOPE_API_KEY=sk-你的真实Key
```
## 所用模型和平台

| 用途 | 模型 | 平台 |
| :--- | :--- | :--- |
| 文本向量化 | paraphrase-multilingual-MiniLM-L12-v2 | SentenceTransformers（本地） |
| 大模型生成 | qwen3.8-max | 阿里云百炼 |
## 程序局限性 
 由于划分文块的不同方式 会影响到模型去查询知识库的能力 无法获取到应有答案的信息 当前程序针对markdown格式做了针对化划分处理
# 思考题
1. 检索就是程序使用问题的向量去搜寻与之最相似的几个文本块向量，生成就是大模型依据问题和检索到的文本块向量生成答案
2. 大模型能容纳的文字容量是有限的，且过多token收费也很昂贵
3. 微调需要重新训练模型 调整参数 而rag只是模型调用外部知识库来推理问题
4. 如果没有严格约束 模型可能会编造虚假答案，或者说内容不存在