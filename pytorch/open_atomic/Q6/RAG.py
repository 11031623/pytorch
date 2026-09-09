import os

knowledge_base_path = r"D:\Learning\machine-learning\pytorch\knowledge_base\knowledge_base"

files = ["01_overview.md", "02_tasks.md", "03_submission.md", "04_faq.md"]

for file_name in files:
    file_path = os.path.join(knowledge_base_path, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    print(f"已读取文件：{file_name}，字符数：{len(content)}")