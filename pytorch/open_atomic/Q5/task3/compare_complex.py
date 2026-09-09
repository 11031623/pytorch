from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv(r"D:\Learning\machine-learning\pytorch\.env")
try:
    client = OpenAI(
       
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://llm-9tvuwidf1m142kyd.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )
    user_input=input('请输入校园信息：')
    prompt = f"""
   你是一个专业的校园信息处理助手。

你的任务是从用户提供的文本中提取关键信息，并严格按照以下格式输出。

输出格式（必须严格遵循）：
1. 一句话总结：<用一句话概括核心内容>
2. 关键信息：<列出所有重要信息点，用分号分隔>
3. 待办事项：<列出需要采取的行动，如果没有则填"无">

约束条件：
- 总结不超过30个字
- 关键信息必须包含时间、地点、对象、内容（如果有）
- 待办事项要具体、可执行
- 如果输入中没有明确提到某个字段，填写"未提及"

    输入内容：
    {user_input}
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