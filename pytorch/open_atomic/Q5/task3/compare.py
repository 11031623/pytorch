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
   请总结以下内容

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