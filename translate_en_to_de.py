import os
from openai import OpenAI
import pandas as pd

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key="sk-71d0d11bec274377b20a14c5a93f2f0c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

test_df = pd.read_csv("Omnilex-Agentic-Retrieval-Competition/data/test.csv")
test_df['query_en'] = test_df['query']

query_de_l = []

for idx, q in enumerate(test_df['query_en']):
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"你是一名精通英语和德语的法律顾问，请将下面文字翻译成德语: {q}"},
        ]
    )
    # print(completion.choices[0].message.content)
    query_de_l.append(completion.choices[0].message.content)

    print(idx)

test_df['query'] = query_de_l

test_df.to_csv("Omnilex-Agentic-Retrieval-Competition/data/test.csv", index=False)