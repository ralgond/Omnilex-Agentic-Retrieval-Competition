
import re
import os
from openai import OpenAI
import pandas as pd
from tqdm import tqdm

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key="sk-71d0d11bec274377b20a14c5a93f2f0c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

rewrite_l = []

query_id_l = []
query_l = []

test_df = pd.read_csv("./data/test_rewrite_001.csv")
for query_id, query in tqdm(zip(test_df['query_id'].tolist(), test_df['query'].tolist()), total=len(test_df)):
    query_id_l.append(query_id)
    query_l.append(query)
    prompt=f''' # Role
                你是一位专业的德语语言专家。
                
                # Task
                请对下面的【德语文本】进行改写。
                
                # Constraints
                1. 保持原意不变，核心信息不能丢失。
                2. 主要通过**替换同义词**来改变表达形式。
                3. 语法必须正确，符合德语习惯。
                4. 不要翻译成中文，输出必须仍然是德语。
                5. 请提供 3 个不同的改写版本。
                
                # Input Text
                {query}
                
                # Output Format
                版本 1: ...
                版本 2: ...
                版本 3: ...'''

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        top_p=0.9
    )
    
    l = [p.strip() for p in re.split(r"版本 \d:", completion.choices[0].message.content) if len(p.strip()) > 0]

    for p in l:
        query_id_l.append(query_id)
        query_l.append(p)

test_002 = pd.DataFrame({'query_id':query_id_l, 'query':query_l})
test_002.to_csv("data/test_rewrite_002.csv", index=False)

