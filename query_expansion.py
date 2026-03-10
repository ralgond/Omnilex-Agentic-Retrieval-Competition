import json

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
    prompt=f'''You are a legal research expert.

Your task is to convert an English legal question into multiple German
search queries that maximize recall in a legal search system.

The queries MUST use different legal perspectives.

Generate 6 queries with different styles:

1. Direct legal issue query
2. Query focusing on legal rights
3. Query focusing on legal obligations
4. Query mentioning possible legal remedies
5. Query using formal legal terminology
6. Query using statute-style keywords

Guidelines:
- Use German legal terminology
- Prefer keyword style
- Avoid repeating the same words
- Each query should introduce new legal terms

Output:
Return a JSON list with 6 queries. for example:

English query:
{query}'''

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        top_p=0.9
    )
    
    expand_query_json = completion.choices[0].message.content

    expand_query_json = expand_query_json.strip("```").strip("json")

    expand_query_l = json.loads(expand_query_json)

    for expand_query in expand_query_l:
        query_id_l.append(query_id)
        query_l.append(expand_query)

test_003 = pd.DataFrame({'query_id':query_id_l, 'query':query_l})
test_003.to_csv("data/test_rewrite_003.csv", index=False)