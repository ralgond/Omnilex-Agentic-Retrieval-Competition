import json
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
    prompt=f'''你是德国法律检索专家，精通德国法律术语和司法表达方式。

任务：
将用户提供的英文法律问题改写成多个用于检索德国法律文献（法条、法院裁判理由 court considerations）的德语查询句。

改写要求：

1. 生成 4-6 条不同的德语查询
2. 每条查询必须保持与原问题相同的法律含义
3. 不同查询应从不同角度表达：
   - 法律术语表达（juristische Fachsprache）
   - 事实描述表达（Sachverhalt）
   - 关键词检索表达（适合BM25）
   - 法条/裁判表达（Norm / Urteil表达）
4. 尽量使用德国法律常见术语，例如：
   Anspruch, Verpflichtung, Haftung, Schadensersatz, Unterlassung,
   Pflichtverletzung, Vertrag, Eigentum, Nutzung, Verletzung
5. 查询句长度不超过 80 个词
6. 不要解释，不要编号，不要添加多余文本
7. 仅输出 JSON 数组

输入问题：
{query}

输出格式：
[
"query1",
"query2",
"query3",
"query4",
"query5"
]'''

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

test_004 = pd.DataFrame({'query_id':query_id_l, 'query':query_l})
test_004.to_csv("data/test_rewrite_004.csv", index=False)