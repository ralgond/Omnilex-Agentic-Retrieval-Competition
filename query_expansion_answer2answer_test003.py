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
    prompt=f'''你是一名精通德国法律的法律研究助手。

任务：
用户会提供一个英文法律问题。请你根据问题，用德语写出一段可能出现在德国法院判决书或法律论证中的“法律分析”。这段文本将用于法律检索，因此需要尽可能包含相关的法律概念、法律术语和法律关系。

要求：
1. 输出语言必须是德语。
2. 内容应像德国法院判决书中的“法院论证（court consideration）”或法律评论。
3. 可以合理假设事实，并进行法律分析。
4. 尽量使用德国法律常见术语，例如：
   - Anspruch
   - Voraussetzung
   - Rechtsfolge
   - Anspruchsgrundlage
   - Schadensersatz
   - Verletzung
   - Pflicht
   - Haftung
   - Vertrag
5. 可以提到可能适用的法律规范，例如：
   - BGB
   - StGB
   - UrhG
   - GG
6. 文本长度控制在 60–100 字。
7. 不要解释任务，也不要输出任何额外说明，只输出德语法律分析文本。

用户问题：
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
    
    expand_query = completion.choices[0].message.content

    query_id_l.append(query_id)
    query_l.append(expand_query)

test_003 = pd.DataFrame({'query_id':query_id_l, 'query':query_l})
test_003.to_csv("data/test_rewrite_003.csv", index=False)