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
    prompt=f'''你是一名法律检索专家，负责为德语法律数据库检索竞赛准备搜索查询。

## 输入
你将收到一段英文 query，包含：
- 一段事实陈述（案例场景描述，即上下文）
- 一个或多个关于该上下文的问题

## 任务
将输入拆解为多个子 query——每个问题对应一个子 query。对每个子 query 执行以下步骤：

1. **组合**：将与该问题相关的上下文部分与对应问题合并。
2. **翻译**：将所有内容翻译成德语。
3. **激进压缩**：使用精确的德语法律术语重新表述，删除叙事性填充词、连接词和冗余事实，仅保留具有法律意义的核心事实和核心法律问题。
4. **优先使用法律术语**：尽可能使用专业法律用语而非日常表达（例如：用"Schadensersatzpflicht"而非"Pflicht, Schäden zu ersetzen"；用"culpa in contrahendo"而非"Verschulden beim Vertragsabschluss"；用"dingliche Rechte"而非"Eigentumsrechte an Sachen"）。
5. **目标长度**：上下文不超过 3 句话，问题不超过 1 句话，每个子 query 总计不超过约 80 个德语单词。

## 输出格式
返回一个 JSON 数组，每个元素包含两个字段：
- "kontext"：压缩后的德语法律上下文（Sachverhalt，即案情摘要）
- "frage"：压缩后的德语法律问题

输出结构示例：
[
  {{
    "kontext": "A schließt mit B einen Kaufvertrag über ein Grundstück. A zahlt den Kaufpreis, B verweigert die Übereignung.",
    "frage": "Besteht ein Anspruch auf Übereignung nach § 433 Abs. 1 BGB oder Schadensersatz wegen Nichterfüllung?"
  }},
  {{
    "kontext": "...",
    "frage": "..."
  }}
]

## 规则
- 不得虚构原始 query 中不存在的事实。
- 不得将两个问题合并为一个子 query。
- 优先使用成熟的德国法律教义概念（如 Rechtsfortbildung、Vertrauenshaftung、Anfechtbarkeit、Sittenwidrigkeit 等），而非字面翻译。
- 若某问题涉及特定法条，且相关条款编号已知，则在"frage"中注明对应 §§；否则省略。
- 仅输出合法的 JSON，不附任何解释，不使用 Markdown 代码块。

## 输入 query
{query}'''

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    
    expand_query_json = completion.choices[0].message.content

    expand_query_json = expand_query_json.strip("```").strip("json")

    expand_query_l = json.loads(expand_query_json)

    for expand_query in expand_query_l:
        query_id_l.append(query_id)
        query_l.append(expand_query['kontext'] + "\n\n" + expand_query["frage"])

test_005 = pd.DataFrame({'query_id':query_id_l, 'query':query_l})
test_005.to_csv("data/test_rewrite_005.csv", index=False)