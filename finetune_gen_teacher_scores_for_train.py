
import re
import os
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import json

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key="sk-71d0d11bec274377b20a14c5a93f2f0c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key="sk-ddff266fc036492b8854d5411ff0e5d5",
    base_url="https://api.deepseek.com",
)


def get_score_from_qwen(query, passage):
    prompt=f'''你是一个专业的德语法律文本相关性评估专家。你的任务是评估一段法律文本（passage）与检索问题（query）之间的语义距离。

## 输入

**Query（检索问题）：**
{query}

**Passage（待评估文本）：**
{passage}

## 输出要求

请严格按照以下 JSON 格式输出，不要添加任何额外内容：

{{
  "score": <0到1之间的浮点数，保留两位小数>
}}'''
    
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        # model="qwen-plus",
        model='deepseek-chat',
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    score = float(json.loads(completion.choices[0].message.content.strip())['score'])
    return score


def write_to_jsonl(jsonl):
    with open("./ft_data/train_with_scores.jsonl", 'w', encoding='utf-8') as f:
        for item in jsonl:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

d_l = []
with open("./ft_data/train.jsonl", 'r', encoding='utf-8') as inf:
    lines = []
    for line in inf:
        lines.append(line.strip())

    for idx, line in tqdm(enumerate(lines), total=len(lines)):
        d = json.loads(line.strip())
        query = d['query']
        poss = d['pos']
        negs = d['neg']

        pos_scores = []
        neg_scores = []

        for pos in poss:
            pos_scores.append(get_score_from_qwen(query, pos))

        for neg in negs:
            neg_scores.append(get_score_from_qwen(query, neg))

        print(pos_scores)
        print(neg_scores)
        d['pos_scores'] = pos_scores
        d['neg_scores'] = neg_scores

        d_l.append(d)

        write_to_jsonl(d_l)


        
        


