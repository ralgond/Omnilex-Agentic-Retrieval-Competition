import os
import json
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from sklearn.metrics import ndcg_score


def inference_demo(base_model_name: str, lora_dir: str):
    """演示如何加载 LoRA 权重进行推理"""
    tokenizer = AutoTokenizer.from_pretrained(lora_dir)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=1
    )
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()

    pairs = [
        ("什么是机器学习？", "机器学习是人工智能的一个分支，通过数据驱动算法让计算机自动学习。"),
        ("什么是机器学习？", "今天天气晴朗，适合出门散步。"),
    ]

    for query, passage in pairs:
        inputs = tokenizer(query, passage, return_tensors="pt",
                           max_length=512, truncation=True)
        with torch.no_grad():
            score = model(**inputs).logits.squeeze().item()
        print(f"Query : {query}")
        print(f"Passage: {passage[:40]}...")
        print(f"Score  : {score:.4f}\n")

if __name__ == "__main__":
    inference_demo("/root/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3", "../ft_data/lora_reranker_output")