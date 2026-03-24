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

class Reranker:
    def __init__(self, base_model_name: str, lora_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(lora_dir)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=1
        )
        self.model = PeftModel.from_pretrained(self.base_model, lora_dir)
        self.model.to("cuda")

    def compute_score(self, pairs):
        self.model.eval()
        idx_l = []
        for idx, (query, passage) in enumerate(pairs):
            inputs = self.tokenizer(query, passage, return_tensors="pt", max_length=1024, truncation=True)
            inputs.to("cuda")
            with torch.no_grad():
                score = self.model(**inputs).logits.squeeze().item()
                idx_l.append((idx, score))
        return sorted(idx_l, key=lambda x: x[1], reverse=True)
        
        
def inference_demo(base_model_name: str, lora_dir: str):
    """演示如何加载 LoRA 权重进行推理"""
    tokenizer = AutoTokenizer.from_pretrained(lora_dir)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=1
    )
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()

    

    

if __name__ == "__main__":
    r = Reranker("/root/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3", "../ft_data/lora_reranker_output")
    ret = r.compute_score(pairs = [
        ("什么是机器学习？", "机器学习是人工智能的一个分支，通过数据驱动算法让计算机自动学习。"),
        ("什么是机器学习？", "今天天气晴朗，适合出门散步。"),
    ])
    print(ret)