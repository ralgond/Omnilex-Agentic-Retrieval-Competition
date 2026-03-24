"""
LoRA 微调 bge-reranker-v2-m3 完整脚本
用途：对 query-passage 对进行相关性打分的重排序模型微调

依赖安装：
    pip install transformers peft datasets torch accelerate bitsandbytes scikit-learn
"""

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


# ─────────────────────────────────────────────
# 1. 数据集
# ─────────────────────────────────────────────

class RerankerDataset(Dataset):
    """
    期望的 JSONL 格式（每行一条）：
    {"query": "...", "passage": "...", "label": 1}   # 1=相关, 0=不相关
    也支持软标签，如 label=0.7
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["query"]
        passage = item["passage"]
        label = float(item["label"])

        # bge-reranker 使用 [CLS] query [SEP] passage [SEP] 格式
        encoding = self.tokenizer(
            query,
            passage,
            max_length=self.max_length,
            padding=False,          # 交给 DataCollator 处理
            truncation=True,
            return_tensors=None,
        )
        encoding["labels"] = label
        return encoding


# ─────────────────────────────────────────────
# 2. 评估指标
# ─────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits shape: (N,) or (N, 1)
    scores = logits.squeeze(-1) if logits.ndim == 2 else logits

    # 二值化预测
    preds = (scores > 0).astype(int)
    labels_bin = (labels > 0.5).astype(int)

    accuracy = (preds == labels_bin).mean()

    # 近似 NDCG（把所有样本当作一个查询列表，仅供参考）
    try:
        ndcg = ndcg_score([labels], [scores])
    except Exception:
        ndcg = 0.0

    return {
        "accuracy": float(accuracy),
        "ndcg": float(ndcg),
    }


# ─────────────────────────────────────────────
# 3. LoRA 配置
# ─────────────────────────────────────────────

def build_lora_model(model_name: str, lora_r: int, lora_alpha: int,
                     lora_dropout: float, target_modules: List[str],
                     torch_dtype=torch.float32):
    """加载模型并注入 LoRA 适配器"""

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # torch_dtype 与训练精度保持一致，节省显存。
    # 注意：不能用 FP16 加载 + fp16 训练，会触发 GradScaler unscale 报错。
    # BF16 加载 + bf16 训练 或 FP32 加载 + fp16 训练 均可正常工作。
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        torch_dtype=torch_dtype,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,  # 见下方说明
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


import math
from torch.utils.data import DataLoader, Subset

class BCERerankerTrainer(Trainer):
    
    SAMPLES_PER_EPOCH = 50_000   # ← 每 epoch 采样数，可改成参数传入

    def get_train_dataloader(self):
        """每次调用（即每个 epoch 开始前）都重新随机采样"""
        dataset = self.train_dataset
        n = len(dataset)
        k = min(self.SAMPLES_PER_EPOCH, n)          # 不超过总量
        indices = torch.randperm(n)[:k].tolist()    # 无放回随机采样
        subset = Subset(dataset, indices)

        return DataLoader(
            subset,
            batch_size=self._train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=True,                            # subset 内部再 shuffle
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss
        
# ─────────────────────────────────────────────
# 4. 主训练流程
# ─────────────────────────────────────────────

def train(args):
    # 4.1 构建模型
    # bge-reranker-v2-m3 基于 XLM-RoBERTa，注意力层名称为：
    #   query, key, value -> q_proj, k_proj, v_proj (MistralLike)
    # 或（标准 Roberta）: query, key, value
    target_modules = args.target_modules or ["query", "key", "value"]

    # bf16 加载节省显存且训练稳定；fp16 模式仍用 FP32 加载避免 GradScaler 报错
    if args.bf16:
        load_dtype = torch.bfloat16
    else:
        load_dtype = torch.float32   # fp16 训练也用 FP32 加载，Trainer 自动处理转换

    model, tokenizer = build_lora_model(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        torch_dtype=load_dtype,
    )

    # 4.2 数据集
    train_dataset = RerankerDataset(args.train_data, tokenizer, args.max_length)
    eval_dataset  = RerankerDataset(args.eval_data,  tokenizer, args.max_length) \
                    if args.eval_data else None

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # 4.3 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        # 精度优先级：命令行显式指定 > 自动检测
        bf16=args.bf16 if (args.bf16 or args.fp16) else (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        fp16=args.fp16 if (args.bf16 or args.fp16) else (torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        logging_steps=50,
        evaluation_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="ndcg",
        greater_is_better=True,
        report_to="none",       # 改为 "wandb" 可接入 WandB
        dataloader_num_workers=4,
    )

    # 4.4 自定义 Trainer，使用 BCE Loss
    # class BCERerankerTrainer(Trainer):
    #     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    #         labels = inputs.pop("labels").float()
    #         outputs = model(**inputs)
    #         logits = outputs.logits.squeeze(-1)   # (batch_size,)
    #         loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
    #         return (loss, outputs) if return_outputs else loss

    BCERerankerTrainer.SAMPLES_PER_EPOCH = args.samples_per_epoch

    trainer = BCERerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_dataset else None,
    )

    trainer.train()

    # 4.5 保存 LoRA 权重（仅适配器，体积极小）
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ LoRA 适配器已保存至：{args.output_dir}")


# ─────────────────────────────────────────────
# 5. 推理示例
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# 6. 命令行入口
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA 微调 bge-reranker-v2-m3")

    # 模型
    parser.add_argument("--model_name", default="/root/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3")
    parser.add_argument("--output_dir", default="../ft_data/lora_reranker_output")

    # 数据
    parser.add_argument("--train_data", required=True, help="训练集 JSONL 路径")
    parser.add_argument("--eval_data",  default=None,  help="验证集 JSONL 路径（可选）")
    parser.add_argument("--max_length", type=int, default=784)

    # LoRA 超参
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", nargs="+", default=None,
                        help="注入 LoRA 的层名，默认 ['query','key','value']")

    # 训练超参
    parser.add_argument("--epochs",     type=int,   default=2)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--grad_accum", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=2e-4)

    # 训练精度（二选一，不传则自动检测）
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true",
                           help="使用 BF16 训练（推荐安培架构：A100/3090/4090）")
    precision.add_argument("--fp16", action="store_true",
                           help="使用 FP16 训练（适合 V100/T4 等不支持 BF16 的卡）")
 
    parser.add_argument("--samples_per_epoch", type=int, default=50_000,
                    help="每个 epoch 从训练集中随机采样的样本数")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)