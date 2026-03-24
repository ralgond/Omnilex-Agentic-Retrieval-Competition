"""
LoRA 微调 bge-reranker-v2-m3 完整脚本
用途：对 query-passage 对进行相关性打分的重排序模型微调

数据集格式：
    {"query": "...", "passage": "...", "label": 1, "type": "random"}
    label=1 为正样本，label=0 时 type 区分难度：random / medium / hard

课程学习策略（Curriculum Learning）：
    epoch 1 → 正样本 + random  负样本           (pos_random)
    epoch 2 → 正样本 + random + medium 负样本   (pos_random_medium)
    epoch 3 → 正样本 + random + medium + hard  (pos_random_medium_hard)

依赖安装：
    pip install transformers peft datasets torch accelerate bitsandbytes scikit-learn
"""

import os
import json
import argparse
import numpy as np
from typing import Optional, List

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerControl,
    TrainerState,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from sklearn.metrics import ndcg_score


# ─────────────────────────────────────────────
# 1. epoch → 允许负样本类型 的映射
# ─────────────────────────────────────────────

EPOCH_NEG_TYPES: dict[int, List[str]] = {
    1: ["random"],                       # pos + random
    2: ["random", "medium"],             # pos + random + medium
    3: ["random", "medium", "hard"],     # pos + random + medium + hard
    4: ["random", "medium", "hard"],     # pos + random + medium + hard
}


# ─────────────────────────────────────────────
# 2. 数据集
# ─────────────────────────────────────────────

class RerankerDataset(Dataset):
    """
    JSONL 格式（每行一条）：
        {"query": "...", "passage": "...", "label": 1, "type": "random"}

    label=1：正样本（type 字段可为空或任意值，始终保留）
    label=0：负样本，根据 allowed_neg_types 过滤：
        - "random" : 随机负样本（最简单）
        - "medium" : 中等难度负样本
        - "hard"   : 困难负样本（最难）

    allowed_neg_types=None 时不过滤，保留全部负样本。
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        allowed_neg_types: Optional[List[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data: List[dict] = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                # 正样本始终保留；负样本按 type 过滤
                if item["label"] == 1:
                    self.data.append(item)
                elif allowed_neg_types is None or item.get("type") in allowed_neg_types:
                    self.data.append(item)

        pos = sum(1 for d in self.data if d["label"] == 1)
        neg = len(self.data) - pos
        print(
            f"[Dataset] 加载完成：共 {len(self.data)} 条"
            f"（正样本 {pos}，负样本 {neg}，"
            f"允许负样本类型：{allowed_neg_types}）"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        # bge-reranker 使用 [CLS] query [SEP] passage [SEP] 格式
        encoding = self.tokenizer(
            item["query"],
            item["passage"],
            max_length=self.max_length,
            padding=False,       # 交给 DataCollator 处理
            truncation=True,
            return_tensors=None,
        )
        encoding["labels"] = float(item["label"])
        return encoding


# ─────────────────────────────────────────────
# 3. 评估指标
# ─────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    scores = logits.squeeze(-1) if logits.ndim == 2 else logits

    preds = (scores > 0).astype(int)
    labels_bin = (labels > 0.5).astype(int)
    accuracy = (preds == labels_bin).mean()

    try:
        ndcg = ndcg_score([labels], [scores])
    except Exception:
        ndcg = 0.0

    return {"accuracy": float(accuracy), "ndcg": float(ndcg)}


# ─────────────────────────────────────────────
# 4. LoRA 模型构建
# ─────────────────────────────────────────────

def build_lora_model(
    model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    torch_dtype=torch.float32,
):
    """加载基座模型并注入 LoRA 适配器"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # bf16 加载节省显存且稳定；fp16 训练时仍用 fp32 加载，Trainer 自动处理
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
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


from transformers import TrainerCallback

from transformers import TrainerCallback

class CurriculumCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer   # ✅ 手动注入

    def on_epoch_begin(self, args, state, control, **kwargs):
        raw_epoch = state.epoch if state.epoch is not None else 0
        epoch_num = round(raw_epoch) + 1

        max_epoch = max(EPOCH_NEG_TYPES.keys())
        allowed_types = EPOCH_NEG_TYPES.get(epoch_num, EPOCH_NEG_TYPES[max_epoch])

        print(
            f"\n{'='*60}\n"
            f"[Callback] ▶ Epoch {epoch_num} 开始\n"
            f"           负样本类型 = {allowed_types}\n"
            f"{'='*60}"
        )

        # ✅ 用 self.trainer
        self.trainer.train_dataset = RerankerDataset(
            data_path=self.trainer._full_data_path,
            tokenizer=self.trainer._base_tokenizer,
            max_length=self.trainer._base_max_length,
            allowed_neg_types=allowed_types,
        )

        # ✅ 强制刷新 dataloader（关键）
        self.trainer._train_dataloader = None
        
# ─────────────────────────────────────────────
# 5. 自定义 Trainer：课程学习 + 动态采样 + BCE Loss
# ─────────────────────────────────────────────

class BCERerankerTrainer(Trainer):

    SAMPLES_PER_EPOCH: int = 50_000

    def __init__(
        self,
        full_data_path: str,
        base_tokenizer,
        base_max_length: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._full_data_path   = full_data_path
        self._base_tokenizer   = base_tokenizer
        self._base_max_length  = base_max_length
        self._current_epoch_int = -1  # 记录上次重建时的 epoch，避免重复重建

    def get_train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset
        n = len(dataset)
        k = min(self.SAMPLES_PER_EPOCH, n)
    
        indices = torch.randperm(n)[:k].tolist()
        subset = Subset(dataset, indices)
    
        print(f"[Trainer] 当前数据集大小={n}，本 epoch 采样={k} 条")
    
        return DataLoader(
            subset,
            batch_size=self._train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=True,
        )
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits  = outputs.logits.squeeze(-1)
        loss    = torch.nn.BCEWithLogitsLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────
# 6. 主训练流程
# ─────────────────────────────────────────────

def train(args):
    # 6.1 构建 LoRA 模型
    # bge-reranker-v2-m3 基于 XLM-RoBERTa，注意力层名称：
    #   标准 Roberta: query / key / value
    target_modules = args.target_modules or ["query", "key", "value"]

    load_dtype = torch.bfloat16 if args.bf16 else torch.float32

    model, tokenizer = build_lora_model(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        torch_dtype=load_dtype,
    )

    # 6.2 初始数据集（epoch 1 配置）
    train_dataset = RerankerDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        allowed_neg_types=EPOCH_NEG_TYPES[1],
    )
    eval_dataset = (
        RerankerDataset(
            data_path=args.eval_data,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        if args.eval_data
        else None
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # 6.3 训练参数
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
        # 精度优先级：命令行显式指定 > 自动检测 GPU 能力
        bf16=(
            args.bf16
            if (args.bf16 or args.fp16)
            else (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        ),
        fp16=(
            args.fp16
            if (args.bf16 or args.fp16)
            else (torch.cuda.is_available() and not torch.cuda.is_bf16_supported())
        ),
        logging_steps=50,
        evaluation_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="ndcg",
        greater_is_better=True,
        report_to="none",       # 改为 "wandb" 可接入 WandB
        dataloader_num_workers=4,
    )

    # 6.4 实例化自定义 Trainer
    BCERerankerTrainer.SAMPLES_PER_EPOCH = args.samples_per_epoch

    trainer = BCERerankerTrainer(
        # ↓ 课程学习专用参数
        full_data_path=args.train_data,
        base_tokenizer=tokenizer,
        base_max_length=args.max_length,
        # ↓ 标准 Trainer 参数
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_dataset else None,
    )

    # ⚠️ 这里把 trainer 传进去
    trainer.add_callback(CurriculumCallback(trainer))

    trainer.train()

    # 6.5 保存 LoRA 权重（仅适配器，体积极小）
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ LoRA 适配器已保存至：{args.output_dir}")


# ─────────────────────────────────────────────
# 7. 推理示例
# ─────────────────────────────────────────────

def inference_demo(base_model_name: str, lora_dir: str):
    """演示如何加载已保存的 LoRA 权重进行推理"""
    tokenizer  = AutoTokenizer.from_pretrained(lora_dir)
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
        inputs = tokenizer(
            query, passage, return_tensors="pt", max_length=512, truncation=True
        )
        with torch.no_grad():
            score = model(**inputs).logits.squeeze().item()
        print(f"Query  : {query}")
        print(f"Passage: {passage[:40]}...")
        print(f"Score  : {score:.4f}\n")


# ─────────────────────────────────────────────
# 8. 命令行入口
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA 微调 bge-reranker-v2-m3（课程学习版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 模型路径
    parser.add_argument(
        "--model_name",
        default="/root/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3",
        help="基座模型路径或 HuggingFace Hub ID",
    )
    parser.add_argument(
        "--output_dir",
        default="../ft_data/lora_reranker_output",
        help="LoRA 适配器保存目录",
    )

    # 数据路径
    parser.add_argument(
        "--train_data",
        required=True,
        help=(
            "训练集 JSONL 路径（包含所有 type 的完整数据集）\n"
            "格式：{\"query\":\"...\", \"passage\":\"...\", \"label\":0/1, \"type\":\"random/medium/hard\"}"
        ),
    )
    parser.add_argument(
        "--eval_data",
        default=None,
        help="验证集 JSONL 路径（可选，格式同训练集）",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=784,
        help="tokenizer 最大序列长度",
    )

    # LoRA 超参
    parser.add_argument("--lora_r",       type=int,   default=16,  help="LoRA 秩")
    parser.add_argument("--lora_alpha",   type=int,   default=32,  help="LoRA alpha 缩放系数")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=None,
        help="注入 LoRA 的层名，默认 ['query', 'key', 'value']",
    )

    # 训练超参
    parser.add_argument("--epochs",     type=int,   default=3,    help="训练总 epoch 数（建议 ≥3 以充分利用课程学习）")
    parser.add_argument("--batch_size", type=int,   default=8,    help="每卡 batch size")
    parser.add_argument("--grad_accum", type=int,   default=4,    help="梯度累积步数")
    parser.add_argument("--lr",         type=float, default=2e-4, help="学习率")
    parser.add_argument(
        "--samples_per_epoch",
        type=int,
        default=50_000,
        help="每个 epoch 从当前数据集随机采样的最大条数",
    )

    # 训练精度（二选一，不传则自动检测 GPU 能力）
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument(
        "--bf16",
        action="store_true",
        help="使用 BF16 训练（推荐安培架构：A100 / 3090 / 4090）",
    )
    precision.add_argument(
        "--fp16",
        action="store_true",
        help="使用 FP16 训练（适合 V100 / T4 等不支持 BF16 的卡）",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)