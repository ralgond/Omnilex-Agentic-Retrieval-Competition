from FlagEmbedding import FlagReranker
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

import os
os.environ["ACCELERATE_USE_FSDP"] = "false"
os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"

model_name = "/root/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3"

# 加载模型
model = FlagReranker(model_name, use_fp16=True)

# LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "key", "value"],  # 关键
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

model.model = get_peft_model(model.model, lora_config)

# 数据
dataset = load_dataset("json", data_files={"train": "./ft_data/train.jsonl"})

# 训练参数
training_args = TrainingArguments(
    output_dir="./ft_data/reranker_lora",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=2,
    logging_steps=50,
    save_steps=500,
    fp16=True,
    dataloader_num_workers=0,  # 🔑 避免 Arrow Dataset 多进程空 batch
    remove_unused_columns=False # 🔑 保留全部列，避免 Trainer 自动删除造成 size=0
)

print(len(dataset["train"]))

# 2️⃣ 定义 tokenize_fn
def tokenize_fn(batch):
    # batch 是 dict，包含 query / pos / neg
    query_enc = model.tokenizer(batch["query"], padding=True, truncation=True)
    pos_enc = model.tokenizer(batch["pos"], padding=True, truncation=True)
    neg_enc = model.tokenizer(batch["neg"], padding=True, truncation=True)

    return {
        "input_ids": query_enc["input_ids"],
        "attention_mask": query_enc["attention_mask"],
        "pos_input_ids": pos_enc["input_ids"],
        "pos_attention_mask": pos_enc["attention_mask"],
        "neg_input_ids": neg_enc["input_ids"],
        "neg_attention_mask": neg_enc["attention_mask"],
    }

# 3️⃣ 用 map tokenization
dataset["train"] = dataset["train"].map(tokenize_fn, batched=True)

# 4️⃣ 转成 torch tensor 格式
dataset["train"].set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "pos_input_ids",
        "pos_attention_mask",
        "neg_input_ids",
        "neg_attention_mask"
    ]
)

trainer = Trainer(
    model=model.model,
    args=training_args,
    train_dataset=dataset["train"]
)

trainer.train()