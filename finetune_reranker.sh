#!/bin/sh

torchrun --nproc_per_node 1 \
	-m FlagEmbedding.finetune.reranker.encoder_only.base \
	--model_name_or_path /root/.cache/modelscope/hub/models/BAAI/bge-reranker-v2-m3 \
    --cache_dir ./cache/model \
    --train_data ./ft_data/train.jsonl \
    --cache_path ./cache/data \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
	--output_dir ./ft_data/bge-reranker-v2-m3-finetune \
    --overwrite_output_dir \
    --learning_rate 2e-5 \
    --fp16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --weight_decay 0.01 \
    --logging_steps 1 \
    --save_steps 1000 \
    --deepspeed ./ft_data/ds_stage0.json \
    --trust_remote_code True \
    --save_safetensors False