from modelscope import snapshot_download

# model_dir = snapshot_download('Xorbits/Mistral-7B-Instruct-v0.2-GGUF')
# model_dir = snapshot_download('Qwen2.5-32B Q4_K_M', cache_dir='/root/autodl-tmp')
# model_dir = snapshot_download('BAAI/bge-m3')
# model_dir = snapshot_download('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
# model_dir = snapshot_download('BAAI/bge-reranker-v2-m3')
model_dir = snapshot_download('BAAI/bge-reranker-v2-gemma')
print(f"{model_dir}")