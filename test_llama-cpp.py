from llama_cpp import Llama

llm = Llama(
    model_path="/root/.cache/modelscope/hub/models/Xorbits/Mistral-7B-Instruct-v0___2-GGUF/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_gpu_layers=-1,  # 全部层加载到 GPU
    verbose=True
)