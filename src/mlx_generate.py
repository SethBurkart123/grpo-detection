from mlx_parallm.utils import load, batch_generate

model, tokenizer = load("mlx-community/dolphin-2.9.1-llama-3-8b-8bit")
prompts = ["prompt_0", ..., "prompt_k"]
responses = batch_generate(model, tokenizer, prompts=prompts_raw[:10], max_tokens=100, verbose=True, format_prompts=True, temp=0.0)
