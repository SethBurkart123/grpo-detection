from mlx_lm import load, generate

model, tokenizer = load("abalogh/Qwen3-0.6B-Base-4bit")

prompt = "When I was walking"

#if tokenizer.chat_template is not None:
#    messages = [{"role": "user", "content": prompt}]
#    prompt = tokenizer.apply_chat_template(
#        messages, add_generation_prompt=True
#    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)