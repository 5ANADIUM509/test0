from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 加载已经微调的模型
final_save_path = "final_saved_path"
model = AutoModelForCausalLM.from_pretrained(final_save_path)
tokenizer = AutoTokenizer.from_pretrained(final_save_path)

# 构建推理pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "tell me some singing skills"

generated_texts = pipe(
    prompt,
    max_new_tokens=300,
    num_return_sequences=1,
    truncation=True
)

print("开始回答:----", generated_texts[0]["generated_text"])