from transformers import AutoTokenizer

# 测试tokenizer是否能正常加载
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print("Tokenizer loaded successfully!")