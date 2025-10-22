# -*- coding: utf-8 -*-
import torch
print("GPU是否可用:", torch.cuda.is_available())  # 输出True表示识别成功，False表示失败
print("GPU设备数量:", torch.cuda.device_count())
print("当前GPU名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")

import json

from transformers import AutoTokenizer, AutoModelForCausalLM

from data_prepare import samples

model_name = "E:/大学资料/大四上/大模型/lora-deepseek/deepseekr1-1.5b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForCausalLM.from_pretrained(model_name).tu("cuda")

# 制作数据集
with open("datasets.json","w",encoding="utf-8") as f:
    for s in samples:
        json_line = json.dumps(s,ensure_ascii=False)
        f.write(json_line+"\n")
    else:
        print("prepare data finished")

# 准备训练集和测试集
from datasets import load_dataset
dataset = load_dataset(path="json", data_files={"train": "datasets.json"}, split="train")
print("数据的数量:",len(dataset))

train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print(f"train dataset len:{len(train_dataset)}")
print(f"test dataset len:{len(eval_dataset)}")

print("---完成训练数据准备---")

#编写tokenizer处理工具
def tokenizer_function(many_samples):
    texts = [f"{prompt}\n{completion}"for prompt, completion in zip(many_samples["prompt"], many_samples["completion"])]
    tokens = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()

    return tokens
tokenized_train_dataset = train_dataset.map(tokenizer_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenizer_function, batched=True)

print("---完成tokenizing---")
#print(tokenized_train_dataset[0])

#量化设置
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=quantization_config,device_map="auto")
print("---完成量化模型加载---")

#LoRA微调设置
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    r=4,  # 降低秩以减少显存占用
    lora_alpha=8,  # 配合r调整
    lora_dropout=0.05, task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model,lora_config)
model.print_trainable_parameters()
print("---lora微调设置完毕---")

#设置训练参数
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./finetuned_models",
    num_train_epochs=5,  # 减少训练轮次
    per_device_train_batch_size=2,  # 减小批次大小
    gradient_accumulation_steps=16,  # 增加梯度累积抵消批次影响
    fp16=True,
    logging_steps=50,  # 降低日志频率
    eval_strategy="steps",
    save_steps=1,
    save_total_limit=1,
    eval_steps=20,  # 降低验证频率
    learning_rate=3e-5,
    logging_dir="./logs",
    run_name="deepseek-r1-finetune"
)

print("---训练参数设置完毕---")

#定义训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset
)

print("---开始训练---")
trainer.train()
print("---训练完成---")

# 保存模型
# 保存LoRA模型
save_path = "./saved_models"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("---LoRA模型已保存---")

# 保存全量模型
final_save_path = "./final_saved_path"

from peft import PeftModel
# 释放训练模型显存
del model, trainer
torch.cuda.empty_cache()

# 4bit量化加载base模型以减少显存占用
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, save_path)
model = model.merge_and_unload()

model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)

print("---全量模型已经保存---")