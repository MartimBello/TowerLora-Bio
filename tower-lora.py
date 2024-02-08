from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--source", required=True)
parser.add_argument("-t", "--target", required=True)
arguments = parser.parse_args()


model_path = "Unbabel/TowerInstruct-7B-v0.1"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    #quantization_config=quantization_config,
    use_cache=False,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)

model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained('Unbabel/TowerInstruct-7B-v0.1')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

breakpoint()
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=32,
    bias="none",
    task_type="CAUSAL_LM"
)

#model = prepare_model_for_kbit_training(model)
breakpoint()
args = TrainingArguments(
    output_dir="/mnt/data/martimbelo/models/TowerInstruct-LoRA-Biomedical",
    # just for demo purposes
    num_train_epochs=4,
    # trying to max out resources on colab
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    logging_steps=25,
    save_strategy="steps",
    save_steps=500,
    learning_rate=7e-6,
    fp16=True,
    weight_decay=0.01,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    disable_tqdm=False,
    save_safetensors=True
)

model = get_peft_model(model, peft_config)
breakpoint()
def formatting_prompts_func(example):
    output_texts = []
    lang_codes = {"en": "English", "ru": "Russian", "pt": "Portuguese", "it": "Italian", "de": "German", "fr": "French"}
    for i in range(len(example['src'])):
        text = f"<|im_start|>user\n{lang_codes[arguments.source]}: {example['src'][i]}\n{lang_codes[arguments.target]}: <|im_end|>\n<|im_start|>assistant\n{example['tgt'][i]}"
        output_texts.append(text)
    return output_texts

response_template = "<|im_start|>assistant\n"

collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=False)

dataset = load_dataset('json', data_files=f'/mnt/data/martimbelo/TowerEval/data/raw_data/mt/wmt21bio.{arguments.source}-{arguments.target}/dev.jsonl', split='train')

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=512
)
 
breakpoint()
trainer.train()

