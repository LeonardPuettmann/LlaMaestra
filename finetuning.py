from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format

from huggingface_hub import login
from datasets import *

import os
import pathlib 
import json
import bitsandbytes as bnb

import pandas as pd
from sklearn.model_selection import train_test_split

hf_token = os.getenv("HUGGINGFACE_API_KEY")
login(token=hf_token)

# wandb.login(key=wb_token)
# run = wandb.init(
#     project='Fine-tune Llama 3.2 on Customer Support Dataset', 
#     job_type="training", 
#     anonymous="allow"
# )

torch_dtype = torch.float16
attn_implementation = "eager"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)
# Load model
base_model = "unsloth/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model, 
    trust_remote_code=True,
    padding=True,
    padding_side="right",
    truncation=True
    )

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

modules = find_all_linear_names(model)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules
)
#model, tokenizer = setup_chat_format(model, tokenizer)
model = get_peft_model(model, peft_config) 

new_model = "llamaestro-3.2-1b-it-finetuned"

def load_translation_data(file_path):
    """
    Load translation dataset from a.txt file and separate English and Italian sentences.
    
    Args:
        file_path (str): Path to the.txt file containing the translation dataset
    
    Returns:
        list: List of tuples containing English and Italian sentences
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [line.strip().split('\t') for line in file.readlines()]
    
    return data

file_path = './en-it-cleaned.txt'
data = load_translation_data(file_path)

# Create a pandas DataFrame with English and Italian sentences
df = pd.DataFrame(data, columns=['english', 'italian'])
df.head()

# split the data into training and test sets
train_text, val_text, train_labels, val_labels = train_test_split(df['english'], df['italian'], test_size=0.2, random_state=42)

# create a DatasetDict with the train and test data
dataset = DatasetDict({
    'train': Dataset.from_pandas(pd.DataFrame({'english': train_text, 'italian': train_labels})),
    'test': Dataset.from_pandas(pd.DataFrame({'english': val_text, 'italian': val_labels}))
})

base_model = "unsloth/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# Define the chat template functions
def format_chat_template_en_it(row):
    instruction = "Translate the following sentence from English to Italian:"
    row_json = [{"role": "system", "content": instruction},
                {"role": "user", "content": row["english"]},
                {"role": "assistant", "content": row["italian"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False, padding=True, truncation=True)
    return row

def format_chat_template_it_en(row):
    instruction = "Translate the following sentence from Italian to English:"
    row_json = [{"role": "system", "content": instruction},
                {"role": "user", "content": row["italian"]},
                {"role": "assistant", "content": row["english"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False, padding=True, truncation=True)
    return row

# Create two separate datasets
dataset_en_it = dataset.map(format_chat_template_en_it)
dataset_it_en = dataset.map(format_chat_template_it_en)

merged_dataset = DatasetDict({
    'train': concatenate_datasets([dataset_en_it['train'], dataset_it_en['train']]),
    'test': concatenate_datasets([dataset_en_it['test'], dataset_it_en['test']])
})

dataset = merged_dataset.shuffle(seed=42)
print(dataset)

#Hyperparamter
training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to=None,
    remove_unused_columns=False
)

# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    max_seq_length= 512,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=True,
)

trainer.train()
#wandb.finish()

# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
#trainer.model.push_to_hub(new_model, use_temp_dir=False)