import torch
import keras
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import argparse
import warnings
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

base_model_id = "unsloth/Llama-3.2-1B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

parser = argparse.ArgumentParser(description="Chat with the model")
parser.add_argument("--cpu", action="store_true", help="Run the model on CPU")
parser.add_argument("--gpu", action="store_true", help="Run the model on GPU (default)")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()

if not args.debug:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('keras').setLevel(logging.FATAL)
    logging.getLogger('torch').setLevel(logging.FATAL)
    warnings.filterwarnings("ignore")
		
device = "cpu" if args.cpu else "cuda"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    quantization_config=bnb_config, 
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
ft_model = PeftModel.from_pretrained(base_model, "LeonardPuettmann/LlaMaestro-3.2-1B-Instruct-v0.1-4bit")

ft_model.generation_config.pad_token_id = tokenizer.pad_token_id
ft_model.generation_config.pad_token_id = tokenizer.eos_token_id

def get_response(user_input):
    row_json = [
        {"role": "system", "content": "Your job is to return translations for sentences or words from either Italian to English or English to Italian."},
        {"role": "user", "content": user_input}
    ]

    prompt =  tokenizer.apply_chat_template(row_json, tokenize=False)
    model_input = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        response = ft_model.generate(**model_input, max_new_tokens=1024)[0]
        response_decoded = tokenizer.decode(response, skip_special_tokens=False)
        return response_decoded.rsplit("<|end_header_id|>", 1)[-1].replace("<|eot_id|>", "").strip()

def chat():
    print("Loading model... \n")
    print(r"""
     _      _       __  __                _             
    | |    | |     |  \/  |              | |            
    | |    | | __ _| \  / | __ _  ___ ___| |_ _ __ ___  
    | |    | |/ _` | |\/| |/ _` |/ _ / __| __| '__/ _ \ 
    | |____| | (_| | |  | | (_| |  __\__ | |_| | | (_) |
    |______|_|\__,_|_|  |_|\__,_|\___|___/\__|_|  \___/                       
    """)                                                    
    print("\nType '/quit' to exit")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "/quit":
            break
        response = get_response(user_input)
        print("Assistant:", response)

if __name__ == "__main__":
    chat()