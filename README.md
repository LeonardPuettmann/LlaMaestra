# A tiny Llama model tuned for text translation
```html
 _      _       __  __                _             
| |    | |     |  \/  |              | |            
| |    | | __ _| \  / | __ _  ___ ___| |_ _ __ ___  
| |    | |/ _` | |\/| |/ _` |/ _ / __| __| '__/ _ \ 
| |____| | (_| | |  | | (_| |  __\__ | |_| | | (_) |
|______|_|\__,_|_|  |_|\__,_|\___|___/\__|_|  \___/ 
```

## Model Card 
This model was finetuned with roughly 300.000 examples of translations from English to Italian and Italian to English. The model was finetuned in a way to more directly provide a translation without much explanation.

Due to its size, the model runs very well on CPUs. 
![A very italian Llama model](llamaestro-sm-bg.png)

## Usage 

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "unsloth/Llama-3.2-1B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

ft_model = PeftModel.from_pretrained(base_model, "LeonardPuettmann/LlaMaestro-3.2-1B-Instruct-v0.1-4bit")

row_json = [
    {"role": "system", "content": "Your job is to return translations for sentences or words from either Italian to English or English to Italian."},
    {"role": "user", "content": "Scontri a Bologna, la destra lancia l'offensiva contro i centri sociali."}
]

prompt =  tokenizer.apply_chat_template(row_json, tokenize=False)
model_input = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    print(tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=1024)[0]))
```

## Data used 
The source for the data were sentence pairs from tatoeba.com. The data can be downloaded from here: https://tatoeba.org/downloads

## Credits

Base model: `unsloth/Llama-3.2-1B-Instruct` derived from `meta-llama/Llama-3.2-1B-Instruct`
Finetuned by: Leonard Püttmann https://www.linkedin.com/in/leonard-p%C3%BCttmann-4648231a9/