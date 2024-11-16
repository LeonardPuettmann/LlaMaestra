import time
import os
from huggingface_hub import InferenceClient

def call_llama_hf(system_prompt, user_prompt, temperature=0.0):
    client = InferenceClient(
        "meta-llama/Llama-3.1-70B-Instruct",
        token=os.getenv("HUGGINGFACE_API_KEY"),
    )
    max_retries = 3
    retries = 0

    while retries < max_retries:
        try:
            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=4096,
                stream=False,
                temperature=temperature
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            retries += 1
            if retries < max_retries:
                print(f"Error occurred: {e}. Retrying in 3 seconds...")
            else:
                print(f"Max retries reached. Last error: {e}")
                raise

def load_translation_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [line.strip().split('\t') for line in file.readlines()]
    
    return data