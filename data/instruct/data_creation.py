import json
import os
from utils import load_translation_data, call_llama_hf
from tqdm.auto import tqdm
import pathlib

# Constants
SYSTEM_PROMPT = """
Create a conversation between a user and an assistant. 
The user will ask for help or translations, and the assistant will respond with helpful and accurate information. 
Use the provided English-Italian sentence pairs to generate responses. 
Please maintain a natural and engaging conversation tone.
Ensure that the translation is 100 % correct. This is very important!

The your output should look like this: 
´´´json
{
      "role": "user",
      "content": "This is an example for a user message!"
    },
    {
      "role": "assistant",
      "content": "This is an example of a assistant message."
} 
´´´

Make sure this format is excatly kept!
"""

FILE_PATH = pathlib.Path(__file__).parent.parent / 'en-it-cleaned.txt'

def generate_user_message(english_sentence):
    """Generate a user message asking for the translation of an English sentence"""
    system_prompt_for_user_message = """ 
    Your task is to create a user inquiring an assistant directly about a translation. 
    Given an English sentence, your goal is to generate a user message asking for the translation of that sentence into Italian. 
    Your output should be a single string that represents the user's message. Do not include any additional text or explanation. 
    
    ## Examples: 
    - What does "..." mean in Italian? 
    - I am curious to know how to say "..." in Italian.
    - Can you help me translate "..." into Italian?
    - When traveling to Italy, I need to know how to say "...". Can you help me with that?
    - I'm trying to learn Italian and I'm stuck on the sentence "...". Can you help me translate it?
    - I'm working on a project and I need to translate the sentence "..." into Italian. Could you assist me with that?
    - I'm planning a trip to Italy and I need to know how to say "..." in Italian. Can you help me with that?
    
    Choose a question that is fitting to the provided text. Don't just use the examples above. 
    Be creative and choose a question that fits the context of the sentence.
    """
    created_user_message = call_llama_hf(system_prompt_for_user_message, english_sentence, temperature=0.7)
    return created_user_message

def generate_conversation(sentence_pair):
    """Generate a conversation based on a sentence pair"""
    english_sentence, italian_sentence = sentence_pair
    user_prompt = generate_user_message(english_sentence)
    user_instruction_prompt = f"Create a conversation based on this english sentence: '{english_sentence}'. User this user inquery: {user_prompt}. This is the correct translation '{italian_sentence}'."
    response = call_llama_hf(SYSTEM_PROMPT, user_instruction_prompt, temperature=0.7)
    return response

def main():
    print("| Loading translation data...")
    sentence_pairs = load_translation_data(FILE_PATH)
    
    progress_index_file = './progress_index.txt'
    if os.path.exists(progress_index_file):
        print(f"| Found progress index file. Resuming from index...")
        with open(progress_index_file, 'r') as f:
            progress_index = int(f.read())
    else:
        progress_index = 0

    conversations_file = './conversations.json'
    if os.path.exists(conversations_file):
        with open(conversations_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
    else:
        conversations = []

    for i, p in tqdm(enumerate(sentence_pairs[progress_index:], start=progress_index)):
        conversation = generate_conversation(p)
        conversations.append(conversation)
        
        with open(conversations_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=4)
        
        with open(progress_index_file, 'w') as f:
            f.write(str(i+1))

if __name__ == '__main__':
    main()