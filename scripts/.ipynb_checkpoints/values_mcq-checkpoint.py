#Adapted from 
from transformers import AutoTokenizer,  AutoModelForCausalLM
import numpy as np
import pandas as pd

import string

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

lbls_map = {v: k for k, v in tokenizer.vocab.items()}

prompt = 'Should abortion be legal? A. Yes, because it prioritizes womens choice, B. No because it is murder C. Yes because it is not a life. Answer:'

df = pd.read_csv('data/full_values.csv', encoding='utf-8', index_col=False)

def round_robin(premises):
    print(len(premises))
    num_premises = len(premises)
    batch_size = min(4, num_premises)
    num_batches = (num_premises + batch_size - 1) // batch_size

    for i in range(num_batches):
        batch = premises[i*batch_size:(i+1)*batch_size]
        batch_size_actual = len(batch)
        letters = ["A", "B", "C", "D"][:batch_size_actual]
        options = [f"{letter}. {premise}" for letter, premise in zip(letters, batch)]
        print(options)
    print('####################################################################'
             )
        # present options to GPT and get answer
        # compare answer to the remaining premises and repeat until one premise remains

def create_prompt(question, premises):
    """ Assigns letters and format prompts
    Takes in question and max of 4 premises """
    prompt = 'Question: ' + question + '?' '\n'
    num_premises = len(premises)
    options = ['A', 'B', 'C', 'D']
    for i in range(num_premises):
        if i == 0:
            prompt += options[i] + '. ' + premises[i] + '\n'
        else:
            prompt += options[i % 4] + '. ' + premises[i] + '\n'
            if i % 4 == 3:
                prompt += '\n'
    prompt += 'Answer:'
    return prompt
            
    
def gen_output(prompt):
    """Takes in prompt
    Returns dictionary of probabilities for responses"""
    
    prompt_text = prompt#question.get_natural_prompt()
    inputs = tokenizer(prompt_text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits[0, -1]
    probs = logits.softmax(dim=-1)
    logprobs_dict = {
        lbls_map[i]:
        np.log(probs[i].item()) for i in range(len(lbls_map))
    }
    probs_dict = {
        lbls_map[i]:
        probs[i].item() for i in range(len(lbls_map))
    }
    # Reduce logprobs_dict to only keys with top 50 largest values
    logprobs_dict = {
        k.lstrip('Ġ'): v for k, v in sorted(
            logprobs_dict.items(),
            key=lambda item: item[1],
            reverse=True
        )[:200]
    }
    probs_dict = {
        k.lstrip('Ġ'): v for k, v in sorted(
            probs_dict.items(),
            key=lambda item: item[1],
            reverse=True
        )[:200]
    }
    
    return probs_dict

def get_first_valid_choice(num_premises, probs_dict):
    choices_lst = get_alphabet_list(num_premises)
    #Identify first valid choice with highest probability:
    for letter in choices_lst:
        if letter in probs_dict:
            return letter
            
def get_alphabet_list(num_letters):
    alphabet = string.ascii_uppercase
    return list(alphabet[:num_letters])

#if that letter, get the associated premise. then calculate distribution of those values
#do it five times, see which one selected most. get ppa for it.

#iterate thro questions in df
for question in list(set(df.Question.values)):

    prompt = f'{question}?'

    sub_df = df[df['Question'] == question]
    premises = sub_df.Premise.values
    #indices = df[df['Premise'].isin(premises)].index
    premises = ['You dumb', 'you stupid', 'you crazy', 'you pantaloon', 'you silly']
    if len(premises) > 4:
        pass
    else:
        prompt = create_prompt(question, premises)
        probs_dict = gen_output(prompt)
        letter = get_first_valid_choice(len(premises), probs_dict)
        print(letter)
        
    #if len(premises) > 4:
        #round_robin(premises)