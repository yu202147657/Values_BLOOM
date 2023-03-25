#Adapted from 
from transformers import AutoTokenizer,  AutoModelForCausalLM
import numpy as np
import pandas as pd

import string
import random


tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

lbls_map = {v: k for k, v in tokenizer.vocab.items()}

prompt = 'Should abortion be legal? A. Yes, because it prioritizes womens choice, B. No because it is murder C. Yes because it is not a life. Answer:'

df = pd.read_csv('data/full_values.csv', encoding='utf-8', index_col=False)

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

def get_first_valid_choice(num_premises, premises, probs_dict):
    choices_lst = get_alphabet_list(num_premises)
    #Identify first valid choice with highest probability:
    for letter in choices_lst:
        if letter in probs_dict:
            #get corresponding premise for letter
            index = ord(letter) - ord('A')
            premise = premises[index]
            return letter, premise
            
def get_alphabet_list(num_letters):
    alphabet = string.ascii_uppercase
    return list(alphabet[:num_letters])


def select_premises(question, premises):
    num_premises = len(premises)
    i = 0
    while num_premises > 1:
        # get a batch of premises
        if num_premises >= 4:
            batch_size = 4
        else:
            batch_size = num_premises
        batch = premises[i:i+batch_size]
        
        prompt = create_prompt(question, batch)
        probs_dict = gen_output(prompt)
        letter, winning_premise = get_first_valid_choice(len(batch), batch, probs_dict)
        
        # randomly add winning premise to the next batch
        if num_premises >= 3:
            next_batch = premises[i+batch_size:i+batch_size+3]
            next_batch.insert(random.randint(0, 2), winning_premise)
        else:
            next_batch = premises[i+batch_size:i+batch_size+2]
            next_batch.insert(random.randint(0, 1), winning_premise)
        
        # update premises and index
        premises[i:i+batch_size+1] = batch + next_batch
        i += batch_size
        num_premises -= batch_size
    return premises[0]



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
        #initial batch of 4
        batch = premises[:4]
        num_premises = len(premises)
        i = 0
        while num_premises > 1:
            # Get the batch size for this iteration
            batch_size = 4 if num_premises >= 4 else 3

            # Check if there are enough premises remaining for another batch
            if num_premises - batch_size < 0:
                batch_size = num_premises

            # Create the batch
            batch = premises[i:i+batch_size]
            print(batch)
        
            prompt = create_prompt(question, batch)
            probs_dict = gen_output(prompt)
            letter, winning_premise = get_first_valid_choice(len(batch), batch, probs_dict)
            
            num_premises -= batch_size - 1
            print(num_premises)

        for i in range(4, len(premises)+1, 3):
            #create batch of 3
            batch = premises[i:i+3]
            # Generate a random index to insert the winning premise
            index = random.randint(0, len(batch))
            # Insert the winning premise at the random index
            batch.insert(index, winning_premise)
            print(batch)
            prompt = create_prompt(question, batch)
            probs_dict = gen_output(prompt)
            letter, winning_premise = get_first_valid_choice(len(batch), batch, probs_dict)


        
        #on first pass, batch 4, 
        #yield winning choice
        #on second pass, get winning choice randomly in list with other 3. 
        #select_premises(question, premises)
    else:
        prompt = create_prompt(question, premises)
        probs_dict = gen_output(prompt)
        letter, winning_premise = get_first_valid_choice(len(premises), premises, probs_dict)
        
        
    #if len(premises) > 4:
        #round_robin(premises)