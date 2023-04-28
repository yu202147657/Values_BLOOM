#Adapted from 
from transformers import AutoTokenizer,  AutoModelForCausalLM
import numpy as np
import pandas as pd

import string
import random
from collections import Counter
import math
import itertools


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
            
    
def gen_output(prompt, model_name, lbls_map):
    """Takes in prompt
    Returns dictionary of probabilities for responses"""
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
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


def select_premises(question, premises, model_name, lbls_map):
    
    #initial batch of 4
    batch = premises[:4]
    num_premises = len(premises)
    i = 0
    while num_premises > 0:
        # Get the batch size for this iteration
        batch_size = 4 if i == 0 else 3

        # Check if there are enough premises remaining for another batch
        if num_premises - batch_size < 0:
            batch_size = num_premises

        # Create the batch
        if batch_size == 4:
            batch = premises[i:i+batch_size]
            
            prompt = create_prompt(question, list(batch))
            probs_dict = gen_output(prompt, model_name, lbls_map)
            letter, winning_premise = get_first_valid_choice(len(batch), list(batch), probs_dict)
        else:
            batch = premises[i:i+batch_size]
            #get index to randomly insert winning premise in
            index = random.randint(0, len(batch))
            batch = np.insert(batch, index, winning_premise)
            
            prompt = create_prompt(question, list(batch))
            probs_dict = gen_output(prompt, model_name, lbls_map)
            letter, winning_premise = get_first_valid_choice(len(batch), list(batch), probs_dict)

        #update round
        i+=batch_size
        num_premises -= batch_size
        print(i, num_premises)
    return winning_premise


#if that letter, get the associated premise. then calculate distribution of those values
#do it five times, see which one selected most. get ppa for it.

def get_mcq(model_name):
    """Get winning premise for 5 runs. Highest probability premise returned
    from multiple choice prompt"""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    lbls_map = {v: k for k, v in tokenizer.vocab.items()}

    df = pd.read_csv('data/full_values.csv', encoding='utf-8', index_col=False)
    
    #iterate thro questions in df
    for question in list(set(df.Question.values)):

        sub_df = df[df['Question'] == question]
        premises = sub_df.Premise.values
        print(len(premises), question)
        indices = df[df['Premise'].isin(premises)].index
        #premises = ['You dumb', 'you stupid', 'you crazy', 'you pantaloon', 'you silly']

        winning_premises = []
        #run 5 times and get most frequent
        

        if len(premises) > 4:
            num_it = 5 #math.factorial(4)
            
            seen_shufflings = set()
            
            for j in range (0, num_it):
                #iterating thro 24, not entire ordering, bc too many combinations to do all permutations
                #instead shuffle, but keep track to avoid repeating shuffles
                shuffle = tuple(random.sample(premises.tolist(), len(premises)))
                if shuffle not in seen_shufflings:
                    seen_shufflings.add(shuffle)
                    premises_shuffled = list(shuffle)
                    #do function
                    winning_premise = select_premises(question, premises_shuffled, model_name, lbls_map)
                    winning_premises.append(winning_premise)
                else:
                    # shuffle has already been seen, try again
                    continue
        else:
            num_it = 5 #math.factorial(len(premises))
            #iterates through every unique combination of premises
            combinations = itertools.permutations(premises)
            for premises in combinations:
                prompt = create_prompt(question, list(premises))
                probs_dict = gen_output(prompt, model_name, lbls_map)
                letter, winning_premise = get_first_valid_choice(len(premises), list(premises), probs_dict)
                winning_premises.append(winning_premise)

        #get most frequent item in list, and num times it occurs
        counter = Counter(winning_premises)
        most_frequent = counter.most_common(1)[0][0]
        num_most_frequent = counter[most_frequent]

        
        df.loc[df.index.isin(indices), 'winning_premise'] = most_frequent
        df.loc[df.index.isin(indices), 'num_it'] = num_it
        df.loc[df.index.isin(indices), 'num_premises'] = len(premises)
        df.loc[df.index.isin(indices), 'num_most_frequent'] = num_most_frequent

    df.to_csv(f'results/{model_name}_values_mcq.csv', index=False, encoding='utf-8', sep='\t')

