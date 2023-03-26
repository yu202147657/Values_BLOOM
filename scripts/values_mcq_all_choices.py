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
    english_lower_upper = list(string.ascii_uppercase) + list(string.ascii_lowercase)
    greek_lower_upper = list(''.join(chr(i) for i in range(945, 969))) + list(''.join(chr(i) for i in range(913, 937)))
    cyrillic_lower_upper = list(''.join(chr(i) for i in range(1072, 1104))) + list(''.join(chr(i) for i in range(1040, 1072)))
    
    #get only unique greek letters
    unique_greek_letters = list(set(greek_lower_upper) - set(english_lower_upper))
    unique_cyrillic_letters = list(set(cyrillic_lower_upper) - set(english_lower_upper))
    letters = english_lower_upper + unique_greek_letters + unique_cyrillic_letters

    for i, premise in enumerate(premises):
        prompt += letters[i] + '. ' + premises[i] + '\n'

    prompt += 'Answer:'

    return prompt, list(letters[:len(premises)])
            
    
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

def get_first_valid_choice(letters_lst, premises, probs_dict):

    #Identify first valid choice with highest probability:
    for letter in letters_lst:
        if letter in probs_dict:
            #get corresponding premise for letter
            index = ord(letter) - ord('A')
            premise = premises[index]
            return letter, premise


def get_mcq(model_name):
    """Get winning premise for 5 runs. Highest probability premise returned
    from multiple choice prompt"""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    lbls_map = {v: k for k, v in tokenizer.vocab.items()}

    df = pd.read_csv('data/full_values.csv', encoding='utf-8', index_col=False)
    
    #iterate thro questions in df
    for question in list(set(df.Question.values)):
        print(question)
        sub_df = df[df['Question'] == question]
        premises = list(sub_df.Premise.values)
        indices = df[df['Premise'].isin(premises)].index
        
        #premises = ['You dumb', 'you stupid', 'you crazy', 'you pantaloon', 'you silly']
        
        winning_premises = []
        seen_shufflings = set()
        
        for j in range(0, 5):
            #shuffle, but keep track to avoid repeating shuffles
            shuffle = tuple(random.sample(premises, len(premises)))
            if shuffle not in seen_shufflings:
                seen_shufflings.add(shuffle)
                premises_shuffled = list(shuffle)
        
                threshold = 35
                if len(premises) > threshold:
                    sublists = [premises_shuffled[i:i+threshold] for i in range(0, len(premises_shuffled), threshold)]
                else:
                    sublists = [premises_shuffled]

                for i, sublist in enumerate(sublists):
                    prompt, letters_lst = create_prompt(question, list(sublist))
                    print(len(letters_lst))
                    probs_dict = gen_output(prompt, model_name, lbls_map)
                    letter, winning_premise = get_first_valid_choice(letters_lst, list(sublist), probs_dict)
                    #if i less than list length
                    if i < len(sublists) - 1:
                        sublists[i+1].append(winning_premise)
            else:
                # shuffle has already been seen, try again
                continue

            winning_premises.append(winning_premise)

        #get most frequent item in list, and num times it occurs
        counter = Counter(winning_premises)
        most_frequent = counter.most_common(1)[0][0]
        num_most_frequent = counter[most_frequent]
        print(winning_premises, num_most_frequent)


        df.loc[df.index.isin(indices), 'winning_premise'] = most_frequent
        #df.loc[df.index.isin(indices), 'num_it'] = num_it
        df.loc[df.index.isin(indices), 'num_premises'] = len(premises)
        df.loc[df.index.isin(indices), 'num_most_frequent'] = num_most_frequent

    df.to_csv(f'results/{model_name}_values_mcq.csv', index=False, encoding='utf-8', sep='\t')
    
    
for model_name in ['EleutherAI/gpt-neox-20b']:#, 'bigscience/bloom-560M', 'EleutherAI/gpt-neo-125M', 'gpt2']:
    print(model_name)
    get_mcq(model_name)


