import torch

from transformers import AutoTokenizer, AutoModelForMultipleChoice, AutoModelForCausalLM
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

import numpy as np
import pandas as pd

def get_entropy(model_name):

    # Load the GPT-3 model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #model = GPTNeoForCausalLM.from_pretrained(model_name)
    #tokenizer = GPT2Tokenizer.from_pretrained(model_name)


    df = pd.read_csv('data/full_values.csv', encoding='utf-8', index_col=False)

    # Set the device to use
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Set the generation parameters
    max_length = 100
    num_return_sequences = 2
    temperature = 1.0
    

    # Evaluate the bias
    entropy_lst = []

    #iterate thro rows in df (pairs for crowspairs
    for question in list(set(df.Question.values)):
        
        prompt = f'{question}?'
        
        sub_df = df[df['Question'] == question]
        premises = sub_df.Premise.values
        indices = df[df['Premise'].isin(premises)].index
        
        completions = premises
        
        # Calculate the log probabilities of each choice completion
        max_log_prob = 0
        max_completion = None
        log_probs = []

        for completion in completions:

            #encode
            encoded_completion = tokenizer.encode(completion, add_special_tokens=False)
            encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False)

            input_ids = torch.tensor(encoded_prompt + encoded_completion).unsqueeze(0).to(device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            log_prob = outputs.logits[0, -len(encoded_completion), encoded_completion[-1]].item()
            log_probs.append(log_prob)
            
        # Convert log probabilities to probabilities
        probs = np.exp(log_probs)
        # normalize probabilities to sum to 1
        probs /= np.sum(probs)
        
        #get max probability
        max_prob = np.max(probs)
        #get completion of max prob
        max_completion = completions[np.argmax(probs)]


        # calculate entropy
        entropy = -np.sum(probs * np.log2(probs))
        # calculate maximum possible entropy
        max_entropy = np.log2(len(probs))
        # normalize entropy
        norm_entropy = entropy / max_entropy

        entropy_lst.append(norm_entropy)
        
        #add entropy to the premises in each question 
        #(entropy is same for each premise/question combination)
        df.loc[df.index.isin(indices), 'entropy'] = np.mean(entropy_lst)
        df.loc[df.index.isin(indices), 'max_prob'] = max_prob
        df.loc[df.index.isin(indices), 'max_completion'] = max_completion
        df.loc[df.index.isin(indices), 'num_premises'] = len(indices)
    
    #write df with entropy for each premise
    df.to_csv(f'results/{model_name}_full_values_entropy.csv', index=False, encoding='utf-8', sep='\t')
    
    #get avg entropy for entire dataset
    #first drop duplicates bc entropy is unique per Q
    df = df.drop_duplicates(subset=['Question'])
    avg_entropy = df['entropy'].mean()
    df = pd.DataFrame({'entropy': [np.round(avg_entropy, 3)]})
    df.to_csv(f'results/{model_name}_values_entropy.csv', index=False, encoding='utf-8', sep='\t')

        

for model_name in ['EleutherAI/gpt-neo-125M', 'bigscience/bloom-560M']:
    print(model_name)
    get_entropy(model_name)
