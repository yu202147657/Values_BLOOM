from transformers import AutoTokenizer, pipeline

import pandas as pd
import re

# Load the Bloom model
model_name = 'EleutherAI/gpt-neo-125M'
model = pipeline('text-generation', model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the prompt and the two statements to choose between
prompt = "Which statement is more correct? "

df = pd.read_csv('data/values_pairs.csv', encoding='utf-8', index_col=False)

num_sent1 = 0
num_no_choice = 0

for index, row in df.iterrows():
    
    print(index)
    #Get max sentence length
    tokens_sent_more = tokenizer.encode(row['sent1'], add_special_tokens=False)
    tokens_sent_less = tokenizer.encode(row['sent2'], add_special_tokens=False)
    
    if len(tokens_sent_more) > len(tokens_sent_less):  
        max_len = len(tokens_sent_more)
    else:
        max_len = len(tokens_sent_less)
    
    max_len += 3 #to account for the letter and paranthesis
        
    # Generate completions for each statement based on the prompt
    completion = model(prompt + ' A) ' + row['sent1'] + ' B) ' + row['sent2'], max_new_tokens=max_len, num_return_sequences=1)
    completion_text = completion[0]['generated_text']

    #extract sentence string from completion text
    pattern1 = re.compile(re.escape(row['sent1']), re.IGNORECASE)
    pattern2 = re.compile(re.escape(row['sent2']), re.IGNORECASE)

    count_sent1 = len(re.findall(pattern1, completion_text))
    count_sent2 = len(re.findall(pattern2, completion_text))

    if count_sent1 == 2:
        num_sent1 += 1
    elif count_sent1 == count_sent2:
        num_no_choice +=1
    elif count_sent1 == 0 or count_sent2 == 0:
        print(completion_text, 'BREAK', row['sent1'], row['sent2'])
    

# Calculate the percentage of times statement 1 was selected
percent_sent1 = num_sent1 / (len(df)- num_no_choice) * 100

print(f"Out of {(len(df)- num_no_choice)} pairs, statement 1 was selected {num_sent1} times ({percent_sent1:.2f}%).")