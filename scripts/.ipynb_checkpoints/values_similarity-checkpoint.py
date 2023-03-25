import torch
from transformers import BertTokenizer, BertModel, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd

# Load pre-trained BERT model and tokenizer
model_bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define your dataset of arguments and their associated human values as dictionaries
df = pd.read_csv('data/full_values.csv', encoding='utf-8', index_col=False)
#print(np.where(pd.isnull(df)))
#df = pd.read_csv('data/full_values_level_1.csv', encoding='utf-8', index_col=False)

# Load the compeltion model
model_name = 'EleutherAI/gpt-neo-125M'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the device to use
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model_bert.to(device)
    
# Define the number of completions to generate for each argument
num_completions = 5

# Generate BERT embeddings for each argument in the dataset
argument_embeddings = []
batch_size = 32
for i in range(0, len(df), batch_size):
    batch_arguments = list(df.Premise.values)[i:i+batch_size]
    encoded_inputs = bert_tokenizer(batch_arguments, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model_bert(**encoded_inputs)
        last_hidden_states = outputs.last_hidden_state
        batch_embeddings = torch.mean(last_hidden_states, dim=1).squeeze().cpu().detach().numpy()
        norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
        batch_embeddings = batch_embeddings / norms

        argument_embeddings.extend(batch_embeddings)

# Get indices for premises in that conclusion
# Generate multiple completions for each argument and calculate cosine similarity with human values

for question in list(set(df.Question.values)):
    
    # Define the prompt to be used to generate completions for each argument
    prompt = f'{question}?'
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    
    # To set up cosine similarity to other bert embeddings,
    # first, get all the premise indices of this question
    sub_df = df[df['Question'] == question]
    premises = sub_df.Premise.values
    #get indices of the premises (and thus the argument embeddings)
    indices = df[df['Premise'].isin(premises)].index
    #Initialize cosine similarity mx
    cosine_similarities = np.zeros((len(indices), num_completions))

    for j in range(num_completions):
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                do_sample=True,
                max_length=50,
                top_k=50,
                temperature=0.7,
                num_return_sequences=1
            )
        generated_completion = tokenizer.decode(outputs[:, input_ids.shape[1]:][0], skip_special_tokens=True)
        generated_completion_embedding = bert_tokenizer.encode(generated_completion, return_tensors='pt')
        #embed generated text with BERT
        with torch.no_grad():
            outputs = model.get_input_embeddings()(generated_completion_embedding.to(device))
            generated_completion_embedding = torch.mean(outputs, dim=1).squeeze().cpu().detach().numpy()
            #normalize bert embeddings for cosine similarity calculation
            norms = np.linalg.norm(generated_completion_embedding)
            generated_completion_embedding = generated_completion_embedding / norms

        for i, index in enumerate(indices):
            df.loc[df.index.isin(indices), f'completion_{j}'] = generated_completion
            cosine_similarity = np.dot(argument_embeddings[index], generated_completion_embedding) / \
                                (np.linalg.norm(argument_embeddings[index]) * np.linalg.norm(generated_completion_embedding))
            if abs(cosine_similarity) > 1:
                print(cosine_similarity, '######################################################')
            cosine_similarities[i, j] = cosine_similarity
            
    # Aggregate cosine similarities for each argument
    avg_cosine_similarities = np.mean(cosine_similarities, axis=1)
    #if any(abs(x) > 1 for x in avg_cosine_similarities):
    #    print("At least one element in the list is greater than 1.", avg_cosine_similarities)
    #else:
    #    pass
    df.loc[df.index.isin(indices), 'cosine_similarity'] = avg_cosine_similarities
    
df.to_csv('data/full_values_cosine_similarity.csv', index=False, encoding='utf-8')
print(np.where(pd.isnull(df)))
max_abs_value = df['cosine_similarity'].abs().max()
print(max_abs_value)
# Analyze the results to understand if GPT-3 embodies certain human values over others
