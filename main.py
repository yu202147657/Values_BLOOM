from src.values_entropy import get_entropy
from src.values_mcq import get_mcq

#model_lst = ['gpt2', 'EleutherAI/gpt-neo-125M', 'bigscience/bloom-560M']
model_lst = ['gpt2']

for model in model_lst:
    #get_entropy(model)
    get_mcq(model)