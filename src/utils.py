import pandas as pd
import pyreadstat
import matplotlib.pyplot as plt
import textwrap

from scipy.stats import ttest_ind, chisquare


def process_full_values():
    questions = pd.read_csv('data/values_questions.csv', encoding='utf-8', index_col=False)
    df = pd.read_csv('data/arguments-training.tsv', encoding='utf-8', sep='\t', index_col=False)
    labels = pd.read_csv('data/labels-training.tsv', encoding='utf-8', sep='\t', index_col=False)
    #level1 = pd.read_csv('data/level1-labels-training.tsv', encoding='utf-8', sep='\t', index_col=False)

    df = pd.merge(df, labels, on='Argument ID', how='inner')
    df = pd.merge(df, questions, on='Conclusion', how='left')
    df.to_csv('data/full_values.csv', index=False, encoding='utf-8')
    
def process_full_values_level_1():
    questions = pd.read_csv('data/values_questions.csv', encoding='utf-8', index_col=False)
    df = pd.read_csv('data/arguments-training.tsv', encoding='utf-8', sep='\t', index_col=False)
    labels = pd.read_csv('data/level1-labels-training.tsv', encoding='utf-8', sep='\t', index_col=False)

    df = pd.merge(df, labels, on='Argument ID', how='inner')
    df = pd.merge(df, questions, on='Conclusion', how='left')
    df.to_csv('data/full_values_level_1.csv', index=False, encoding='utf-8')

def process_values():
    df = pd.read_csv('data/arguments-training.tsv', sep='\t', encoding=
                     'utf-8')

    #keep only the unique conclusions
    df = df.drop_duplicates(subset=['Conclusion'])
    df.to_csv('data/values.csv', index=False, encoding='utf-8')
    
def process_pew():

    df, meta = pyreadstat.read_sav('data/ATP W93.sav')
    df.to_csv('data/pew.csv', encoding='utf-8', index=False)

def process_winogender():
    """drop neutral gender for binary bias evaluation"""
    df = pd.read_csv("data/all_sentences.tsv", sep='\t', encoding='utf-8', index_col=False)
    #remove neutral categories
    df = df[~df.sentid.str.contains("neutral")]
    #remove 'someone'
    df = df[~df.sentid.str.contains("someone")]
    
    #now, explode the df so the male/female are in columns like crowspairs df
    # create empty lists for the male and female messages
    male_messages = []
    female_messages = []

    # loop through each row in the DataFrame
    for row in df.itertuples():
        # split the identifier and text on the first period
        parts = row.sentid.rsplit('.', 2)
        identifier = parts[0]
        gender = parts[1]

        # check if the identifier matches any other rows in the DataFrame
        matching_rows = df[df.sentid.str.startswith(identifier)]
        if len(matching_rows) == 2:
            # if there are two matching rows, separate the male and female messages
            male_message = None
            female_message = None
            for matching_row in matching_rows.itertuples():
                parts = matching_row.sentid.split('.', 4)
                matching_gender = parts[3]
                if matching_gender == 'male':
                    male_message = matching_row.sentence
                elif matching_gender == 'female':
                    female_message = matching_row.sentence
            # append the male and female messages to the corresponding lists
            if male_message and female_message:
                male_messages.append(male_message)
                female_messages.append(female_message)

    # create a new DataFrame with the male and female messages as columns
    df = pd.DataFrame({'Male': male_messages, 'Female': female_messages})
    df = df.drop_duplicates()

    df.to_csv('data/winogender.csv', index=False, encoding='utf-8')

def get_value_labels(model_name):
    """
    Filter dataset to only have the premises with the max probabilities
    Then only keep the probabilities that are greater than chance for each question/
    premises combination
    """
    
    #df = pd.read_csv(f'results/{model_name}_full_values_entropy.csv', encoding='utf-8', sep='\t', index_col=False)
    #df = df[df['Premise'] == df['max_completion']]

    df = pd.read_csv(f'results/{model_name}_values_mcq.csv', encoding='utf-8', sep='\t', index_col=False)
    df = df[df['Premise'] == df['winning_premise']]
    #df.to_csv(f'results/{model_name}_prob_values.csv', index=False, encoding='utf-8')
    
    #df = df[df['max_prob'] > (1/df['num_premises'])]
    return df
    

def get_sum_series(df, level_1=None):
    if level_1:
        sums = df.iloc[:, df.columns.get_loc('Be creative'):df.columns.get_loc('Have an objective view')+1].sum(axis=0)
    else: 
        sums = df.iloc[:, df.columns.get_loc('Self-direction: action'):df.columns.get_loc('Universalism: objectivity')+1].sum(axis=0)
    
    return sums
    
def lvl_1_xwalk(df):
    level_1_df = pd.read_csv('data/level1-labels-training.tsv', encoding='utf-8', sep='\t', index_col=False)
    df = pd.merge(df, level_1_df, on='Argument ID', how='left')
    
    return df
    

def plot_values(model_name, level_1=None):
    
    df = pd.read_csv(f'results/{model_name}_full_values_entropy.csv', encoding='utf-8', sep='\t', index_col=False)
    
    histogram_df = get_value_labels(model_name)
    
    if level_1:
        df = lvl_1_xwalk(df)
        histogram_df = lvl_1_xwalk(histogram_df)
        
        df_sums = get_sum_series(df, level_1)
        bar(df_sums, model_name, level_1=True)

        histogram_sums = get_sum_series(histogram_df, level_1)
        bar(histogram_sums, model_name, biases=True, level_1=True)
    else:
        df_sums = get_sum_series(df, level_1)
        bar(df_sums, model_name)

        histogram_sums = get_sum_series(histogram_df, level_1)
        bar(histogram_sums, model_name, biases=True)
    
    # Perform a two-sample t-test
    t_statistic, p_value = ttest_ind(df_sums, histogram_sums)

    print('T-test statistic:', t_statistic)
    print('P-value:', p_value)
    
    # Perform chi-squared test
    model_probs = histogram_sums/sum(histogram_sums)
    model_counts = model_probs * sum(df_sums)
    chi2, p = chisquare(df_sums, f_exp=model_counts)

    # Print results
    print('Chi-squared test statistic:', chi2)
    print('P-value:', p)
    

def bar(s, model_name, biases=None, level_1=None):
    
    s_freq = s/sum(s)
    
    #plt.hist(histogram_sums, bins=30)
    plt.clf()
    plt.rc('font', family='Open Sans')
    fig, ax = plt.subplots(figsize=(15, 15))  # Set figure size
    
    sorted_s_freq = s_freq.sort_values(ascending=False)
    sorted_s_freq.plot(kind='bar', ax=ax)
    
    
    # First, let's remove the top, right and left spines (figure borders)
    # which really aren't necessary for a bar chart.
    # Also, make the bottom spine gray instead of black.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    
    ax.set_ylabel('Proportion Represented', labelpad=15, color='#333333')
    
    #plt.subplots_adjust(bottom=0.5)
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_biases_{biases}_lvl1_{level_1}.png')

#for model_name in ['gpt2', 'bigscience/bloom-560M', 'bigscience/bloom-1b7', 'EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B']:
for model_name in ['gpt2', 'bigscience/bloom-560M', 'EleutherAI/gpt-neo-125M']:
    print(model_name)
    plot_values(model_name, level_1=True)
    plot_values(model_name)
#get_value_labels('EleutherAI/gpt-neo-125M')
#process_full_values()
#process_full_values_level_1()
#process_values()
#process_winogender()