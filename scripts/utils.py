import pandas as pd
import pyreadstat

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
    
process_full_values()
process_full_values_level_1()
#process_values()
#process_winogender()