import argparse
import pandas as pd
import pickle
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import math
from numpy.linalg import norm

lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

parser = argparse.ArgumentParser(description='Processing contexts.')
parser.add_argument('--path', help='path do SQuAD dataset', dest='path_to_data', default='./squad1.1/train-v1.1.json')
parser.add_argument('--nb', type=int, help= "number of contexts to process (positive int)", dest='nb_of_contexts', default=-1)


def create_dict_of_indices(chunk_size, list_contexts):
    '''
    Creates a dictionary of chunk indices to original context indices (to find the context from the chunk)
    '''
    lengths = [math.ceil(len(sentence.split())/chunk_size) for sentence in list_contexts]

    def cumulativeSum(arr):
        for i in range(1, len(arr)):
            arr[i] += arr[i - 1]
        return arr
  
    indices_before_range = [0] + cumulativeSum(lengths) # add 0 for range later (below)
    indices = [list(range(indices_before_range[i-1], indices_before_range[i])) for i in range(1, len(indices_before_range))]
    chunk_to_context = {indices[i][j]: i for i in range(len(indices)) for j in range(len(indices[i]))}

    return chunk_to_context

def process_data(context):
    '''
    Steps of contexts processing involve: dropping duplicates, lowering, removing stopwords, lemmatizing, cutting in chunks
    '''
    chunks_length = 20
    context_set = (context.copy().drop_duplicates().reset_index(drop=True))#.tolist()

    context_set = context_set.apply(lambda s: s.lower())
    context_set = [(" ".join([lem.lemmatize(word) for word in cte.split() if word not in stop_words])) for cte in context_set]

  
    # dict linking context indices with chunk indices
    indice_chunk_to_context = create_dict_of_indices(chunks_length, context_set)

    # divide into chiunks
    context_set_split = [" ".join(context_set[j].split()[i:i+chunks_length]) for j in range(len(context_set)) for i in range(0, len(context_set[j].split()), chunks_length)]

    with open('processed_contexts.pickle', 'wb') as file:
        pickle.dump(context_set_split, file)

    with open('dict_indices.pickle', 'wb') as file:
        pickle.dump(indice_chunk_to_context, file)

if __name__ == '__main__':

    args = parser.parse_args()
    nb_of_contexts = args.nb_of_contexts
    path_to_data = args. path_to_data

    with open(path_to_data) as file:
        train_df = json.load(file)
    data = train_df['data'].copy()

    # extract contexts and corresponding questions
    df = pd.DataFrame([])
    for i in range(len(data)):
        for j in range(len(data[i]['paragraphs'])):
            df = pd.concat((df, pd.DataFrame.from_dict(data[i]['paragraphs'][j])))

    df = df.reset_index(drop=True).rename({'qas': 'questions'}, axis='columns')
    df['questions'] = df['questions'].map(lambda x: x['question'])

    with open('contextsAndQuestions.pickle', 'wb') as file:
        pickle.dump(df, file)

    if nb_of_contexts !=- 1:
        process_data(df['context'][:nb_of_contexts])
    else:
        process_data(df['context'])
    
    print("Data processed sucessfully!")
    
    

    
    