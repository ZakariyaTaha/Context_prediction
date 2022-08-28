import argparse
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KDTree
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

parser = argparse.ArgumentParser(description='Predicting context.')
parser.add_argument('--index', type=int, help='indice of the question in the dataset', dest='question_ind', default=-1)
parser.add_argument('--question', type=str, help= "new question", dest='question', default=" ")

def cosine(A, B):
    return np.dot(A,B)/(norm(A)*norm(B))


def predict(question, context_set):

    with open('./processed_contexts.pickle', 'rb') as file:
        processed_contexts = pickle.load(file)
    with open('./dict_indices.pickle', 'rb') as file:
        indice_chunk_to_context = pickle.load(file)

    question_processed = question.lower()
    question_processed = [" ".join(lem.lemmatize(word) for word in question_processed.split() if word not in stop_words)][0]
    processed_contexts.append(question_processed)

    tfidf = TfidfVectorizer()
    train_tfidf = tfidf.fit_transform(processed_contexts)
    train_tfidf = list(train_tfidf.toarray())

    kdtree = KDTree(train_tfidf, leaf_size=10)
    distance, idx = kdtree.query([train_tfidf[len(train_tfidf)-1]], k=4)



    actual_indices = np.array(list(indice_chunk_to_context.values()))[idx[0][1:]]
    corresponding_contexts = np.array(context_set)[actual_indices]
    corresponding_contexts = np.append(corresponding_contexts, question)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(corresponding_contexts)

    scores = []
    for i in range(len(embeddings)-1):
        scores.append(cosine(embeddings[i], embeddings[len(embeddings)-1]))

    ind = np.argmax(scores)
    return corresponding_contexts[ind]

if __name__ == '__main__':

    args = parser.parse_args()
    question = args.question
    question_ind = args. question_ind

    try:
        with open('contextsAndQuestions.pickle', 'rb') as file:
                df = pickle.load(file)
    except:
        raise Exception("Run the data processing file first")

    # either a question is given or an index, in case of an index, we assign the corresponding question
    if not (question_ind != -1)^(question != " "):
        raise Exception("Give either a question or an index not both or neither")
    else:
        if question_ind != -1:
            question = df['questions'][question_ind]

    context_set = (df['context'].copy().drop_duplicates().reset_index(drop=True)).tolist()
    context = predict(question, context_set)

    print(context)