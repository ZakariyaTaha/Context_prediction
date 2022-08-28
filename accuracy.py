from tqdm import tqdm
from predict import predict
import pickle
import argparse


parser = argparse.ArgumentParser(description='Computing accuracy.')
parser.add_argument('--n', type=int, help='number of questions to evaluate', dest='number_of_questions', default=-1)

if __name__ == '__main__':
    args = parser.parse_args()
    number_of_questions = args.number_of_questions
    count = 0

    try:
        with open('contextsAndQuestions.pickle', 'rb') as file:
                df = pickle.load(file)
    except:
        raise Exception("Run the data processing file first")

    context_set = (df['context'].copy().drop_duplicates().reset_index(drop=True)).tolist()
    if number_of_questions == -1:
        number_of_questions = len(df['questions'])

    for i in tqdm(range(number_of_questions)):
        
    
        context = predict(df['questions'][i], context_set)
        eq = context in list(df['context'][df['questions'] == df['questions'][i]])
        if eq:
            count+=1
    
    print(count/number_of_questions)
