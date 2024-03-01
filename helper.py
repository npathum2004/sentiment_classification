import numpy as np
import pandas as pd
import re
import string
from nltk.stem import PorterStemmer
import pickle
ps = PorterStemmer()


#Load model
with open('static/model/imdb_lr_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Load Stopwords
with open('static/model/corpora/stopwords/english', 'r') as file:
    sw = file.read().splitlines()

# Load Tokens
vocab = pd.read_csv('static/model/imdb_vocabulary.txt', header=None)
tokens = vocab[0].tolist()

def preprocessing(text):
    data = pd.DataFrame([text], columns=['tweet'])
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '' ,x, flags=re.MULTILINE) for x in x.split()))
    data['tweet'] = data['tweet'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    data['tweet'] = data['tweet'].str.replace('\d+','',regex=True)
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
    return data['tweet']

def vectorizer(df):
    vectorized_lst = []
    for sentence in df:
        sentence_lst = np.zeros(len(tokens))
        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_lst[i] = 1
        vectorized_lst.append(sentence_lst)
    vectorized_lst_new = np.asarray(vectorized_lst, dtype=np.float32)
    return vectorized_lst_new

def get_prediction(vect_text):
    prediction = model.predict(vect_text)
    if prediction == 1:
        return 'negative'
    else:
        return 'positive'