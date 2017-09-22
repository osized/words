import gensim.models
import pandas as pd
import re

import  numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

train = pd.read_csv("C:\\Users\\AYudin.NBKI\\Downloads\\X_train.csv", na_filter=False)

print("--------------start dict loading--------------")
w2v_fpath = "C:\\Users\\AYudin.NBKI\\Desktop\\Kaggle\\tenth.norm-sz500-w7-cb0-it5-min5.w2v"
w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_fpath, binary=True, unicode_errors='ignore')
w2v.init_sims(replace=True)
print("--------------dict loaded--------------")

def vectorize_ratings(ratings):
    y = np.zeros((len(ratings), 5), dtype=np.bool)
    for i, rating in enumerate(ratings):
        y[i, rating - 1] = 1
    return y

def get_dict_words(comment):
    sentence = []
    comment = comment.lower().replace('ё', 'е')

    comment = re.sub('[1]', ' один ', comment)
    comment = re.sub('[2]', ' два ', comment)
    comment = re.sub('[3]', ' три ', comment)
    comment = re.sub('[4]', ' четыре ', comment)
    comment = re.sub('[5]', ' пять ', comment)
    comment = re.sub('[6]', ' шесть ', comment)
    comment = re.sub('[7]', ' семь ', comment)
    comment = re.sub('[8]', ' восемь ', comment)
    comment = re.sub('[9]', ' девять ', comment)

    replaced_comment = re.sub('[^а-яa-z]', ' ', comment)

    for word in replaced_comment.split(' '):
        if word:
            if word in w2v.wv.vocab:
                sentence.append(w2v.wv[word])
            else:
                sentence.append(np.zeros(500))
    return sentence

print("--------------start--------------")

sentences = []
ratings = []

for index, row in train.iterrows():
    sentence = get_dict_words(row.comment + " " + row.commentNegative + " " + row.commentPositive)
    sentences.append(np.mean(np.array(sentence), axis=0))
    ratings.append(int(row.reting))

ratings = vectorize_ratings(ratings)

ratings_train = ratings[:-256]
sentences_train = sentences[:-256]

ratings_test = ratings[-256:]
sentences_test = sentences[-256:]

print("--------------train--------------")
# fit model no training data
model = XGBClassifier()
model.fit(np.array(sentences_train), np.array(ratings_train))
# make predictions for test data
ratings_pred = model.predict(np.array(sentences_test))
predictions = [round(value) for value in ratings_pred]
# evaluate predictions
accuracy = accuracy_score(np.array(ratings_test), np.array(predictions))
print("Accuracy: %.2f%%" % (accuracy * 100.0))