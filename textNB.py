#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multinomial Naive Bayes algorithm with Laplacian correction
Created on Sat Mar  4 23:45:23 2023

@author: okursun
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

newsgroups_train = fetch_20newsgroups(subset='train', 
                                      remove=('headers', 'footers', 'quotes'))
y_train = newsgroups_train.target


newsgroups_test = fetch_20newsgroups(subset='test', 
                                      remove=('headers', 'footers', 'quotes'))
y_test = newsgroups_test.target

#%%

print(newsgroups_train.data[11310])
expected_output = y_train[11310]
print(newsgroups_train.target_names[expected_output])

vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
clf = MultinomialNB(alpha=1)
clf.fit(vectors, y_train)
vectors_test = vectorizer.transform(newsgroups_test.data)
pred = clf.predict(vectors_test)
acc = (pred == y_test).mean()
print(f'{acc=}')
print('f1-score=',f1_score(newsgroups_test.target, pred, average='macro'))

probs = clf.predict_proba(vectors_test)
print('alternatively computed acc=', (np.argmax(probs,axis=1) == y_test).mean())
from sklearn.metrics import top_k_accuracy_score
topk_acc = top_k_accuracy_score(y_test, probs, k=2)
print(f'{topk_acc=}')

#%%
#moving from sparse to dense using toarray() is not a good idea but we use it for a simple demonstration of the algorithm
#even for the testing phase it is too slow

#TRAINING
alpha = 1   #works!
#alpha=0 leads to divide by zero, you can set those to zero but then the accuracy is still low around 4%
num_classes = len(newsgroups_train.target_names)
dims = vectors.shape[1]
priors = np.zeros(num_classes)
likelihoods = np.zeros((num_classes, dims))
for clas in range(num_classes):
    members = y_train==clas
    priors[clas] = members.mean()
    counts_in_clas = vectors[members].toarray().sum(axis=0)
    likelihoods[clas] = np.log((counts_in_clas + alpha) / (counts_in_clas.sum() + dims*alpha))
    
#%%

#TESTING - SLOW
from timeit import default_timer as timer
start = timer()
correct_classification = 0
N_test = vectors_test.shape[0]
for i in range(N_test):
    posteriors = -np.inf * np.ones(num_classes)
    x = vectors_test[i].toarray().flatten()
    for clas in range(num_classes):
        likelihood = np.dot(likelihoods[clas], x)
        posteriors[clas] = priors[clas]+likelihood
    predicted_class = np.argmax(posteriors)
    if predicted_class == y_test[i]:
        correct_classification = correct_classification + 1
end = timer()
print(end - start, 'secs') # Time in seconds, e.g. 5.38091952400282        
print(correct_classification/N_test)

#%%

#TESTING - less SLOW
start = timer()
posteriors = np.matmul(likelihoods,vectors_test.transpose().toarray()) + priors.reshape((-1,1))
end = timer()
print(end - start, 'secs') # Time in seconds, e.g. 5.38091952400282        
print((np.argmax(posteriors, axis=0) == y_test).mean())
