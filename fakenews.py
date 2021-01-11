# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 17:40:24 2021

@author: lukev
"""

import numpy as np
import pandas as pd
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk as nltk
from nltk.stem.porter import PorterStemmer 
import string
from itertools import chain
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix


def FilterText(data):
    data['text'] = list(data['text'].str.lower()) #Make everything lower case
    stopwordslist = stopwords.words('english') + ["would", "said"] #Create list of stopwords
    specchar = string.punctuation + "’" + "." + "“"
    corpus = []
    porstem = PorterStemmer()
    #print(specchar)
    for i in range(len(data['text'])):
        lists = data['text'][i].split()
        lists = [porstem.stem(word) for word in lists if not word in stopwordslist if not any(not letter.isalnum() for letter in word) if not any(letter.isdigit() for letter in word)]
        lists = ' '.join(lists)
        corpus.append(lists)
    return corpus

def MostFrequentWords(corpus, labels):
    npcorpus = np.array(corpus)
    corpusflat = npcorpus.ravel()
    total = []
    for i in range(len(corpusflat)):
        total = total + corpusflat[i].split()
    most_frequent = Counter(total).most_common(5)
    label_names = []
    values = []
    for tuplevalue in most_frequent:
        label_names.append(tuplevalue[0])
        values.append(tuplevalue[1])
    fig = plt.figure()
    axes = fig.add_axes([0,0,1,1])
    axes.bar(label_names, values)
    axes.set_title('Most frequent words in corpus')
    for i in range(len(label_names)):
        boolarray = []
        countfake = 0
        counttrue = 0
        for j in range(len(corpus)):
            if (label_names[i] in corpus[j]):
                boolarray.append(labels[j])
                add = corpus[j].count(label_names[i])
                if(labels[j]):
                    countfake = countfake + add
                else:
                    counttrue = counttrue + add
        print("Amount of true occurences for word " + str(label_names[i]) + ": " + str(counttrue))
        print("Amount of fake occurences for word " + str(label_names[i]) + ": " + str(countfake))
    plt.show()
    print("Most frequent = ] "+ str(most_frequent))

def ReadData():
    fake = pd.read_csv('data/Fake.csv')
    fake = fake[:1000]
    true = pd.read_csv('data/True.csv')
    true = true[:1000]
    true['labels'] = True 
    fake['labels'] = False
    size = len(true)-1
    size2 = size + len(fake)
    newindex = np.arange(size+1, size2+1)
    fake = fake.set_index(newindex)
    combined = pd.concat([true,fake])
    data = combined.drop(['title', 'date', 'subject'], axis = 1)
    return data

data = ReadData()
data = data.dropna() #Remove non existent values
corpus = FilterText(data)
sns.countplot(data.labels)
#MostFrequentWords(corpus, data.labels)
X_train, X_test, y_train, y_test = train_test_split(corpus, data.labels, test_size=0.20, random_state=0)  
vector = TfidfVectorizer()
vectors = vector.fit(X_train)
feature_names = vector.get_feature_names()
#print(feature_names)
xtrain = vector.transform(X_train)
xtest = vector.transform(X_test)
clf = RandomForestClassifier(max_depth=10, random_state=0).fit(xtrain, y_train)
print("Score random forest classifier: " + str(clf.score(xtest, y_test)))
results1 = clf.predict(xtest)
clf2 = LogisticRegression(random_state=0).fit(xtrain, y_train)
print("Score logistic regression: " + str(clf2.score(xtest, y_test)))
results2 = clf2.predict(xtest)
print("Classification report random forest classifier:")
print(classification_report(y_test, results1))
print("Classification report logistic regression:")
print(classification_report(y_test, results2))
plot_confusion_matrix(clf, xtest, y_test)
plot_confusion_matrix(clf2, xtest, y_test)




