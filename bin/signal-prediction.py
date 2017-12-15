#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 13:55:45 2017

@author: jiemo
"""

from sys import platform
import os
import numpy as np
import glob
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB 
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support


def X_input_check(X):
    counter = {}
    for idx,seq in enumerate(X):
        if(len(seq) <= 2):
            assert 1 == 2  # check if there is seq < 2 char
        for char in seq:
            if char in counter.keys():
                counter[char] +=1
            else:
                counter[char] = 1
    assert len(counter.keys()) <= 20 # check if there are less than 20 types of proteins


def get_file_sequences(file_path):
    with open(file_path,'r') as file:
        lines =  file.readlines()
    accession = {}
    current_key = None
    for line in lines:
        if line.startswith(">"):
            current_key = line[1:].strip()
            current_key = current_key.split(' ',1)[0]
            accession[current_key] = ""
        elif line.startswith("#"):
            continue
        elif current_key is not None:
            accession[current_key]+=line.strip()
    return accession

def load_data(load_tm = False):
    #load file into dictionaries
    negative_sample_path = "../data/negative_examples/"
    positive_sample_path = "../data/positive_examples/"
    cwd = os.getcwd()
    if platform=='win32':
        negative_sample_path = cwd+"\\"+negative_sample_path
        positive_sample_path = cwd+"\\"+positive_sample_path
    else:
        negative_sample_path = cwd+'/'+ negative_sample_path.replace('\\','/')
        positive_sample_path = cwd+"/"+positive_sample_path.replace('\\','/')
        
    if load_tm:
        tm_adder = "tm/*"
    else:
        tm_adder = "non_tm/*"
    y = []
    X = []
    pattern = re.compile("[X]+")
    negative_files = {} #probs not useful
    for file in glob.glob(negative_sample_path+tm_adder):
        fasta_sequences = get_file_sequences(file)
        for name,sequence in fasta_sequences.items():
            X.append(pattern.sub('',sequence))
            y.append(0)
            negative_files[file] = fasta_sequences
    positive_files = {} 
    for file in glob.glob(positive_sample_path+tm_adder):
        fasta_sequences = get_file_sequences(file)
        for name,sequence in fasta_sequences.items():
            X.append(pattern.sub('',sequence))
            y.append(1)
            positive_files[file] = fasta_sequences
    
#    X_input_check(X) #check there is less or equal to 20 different types of proteins
    #Vectorised X
    transformer = CountVectorizer(lowercase = False,analyzer = "char",ngram_range=[3,4])
    X_sparse = transformer.fit_transform(X)
    print("X_shape: ",X_sparse.shape)
#    negative_ending = 1087 #246 for tm
#    plot_X(X_sparse[:negative_ending].todense()) #all negative samples
#    plot_X(X_sparse[negative_ending+1:].todense()) #all positive samples
    
    #split and shuffle
    X_train, X_test, y_train, y_test = train_test_split(X_sparse, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data(True)

def check_acc(trained_clf,clf_name):
    #print training and testing acc automatically 
    
    #OUTPUTS
    #produce a accuracy chart for different algorithms
    #A distribution graph (TM/NON-TM)
    #Result - prediction (using TM/NON-TM/Mixed training data)- input training model 
    
    #training acc
    result_sk = trained_clf.predict(X_train)
    true_label =np.array(list(map(int,y_train)))
    result_sk = np.array(list(map(int,result_sk)))
    acc = np.mean(result_sk == true_label) * 100.
    print("Training Accuracy, %s : %.2f%%" % (clf_name,acc))
        
    #testing acc
    result_sk = trained_clf.predict(X_test)
    true_label =np.array(list(map(int,y_test)))
    result_sk = np.array(list(map(int,result_sk)))
    acc = np.mean(result_sk == true_label) * 100.
    print("Testing Accuracy, %s : %.2f%%" % (clf_name,acc))
    
    precision, recall, _,_ = precision_recall_fscore_support(true_label,result_sk,labels=[0,1])
    print("neg precision %.4f, neg recall %.4f" % (precision[0], recall[0]))
    print("pos precision %.4f, pos recall %.4f\n" % (precision[1], recall[1]))
#    print("neg precision %.4f, neg recall %.4f\n" % (precision[2], recall[2]))


    
text_clf = BernoulliNB()
text_clf.fit(X_train,y_train)
check_acc(text_clf,"BernoulliNB")

#winner  
text_clf = MultinomialNB()
text_clf.fit(X_train,y_train)
check_acc(text_clf,"MultinomialNB")

text_clf = LogisticRegression(penalty = 'l2') #works better on high dimentional sparse dataset
text_clf.fit(X_train,y_train)
check_acc(text_clf,"Logistic Regression")

linsvc = svm.SVC(C=10, kernel='linear', decision_function_shape='ovo')
linsvc.fit(X_train, y_train)
check_acc(linsvc, "SVC with linear kernel")

    
    
    
    
    
    
    
    
    
    
    
    