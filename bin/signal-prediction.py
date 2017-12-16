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
import matplotlib.pyplot as plt
import sys
import re
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support

def X_input_check(X):
    '''Check input X for validility
    Parameter:
        X - arrary with dimension M * N, where N is number of features
    and M is the number of samples
    Returns: None.
    Usage:
        Call this method to check input, will throw assert error if there is a
        sequence <= 2 char OR the number of distinct proteins is <= 20
    '''
    counter = {}
    for idx, seq in enumerate(X):
        if(len(seq) <= 2):
            assert 1 == 2  # check if there is seq < 2 char
        for char in seq:
            if char in counter.keys():
                counter[char] += 1
            else:
                counter[char] = 1
    assert len(counter.keys()) <= 20 # check if there are less than 20 types of proteins


def get_file_sequences(file_path):
    '''Parse FASTA format files into dictionaries, with accession as keys and sequence as values
    Parameters:
        file_path: the path of the FASTA file to read
    Returns:
        dictionary containing accessions and sequences
    Exception:
        if file not exist, will print No such file exists in stderr
    '''
    try:
        with open(file_path,'r') as file:
            lines =  file.readlines()
    except FileNotFoundError:
        print(FileNotFoundError,file = sys.stderr)
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

input_settings_note = ""
def load_data(load_tm = False,split = 0.2):
    ''' Process data into training and testing set. Shuffle is applied automatically
    Split based on 20% testing and 80% training
    Parameters:
        load_tm: indicate whether to load TM data
    Returns:
        X_train, X_test, y_train, y_test
    '''
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
    pattern = re.compile("[X]+") #remove breaks in the sequence
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
    ngram = [3,4]
    transformer = CountVectorizer(lowercase = False,analyzer = "char",ngram_range= ngram)
    X_sparse = transformer.fit_transform(X)
    global input_settings_note
    input_settings_note = "use_TM: {3}, total_X_shape: {0}, ngram: {1},\
 split_size: {2}".format(X_sparse.shape,ngram,split,load_tm)
    print(input_settings_note)
#    negative_ending = 1087 #246 for tm
#    plot_X(X_sparse[:negative_ending].todense()) #all negative samples
#    plot_X(X_sparse[negative_ending+1:].todense()) #all positive samples

    #split and shuffle
    X_train, X_test, y_train, y_test = train_test_split(X_sparse, y, test_size=split)
    return X_train, X_test, y_train, y_test


def check_acc(data,trained_clf,clf_name, verbose = True):
    '''produce training /testing and precision/recall
    Input data are fetched from globe
    Parameter:
        trained_clf: A trianed classifier from sklearn package
        clf_name: string, custom name for verbose mode
        verbose: indicate whether to print results
    Returns:
        train_acc,test_acc,precision,recall,f_beta
    '''
    X_train, X_test, y_train, y_test = data
    #training acc
    result_sk = trained_clf.predict(X_train)
    true_label =np.array(list(map(int,y_train)))
    result_sk = np.array(list(map(int,result_sk)))
    train_acc = np.mean(result_sk == true_label) * 100.
    if verbose:
        print("Training Accuracy, %s : %.2f%%" % (clf_name,train_acc))

    #testing acc
    result_sk = trained_clf.predict(X_test)
    true_label =np.array(list(map(int,y_test)))
    result_sk = np.array(list(map(int,result_sk)))
    test_acc = np.mean(result_sk == true_label) * 100.
    if verbose:
        print("Testing Accuracy, %s : %.2f%%" % (clf_name,test_acc))

    precision, recall, f_beta,_ = precision_recall_fscore_support(true_label,result_sk,labels=[0,1])
    if verbose:
        print("neg precision %.4f, neg recall %.2f" % (precision[0], recall[0]))
        print("pos precision %.4f, pos recall %.2f\n" % (precision[1], recall[1]))
    return train_acc,test_acc,precision,recall,f_beta



def model_performance_test(save_path,n_fold = 5,use_tm = False):
    ''' Train and save results in to pdf,
    Models are:
        Bernoulli NB
        Multinomial NB
        Logistic Regression with l2 penalty
        Linear SVM
    Parameters:
        save_path: path of the saved_pdf, ends with .pdf
        use_tm: indicate whether to use TM data, default to False
    Output:
        A accuracy chart that contains 4 models
        4 tables containing the precision and recall score
        Final Table comparing the f-scores, with best-model in red.
        All saved in save_path
    '''
    print("Initiating Model Performance Test...")
    # prepare models
    models = []
    models.append(('Logistic Regression', LogisticRegression(penalty = 'l2')))
    models.append(('Bernoulli NavieBayes', BernoulliNB()))
    models.append(('Multinomia NavieBayes', MultinomialNB()))
    models.append(('Linear SVM', svm.SVC(kernel='linear', decision_function_shape='ovr')))
    X,_, Y, _ = load_data(False,split = 0)
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    print('Running, this may take a few miniutes, Please wait...')
    for name, model in models:
        kfold = KFold(n_fold)
        cv_results = cross_val_score(model, X.toarray(), Y, cv=kfold, scoring=scoring,n_jobs= -1)
        results.append(cv_results)
        names.append(name)
        msg = "%s-> \t mean_acc:%f \t with S.D.:(%f)" % (name.ljust(25)[0:25], cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.2)
    title = "Algorithm Comparison, Based on N-fold:{0}, with_TM_data? {1}".format(n_fold,use_tm)
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.xticks(rotation=20,fontsize = 8)
    plt.ylabel("Accuracy")
#    plt.show()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = save_path+"/model_performance_result.pdf"
    pp = PdfPages(file_path)
    pp.savefig(fig)
    pp.close()
    print("Result chart saved at ",file_path)
    
def model_f_score_test(save_path, mix_tm = True,verbose = False):
    X_train, X_test, y_train, y_test = load_data(False,split = 0.2)
    
    data = X_train, X_test, y_train, y_test
    data_list = np.zeros(shape = [8,4])
    
    PP_clf = BernoulliNB()
    PP_clf.fit(X_train,y_train)
    train_acc,test_acc,precision,recall,f_beta = check_acc(data,PP_clf,"BernoulliNB",verbose= verbose)
    data_list[:,0] = train_acc,test_acc,precision[1],recall[1],f_beta[1],precision[0],recall[0],f_beta[0]
    #winner
    PP_clf = MultinomialNB()
    PP_clf.fit(X_train,y_train)
    train_acc,test_acc,precision,recall,f_beta = check_acc(data,PP_clf,"MultinomialNB",verbose=verbose)
    data_list[:,1] = train_acc,test_acc,precision[1],recall[1],f_beta[1],precision[0],recall[0],f_beta[0]
    
    PP_clf = LogisticRegression(penalty = 'l2') #works better on high dimentional sparse dataset
    PP_clf.fit(X_train,y_train)
    train_acc,test_acc,precision,recall,f_beta  = check_acc(data,PP_clf,"Logistic Regression",verbose=verbose)
    data_list[:,2] = train_acc,test_acc,precision[1],recall[1],f_beta[1],precision[0],recall[0],f_beta[0]
    
    linsvc = svm.SVC(C=10, kernel='linear', decision_function_shape='ovr')
    linsvc.fit(X_train, y_train)
    train_acc,test_acc,precision,recall,f_beta  = check_acc(data,linsvc, "SVC with linear kernel",verbose=verbose)
    data_list[:,3] = train_acc,test_acc,precision[1],recall[1],f_beta[1],precision[0],recall[0],f_beta[0]
    
    columns = ('Bernoulli NB', 'MultinomialNB', 'Logistic Regression', 'SVM with linear kernel')
    rows = ['Training Acc', 'Testing Acc', '+ Precision', '+ Recall','+ F-beta'
            ,'- Precision','- Recall','- F-beta']
    fig = plt.figure()
    fig.subplots_adjust(left=0.25,top=0.9, wspace=2)
    #Table - Main table
    global input_settings_note
    ax = plt.subplot2grid((4,3), (0,0), colspan=3, rowspan=3)
    ax.table(cellText=np.around(data_list,2),
              rowLabels=rows,
              colLabels=columns, loc="upper center")
    ax.set_title("F-score results")
    fig.suptitle(input_settings_note,fontsize=10)
    ax.axis("off")
    fig.set_size_inches(w=6, h=5)
#    plt.show()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = save_path+"/F-score_result.pdf"
    pp = PdfPages(file_path)
    pp.savefig(fig)
    
    pp.close()
    print("Result chart saved at ",file_path)
    
    
    

def train_and_predict(save_path, mix_tm = True):
    pass

if __name__ == "__main__":
#    model_performance_test("../results/2017-12-15/")
    model_f_score_test("../results/2017-12-15/")
















