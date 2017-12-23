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
import argparse

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
def load_data(load_tm = False,split = 0.2,mix_tm = False,verbose = True):
    ''' Process data into training and testing set. Shuffle is applied automatically
    Split based on 20% testing and 80% training
    Parameters:
        load_tm: indicate whether to load TM data
        mix_tm: indicate whether to use TM and Non-TM data, will override load_tm if set to True
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
    if mix_tm:
        tm_adder = "*/*"
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
    input_settings_note = "use_TM: {3}, mix_TM: {4} total_X_shape: {0}, ngram: {1},\
 split_size: {2}".format(X_sparse.shape,ngram,split,load_tm,mix_tm)
    if verbose:
        print(input_settings_note)
#    negative_ending = 1087 #246 for tm
#    plot_X(X_sparse[:negative_ending].todense()) #all negative samples
#    plot_X(X_sparse[negative_ending+1:].todense()) #all positive samples

    #split and shuffle
    X_train, X_test, y_train, y_test = train_test_split(X_sparse, y, test_size=split)
    return X_train, X_test, y_train, y_test,transformer

def __load_proteome(vectoriser,verbose = True):
    #load file into dictionaries
    datapath = "../data/proteome/*"
    cwd = os.getcwd()
    if platform=='win32':
        datapath = cwd+"\\"+datapath
        spliter = '\\'
    else:
        datapath = cwd+'/'+ datapath.replace('\\','/')
        spliter = '/'
#    X = []
    pattern = re.compile("[X]+") #remove breaks in the sequence
    proteome_files = {} 
    for file in glob.glob(datapath):
        fasta_sequences = get_file_sequences(file)
        file_dic = {}
        for name,sequence in fasta_sequences.items():
            seq = pattern.sub('',sequence)
            file_dic[name] = vectoriser.transform([seq])
        if verbose:
            print("Found {0} sequences in {1}".format(len(file_dic),file.split(spliter)[-1]))
        proteome_files[file] = file_dic           
    return proteome_files



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



def model_performance_test(save_path,n_fold = 5,use_TM = False,mix_TM = False,verbose = True):
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
    X,_, Y, _,_ = load_data(split = 0,load_tm= use_TM,mix_tm=mix_TM)
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    if verbose:
        print('Running, this may take a few miniutes, Please wait...')
    for name, model in models:
        kfold = KFold(n_fold)
        cv_results = cross_val_score(model, X.toarray(), Y, cv=kfold, scoring=scoring,n_jobs= -1)
        results.append(cv_results)
        names.append(name)
        msg = "%s-> \t mean_acc:%f \t with S.D.:(%f)" % (name.ljust(25)[0:25], cv_results.mean(), cv_results.std())
        if verbose:
            print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.2)
    title = "Algorithm Comparison, N-fold:{0}, Mixed_TM?:{2}, TM_data?: {1}".format(n_fold,use_TM,mix_TM)
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
    if verbose:
        print("Result chart saved at ",file_path)
    
def model_f_score_test(save_path, use_TM = False, mix_tm = False,verbose = False):
    '''Run a f-score test, save the results in save_path, the saved file will be called F-score_result.pdf
    Parameters:
        save_path: file path such as ../results/2017-12-12, that will be used to save output file
        mix_tm, indicate whether to mix_tm data for the f-score test
    '''
    X_train, X_test, y_train, y_test,_ = load_data(use_TM,split = 0.2,mix_tm = mix_tm,verbose = verbose)
    
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
    if verbose:
        print("Result chart saved at ",file_path)
    
    

def train_and_predict(save_path, threshold = 0.5,mix_tm = True,verbose = True):
    '''Train model using logistic regression, with n-gram 3:4, split 0
    The trained model will be applied to predict proteomes under /data/proteome , must be in fasta format
    Parameters:
        save_path: file path such as ../results/2017-12-12, that will be used to save output files
        mix_tm, indicate whether to mix_tm data for training
    Outputs:
        Each input proteome will associate with an output txt file containing accession and prediction
        EG, >ACADADSA,1 indicate the accession ACADADSA contains a signal peptide in it.
    '''
    if platform=='win32':
        spliter = '\\'
    else:
        spliter = '/'
    
    if verbose:
        print("Load training dataset and training...")
    X_train, _, y_train, _,transformer = load_data(True,split = 0,mix_tm = mix_tm,verbose = verbose)
    PP_clf = LogisticRegression(penalty = 'l2') #works better on high dimentional sparse dataset
    PP_clf.fit(X_train,y_train)
    
    if verbose:
        print("Loading all proteomes, please wait...")
    proteomes = __load_proteome(transformer,verbose)
    
    for file_path,file_dic in proteomes.items():
        file_lines = list()
        true_count = 0
        for acc,vec_seq in file_dic.items():
            r = PP_clf.predict_proba(vec_seq)
            if r[0][1] >= threshold:
                file_lines.append(acc+",1")
                true_count += 1
            else:
                file_lines.append(acc+",0")
        #save the file
        if verbose:
            print("# positive predictions: {0}/{1} ({2:0.2f}%) for file {3}".format(true_count,len(file_dic),(true_count/len(file_dic))*100,
                  file_path.split(spliter)[-1]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = save_path+"{1}{0}_pred.txt".format(file_path.split(spliter)[-1],spliter)
        print(file_path)
        with open(file_path,'w') as file:
            file.write("\n".join(file_lines))
    if verbose:
        print("all files saved under {0}".format(save_path))
        
            
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Signal peptide prediction, Train/Test/Predict the following models:\
                                     Bernoulli NB, \
                                     Multinomial NB, \
                                     Logistic Regression with l2 penalty, \
                                     Linear SVM')
    group = parser.add_mutually_exclusive_group()
    performance_group = parser.add_argument_group()
    parser.add_argument('save_path',
                        help='Path to save outout result, eg ../results/2017-12-31')
    performance_group.add_argument('-pt', "--performance_test",action="store_true",
                        help='Run performance test, save to save_path')
    performance_group.add_argument('--nfold',type = int, help = "indicate how many n-fold",default = 3)
    
    f_test_group = parser.add_argument_group()
    f_test_group.add_argument('-f','--ftest',action="store_true",
                        help = "Run f-score test, save to save_path")
    
    predict_group = parser.add_argument_group()
    predict_group.add_argument('-p',"--prediction",help = "Train on LR and predict proteomes under /data/proteomes/ directory",action = "store_true")
    predict_group.add_argument('-t', "--threshold",default = 0.5, type = float,
                        help = "Indicate the cut-off threshold for positive samples, default 0.5")
    group.add_argument('-i', "--input_mode",default = 0, type = int,
                        help = "input data mix-mode,  1 = non-TM only, 2 = TM only, otherwise mix-mode will be used")
    parser.add_argument("-q",'--quiet', default = False,action="store_true",
                        help = "quiet mode")
    
    args = parser.parse_args()
    save_path = args.save_path
    if args.input_mode == 1:
        mix_tm = False
        use_TM = False
    elif args.input_mode == 2:
        mix_tm = False
        use_TM = True
    else:
        mix_tm = True
        use_TM = True
    verbose = True
    if args.nfold <= 1:
        print("N-fold must be > than 1!",file = sys.stderr)
        sys.exit()
    if args.quiet:
        verbose = False
    if args.ftest:
        model_f_score_test(save_path,use_TM= use_TM,mix_tm=mix_tm,verbose = verbose)
    if args.performance_test:
        model_performance_test(save_path,n_fold=args.nfold,verbose = verbose,use_TM= use_TM,mix_TM= mix_tm)
    if args.prediction:
        train_and_predict(save_path,threshold=args.threshold,verbose = verbose,mix_tm = mix_tm)
#    save_path = "../results/2017-12-15/"
















