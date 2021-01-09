# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import pandas as pd


# our code
import utils

from knn import KNN
from random_forest import RandomForest
from naive_bayes import NaiveBayes
from stacking import Stacking

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier


def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question
    
    if question == '1':
        filename = "wordvec_train.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            dataset = pd.read_csv(f)       
        filename = "wordvec_test.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            datasetTest = pd.read_csv(f)  
            
        X = np.array(dataset)[1:,:-1]
        y = np.array(dataset)[1:,-1].astype(int)
        
        Xtest = np.array(datasetTest)[1:,:-1]
        ytest = np.array(datasetTest)[1:,-1].astype(int)
        
        #model = RandomForest(1, max_depth = np.inf)
        model = RandomForestClassifier(n_estimators=15)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)
        
        y_pred = model.predict(Xtest)
        te_error = np.mean(y_pred != ytest)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)
        
    if question == '2':
        filename = "wordvec_train.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            dataset = pd.read_csv(f)       
        filename = "wordvec_test.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            datasetTest = pd.read_csv(f)  
            
        X = np.array(dataset)[1:,:-1]
        y = np.array(dataset)[1:,-1].astype(int)
        
        Xtest = np.array(datasetTest)[1:,:-1]
        ytest = np.array(datasetTest)[1:,-1].astype(int)
        
        model = NaiveBayes(num_classes = 2)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)
        
        y_pred = model.predict(Xtest)
        te_error = np.mean(y_pred != ytest)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)    
        
        
    
    if question == '3':
        filename = "wordvec_train.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            dataset = pd.read_csv(f)       
        filename = "wordvec_test.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            datasetTest = pd.read_csv(f)  
            
        X = np.array(dataset)[1:,:-1]
        y = np.array(dataset)[1:,-1].astype(int)
        
        Xtest = np.array(datasetTest)[1:,:-1]
        ytest = np.array(datasetTest)[1:,-1].astype(int)
        
        model = KNN(3)

        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)
        
        y_pred = model.predict(Xtest)
        te_error = np.mean(y_pred != ytest)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    if question == '4':
        filename = "wordvec_train.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            dataset = pd.read_csv(f)       
        filename = "wordvec_test.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            datasetTest = pd.read_csv(f)  
            
        X = np.array(dataset)[1:,:-1]
        y = np.array(dataset)[1:,-1].astype(int)
        
        Xtest = np.array(datasetTest)[1:,:-1]
        ytest = np.array(datasetTest)[1:,-1].astype(int)
        
        #estimators =  [('rf', RandomForestClassifier(n_estimators=15)), ('knn',KNeighborsClassifier(n_neighbors=3)), ('nv', GaussianNB())]
        #model = StackingClassifier(estimators=estimators, final_estimator=KNeighborsClassifier(n_neighbors=3))
        model = Stacking()

        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)
        
        y_pred = model.predict(Xtest)
        te_error = np.mean(y_pred != ytest)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)








