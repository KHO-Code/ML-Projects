# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time


# sklearn imports
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression


# our code
import utils

'''
from decision_stump import DecisionStumpErrorRate, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest
'''
import linear_model

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "autoregression":
        data_file = "phase1_training_data.csv"
        with open(os.path.join("..", "data", data_file), "rb") as f:
            data = pd.read_csv(f, sep=',',header=0)

        # Paramaters
        # placeholder
        countries = ["CA", "JP", "US"]
        columns = ["cases_100k", "cases_14_100k", "deaths", "cases"]
        num_preds = 10

        num_days = data[["country_id"]].values
        num_days = num_days[np.argwhere(num_days[:, 0] == "CA"), 0].flatten().shape[0]
        X = np.empty((num_days + num_preds, 0))
        K = np.array([])

        # Group of parameters for feature selection
        err_arr = np.array([])

        for j in range(len(countries)):
            print("countries: ", countries[j])
            for i in range(len(columns)):
                if countries[j] == "CA" and columns[i] == "deaths":
                    continue
                print("column: ", columns[i])
                X_reg = data[["country_id", columns[i]]].values
                X_reg = X_reg[np.argwhere(X_reg[:, 0] == "CA"), 1].flatten()

                N = int(X_reg.shape[0])
                n = int(N/4)

                X_train = X_reg[:N-n-1]
                # Add noise to avoid singular matrix
                noise = np.random.normal(0, 0.001, X_train.shape)
                X_train = X_train + noise

                # Validate
                y_val = X_reg[N-n:]
                min_err = np.inf
                min_k = 0
                for k in range(n-1):
                    model = linear_model.AutoRegress(K=k+1)
                    # try:
                    model.fit(X_train)
                    # except np.linalg.LinAlgError as err:
                    #     if 'Singular matrix' in str(err):
                    #         continue
                    #     else:
                    #         raise
                    y_pred_val = model.predict(y_val.shape[0])
                    err = np.abs(np.mean(y_pred_val - y_val))

                    if err < min_err:
                        min_err = err
                        min_k = k
                
                
                # Predict
                model = linear_model.AutoRegress(K=min_k+1)
                model.fit(X_reg)
                y_pred = model.predict(num_preds)
                column_curr = np.empty((num_days + num_preds, 1))
                column_curr[:, 0] = np.append(X_reg, y_pred)[:]
                X = np.append(X, column_curr, axis=1)
                # X[num_days:, (j * 4) + i] = y_pred
                err_arr = np.append(err_arr, min_err)

        # Drop features with err_rate that's too high
        # err_rate threshold can be up for discussion
        # TODO should probably determine the outliers using Kmeans
        idx = np.argwhere(err_arr > np.mean(err_arr))
        X = np.delete(X, idx, axis=1)
        
        # Validate
        M = int(num_days / 2)
        
        X_train = X[:M, :]
        X_test = X[M:-num_preds, :]
        y = data[["country_id", "deaths"]].values
        y = y[np.argwhere(y[:, 0] == "CA"), 1].flatten().astype(float)
        y_train = y[:M]
        y_test = y[M:]
        print(X_train.astype(int))
        print(y_train.astype(int))
        
        # Add noise to avoid singular matrix
        X_noise = np.random.normal(0, 0.001, X_train.shape)
        X_train = X_train + X_noise
        y_noise = np.random.normal(0, 0.001, y_train.shape)
        y_train = y_train + y_noise
        
        model = linear_model.LeastSquares()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(np.abs(np.mean(y_pred - y_test)))
        

        # Predict
        X_train = X[:num_days, :]
        X_pred = X[num_days:, :]
        y_train = data[["country_id", "deaths"]].values
        y_train = y_train[np.argwhere(y_train[:, 0] == "CA"), 1].flatten().astype(float)

        # Add noise to avoid singular matrix
        X_noise = np.random.normal(0, 0.001, X_train.shape)
        X_train = X_train + X_noise
        y_noise = np.random.normal(0, 0.001, y_train.shape)
        y_train = y_train + y_noise
        
        model = linear_model.LeastSquares()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_pred)
        print(model.w)
        print(y_pred.astype(int))
       

        
    else:
        print("Unknown question: %s" % question)