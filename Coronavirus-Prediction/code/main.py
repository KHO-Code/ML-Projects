# basics
import os
import pickle
import argparse
import pandas as pd
import numpy as np
#10/26/20 -> 10/30/20 PREDICT DEATHS IN CANADA FOR DATES 10/26/20 -> 10/30/20 OCT 26 to OCT 30 2020

import linear_model

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":
        data_file = "dataCS2.csv"
        with open(os.path.join("..", "data", data_file), "rb") as f:
            data = pd.read_csv(f, sep=',',header=0)

        # Paramaters
        # placeholder
        countries = ["CA"]
        columns = ["cases_100k", "cases_14_100k", "deaths", "cases"]
        num_preds = 5

        num_days = data[["country_id"]].values
        num_days = num_days[np.argwhere(num_days[:, 0] == "CA"), 0].flatten().shape[0]
        X = np.zeros((num_days + num_preds, len(countries) * len(columns)))
        K = np.array([])


        for j in range(len(countries)):
            for i in range(len(columns)):
                #print("column: ", columns[i])
                X_reg = data[["country_id", columns[i]]].values
                X_reg = X_reg[np.argwhere(X_reg[:, 0] == "CA"), 1].flatten()
                X[:num_days, (j * 4) + i] = X_reg

                N = int(X_reg.shape[0])
                n = int(N/3)

                X_train = X_reg[:N-n-1]

                # Validate
                y_val = X_reg[N-n:]
                min_err = np.inf
                min_k = 0
                for k in range(n-1):
                    model = linear_model.AutoRegress(K=k+1)
                    try:
                        model.fit(X_train)
                    except np.linalg.LinAlgError as err:
                        if 'Singular matrix' in str(err):
                            continue
                        else:
                            raise
                    y_pred_val = model.predict(y_val.shape[0])
                    err = np.abs(np.mean(y_pred_val - y_val))

                    if err < min_err:
                        min_err = err
                        min_k = k
                    
                    # prevents overflow
                    if min_err**min_err < err:
                        break
                
                # Predict
                model = linear_model.AutoRegress(K=min_k+1)
                model.fit(X_reg)
                y_pred = model.predict(num_preds)
                X[num_days:, (j * 4) + i] = y_pred
        
        # Validate
        M = int(num_days / 2)
        
        X_train = X[:M, :]
        X_test = X[M:-num_preds, :]
        y = data[["country_id", "deaths"]].values
        y = y[np.argwhere(y[:, 0] == "CA"), 1].flatten().astype(float)
        y_train = y[:M]
        y_test = y[M:]
        
        model = linear_model.LeastSquares()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        #print(np.abs(np.mean(y_pred - y_test)))
        #print(y_pred)
        

        # Predict
        X_train = X[:num_days, :]
        X_pred = X[num_days:, :]
        y_train = data[["country_id", "deaths"]].values
        y_train = y_train[np.argwhere(y_train[:, 0] == "CA"), 1].flatten().astype(float)
        
        model = linear_model.LeastSquares()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_pred)
        #print(model.w)
        #print total deaths each predicted day
        print(y_pred.astype(int))
       

        
 