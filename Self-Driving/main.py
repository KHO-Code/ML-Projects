#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import os
import time
import pickle
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import solve



# ---------------------------------------------------new------------------------------------------------------------------------

# master dataset construction that finds where the agent x.y values are
# and puts agent x.y values in the first two columns
# the x.y columns for the other vehicles follow in the same order as they are found

# run this whole block once

num_files = 2308

master_x = []
for i in range(num_files):
    x = pd.read_csv(r"C:/Users/Kevin/Desktop/group/340\train\X\X_{num}.csv".format(num=i))
    
    # extracting where the agent columns are
    for c in range(10):
        if x[' role{n}'.format(n=c)][0] == ' agent':
            break
    
    other_columns = []
    agent_columns = []
    for k in range(10):
        if k==c:
            agent_columns.append(3*(2*k + 1) + 1)
            agent_columns.append(3*(2*k + 1) + 2)
        else:
            other_columns.append(3*(2*k + 1) + 1)
            other_columns.append(3*(2*k + 1) + 2)

    # rearranging the columns such that agent columns are the first two columns
    # and the rest follow in the same order
    agent_columns.extend(other_columns)
    train_columns = x.columns[agent_columns]
    
    N = len(x)
    D = len(train_columns)
    
    # populating a raw matrix from the csv file based on the columns in train_columns
    raw_x = np.zeros((N, D))
    for d in range(D):
        for i in range(N):
            raw_x[i][d] = x[train_columns[d]][i]
    
    master_x.append(raw_x)

time_x = np.array(x['time step'])
master_x = np.array(master_x)

# takes the output x.y values for the agent for the next 3 seconds
# from the Y_## files in train/Y/
# just maps them into master_y

master_y = []
for i in range(num_files):
    y = pd.read_csv(r"C:/Users/Kevin/Desktop/group/340\train\y\y_{num}.csv".format(num=i))
    raw_y = np.array(y[[' x', ' y']])
    master_y.append(raw_y)

time_y = np.array(y['time step'])
master_y = np.array(master_y)

# k specifies the number of rows (past time steps) we want to take
# to predict the future value
# min.k = 1, max.k = T-1
 

N = len(master_x)        # 2308
D = master_x[0].shape[1] # 20
T = master_x[0].shape[0] # 11
k = 2             # lag
d = 2             # how many x.y coordinates to use - 1 means x coordinate of agent, 2 means x and y coordinate of agent

auto_regressed_x = []
auto_regressed_y = []

for i in range(N):
    x = master_x[i]
    raw_x = np.zeros((T-k, D, k)) # 9 x 20 x 2
    raw_y = np.zeros((T-k, D, 1)) # 9 x 20 x 1
    for j in range(T-k):
        inp = x[j:(j+k)].T
        out = x[(j+k)].T
        raw_x[j] = inp
        raw_y[j] = out.reshape((D, 1))
    auto_regressed_x.append(raw_x)
    auto_regressed_y.append(raw_y)

auto_regressed_x = np.array(auto_regressed_x)
auto_regressed_y = np.array(auto_regressed_y)

def appendpop(inp, pred, d, k):
    for selector in range(k*(d-1), -1, -k):
        inp = np.delete(inp, selector)
    
    if d==1:
        inp = np.append(inp, pred)
    else:
        for selector in range(d, 0, -1):
            inp = np.insert(inp, selector, pred[0][selector-1])
    return inp

newx = []
for i in range(N):
    newx.append(auto_regressed_x[i][:,0])
    
newx = np.array(newx)
newx = newx.reshape((N*(T-k), k))

newy = []

for i in range(N):
    newy.append(auto_regressed_y[i][:,0])

newy = np.array(newy)
newy = newy.reshape((N*(T-k), 1))

boss_x = np.copy(newx)
boss_y = np.copy(newy)

for dd in range(1,D):
    newx = []
    for i in range(N):
        newx.append(auto_regressed_x[i][:,dd])
    
    newx = np.array(newx)
    newx = newx.reshape((N*(T-k), k))

    newy = []

    for i in range(N):
        newy.append(auto_regressed_y[i][:,dd])

    newy = np.array(newy)
    newy = newy.reshape((N*(T-k), 1))
    
    boss_x = np.append(boss_x, newx, 1)
    boss_y = np.append(boss_y, newy, 1)



class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w


model = LeastSquares()
model.fit(np.array(boss_x), np.array(boss_y))

boss_predictions = []

for i in range(N):
    x_in = boss_x[(T-k)*(i+1)-1]
    y_in = boss_y[(T-k)*(i+1)-1]
    app = appendpop([x_in], [y_in], d, k)
    
    pred = []
    for i in range(30):
        pred1 = model.predict([app])
        pred.append(pred1[0])
        app = appendpop(app, pred1, d, k)

    pred = np.array(pred)
    boss_predictions.append(pred)

boss_predictions = np.array(boss_predictions)

exp_pred = boss_predictions[0]
exp_output = master_y[0]
boss_error = np.mean(np.mean((exp_pred[:,0] - exp_output[:,0])**2) + np.mean(exp_pred[:,1] - exp_output[:,1])**2)
print(boss_error)

# testing stuff. once anything below this is executed, it's done for

# ---------------------------------------------------new------------------------------------------------------------------------

# master dataset construction that finds where the agent x.y values are
# and puts agent x.y values in the first two columns
# the x.y columns for the other vehicles follow in the same order as they are found

# run this whole block once

num_files_test = 20

master_x_test = []
for i in range(num_files_test):
    x_test = pd.read_csv(r"C:/Users/Kevin/Desktop/group/340\test\X\X_{num}.csv".format(num=i))
    
    # extracting where the agent columns are
    for c in range(10):
        if x_test[' role{n}'.format(n=c)][0] == ' agent':
            break
    
    other_columns_test = []
    agent_columns_test = []
    for k in range(10):
        if k==c:
            agent_columns_test.append(3*(2*k + 1) + 1)
            agent_columns_test.append(3*(2*k + 1) + 2)
        else:
            other_columns_test.append(3*(2*k + 1) + 1)
            other_columns_test.append(3*(2*k + 1) + 2)

    # rearranging the columns such that agent columns are the first two columns
    # and the rest follow in the same order
    agent_columns_test.extend(other_columns_test)
    train_columns_test = x_test.columns[agent_columns_test]
    
    N_test = len(x_test)
    D_test = len(train_columns_test)
    
    # populating a raw matrix from the csv file based on the columns in train_columns
    raw_x_test = np.zeros((N_test, D_test))
    for d in range(D_test):
        for i in range(N_test):
            raw_x_test[i][d] = x_test[train_columns_test[d]][i]
    
    master_x_test.append(raw_x_test)

time_x_test = np.array(x_test['time step'])
master_x_test = np.array(master_x_test)



# k specifies the number of rows (past time steps) we want to take
# to predict the future value
# min.k = 1, max.k = T-1
 

N_test = len(master_x_test)        # 2308
D_test = master_x_test[0].shape[1] # 20
T_test = master_x_test[0].shape[0] # 11
k_test = 2             # lag
d_test = 2             # how many x.y coordinates to use - 1 means x coordinate of agent, 2 means x and y coordinate of agent

auto_regressed_x_test = []
auto_regressed_y_test = []

for i in range(N_test):
    x_test = master_x_test[i]
    raw_x_test = np.zeros((T_test-k_test, D_test, k_test)) # 9 x 20 x 2
    raw_y_test = np.zeros((T_test-k_test, D_test, 1)) # 9 x 20 x 1
    for j in range(T_test-k_test):
        inp_test = x_test[j:(j+k_test)].T
        out_test = x_test[(j+k_test)].T
        raw_x_test[j] = inp_test
        raw_y_test[j] = out_test.reshape((D_test, 1))
    auto_regressed_x_test.append(raw_x_test)
    auto_regressed_y_test.append(raw_y_test)

auto_regressed_x_test = np.array(auto_regressed_x_test)
auto_regressed_y_test = np.array(auto_regressed_y_test)


'''
newx_test = []
for i in range(N_test):
    newx_test.append(auto_regressed_x_test[i][:,0])
    
newx_test = np.array(newx_test)
newx_test = newx_test.reshape((N_test*(T_test-k_test), k_test))

newy_test = []

for i in range(N_test):
    newy_test.append(auto_regressed_y_test[i][:,0])

newy_test = np.array(newy_test)
newy_test = newy_test.reshape((N_test*(T_test-k_test), 1))



newx2_test = []
for i in range(N_test):
    newx2_test.append(auto_regressed_x_test[i][:,1])
    
newx2_test = np.array(newx2_test)
newx2_test = newx2_test.reshape((N_test*(T_test-k_test), k_test))

newy2_test = []

for i in range(N_test):
    newy2_test.append(auto_regressed_y_test[i][:,1])

newy2_test = np.array(newy2_test)
newy2_test = newy2_test.reshape((N_test*(T_test-k_test), 1))



boss_x_test = np.append(newx_test, newx2_test, 1)
boss_y_test = np.append(newy_test, newy2_test, 1)
'''

newx_test = []
for i in range(N_test):
    newx_test.append(auto_regressed_x_test[i][:,0])
    
newx_test = np.array(newx_test)
newx_test = newx_test.reshape((N_test*(T_test-k_test), k_test))

newy_test = []

for i in range(N_test):
    newy_test.append(auto_regressed_y_test[i][:,0])

newy_test = np.array(newy_test)
newy_test = newy_test.reshape((N_test*(T_test-k_test), 1))

boss_x_test = np.copy(newx_test)
boss_y_test = np.copy(newy_test)

for dd in range(1,D_test):
    newx_test = []
    for i in range(N_test):
        newx_test.append(auto_regressed_x_test[i][:,dd])
    
    newx_test = np.array(newx_test)
    newx_test = newx_test.reshape((N_test*(T_test-k_test), k_test))

    newy_test = []

    for i in range(N_test):
        newy_test.append(auto_regressed_y_test[i][:,dd])

    newy_test = np.array(newy_test)
    newy_test = newy_test.reshape((N_test*(T_test-k_test), 1))
    
    boss_x_test = np.append(boss_x_test, newx_test, 1)
    boss_y_test = np.append(boss_y_test, newy_test, 1)


boss_predictions_test = []

for i in range(N_test):
    x_in = boss_x_test[(T_test-k_test)*(i+1)-1]
    y_in = boss_y_test[(T_test-k_test)*(i+1)-1]
    app_test = appendpop([x_in], [y_in], d_test, k_test)
    
    pred_test = []
    for i in range(30):
        pred1_test = model.predict([app_test])
        pred_test.append(pred1_test[0])
        app_test = appendpop(app_test, pred1_test, d_test, k_test)

    pred_test = np.array(pred_test)
    boss_predictions_test.append(pred_test)

boss_predictions_test = np.array(boss_predictions_test)

boss_predictions_test.shape

submission = []
for i in range(N_test):
    file = boss_predictions_test[i]
    for row in file:
        submission.append(row[0])
        submission.append(row[1])

print(len(submission))

submission = np.array(submission)
sub = pd.DataFrame(submission)
sub.to_csv(r"C:/Users/Kevin/Desktop/submission3.csv")


# In[ ]:




