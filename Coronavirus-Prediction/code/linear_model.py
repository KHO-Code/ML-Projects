import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares): # inherits the predict() function from LeastSquares
    def fit(self,X,y,z):
        n, d = X.shape

        self.w = solve(X.T@z@X, X.T@z@y)

'''
class LinearModelGradient(LeastSquares):

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))
'''
class LinearModelGradient(LeastSquares):

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d, dtype=np.float128)

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):

        def enum(n):
            return np.exp(w.T@X[n] - y[n])

        def denom(n):
            return np.exp(y[n] - w.T@X[n])

        f = 0
        for n in range(X.shape[0]):
            f = f + np.log(enum(n) + denom(n))

        g = 0
        for n in range(X.shape[0]):
            g = g + (X[n] * (enum(n) - denom(n))) / (enum(n) + denom(n))

        return (f,g)

'''
    def funObj(self,w,X,y):

        # Calculate the function value
        f = 0.5*np.sum((X@w - y)**2)

        # Calculate the gradient value
        g = X.T@(X@w-y)

        return (f,g)
'''
    


# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        Z = np.c_[np.ones(X.shape[0]), X]
        self.w = solve(Z.T@Z, Z.T@y)

    def predict(self, X):
        return np.c_[np.ones(X.shape[0]), X]@self.w

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        Z = self.__polyBasis(X)
        self.w = solve(Z.T@Z, Z.T@y)

    def predict(self, X):
        return self.__polyBasis(X)@self.w

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        Z = np.ones((X.shape[0], self.p + 1))
        for i in range(X.shape[0]):
            for j in range(self.p + 1):
                Z[i, j] = X[i]**j
        return Z


class AutoRegress:
    
    def __init__(self, K=200):
        self.K = K

    def fit(self,X):
        N = X.shape[0]
        K = self.K

        X_train = np.ones((N - K, K + 1))
        y_train = X[K:N].astype(int)
        for k in range(K):
            X_train[:, k+1] = X[k:N-K+k]

        # least squares can/needs to be changed into a better fit regression model
        model = LeastSquares()
        model.fit(X_train,y_train)

        self.X = X_train
        self.y = X
        self.model = model
        self.w = model.w

    def step(self):
        N = self.X.shape[0]
        K = self.K

        X_new = np.ones(K + 1)
        X_new[1:] = self.y[-K:]
        y_pred = int(self.model.predict(X_new))

        self.X = np.vstack([self.X, X_new])
        self.y = np.append(self.y, y_pred)

        return y_pred

    def predict(self, num_preds):
        y_pred = np.zeros(num_preds)

        for n in range(num_preds):
            y_pred[n] = self.step()

        return y_pred
