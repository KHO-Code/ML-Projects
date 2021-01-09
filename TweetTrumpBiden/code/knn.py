"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils
from sklearn.metrics.pairwise import cosine_distances

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y

    def predict(self, Xtest):
        X = self.X
        y = self.y
        n = X.shape[0]
        t = Xtest.shape[0]
        k = min(self.k, n)

        # Compute cosine_distance distances between X and Xtest
        dist2 = self.cosine_distance(X, Xtest)


        # yhat is a vector of size t with integer elements
        yhat = np.ones(t, dtype=np.uint8)
        for i in range(t):
            # sort the distances to other points
            inds = np.argsort(dist2[:,i])

            # compute mode of k closest training pts
            yhat[i] = stats.mode(y[inds[:k]])[0][0]

        return yhat

    def cosine_distance(self,X1,X2):
         NormX1 = np.linalg.norm(X1)
         NormX2 = np.linalg.norm(X2)
         
         numerator = np.dot(X1,X2.transpose())
         denominator = NormX1 * NormX2
         
         cosineSimilarity = numerator/denominator
         cosineDist = 1 - cosineSimilarity
         #return cosineDist
         return cosine_distances(X1,X2)
         
         if NormX1 == 0 or NormX2 == 0:
              return 0
        

