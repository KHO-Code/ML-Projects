import numpy as np


class NaiveBayes:

    def __init__(self, num_classes, beta = 0):
        self.num_classes = num_classes
        self.beta = beta
        
    def fit(self, X, y):
        N, D = X.shape
        p_y = np.bincount(y) / N
        mean = np.zeros((self.num_classes,D))
        var = np.zeros((self.num_classes,D))

        for c in range(self.num_classes):
            for d in range(D):
                mean[c,d] = np.mean(X[y==c,d])
                var[c,d] = np.var(X[y==c,d])
                
                
        self.mean = mean
        self.var = var
        self.p_y = p_y
        
    def predict(self, X):
        p_y = self.p_y
        N, D = X.shape
        mean = self.mean
        y_pred = np.zeros(N)
        var = self.var
    
        for n in range(N):
            probs = p_y.copy()
            for d in range(D):
                if X[n, d] != 0:
                   probs += -1 *((0.5*((X[n,d]-mean[:,d])/np.sqrt(var[:,d]))**2)+np.log(np.sqrt(var[:,d])*np.sqrt(2*np.pi)))
                else:
                     probs += 1 - probs
                
            y_pred[n] = np.argmax(probs)
            
        return y_pred






