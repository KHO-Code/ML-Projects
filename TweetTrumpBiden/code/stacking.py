import numpy as np
import utils
from random_forest import RandomForest
from knn import KNN
from naive_bayes import NaiveBayes
from random_forest import DecisionTree

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
class Stacking():

    def __init__(self):
        pass

    def fit(self, X, y):
        N,D = X.shape
        rfModel = RandomForestClassifier(n_estimators=50)
        nbModel = NaiveBayes(num_classes = 2)
        knnModel = KNN(3)
        
        knnModel.fit(X,y)
        knn_y_pred = knnModel.predict(X).astype(int)
        
        nbModel.fit(X,y)
        nb_y_pred = nbModel.predict(X).astype(int)
        
        rfModel.fit(X,y)
        rf_y_pred = rfModel.predict(X).astype(int)

        Xy_label_combined = np.array((knn_y_pred,nb_y_pred,rf_y_pred)).transpose()
        
        self.Xy_label_combined = Xy_label_combined
        self.y = y
        
        
    def predict(self, X):
        N, D = X.shape
        y_pred = np.zeros(N)
        Xy_label_combined = self.Xy_label_combined
        
        y = self.y
        
        model = DecisionTreeClassifier()
        model.fit(Xy_label_combined, y)
        y_pred = model.predict(Xy_label_combined)
        
        return y_pred
    
    
    
    