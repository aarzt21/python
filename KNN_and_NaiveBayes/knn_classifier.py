import os
import numpy as np


class KNNClassifier:
    '''
    A class object that implements the methods of a k-Nearest Neighbor classifier
    The class assumes there are only two labels, namely POS and NEG

    Attributes of the class
    -----------------------
    k : Number of neighbors
    X : A matrix containing the data points (train set)
    y : A vector with the labels
    dist : Distance metric used. Possible values are: 'euclidean', 'hamming', 'minkowski', and others
           For a full list of possible metrics have a look at:
           http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

    HINT: for using the attributes of the class in the class' methods, you can use: self.attribute_name
          (e.g. self.X for accessing the value of the attribute named X)
    '''     
    def __init__(self, X, y, metric):
        '''
        Constructor when X and Y are given.
        
        Parameters
        ----------
        X : Matrix with data points
        Y : Vector with class labels
        metric : Name of the distance metric to use
        '''
        # Default values
        self.verbose = False
        self.k = 1

        # Parameters
        self.X = X
        self.y = y
        self.metric = metric


    def debug(self, switch):
        '''
        Method to set the debug mode.
        
        Parameters
        ----------
        switch : String with value 'on' or 'off'
        '''
        self.verbose = True if switch == "on" else False


    def set_k(self, k):
        '''
        Method to set the value of k.
        
        Parameters
        ----------
        k : Number of nearest neighbors
        '''
        self.k = k


    def _compute_distances(self, X, x):
        '''
        Private function to compute distances. 
        Compute the distance between x and all points in X
    
        Parameters
        ----------
        x : a vector (data point)
        '''
        dis = np.zeros(len(X))
        
        
        if self.metric == "euclidean":
            
            for i in range(len(X)):
                d = np.subtract(x,X[i,:])
                d2 = d**2
                sd2 = d2.sum()
                dis[i]= np.sqrt(sd2)
                
        else:
            pass
        
        return dis


    def predict(self, x):
        '''
        Method to predict the label of one data point.
        Here you actually code the KNN algorithm.
       
        Hint: for calling the method _compute_distance 
              (which is private), you can use: 
              self._compute_distances(self.X, x) 
        
        Parameters
        ----------
        x : Vector from the test data.
        '''
        diss = self._compute_distances(self.X, x) 
        k=self.k
        idx = np.argpartition(diss, k)
        idx=idx[0:k]
        labels = self.y[idx]
        labels=list(labels)
        pred = 1
        
        if labels.count("+") > labels.count("-"):
            pred = 0
            
        #if tie
        elif labels.count("+") == labels.count("-"):
            idx2 = idx[0:(k-1)]
            labels2 = self.y[idx2]
            labels2 = list(labels2)
            if labels2.count("+") > labels2.count("-"):
                pred = 0
            else:
                pred = 1
                
        else: 
            pred =1
            
        return pred
    
            
        
            
