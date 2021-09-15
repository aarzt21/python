#!/usr/bin/env python3

'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 2: Decision Trees

Authors: Anja Gumpinger, Bastian Rieck
'''

import numpy as np
import sklearn.datasets
from math import log2


if __name__ == '__main__':

    iris = sklearn.datasets.load_iris()
    X = iris.data
    y = iris.target

    feature_names = iris.feature_names
    num_features = len(set(feature_names))
    
    # helper functions for main function
    def split_data(X, y, attribute_index, theta):
        """helper function to split data"""
    
        sub = attribute_index 
        idx1 = X[:,sub] < theta
        idx2 = X[:,sub] >= theta
        
        X1, X2 = X[idx1] , X[idx2]  
        y1, y2 = y[idx1], y[idx2]
    
        return [X1, X2, y1, y2]
        
    
    def compute_information_content(y):
        """
        helper function that computes the information content of a subset X with
        labels y
        """
        c = np.bincount(y)
        c = c[np.nonzero(c)]
        probs = c/len(y)
        info = -sum(probs*np.log2(probs))
        return info

        
    def compute_information_a(X, y, attribute_index, theta):
        """
        helper function to compute Info_A(X) for dataset X with labels y that is split
        according to the split defined by the pair (attribute_index, theta)
        """
        X1, X2, y1, y2 = split_data(X, y, attribute_index, theta)
        inf1 = compute_information_content(y1)
        inf2 = compute_information_content(y2)
        
        info_a = ((len(X1)* inf1 +  len(X2)*inf2)/ len(X)  )
        return info_a
    
    
    # Main function
    
    def compute_information_gain(X, y, attribute_index, theta):
        """
        This function computes the information gain of a split for a feature
        of the Iris data set.
        
        attribute_index = 0: sepal length
        attribute_index = 1: sepal width
        attribute_index = 2: petal length
        attribute_index = 3: sepal width
        
        E.g.: If attribute_index = 3 and theta = 1.0 then split is "is petal width < 1.0 ?"
        
        """
        info = compute_information_content(y)
        info_a = compute_information_a(X, y, attribute_index, theta)
        res = info-info_a
        
        return res 

    a1 = round(compute_information_gain(X, y, attribute_index=0, theta=5),2)
    a2 = round(compute_information_gain(X, y, attribute_index=1, theta=3),2)
    a3 = round(compute_information_gain(X, y, attribute_index=2, theta=2.5),2)
    a4 = round(compute_information_gain(X, y, attribute_index=3, theta=1.5),2)

    print('Exercise 2.b')
    print('------------')
    print("Split ( sepal length (cm) < 5.0 ): information gain = {}".format(a1))
    print("Split ( sepal width (cm) < 3.0 ): information gain = {}".format(a2))
    print("Split ( petal length (cm) < 2.5 ): information gain = {}".format(a3))
    print("Split ( petal width (cm) < 1.5 ): information gain = {}".format(a4))
    

    print('')

    print('Exercise 2.c')
    print('------------')
    print("I would select ( petal length (cm) < 2.5 ) to be the first split, because it has the largest information gain (0.92), i.e. it reduces the cost the most. ")
    print('')

    ####################################################################
    # Exercise 2.d
    ####################################################################


    print('Exercise 2.d')
    print('------------')
    # Do _not_ remove this line because you will get different splits
    # which make your results different from the expected ones...
    np.random.seed(42)
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import KFold
    from sklearn.metrics import  accuracy_score
    
    cv = KFold(n_splits=5, shuffle=True)
    clf = DecisionTreeClassifier()
    
    i = 0
    scores = np.zeros((5,))
    importances = np.zeros((5,4))
    
    for train_idx, test_idx in cv.split(X):
        #print("TRAIN:", train_idx, "TEST:", test_idx)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf = DecisionTreeClassifier()
        clf.fit(X_train,y_train)
        importances[i] = clf.feature_importances_
        
        preds = clf.predict(X_test)
        scores[i] = accuracy_score(y_test,preds)
        i += 1
        
    a5 = round(np.mean(scores)*100,2) 
    a6 = importances.mean(axis=0)
    
    print("")
    print('The mean accuracy is {}'.format(a5))
 
    print('')
    print('For the original data, the two most important features are:')
    print(" ")
    print("-> petal width with the following feature importance scores in each training fold of the cross-validation: {}".format(importances[:, 3]))
    print(" ")
    print("-> petal length with the following feature importance scores in each training fold of the cross-validation: {}".format(importances[:, 2]))
    
    print("")
    print("The feature importance scores obtained from each training set of the cross validation are (rows: data set, columns: feature (index)):")
    print("")
    print(importances)
    print("")
    print("As we can see, petal width (column 4) and petal length (column 3) are more important than the other features over all training sets. ")


    print("")
    print('')
    print('For the reduced data, the most important feature is:')
    
    
    
    X = X[y != 2]
    y = y[y != 2]
    np.random.seed(42)
    cv = KFold(n_splits=5, shuffle=True)
    i = 0
    scores = np.zeros((5,))
    importances = np.zeros((5,4))
    
    
    for train_idx, test_idx in cv.split(X):
        #print("TRAIN:", train_idx, "TEST:", test_idx)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf = DecisionTreeClassifier()
        clf.fit(X_train,y_train)
        importances[i] = clf.feature_importances_
        
        preds = clf.predict(X_test)
        scores[i] = accuracy_score(y_test,preds)
        i += 1
        
    round(np.mean(scores)*100,2) 
    a7 = importances.mean(axis=0)
        
    print("-> petal length with the following feature importance scores in each training fold of the cross-validation: {}".format(importances[:, 2]))
        
    print("")
    print("The feature importance scores obtained from each reduced training set of the cross validation are (rows: data set, columns: feature (index)):")
    print("")
    print(importances)
    print("")
    
    
    a8 = ("As we can see, petal length (column 3) is able to seperate 4 out of 5 training sets by just itself. This implies that it's likely the only relevant feature for the reduced data set."
          " Further, all hold out folds are perfectly separable since the mean accuracy from the cross validation is 100 percent.")
    print(a8)
    
    
















