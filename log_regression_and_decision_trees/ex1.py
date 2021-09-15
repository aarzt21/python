'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 1: Logistic Regression

Authors: Anja Gumpinger, Dean Bodenham, Bastian Rieck
'''

#!/usr/bin/env python3

import pandas as pd
import numpy as np

import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Constants
train_data = "diabetes_train.csv"
test_data = "diabetes_test.csv"

def load_data(path,typ):
    
    """
    Function for loading the data.
    
    """
    file = "{}/{}".format(path,typ)    

    df = pd.read_csv(file)
    X = df.iloc[:, 0:7].values
    y = df.iloc[:, 7].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    ret = [X,y]
    return ret
    
  
    
def load_data2(path,typ):
    
    """
    Function for loading the data.
    
    """
    file = "{}/{}".format(path,typ)    

    df = pd.read_csv(file)
    X = df.iloc[:, [0,1,2,4,5,6]].values
    y = df.iloc[:, 7].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    ret = [X,y]
    return ret
    




def compute_metrics(y_true, y_pred):
    '''
    Computes several quality metrics of the predicted labels and prints
    them to `stdout`.

    :param y_true: true class labels
    :param y_pred: predicted class labels
    '''

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print('TP: {0:d}'.format(tp))
    print('FP: {0:d}'.format(fp))
    print('TN: {0:d}'.format(tn))
    print('FN: {0:d}'.format(fn))
    print('Accuracy: {0:.3f}'.format(accuracy_score(y_true, y_pred)))


if __name__ == "__main__":

    ###################################################################
    # Your code goes here.
    ###################################################################

    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(
        description="Run Logistic Regression")
    
    parser.add_argument(
        "--datadir",
        required=True,
        help="Path to training data")
    

    args = parser.parse_args()

    # Set the paths
    direc = args.datadir


    # Read the training and test data (features are preprocessed inside load_data())
    X_train, y_train = load_data(direc, train_data)
    X_test, y_test = load_data(direc, test_data)

    #Fit model on training data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    #Do predictions on test data
    preds = model.predict(X_test)
    
    #Get performance
    print('Exercise 1.a')
    print('------------')
    
    compute_metrics(y_test,preds)
    
    print(" ")
    print('Exercise 1.b')
    print('------------')
    string = ("For the diabetes dataset I would choose LDA, because it produces less false negatives (28) "
              "and this is the most important in the context of a dangerous disease "
              "because just one false negative could lead to the death of a (untreated) patient.")
    print(string)




    print(" ")
    print('Exercise 1.c')
    print('------------')
    s = ("For another dataset, I would choose Logistic Regression because"
         " it is more robust since it is a discriminative model which makes no assumptions on the features. "
         " LDA is a generative model which assumes normality of the features and it's not "
         " very robust against a violation of this assumption. ")
    print(s)
    
    
    print(" ")
    print('Exercise 1.d')
    print('------------')
    
    print("The two attributes which appear to contribute the most to the prediction are ")
    print("plasma glucose (glu) and diabetes pedigree function (ped) since they have the largest coefficients.")
    
    print("")
    string2 = ("The coefficient for age is 0.43. Calculating the exponential function results "
          "in 1.54, which amounts to an increase in diabetes risk "
          "of 54% percent per additional year.")
    
    print(string2)
    
    
    X_train2, y_train2 = load_data2(direc, train_data)
    X_test2, y_test2 = load_data2(direc, test_data)
    
    #Fit model on training data
    model2 = LogisticRegression()
    model2.fit(X_train2, y_train2)
    
    #Do predictions on test data
    preds2 = model2.predict(X_test2)
    
    #Get performance
    print("")
    print("Performance on reduced dataset:")
    compute_metrics(y_test2,preds2)
    
    print("")
    string3 = ("By comparing the performance and the coefficients obtained on the reduced"
               " dataset with the ones on the model including all the attributes, I observe"
               " that the performance and coefficients remain the same.")
    print(string3)
    print("")
    string4 = ("My explanation is that the coefficient of skin "
               "was the smallest, i.e. the least stat. significant, "
               "and hence contributed the least to model performance."
               " Since it contributed nothing to performance, it should not "
               "make a difference in performance if we remove this covariate, which was confirmed"
               " by the above comparison.")
    print(string4)

    
 
    
    
    

