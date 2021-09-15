"""
Homework : k-Nearest Neighbor and Naive Bayes
Course   : Data Mining (636-0018-00L)

Auxiliary functions.

This file implements the metrics that are invoked from the main program.

Author: Damian Roqueiro <damian.roqueiro@bsse.ethz.ch>
Extended by: Bastian Rieck <bastian.rieck@bsse.ethz.ch>
"""

import numpy as np


def confusion_matrix(y_true, y_pred):
    '''
    Function for calculating TP, FP, TN, and FN.
    The input includes the vector of true labels
    and the vector of predicted labels
    '''
    m = np.zeros((2, 2))
    t = y_true
    p = y_pred 
    
    for a,b in zip(t,p):
        m[b,a] +=1
    
    return m



def compute_precision(y_true, y_pred):
    """
    Function: compute_precision
    Invoke confusion_matrix() to obtain the counts
    """
    m = confusion_matrix(y_true,y_pred)
    prec = m[0,0]/(m[0,0]+m[0,1])
    return round(prec,2)

def compute_recall(y_true, y_pred):
    """
    Function: compute_recall
    Invoke confusion_matrix() to obtain the counts
    """
    m = confusion_matrix(y_true,y_pred)
    rec = m[0,0]/(m[0,0]+m[1,0])
    return round(rec,2)


def compute_accuracy(y_true, y_pred):
    """
    Function: compute_accuracy
    Invoke the confusion_matrix() to obtain the counts
    """
    
    m = confusion_matrix(y_true,y_pred)
    acc = (m[0,0] + m[1,1])/(m[0,0] + m[1,1] + m[1,0] +m[0,1])
    return round(acc,2)

