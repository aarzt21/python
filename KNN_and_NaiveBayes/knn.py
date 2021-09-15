"""
Homework : k-Nearest Neighbor and Naive Bayes
Course   : Data Mining (636-0018-00L)

Main program for k-NN.
Predicts the labels of the test data using the training data.
The k-NN algorithm is executed for different values of k (user-entered parameter)


Original author: Damian Roqueiro <damian.roqueiro@bsse.ethz.ch>
Extended by: Bastian Rieck <bastian.rieck@bsse.ethz.ch>
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Import the file with the performance metrics 
import evaluation

# Class imports
from knn_classifier import KNNClassifier


# Constants
# 1. Files with the datapoints and class labels
DATA_FILE  = "matrix_mirna_input.txt"
PHENO_FILE = "phenotype.txt"


# 2. Classification performance metrics to compute
PERF_METRICS = ["accuracy", "precision", "recall"]


def load_data(dir_path): 
    """
    Function for loading the data.
    Receives the path to a directory that will contain the DATA_FILE and PHENO_FILE.
    Loads both files into memory as numpy arrays. Matches the patientId to make
    sure the class labels are correctly assigned.

    Returns
     X : a matrix with the data points
     y : a vector with the class labels
    """

    file = "{}/{}".format(dir_path,DATA_FILE)        
    file2 = "{}/{}".format(dir_path, PHENO_FILE)
 
    X = pd.read_csv(file, sep="\t")
    y = pd.read_csv(file2, sep="\t")
    
    if not (y.iloc[:,0]==X.iloc[:,0]).all():
        raise Exception("patientId doesn't match")
        
    else:
        X = X.iloc[:, 1:]
        y = y.iloc[:, 1:]
        X = np.array(X)
        y = np.array(y)
        ret = [X,y]
        return ret
        


def obtain_performance_metrics(y_true, y_pred): # (TO DO)
    """
    Function obtain_performance_metrics
    Receives two numpy arrays with the true and predicted labels.
    Computes all classification performance metrics.
    
    In this function you might call the functions:
    compute_accuracy(), compute_precision(), compute_recall()
    from the evaluation.py file. You can call them by writing:
    evaluation.compute_accuracy, and similarly.

    Returns a vector with one value per metric. The positions in the
    vector match the metric names in PERF_METRICS.
    """
    vec = np.zeros((1,3))
    vec[0,0] = evaluation.compute_accuracy(y_true, y_pred)
    vec[0,1] = evaluation.compute_precision(y_true, y_pred) 
    vec[0,2] = evaluation.compute_recall(y_true, y_pred)
    return vec



#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------

if __name__ == '__main__':
    
    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(
        description="Compute KNN")
    
    parser.add_argument(
        "--traindir",
        required=True,
        help="Path to training data")
    
    parser.add_argument(
        "--testdir",
        required=True,
        help="Path to test data")
    
    parser.add_argument(
        "--outdir",
        required=True,
        help="Path to directory where output_knn.txt will be created")
    
    parser.add_argument(
        "--mink",
        required=True,
        type=int,
        help="min value for k")
    
    parser.add_argument(
        "--maxk",
        required=True,
        type=int,
        help="max value for k")
        
    
    args = parser.parse_args()

    # Set the paths
    out_dir = args.outdir
    train_dir = args.traindir
    test_dir = args.testdir
    mink = args.mink
    maxk = args.maxk
    

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    
    
    # Read the training and test data. For each dataset, get also the true labels.
    # Use the function load_data().
    # Important: Match the patientId between data points and class labels
    X_train, y_train = load_data(train_dir)
    
    X_test, y_test = load_data(test_dir)
    y_true = np.where(y_test == "+", 0, 1)
    
    # Create the output file & write the header as specified in the homework sheet
    try:
        file_name = "{}/output_knn.txt".format(args.outdir)
        f_out = open(file_name, 'w')
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)      
        
    f_out.write('{}\t{}\t{}\t{}\n'.format(
        'Value of k','Accuracy', 'Precision','Recall'))


    ############################## KNN algorithm ####################################

    # Create the k-NN object. (Hint about how to do it in the homework sheet)
    knn = KNNClassifier(X_train,y_train, metric="euclidean")

    # Iterate through all possible values of k:
    # HINT: remember to set the number of neighbors for the KNN object through: knn_obj.set_k(k)
    for k in range(mink, maxk):
        knn.set_k(k)
        print(k)
        
    # 1. Perform KNN training and classify all the test points. In this step, you will
    # obtain a prediction for each test point. 
        preds=np.zeros((len(X_test),1),dtype=int)
        
        for i in range(len(X_test)):
            x = X_test[i,:]
            preds[i] = knn.predict(x)
            #print(knn.predict(x))
        
    # 2. Compute performance metrics given the true-labels vector and the predicted-
    # labels vector (you might consider to use obtain_performance_metrics() function)

        perf = obtain_performance_metrics(y_true, preds)
        print(perf)
    
        
    # 3. Write performance results in the output file, as indicated the in homework
    # sheet. Close the file.

        f_out.write('{}\t{}\t{}\t{}\n'.format(k,perf[0,0],perf[0,1], perf[0,2]))

    f_out.close()