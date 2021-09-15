# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 13:00:16 2020

@author: axarz
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

#Constant
data_file  = "tumor_info.txt"

def load_data(dir_path): 
    """
    Function for loading the data.

    """

    file = "{}/{}".format(dir_path, data_file)        
 
    data = pd.read_csv(file, sep="\t", header=None)
    
    return data 
        






if __name__ == '__main__':
    
    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(
        description="Compute Naive Bayes")
    
    parser.add_argument(
        "--traindir",
        required=True,
        help="Path to training data")
    
    parser.add_argument(
        "--outdir",
        required=True,
        help="Path to directory where output_summary_class_<label>.txt will be created")
    
 
    args = parser.parse_args()

    # Set the paths
    out_dir = args.outdir
    train_dir = args.traindir


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    data = load_data(train_dir)
    data = data.rename(columns={0: 'clump', 1:"uniformity" ,
                                2:"marginal", 3:"mitoses",4:"label"})
    
    data_class_2 = data[data.label==2]
    data_class_2 = data_class_2.iloc[:,0:4]
    
    data_class_4 = data[data.label==4]
    data_class_4 = data_class_4.iloc[:,0:4]
    
    
    #class 2 only for now
    res_2 = np.zeros((10,4))
    
    for i in range(1,11):
        for j in range(0,4):
            res_2[i-1,j] = round(sum(data_class_2.iloc[:,j]==i)/sum(~np.isnan(data_class_2.iloc[:,j]))
                                 ,3)
            
    
    
    res_4 = np.zeros((10,4))
    
    for i in range(1,11):
        for j in range(0,4):
            res_4[i-1,j] = round(sum(data_class_4.iloc[:,j]==i)/sum(~np.isnan(data_class_4.iloc[:,j]))
                                 ,3)
            
    
       # Create the output file & write the header as specified in the homework sheet
    try:
        file_name = "{}/output_summary_class_2.txt".format(out_dir)
        f_out = open(file_name, 'w')
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)      
        
    f_out.write('{}\t{}\t{}\t{}\t{}\n'.format(
        'Value','clump','uniformity',"marginal",'mitoses'))
    
    for i in range(10):
        f_out.write('{}\t{}\t{}\t{}\t{}\n'.format((i+1),res_2[i,0], res_2[i,1], 
                                              res_2[i,2], res_2[i,3]))

    f_out.close() 
    
    
    
    
    
    try:
        file_name = "{}/output_summary_class_4.txt".format(out_dir)
        f_out2 = open(file_name, 'w')
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)      
        
    f_out2.write('{}\t{}\t{}\t{}\t{}\n'.format(
        'Value','clump','uniformity',"marginal",'mitoses'))
    
    for i in range(10):
        f_out2.write('{}\t{}\t{}\t{}\t{}\n'.format((i+1),res_4[i,0], res_4[i,1], 
                                              res_4[i,2], res_4[i,3]))

    f_out2.close() 
    



    
    

    





