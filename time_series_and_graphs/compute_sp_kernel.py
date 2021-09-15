# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:27:01 2020

@author: Alexander Arzt 14-617-070
"""


import numpy as np
import os
import sys
import scipy.io
from shortest_path_kernel import floyd_warshall
from shortest_path_kernel import sp_kernel
import argparse


if __name__ == '__main__':

    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(
        description="Compute SP kernel similarities")
    
    parser.add_argument(
        "--datadir",
        required=True,
        help="Path to input directory containing file MUTAG.mat")
    
    parser.add_argument(
        "--outdir",
        required=True,
        help="Path to directory where graphs_output.txt will be created")

    args = parser.parse_args()

    # Set the paths
    data_dir = args.datadir
    out_dir = args.outdir

    os.makedirs(args.outdir, exist_ok=True)

    # Load the data
    mat = scipy.io.loadmat("{}/{}".format(args.datadir, 'MUTAG.mat'))
    label = np.reshape(mat["lmutag"], (len(mat["lmutag"],)))
    data = np.reshape(mat["MUTAG"]["am"], (len(label), ))     

    # Create the output file
    try:
        file_name = "{}/graphs_output.txt".format(args.outdir)
        f_out = open(file_name, 'w')
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)      
        
    
    f_out.write('{}\t{}\n'.format(
        'Pair of classes',
        'SP'))
    

# Compute the average SP kernel similarities between the mutagenic and the non muta class
# Mutagenic = 1, non-mutagenic = -1
    
    #Mutagenic:Mutagenic
    
    data_m = data[0:125]
    summe_m = 0
    count_m = 0
    
    for i in range(len(data_m)):
        for j in range((i+1),len(data_m)):
            if i != (len(data_m)-1):
                
                Si = floyd_warshall(data_m[i]); Sj = floyd_warshall(data_m[j])
                k = sp_kernel(Si, Sj) 
                
                print(i)
                print(j)
                print("------")
                summe_m += k
                count_m += 1
                
            else:
                pass
                
    print(summe_m/count_m)
    
    
    # #Muta:Non-Muta
    
    data_m = data[0:125]
    data_nm = data[125:]
    summe_mix = 0
    count_mix = 0
    
    for i in range(len(data_m)):
        for j in range(len(data_nm)):
            
            Si = floyd_warshall(data_m[i]); Sj = floyd_warshall(data_nm[j])
            k = sp_kernel(Si, Sj) 
                
            print(i)
            print(j)
            print("------")
            summe_mix += k
            count_mix += 1
                
    print(summe_mix/count_mix)
    


    # #Non-Muta:Non-Muta
    
    summe_nm = 0
    count_nm = 0
    
    for i in range(len(data_nm)):
        for j in range((i+1),len(data_nm)):
            if i != (len(data_nm)-1):
                
                Si = floyd_warshall(data_nm[i]); Sj = floyd_warshall(data_nm[j])
                k = sp_kernel(Si, Sj) 
                
                print(i)
                print(j)
                print("------")
                summe_nm += k
                count_nm += 1
                
            else:
                pass
                
    print(summe_nm/count_nm)
    
    
    # Save the output
    f_out.write(
                '{}\t{}\n'.format("mutagenic:mutagenic",round(summe_m/count_m,2))
            )
    
    f_out.write(
                '{}\t{}\n'.format("mutagenic:non-mutagenic",round(summe_mix/count_mix,2))
            )
    
    f_out.write(
                '{}\t{}\n'.format("non-mutagenic:non-mutagenic",round(summe_nm/count_nm,2))
            )
    
    
    
    
    f_out.close()


    




