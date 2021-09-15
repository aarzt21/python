# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 14:09:12 2020

@author: Alex Arzt 14-610-070


- my driver's car has D=60 stencilled on it
- no more than N_max=1000 cars in Basel

- P(N|D)= P(N)P(D|N) / P(D)

"""
import numpy as np
import argparse
import os
import sys

N_vals = np.arange(60, 1001, 1) # all possible values for N

prior = 1/1000 #prior

cont = np.zeros((len(N_vals),))
for i in range(len(N_vals)):
    cont[i]=(1/N_vals[i]) * prior
        
        
normalizer = sum(cont) #P(D)


#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------

if __name__ == '__main__':
    
    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(
        description="Compute nbayes_uber")
    
    parser.add_argument(
        "--outdir",
        required=True,
        help="Path to directory where output_nbayes_uber.txt & output_nbayes_uber_max.txt will be created")
    
    
    args = parser.parse_args()

    # Set the paths
    outdir = args.outdir

    if not os.path.exists(outdir):
        os.makedirs(outdir)


    try:
        file_name = "{}/output_nbayes_uber.txt".format(outdir)
        f_out = open(file_name, 'w')
    except IOError:
        print("Output file {} cannot be created".format(file_name))
        sys.exit(1)      
  
      
    f_out.write('{}\t{}\n'.format(
        'Value of N','Posterior'))


    post = np.zeros((len(N_vals),))
    for i in range(len(N_vals)):
        res = round((prior*(1/N_vals[i]))/normalizer,6)
        post[i] = res
        print(res)
        
    for i in range(len(post)):
        f_out.write('{}\t{}\n'.format(N_vals[i], post[i]))
                     
    f_out.close()
    
    maxi = max(post)
    
    
    try:
        file= "{}/output_nbayes_uber_max.txt".format(outdir)
        f = open(file, 'w')
    except IOError:
        print("Output file {} cannot be created".format(file))
        sys.exit(1)  
        
    boolA = post==maxi
    N = N_vals[boolA]
    N = N[0]
    
    f.write('Max. Posterior is {} for N={}'.format(maxi,N))
    f.close()
    
    print('Max. Posterior is {} for N={}'.format(maxi,N))




    
    

    



