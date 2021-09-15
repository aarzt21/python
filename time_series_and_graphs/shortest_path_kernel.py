"""Skeleton file for your solution to the shortest-path kernel."""

import numpy as np

def floyd_warshall(A):
    """Implement the Floyd--Warshall on an adjacency matrix A.

    Parameters
    ----------
    A : `np.array` of shape (n, n)
        Adjacency matrix of an input graph. If A[i, j] is `1`, an edge
        connects nodes `i` and `j`.

    Returns
    -------
    An `np.array` of shape (n, n), corresponding to the shortest-path
    matrix obtained from A.
    """
    n=len(A)
    d = np.zeros((n,n)) # auxiliary matrix
    
    for i in range(n):
        for j in range(n):
            if  (i != j and A[i,j] == 0):
                d[i,j] = 50000
                
            elif i==j: 
                d[i,j]=0
            else:
                d[i,j]= 1    #if there's no node then set entry in aux matrix to large value
                
    for k in range(n):
        for i in range(n):
            for j in range(n):
                
                if d[i,j] > d[i,k] + d[k,j]:
                    d[i,j] = d[i,k] + d[k,j]
                              
    return d





def sp_kernel(S1, S2):
    """Calculate shortest-path kernel from two shortest-path matrices.

    Parameters
    ----------
    S1: `np.array` of shape (n, n)
        Shortest-path matrix of the first input graph.

    S2: `np.array` of shape (m, m)
        Shortest-path matrix of the second input graph.

    Returns
    -------
    A single `float`, corresponding to the kernel value of the two
    shortest-path matrices
    """
    t1 = np.triu(S1); t2 = np.triu(S2)
    t1 = t1.flatten(); t2 = t2.flatten()
    
    t1=t1[t1!=0] ; t2=t2[t2!=0]
    t1
    
    count = 0
    
    for i in range(len(t1)):
        for j in range(len(t2)):
            
            if t1[i] == t2[j]:
                count = count + 1
            else:
                pass
    
    count = float(count)
    return count
        

