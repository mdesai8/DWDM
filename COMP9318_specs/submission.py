import numpy as np
import pickle
import time
from copy import deepcopy
from scipy.spatial.distance import cdist


def kmedians(data_1, centroids, K,  max_iter):
    
    # Number of training data
    n = data_1.shape[0]
    # Number of features in the data
    c = len(data_1[0])
    
    centers = deepcopy(centroids)
    centers_old = np.zeros(centers.shape)
    centers_new = deepcopy(centers) 

    #Empty array for distances and clusters. 
    clusters = np.zeros(n)
    distances = np.zeros((n,K))
    
    # Loop stopping criteria. 
    error = False
    count = 0 
    
    #Looping for finding the correct clusters
    while count < max_iter:
        for i in range(K):
            distances[:,i] = np.abs(data_1 - centers_new[i]).sum(-1)
        
        clusters = np.argmin(distances, axis = 1)

        centers_old = deepcopy(centers_new)

        for i in range(K):
            if i in clusters:
                centers_new[i] = np.median(data_1[clusters == i], axis=0)
            else:
                pass  
            
        count += 1
    
            
    
    return centers_new 

def pq(data, P, init_centroids, max_iter, k= 256):
    split_data = np.split(data, P, axis = 1)
    
    codebooks = np.zeros(shape = (P, k, len(split_data[0][0])), dtype = 'float32')
    
    for i in range(P):
        codebooks[i] = kmedians(split_data[i], init_centroids[i], k, max_iter)
    
    codes = np.zeros(shape = (len(data),P), dtype='uint8')
    
    
    for i in range(len(split_data)):
        data_1 = split_data[i]
        coddebuk = codebooks[i]

        dists = cdist(data_1, coddebuk, metric='cityblock')

        for j in range(len(dists)):
            codes[j][i] = np.argmin(dists[j])    

    
    return codebooks, codes