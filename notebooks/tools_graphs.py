import numpy as np

#TO DO:
    #extend for Ndimenional data
    #make it sensitive to 1D data
def get_laplacian(X,sig=1):
    """ Caluclate the Laplacian of a 2D dataset
    Input:
        X: narray with m(samples)*n(dimensions)
        sig: variance of the similarity kernel
    Output:
        L: narray m*m. laplacian of X
    """
    #Similarity matrix
    m,n = X.shape
    S = np.zeros((m,m))
    S = np.zeros((X.shape[0],X.shape[0]))
    for i in range(m):
      for j in range(m):
        S[i,j] = np.exp(-np.linalg.norm(X[i,:]-X[j,:])**2/(sig**2*2))

    #Laplacian matrix
    W = S - np.diag(np.diag(S))
    D = np.diag(np.sum(W,axis=0))
    L = D - W

    return L
