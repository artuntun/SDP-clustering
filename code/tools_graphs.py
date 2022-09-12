import numpy as np
import networkx as nx
from utils import bernoulli_sampler
from scipy.linalg import expm

# TO DO:
# extend for Ndimenional data
# make it sensitive to 1D data
def get_laplacian(X, sig=1):
    """Caluclate the Laplacian of a 2D dataset
    Input:
        X: narray with m(samples)*n(dimensions)
        sig: variance of the similarity kernel
    Output:
        L: narray m*m. laplacian of X
    """
    # Similarity matrix
    m, n = X.shape
    S = np.zeros((m, m))
    S = np.zeros((X.shape[0], X.shape[0]))
    for i in range(m):
        for j in range(m):
            S[i, j] = np.exp(-np.linalg.norm(X[i, :] - X[j, :]) ** 2 / (sig**2 * 2))

    # Laplacian matrix
    W = S - np.diag(np.diag(S))
    D = np.diag(np.sum(W, axis=0))
    L = D - W

    return L


def sample_SBM(n_nodes, p=0.5, q=0.05, bal=0.5):
    """It samples an Stochastic Block Model with 2 groups
    Inputs:
        -n_nodes:(int) number of nodes
        -p:(float) probability that two nodes from same group connect
        -q:(float) prob that two nodes from different groups connect
        -bal(float): precentage of nodes in each group
    Output:
        -net: graph
    """
    net = nx.Graph()
    N1 = int(round(bal * n_nodes))
    labels = [0] * N1 + [1] * (n_nodes - N1)
    # adding ground turth nodes
    for i in range(N1):
        net.add_node(i, attr_dict={"class": "r"})
    for i in range(n_nodes - N1):
        net.add_node(i + N1, attr_dict={"class": "b"})
    # Adding edges
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if net.node[i]["attr_dict"]["class"] == net.node[j]["attr_dict"]["class"]:
                if bernoulli_sampler(p) == 1:
                    net.add_edge(i, j)
            else:
                if bernoulli_sampler(q) == 1:
                    net.add_edge(i, j)
    A = np.array(nx.to_numpy_matrix(net, dtype=np.float64))
    node_color = [d["attr_dict"]["class"] for x, d in net.nodes(data=True)]
    if 0 in expm(A):  # not a fully connected graph
        A, labels, net, node_color = sample_SBM(n_nodes, p, q, bal)
    return A, labels, net, node_color


def sample_random_graph(N, p=0.5):
    """Generate Adjacency of a random graph
    Inputs:
        - N: number of nodes
        - p: probability that each pair of nodes connects
    """
    A = np.zeros((N, N), dtype=np.int16)
    for i in range(N):
        for j in range(i + 1, N):
            if i != j:
                A[i, j] = bernoulli_sampler(0.5)
                A[j, i] = A[i, j]
    return A
