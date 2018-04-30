import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "../")
import solvers as sv
import scipy as sp
import copy

def bernoulli_sampler(p,n=1):
    """" Sampling from bernoulli distribution """
    assert(p<=1)
    assert(p>=0)
    samples = [1 if x<p else 0 for x in np.random.random_sample(n)]
    if n==1:
        samples = samples[0]
    return samples

def get_error(a,b):
    """Takes two list and return the ratio of similarity"""
    assert(len(a)==len(b))
    result = [None]*len(a)
    for i in range(len(a)):
        result[i] = (a[i] == b[i])
    error = result.count(1)/float(len(a))
    return error

def sample_net(n_nodes,intercon=0.5,intracon=0.05,bal=0.25):
    net = nx.Graph()
    N1 = int(round(bal*n_nodes))
    labels = [0]*N1 +[1]* (n_nodes-N1)
    #adding ground turth nodes
    for i in range(N1):
        net.add_node(i,attr_dict={'class':'r'})
    for i in range(n_nodes-N1):
        net.add_node(i+N1,attr_dict={'class':'b'})
    #Adding edges
    for i in range(n_nodes):
        for j in range(i+1,n_nodes):
            if net.node[i]['attr_dict']['class']==net.node[j]['attr_dict']['class']:
                if bernoulli_sampler(intercon)==1:
                    net.add_edge(i,j)
            else:
                if bernoulli_sampler(intracon)==1:
                    net.add_edge(i,j)
    return net,labels

n_nodes = 40
net,labels = sample_net(n_nodes,0.3,0.15,0.5)

A = nx.to_scipy_sparse_matrix(net,dtype=np.float64)

solver = sv.SemidefCluster(assign_labels='sampling')
solver.fit(A)
