import numpy as np
import networkx as nx
from solvers import evd, sampler


def get_karate_dataset(path):
    G = nx.read_gml(path, label="id")
    y_true = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    for i, (n, d) in enumerate(G.nodes(data=True)):
        if y_true[i] == 0:
            d["class"] = "r"
        else:
            d["class"] = "b"
    A_karate = np.array(nx.to_numpy_matrix(G, dtype=np.float64))
    node_color = [d["class"] for x, d in G.nodes(data=True)]
    return A_karate, y_true, G, node_color


def bernoulli_sampler(p, n=1):
    """ " Sampling from bernoulli distribution"""
    assert p <= 1
    assert p >= 0
    samples = [1 if x < p else 0 for x in np.random.random_sample(n)]
    if n == 1:
        samples = samples[0]
    return samples


def get_error(a, b):
    """Takes two list and return the ratio of similarity"""
    assert len(a) == len(b)
    result = [None] * len(a)
    for i in range(len(a)):
        result[i] = a[i] == b[i]
    error = result.count(1) / float(len(a))
    if error <= 0.5:
        error = 1 - error
    return error


def get_ratio(x):
    """get the ratio of ones and zeros"""
    ratio = np.sum(x)
    ratio = ratio / float(len(x))
    if ratio > 0.5:
        ratio = 1 - ratio
    return ratio
