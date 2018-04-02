import numpy as np

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
