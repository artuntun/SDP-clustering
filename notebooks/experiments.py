import matplotlib, matplotlib.pyplot as plt
import sklearn, numpy as np
from solvers import balanced_cut, evd, sampler
from sklearn import datasets, neighbors, cluster

def main():
    K = 2
    spectralKmeans = cluster.SpectralClustering(K,affinity='precomputed',assign_labels='kmeans')
    spectralDiscre = cluster.SpectralClustering(K,affinity='precomputed',assign_labels='discretize')
    kmeans = cluster.KMeans(K)
    semidef_truncated = SemidefCluster(assign_labels='truncated')
    semidef_sampler = SemidefCluster(assign_labels='sampling')
    semidef_kmeans = SemidefCluster(assign_labels='kmeans')
    methods = [('kmeans',kmeans),
               ('spectralKmeans',spectralKmeans),
               ('spectralDiscrete',spectralDiscre),
               ('semidefTruncated',semidef_truncated),
               ('semidefKmeans',semidef_kmeans),
               ('semidefSampler',semidef_sampler)]

    #Data generation
    n_samples = 400
    distances = np.linspace(0.05,0.5,15)
    X,Y = datasets.make_moons(n_samples, noise = 0.15)

    #Different methods for clustering
    experiments = []
    inits = 5
    for name,algorithm in methods:
        results = {'method':name,'error':0,'std':0,'labels':None,'distance':0}
        error_list = []
        #Several initalization for calculating std of solution
        for i in range(inits):
            if name == 'kmeans':
                algorithm.fit(X)
                nu = get_error(algorithm.labels_,Y)
                best = {'error':nu,'labels':algorithm.labels_,'distance':0}
            else:
                best = {'error':0,'labels':None,'distance':0}
                #Getting the best hyperparamters
                for d in distances:
                    A = sklearn.neighbors.radius_neighbors_graph(X, d, mode='distance')
                    algorithm.fit(A)
                    nu = get_error(algorithm.labels_, Y)
                    if nu > best['error']:
                        best['error'] = nu
                        best['labels'] = algorithm.labels_
                        best['distance'] = d
            error_list.append(best['error'])
        results['error'] = np.array(error_list).mean()
        results['std'] = np.array(error_list).std()
        results['labels'] = best['labels']
        results['distance'] = best['distance']
        experiments.append(results)

    import IPython
    IPython.embed()

    ## VISUALIZATION
    for k in range(K):
        plt.subplot(1,len(experiments)+1,1)
        plt.plot(X[Y==k,0], X[Y==k,1], marker='.', color="C"+str(k), lw=0)
        plt.title('Original')
    for i,exp in enumerate(experiments):
        plt.subplot(1,len(experiments)+1,i+2)
        for k in range(K):
            plt.plot(X[exp['labels']==k,0], X[exp['labels']==k,1], marker='.', color="C"+str(k), lw=0)
            plt.title(exp['method']+' '+str(round(exp['distance'],2)))
            plt.text(.99, .01, ('{}+{}'.format(round(exp['error'],2),round(exp['std'],2))).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
    import IPython
    IPython.embed()

def get_error(a,b):
    """Takes two list and return the ratio of similarity"""
    assert(len(a)==len(b))
    result = [None]*len(a)
    for i in range(len(a)):
        result[i] = (a[i] == b[i])   
    error = result.count(1)/float(len(a))
    return error

class SemidefCluster:
    assign_labels = None
    labels_ = None
    def __init__(self,assign_labels='truncated'):
        self.assign_labels = assign_labels

    def fit(self,A,Z=None):
        if Z == None:
            Z = balanced_cut(A, delta = 1.0, cut = 'min', solver = 'mosek')
        if self.assign_labels == 'truncated':
            Vr,wr = evd(Z, scaled=True)
            self.labels_ = np.array([0 if vi >= 0.0 else 1 for vi in list(Vr[:,0])])
        elif self.assign_labels == 'sampling':
            Vr,wr = evd(Z, scaled=True)
            self.labels_ = sampler(A,Vr)
        elif self.assign_labels == 'kmeans':
            Vr,wr = evd(Z, scaled=True)
            self.labels_ = cluster.k_means(np.array(Vr),2)

if __name__ == '__main__':
    main()



