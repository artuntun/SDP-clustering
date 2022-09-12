import matplotlib.pyplot as plt
import sklearn
import numpy as np
from sklearn import datasets, cluster

from sdp_clustering.solvers import SemidefCluster

# TO DO:
# chagne get error function. VEctorize
# add different datasets implementation
# add different levels of noise
# DONE add semidef Kmeans
# DONE implement several inits for std analysis


def main():
    K = 2
    spectralKmeans = cluster.SpectralClustering(
        K, affinity="precomputed", assign_labels="kmeans"
    )
    spectralDiscre = cluster.SpectralClustering(
        K, affinity="precomputed", assign_labels="discretize"
    )
    kmeans = cluster.KMeans(K)
    semidef_truncated = SemidefCluster(assign_labels="truncated")
    semidef_sampler = SemidefCluster(assign_labels="sampling")
    semidef_kmeans = SemidefCluster(assign_labels="kmeans")
    methods = [
        ("kmeans", kmeans),
        ("spectralKmeans", spectralKmeans),
        ("spectralDiscrete", spectralDiscre),
        ("semidefTruncated", semidef_truncated),
        ("semidefKmeans", semidef_kmeans),
        ("semidefSampler", semidef_sampler),
    ]
    noise = [0.15, 0.2, 0.25, 0.30]
    noise = [0.07, 0.10, 0.15]
    data = []
    all_experiments = []
    plot_num = 1
    for sig in noise:
        # Data generation
        n_samples = 400
        experiments = []
        distances = np.linspace(0.05, 0.5, 15)
        # X,Y = datasets.make_moons(n_samples, noise = sig)
        X, Y = datasets.make_circles(n_samples, factor=0.5, noise=sig)
        data.append((X, Y))
        results = {
            "method": "Original",
            "error": 0,
            "std": 0,
            "labels": Y,
            "distance": 0,
        }
        experiments.append(results)

        # Different methods for clustering
        inits = 5
        for name, algorithm in methods:
            results = {
                "method": name,
                "error": 0,
                "std": 0,
                "labels": None,
                "distance": 0,
            }
            error_list = []
            # Several initalization for calculating std of solution
            for i in range(inits):
                if name == "kmeans":
                    algorithm.fit(X)
                    nu = get_error(algorithm.labels_, Y)
                    best = {"error": nu, "labels": algorithm.labels_, "distance": 0}
                else:
                    best = {"error": 0, "labels": None, "distance": 0}
                    # Getting the best hyperparamters
                    for d in distances:
                        A = sklearn.neighbors.radius_neighbors_graph(
                            X, d, mode="distance"
                        )
                        algorithm.fit(A)
                        nu = get_error(algorithm.labels_, Y)
                        if nu > best["error"]:
                            best["error"] = nu
                            best["labels"] = algorithm.labels_
                            best["distance"] = d
                error_list.append(best["error"])
            results["error"] = np.array(error_list).mean()
            results["std"] = np.array(error_list).std()
            results["labels"] = best["labels"]
            results["distance"] = best["distance"]
            experiments.append(results)
        all_experiments.append(experiments)

    # Plotting
    plot_num = 1
    for i in range(len(all_experiments)):
        X = data[i][0]
        Y = data[i][1]
        for j, exp in enumerate(all_experiments[i]):
            plt.subplot(len(noise), len(methods) + 1, plot_num)
            for k in range(K):
                plt.plot(
                    X[exp["labels"] == k, 0],
                    X[exp["labels"] == k, 1],
                    marker=".",
                    color="C" + str(k),
                    lw=0,
                )
                if i == 0:
                    plt.title(exp["method"] + " " + str(round(exp["distance"], 2)))
                plt.text(
                    0.99,
                    0.01,
                    (
                        "{}+{}".format(round(exp["error"], 2), round(exp["std"], 2))
                    ).lstrip("0"),
                    transform=plt.gca().transAxes,
                    size=15,
                    horizontalalignment="right",
                )
                plt.xticks(())
                plt.yticks(())
            plot_num += 1


def get_error(a, b):
    """Takes two list and return the ratio of similarity"""
    assert len(a) == len(b)
    result = [None] * len(a)
    for i in range(len(a)):
        result[i] = a[i] == b[i]
    error = result.count(1) / float(len(a))
    return error


if __name__ == "__main__":
    main()
