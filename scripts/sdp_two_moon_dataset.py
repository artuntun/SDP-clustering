import numpy as np
import sklearn
import matplotlib.pyplot as plt
import cvxpy as cvx

import sdp_clustering.tools_graphs as gr


class experiment:
    X = None  # dataset
    y = None  # dataset labels
    L = None  # Laplacian of dataset
    Z = None  # Optimization problem
    yspec = None
    yopt = None


# Two Moons dataset with different noise levels
n_samples = 400
x = np.zeros([n_samples, 2, 3])
y = np.zeros([n_samples, 3])
plt.figure(1)
for i in range(3):
    x[:, :, i], y[:, i] = sklearn.datasets.make_moons(
        n_samples=n_samples, noise=0.02 + 0.03 * (i + 1), shuffle=False
    )
    n_mid = int(n_samples / 2)
    color = "r" * n_mid + "b" * n_mid

# Get Laplacians
Llist = []
for i in range(3):
    Llist.append(gr.get_laplacian(x[:, :, i]))

L = Llist[2]
# Spectral clustering
k = 2
lamb, V = np.linalg.eig(L)
idx = lamb.argsort()
lambOrder = lamb[idx]
VOrder = V[:, idx]

# Problem data.
np.random.seed(1)
m = L.shape[0]
# Construct the problem.
X = cvx.Semidef(m)
objective = cvx.Minimize(cvx.trace(L * X))
constraints = [X[i, i] == 1 for i in range(m)] + [cvx.sum_entries(X) <= 1]

prob = cvx.Problem(objective, constraints)
result = prob.solve(solver="SCS")
