"""This is a warpper for cvxpy"""
import numpy as np
import cvxpy as cvx
from sklearn.cluster import k_means
from solvers import evd, sampler
from cvxopt import matrix


class SolverCVXPY:
    assign_labels = None
    labels_ = None
    Z = None
    objective = None
    eq_constr = []
    ineq_constr = []

    def __init__(self, assign_labels="truncated"):
        self.assign_labels = assign_labels

    def fit(self, Z, obj, eq_constr, ineq_constr):
        prob = cvx.Problem(obj, eq_constr + ineq_constr)
        prob.solve()
        self.Z = matrix()
        if self.assign_labels == "truncated":
            Vr, wr = evd(self.Z, scaled=True)
            self.labels_ = np.array([0 if vi >= 0.0 else 1 for vi in list(Vr[:, 0])])
        elif self.assign_labels == "sampling":
            Vr, wr = evd(self.Z, scaled=True)
            self.labels_ = sampler(Z, Vr)
        elif self.assign_labels == "kmeans":
            Vr, wr = evd(self.Z, scaled=True)
            inl, self.labels_, c = k_means(np.array(Vr), 2)
