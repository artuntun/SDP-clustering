from cvxopt import (
    solvers,
    matrix,
    spmatrix,
    sparse,
    msk,
    blas,
    lapack,
    normal,
    amd,
    mul,
    misc,
)
from math import sqrt
import numpy as np
from sklearn.cluster import k_means


class SemidefCluster:
    assign_labels = None
    labels_ = None
    Z = None

    def __init__(self, assign_labels="truncated"):
        self.assign_labels = assign_labels

    def fit(self, A, Z=None):
        if type(Z) == type(None):
            self.Z = balanced_cut(A, delta=1.0, cut="min", solver="mosek")
        else:
            self.Z = Z
        if self.assign_labels == "truncated":
            Vr, wr = evd(self.Z, scaled=True)
            self.labels_ = np.array([0 if vi >= 0.0 else 1 for vi in list(Vr[:, 0])])
        elif self.assign_labels == "sampling":
            Vr, wr = evd(self.Z, scaled=True)
            self.labels_ = sampler(A, Vr)
        elif self.assign_labels == "kmeans":
            Vr, wr = evd(self.Z, scaled=True)
            inl, self.labels_, c = k_means(np.array(Vr), 2)


def custom_kkt(W):
    """
    Custom KKT solver for the following conic LP
    formulation of the Schur relaxation of the
    balance-constrained min/max cut problem

        maximize     Tr(C,X)
        subject to
            X_{ii} = 1, i=1...n
            sum(X) + x = const
            x >= 0, X psd
    """
    r = W["rti"][0]
    N = r.size[0]
    e = matrix(1.0, (N, 1))

    # Form and factorize reduced KKT system
    H = matrix(0.0, (N + 1, N + 1))
    blas.syrk(r, H, n=N, ldC=N + 1)
    blas.symv(H, e, H, n=N, ldA=N + 1, offsety=N, incy=N + 1)
    H[N, N] = blas.dot(H, e, n=N, offsetx=N, incx=N + 1)
    rr = H[:N, :N]  # Extract and symmetrize (1,1) block
    misc.symm(rr, N)  #
    q = H[N, :N].T  # Extract q = rr*e
    H = mul(H, H)
    H[N, N] += W["di"][0] ** 2
    lapack.potrf(H)

    def fsolve(x, y, z):
        """
        Solves the system of equations

            [ 0  G'*W^{-1} ] [ ux ] = [ bx ]
            [ G  -W'       ] [ uz ]   [ bz ]

        """
        #  Compute bx := bx + G'*W^{-1}*W^{-T}*bz
        v = matrix(0.0, (N, 1))
        for i in range(N):
            blas.symv(z, rr, v, ldA=N, offsetA=1, n=N, offsetx=N * i)
            x[i] += blas.dot(rr, v, n=N, offsetx=N * i)
        blas.symv(z, q, v, ldA=N, offsetA=1, n=N)
        x[N] += blas.dot(q, v) + z[0] * W["di"][0] ** 2
        #  Solve G'*W^{-1}*W^{-T}*G*ux = bx
        lapack.potrs(H, x)

        # Compute bz := -W^{-T}*(bz-G*ux)
        # z -= G*x
        z[1 :: N + 1] -= x[:-1]
        z -= x[-1]
        # Apply scaling
        z[0] *= -W["di"][0]
        blas.scal(0.5, z, n=N, offset=1, inc=N + 1)
        tmp = +r
        blas.trmm(z, tmp, ldA=N, offsetA=1, n=N, m=N)
        blas.syr2k(r, tmp, z, trans="T", offsetC=1, ldC=N, n=N, k=N, alpha=-1.0)

    return fsolve


def balanced_cut_conelp(A, N, delta=1.0, cut="min"):
    """
    Generates problem data for semidefinite relaxation of the
    following balance-constrained min/max cut problem

        min./max.   -Tr(AX)
        subject to  X{ii} == 1,  i = 1,...,n
                    sum(X) + x == delta^2
                    x >= 0,  X p.s.d.
    """
    c = matrix([matrix(-1.0, (N, 1)), -(delta**2)])
    h = matrix([0.0, matrix(A.toarray(), (N**2, 1))], (N**2 + 1, 1))
    if cut == "min":
        blas.scal(-1.0, h)
    elif not cut == "max":
        raise ValueError('cut must be "min" or "max"')
    G = sparse(
        [
            [
                spmatrix(
                    1.0, [1 + i * (N + 1) for i in range(N)], range(N), (N**2 + 1, N)
                )
            ],
            [matrix(1.0, (N**2 + 1, 1))],
        ]
    )
    dims = {"l": 1, "q": [], "s": [N]}
    return c, G, h, dims


def balanced_cut(A, delta=1.0, cut="min", solver="conelp_custom"):
    """
    Solves semidefinite relaxation of the following
    balance-constrained min/max cut problem
    """
    N = A.shape[0]
    prob = balanced_cut_conelp(A, N, delta, cut)
    if solver == "mosek":
        sol = msk.conelp(*prob)
        Z = matrix(sol[2][1:], (N, N), tc="d")
    elif solver == "conelp":
        sol = solvers.conelp(*prob)
        Z = matrix(sol["z"][1:], (N, N), tc="d")
    elif solver == "conelp_custom":
        sol = solvers.conelp(*prob, kktsolver=custom_kkt, options={"refinement": 3})
        Z = matrix(sol["z"][1:], (N, N), tc="d")
    else:
        raise ValueError("Unknown solver")
    return Z


def evd(Z, tol=1e-5, maxr=None, scaled=False):
    """
    Computes eigenvalue decomposition of a symmetrix matrix Z
    and returns a matrix Vr with at most 'maxr' eigenvectors and
    a vector wr with the corresponding eigenvalues. Only eigenvalues
    that satisfy lambda_i > tol*lambda_max are included.
    """
    Zt = +Z
    N = Z.size[0]
    if maxr is None:
        maxr = N
    w = matrix(0.0, (N, 1))
    V = matrix(0.0, (N, N))
    lapack.syevr(Zt, w, jobz="V", Z=V)
    if scaled:
        Vr = matrix(
            [
                [V[:, i] * sqrt(w[i])]
                for i in range(N - 1, N - 1 - maxr, -1)
                if w[i] >= tol * w[-1]
            ]
        )
    else:
        Vr = matrix(
            [[V[:, i]] for i in range(N - 1, N - 1 - maxr, -1) if w[i] >= tol * w[-1]]
        )
    wr = matrix([w[i] for i in range(N - 1, N - 1 - maxr, -1) if w[i] >= tol * w[-1]])
    return Vr, wr


def sampler(A, Vr):
    # Generate 1000 samples and apply rounding
    A = A.toarray()
    A[A == 0.0] = -1
    Am = matrix(A)
    vals = []
    fbest = float("-inf")
    xbest = None
    for k in range(10000):
        x = matrix([-1.0 if xi >= 0 else 1.0 for xi in Vr * normal(Vr.size[1], 1)])
        f = blas.dot(x, Am * x)
        if f > fbest:
            fbest = f
            xbest = x
        vals.append(f)
    x1 = np.array([0 if xi >= 0.0 else 1 for xi in xbest])
    return x1
