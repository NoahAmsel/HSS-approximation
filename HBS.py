from itertools import product
from joblib import Parallel, delayed
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import aslinearoperator as alo, factorized, LinearOperator
from tqdm import tqdm


class BlockMatrix(LinearOperator):
    pass


class Block22Matrix(LinearOperator):
    def __init__(self, A11: LinearOperator, A12: LinearOperator, A21: LinearOperator, A22: LinearOperator):
        assert A11.shape[0] + A21.shape[0] == A12.shape[0] + A22.shape[0]
        assert A11.shape[1] + A12.shape[1] == A21.shape[1] + A22.shape[1]
        self.A11 = A11
        self.A12 = A12
        self.A21 = A21
        self.A22 = A22

    @property
    def shape(self):
        return (self.A11.shape[0] + self.A21.shape[0], self.A11.shape[1] + self.A12.shape[1])

    def _matmat(self, other):
        top_half = other[:self.A11.shape[1]]
        bottom_half = other[self.A11.shape[1]:]
        return self.A11 @ top_half + self

    def _adjoint(self):
        pass

# WRONG! I don't think this is actually what we want
class BranchMatrix:
    def __init__(self, A11, B12, B21, A22):
        """All inputs should implement LinearOperator interface"""
        self.A11 = A11
        self.B12 = B12
        self.B21 = B21
        self.A22 = A22
    # block with arbitrary A11 and A22, but A12 = [[1,1], [1,1]] \kron B12 and likewise A21 = [[1,1], [1,1]] \kron B21

    def _matmat(self, other):
        top_half = other[:self.A11.shape[1]]
        bottom_half = other[self.A11.shape[1]:]
        out = np.concat([self.A11 @ top_half, self.A22 @ bottom_half])
        self.B12 @ top_half.reshape(-1, 2).sum(axis=1)

class HBS:
    def __init__(U, V, A: BranchMatrix):
        """
        U: N by r
        V: N by r
        As
        """
        assert U.shape == V.shape
        N, r = U.shape


def rowblock_x_diag(A, level, block_i):
    blocksize = A.shape[0] // 2**level
    start = block_i * blocksize
    end = start + blocksize
    return A[start:end, np.r_[:start, end:A.shape[0]]]

def colblock_x_diag(A, level, block_i):
    return rowblock_x_diag(A.T, level, block_i).T

def diagblock(A, level, block_i):
    blocksize = A.shape[0] // 2**level
    start = block_i * blocksize
    end = start + blocksize
    return A[start:end, start:end]

# A = np.arange(64).reshape(8, 8)
# rowblock_x_diag(A, 2, 2)
# colblock_x_diag(A, 2, 2)


    def __init__(self, A, dtype=None):
        assert A.shape[0] == A.shape[1]
        super().__init__(dtype, A.shape)
        self.Ainv = factorized(A)
        self.ATinv = factorized(A.T)
        self._init_dtype()

# Don't actually need this? just use regular arithmetic on LinearOperator's
# But if we want to access the factors,
class FourPartLens(LinearOperator):
    def __init__(self, U, A, V, D, dtype=None):
        """
        A factorization of the form UAV^T + D.
        Assumes nothing about the four pieces other than:
            1. They implement the LinearOperator interface
            2. They have a property called .size that says how many parameters they contain
            3. Their dimensions are compatible
        """
        assert U.shape[0] == D.shape[0]
        assert U.shape[1] == A.shape[0]
        assert A.shape[1] == V.T.shape[0]
        assert V.T.shape[1] == D.shape[1]
        self.U = U
        self.V = V
        self.A = A
        self.D = D
        super().__init__(dtype, D.shape)
        self._init_dtype()

    def _matmat(self, other):
        return self.U @ (self.A @ (self.V.T @ other)) + self.D @ other

    def _adjoint(self):
        type(self)(self.V, self.A.T, self.U, self.D.T)

    def toarray(self):
        if isinstance(self.A, FourPartLens):
            A_array = self.A.toarray()
        else:
            A_array = self.A
        return self.U @ A_array @ self.V.T + self.D

    @property
    def size(self):
        return self.U.size + self.A.size + self.V.size + self.D.size


def random_access_greedy_alg(A: np.ndarray, level: int, r: int):
    if level == 0:
        return A
    # replace with sklearn TruncatedSVD or with scipy.sparse.svds
    U_l = sp.block_diag([np.linalg.svd(rowblock_x_diag(A, level, block_i), full_matrices=False).U[:, :r] for block_i in range(2**level)])
    # TODO: take conjugate transpose, not plain transpose. (or better yet, just switch this to be identical to above line but with A^* instead of A)
    V_l = sp.block_diag([np.linalg.svd(colblock_x_diag(A, level, block_i), full_matrices=False).Vh.T[:, :r] for block_i in range(2**level)])
    D_l = sp.block_diag([diagblock(A, level, block_i) for block_i in range(2**level)])
    A_lminus1 = random_access_greedy_alg(np.asarray(U_l.T @ (A - D_l) @ V_l), level-1, r)
    return FourPartLens(U_l, A_lminus1, V_l, D_l)
    # return np.asarray(U_l @ A_lminus1 @ V_l.T + D_l)
    # below is nice and fast but to convert back to an array, must multiply with identity matrix
    # return alo(U_l) @ alo(random_access_greedy_alg(np.asarray(U_l.T @ (A - D_l) @ V_l), level-1, r)) @ alo(V_l).T + alo(D_l)


def nullspace_basis(X):
    # Unlike matlab null() or scipy null_space, this only finds a space of dimension #cols - #rows.
    # The true nullspace may be larger.
    n, _ = X.shape
    return np.linalg.qr(X.T, 'complete').Q[:, n:]

def rowblock(A, level, block_i):
    blocksize = A.shape[0] // 2**level
    start = block_i * blocksize
    end = start + blocksize
    return A[start:end, :]

def row_nullifier(Omega, level, block_i):
    return nullspace_basis(rowblock(Omega, level, block_i))

def right_pseudoinv(X, Y):
    """Computes XY^+"""
    return np.linalg.lstsq(Y.T, X.T, rcond=None)[0].T

def blockwise_right_pseudoinv(X, Y, level):
    """X has row blocks X_i, Y has row blocks Y_i.
    Returns block diagonal matrix with X_i Y_i^+ blocks.
    """
    return sp.block_diag([right_pseudoinv(rowblock(X, level, block_i), rowblock(Y, level, block_i)) for block_i in range(2**level)])


def matvec_alg_unified_sketch_helper(Omega1, AOmega1, Omega2, ATOmega2, level: int, r: int):
    if level == 0:
        # NOTE this isn't symmetric in approximate case
        return right_pseudoinv(AOmega1, Omega1)
    U_l = sp.block_diag([np.linalg.svd(rowblock(AOmega1, level, block_i) @ row_nullifier(Omega1, level, block_i), full_matrices=False).U[:, :r] for block_i in range(2**level)])
    V_l = sp.block_diag([np.linalg.svd(rowblock(ATOmega2, level, block_i) @ row_nullifier(Omega2, level, block_i), full_matrices=False).U[:, :r] for block_i in range(2**level)])
    Dpart1 = blockwise_right_pseudoinv(AOmega1 - U_l @ (U_l.T @ AOmega1), Omega1, level)
    Dpart2 = U_l @ (U_l.T @ blockwise_right_pseudoinv(ATOmega2 - V_l @ (V_l.T @ ATOmega2), Omega2, level).T)
    D_l = Dpart1 + Dpart2
    newOmega1 = V_l.T @ Omega1
    newAOmega1 = U_l.T @ (AOmega1 - D_l @ Omega1)
    newOmega2 = U_l.T @ Omega2
    newATOmega2 = V_l.T @ (ATOmega2 - D_l.T @ Omega2)
    A_lminus1 = matvec_alg_unified_sketch_helper(newOmega1, newAOmega1, newOmega2, newATOmega2, level-1, r)
    return FourPartLens(U_l, A_lminus1, V_l, D_l)


def matvec_alg_unified_sketch(A, level: int, r: int, num_sketches:int):
    left_Omega = np.random.randn(A.shape[0], num_sketches)
    right_Omega = np.random.randn(A.shape[1], num_sketches)
    return matvec_alg_unified_sketch_helper(right_Omega, A @ right_Omega, left_Omega, A.T @ left_Omega, level, r)


def matvec_alg_resketch(A, level: int, r: int, num_sketches_per_level: int):
    Omega1 = np.random.randn(A.shape[1], num_sketches_per_level)
    AOmega1 = A @ Omega1
    Omega2 = np.random.randn(A.shape[0], num_sketches_per_level)
    ATOmega2 = A.T @ Omega2
    if level == 0:
        # NOTE this isn't symmetric in approximate case
        return right_pseudoinv(AOmega1, Omega1)
    U_l = sp.block_diag([np.linalg.svd(rowblock(AOmega1, level, block_i) @ row_nullifier(Omega1, level, block_i), full_matrices=False).U[:, :r] for block_i in range(2**level)])
    V_l = sp.block_diag([np.linalg.svd(rowblock(ATOmega2, level, block_i) @ row_nullifier(Omega2, level, block_i), full_matrices=False).U[:, :r] for block_i in range(2**level)])
    Dpart1 = blockwise_right_pseudoinv(AOmega1 - U_l @ (U_l.T @ AOmega1), Omega1, level)
    Dpart2 = U_l @ (U_l.T @ blockwise_right_pseudoinv(ATOmega2 - V_l @ (V_l.T @ ATOmega2), Omega2, level).T)
    D_l = Dpart1 + Dpart2
    A_lminus1 = matvec_alg_resketch(alo(U_l).T @ (alo(A) - alo(D_l)) @ alo(V_l), level-1, r, num_sketches_per_level)
    return FourPartLens(U_l, A_lminus1, V_l, D_l)


class SparseInverse(LinearOperator):
    def __init__(self, A, dtype=None):
        assert A.shape[0] == A.shape[1]
        super().__init__(dtype, A.shape)
        # self.LU = sp.linalg.splu(A)  # TODO: get rid of below and just use this
        self.Ainv = factorized(A)
        self.ATinv = factorized(A.T)
        self._init_dtype()

    def _matmat(self, other): return self.Ainv(other)
    def _rmatmat(self, other): return self.ATinv(other)
    def toarray(self): return self @ np.eye(self.shape[0])


def banded_gaussian(N, half_bandwidth):
    offsets = list(range(-half_bandwidth, half_bandwidth+1))
    return sp.diags_array([np.random.randn(N-abs(offset)) for offset in offsets], offsets=offsets)


if False:
    N = 2**4; r = 3
    A = np.random.randn(N, r) @ np.random.randn(r, N)
    A_tilde = random_access_greedy_alg(A, 1, 3)
    np.linalg.norm(A - A_tilde.toarray(), np.inf)


if False:
    order = 8
    num_diags_above = 1
    A = SparseInverse(banded_gaussian(2**order, num_diags_above))
    A_tilde = random_access_greedy_alg(A.toarray(), order, 2*num_diags_above)
    print(np.linalg.norm(A.toarray() - A_tilde.toarray(), np.inf))
    A_tilde_matvec = matvec_alg_unified_sketch(A, order, 2*num_diags_above, 3*(2*num_diags_above))
    print(np.linalg.norm(A.toarray() - A_tilde_matvec.toarray(), np.inf))
    A_tilde_resketch = matvec_alg_resketch(A, order, 2*num_diags_above, 3*(2*num_diags_above))
    print(np.linalg.norm(A.toarray() - A_tilde_resketch.toarray(), np.inf))

    # now with a too-small rank
    A_tilde = random_access_greedy_alg(A, order, 1)
    print(np.linalg.norm(A - A_tilde.toarray(), np.inf))
    A_tilde_matvec = matvec_alg_unified_sketch(A, order, 1, 3*(2*num_diags_above))
    print(np.linalg.norm(A - A_tilde_matvec.toarray(), np.inf))
    A_tilde_resketch = matvec_alg_resketch(A, order, 1, 3*(2*num_diags_above))
    print(np.linalg.norm(A - A_tilde_resketch.toarray(), np.inf))


def grid_schur_complement(partitioned_grid_dimension, other_grid_dimension):
    L = sp.linalg.LaplacianNd((partitioned_grid_dimension, other_grid_dimension), dtype=np.float64).tosparse()
    j1 = other_grid_dimension * (partitioned_grid_dimension//2)
    j2 = j1 + other_grid_dimension
    C11 = L[:j1, :j1]
    C13 = L[:j1, j1:j2]
    C22 = L[j2:, j2:]
    C23 = L[j2:, j1:j2]
    C31 = L[j1:j2, :j1]
    C32 = L[j1:j2, j2:]
    C33 = L[j1:j2, j1:j2]
    A = alo(C33) - alo(C31) @ SparseInverse(C11) @ alo(C13) - alo(C32) @ SparseInverse(C22) @ alo(C23)
    return A


def grid_schur_complement_(partitioned_grid_dimension, leaf_size, levels):
    return grid_schur_complement(partitioned_grid_dimension, leaf_size*(2**levels))


def approx_Frob(A, sketch_size):
    return np.linalg.norm(A @ np.random.randn(A.shape[1], sketch_size), ord='fro') / np.sqrt(sketch_size)


if __name__ == "__main__":
    # Schur complement
    # r = 30  # given in fig 7
    # m = 2*r  # given in beginning of experiments section
    # r = 10
    # m = 16
    # # s = max(r+m, 3*r)  # given in beginning of experiments section and beginning of sec. 4
    # levels = 6
    # n = 51
    # A = grid_schur_complement_(n, m, levels)
    # title = f"Grid Schur Complement:\nn={n},m={m},level={levels},r={r}"

    # banded inverse
    levels = 12
    num_diags_above = 5
    A = SparseInverse(banded_gaussian(2**levels, num_diags_above))
    r = 8 #num_diags_above # true rank is 2*num_diags_above
    title = f"Banded:\nnum_diag={num_diags_above},level={levels},r={r}"

    methods = {
        "fresh sketches": lambda s: matvec_alg_resketch(A, levels, r, s),
        "one sketch": lambda s: matvec_alg_unified_sketch(A, levels, r, s),
    }
    def rel_error(A_tilde):
        return approx_Frob(A_tilde - A, int(3e2)) / approx_Frob(A, int(3e2))
    def datum(sketch_dim, method_name, method):
        return dict(
            sketch_dim=sketch_dim,
            method=method_name,
            relative_error=rel_error(method(sketch_dim)),
        )
    results = Parallel(n_jobs=1)(
        delayed(datum)(sketch_dim, method_name, method)
        for _, (method_name, method), sketch_dim in tqdm(list(product(range(1), methods.items(), [16, 32, 64, 128, 256])))
    )
    # random_err = rel_error(random_access_greedy_alg(A.toarray(), levels, r))
    # results += [dict(sketch_dim=sketch_dim, method="random", relative_error=random_err,)
    #     for sketch_dim in [16, 256]
    # ]

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame(results)
    df["total_sketching"] = df.apply(lambda x: x["sketch_dim"] * {"fresh sketches": levels, "one sketch": 1, "random": 1}[x["method"]], axis=1)
    for i, x in enumerate(["sketch_dim", "total_sketching"]):
        plt.figure()
        ax = sns.lineplot(df, x=x, y="relative_error", hue="method", style="method", errorbar=("ci", 95), marker='o')
        plt.xscale("log")
        plt.yscale("log")
        plt.title(title)
        ax.get_figure().savefig(f"{title.replace("\n", "")}_{i}.png", dpi=400)

# from joblib import load, dump, wrap_non_picklable_objects
# filename = os.path.join('joblib_test.mmap')
# _ = dump(wrap_non_picklable_objects(A), filename)
# from scipy.sparse.linalg import SuperLU
# SuperLU

# sns.lineplot(df[df["sketch_dim"] > 8], x="sketch_dim", y="relative_error", hue="method", style="method", errorbar=("ci", 95), marker='o')
# sns.lineplot(df[df["sketch_dim"] > 8], x="total_sketching", y="relative_error", hue="method", style="method", errorbar=("ci", 95), marker='o')
