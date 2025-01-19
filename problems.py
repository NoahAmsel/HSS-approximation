import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import aslinearoperator as alo, factorized, LinearOperator


from HSS_approx import FourPartLens


class SparseInverse(LinearOperator):
    def __init__(self, A, dtype=None):
        assert A.shape[0] == A.shape[1]
        super().__init__(dtype, A.shape)
        # self.LU = sp.linalg.splu(A)  # TODO: get rid of below and just use this
        self.Ainv = factorized(A)
        self.ATinv = factorized(A.T.tocsc())
        self._init_dtype()

    def _matmat(self, other): return self.Ainv(other)
    def _rmatmat(self, other): return self.ATinv(other)
    def toarray(self): return self @ np.eye(self.shape[0])


def banded_gaussian(N, half_bandwidth):
    offsets = list(range(-half_bandwidth, half_bandwidth+1))
    return sp.diags_array([np.random.randn(N-abs(offset)) for offset in offsets], offsets=offsets, format='csc')


def grid_schur_complement(partitioned_grid_dimension, other_grid_dimension):
    L = sp.linalg.LaplacianNd((partitioned_grid_dimension, other_grid_dimension), dtype=np.float64).tosparse().tocsc()
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


def grid_schur_complement_levels(partitioned_grid_dimension, leaf_size, levels):
    return grid_schur_complement(partitioned_grid_dimension, leaf_size*(2**levels))


def factor2_example(N, eps):
    assert N % 2 == 0
    off_diagonal = np.array([[0, 1+eps], [1, 0]])
    eye = np.eye(2)
    zero = np.zeros((2, 2))
    def block(i, j):
        if i == j:
            return zero
        if i + j == N - 1:
            return off_diagonal
        else:
            return eye
    return np.block([[block(i, j) for i in range(N)] for j in range(N)])


def factor2_optimal_solution(N):
    block_diag_ones = sp.block_diag([np.ones((2, 1)) for _ in range(N)])
    return FourPartLens(block_diag_ones, np.ones((N, N)), block_diag_ones, sp.block_diag([-np.ones((2, 2)) for _ in range(N)]))
