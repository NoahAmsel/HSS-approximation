import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, aslinearoperator


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


# Don't actually need this? just use regular arithmetic on LinearOperator's
# But if we want to access the factors, 
class FourPartLens(LinearOperator):
    def __init__(self, U, A, V, D):
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

    @property
    def shape(self):
        return self.D.shape

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
    # return aslinearoperator(U_l) @ aslinearoperator(random_access_greedy_alg(np.asarray(U_l.T @ (A - D_l) @ V_l), level-1, r)) @ aslinearoperator(V_l).T + aslinearoperator(D_l)


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
    A_lminus1 = matvec_alg_resketch(aslinearoperator(U_l).T @ (aslinearoperator(A) - aslinearoperator(D_l)) @ aslinearoperator(V_l), level-1, r, num_sketches_per_level)
    return FourPartLens(U_l, A_lminus1, V_l, D_l)



N = 2**4; r = 3
A = np.random.randn(N, r) @ np.random.randn(r, N)
A_tilde = random_access_greedy_alg(A, 1, 3)
np.linalg.norm(A - A_tilde.toarray(), np.inf)


order = 8
N = 2**order
num_diags_above = 1
offsets = list(range(-num_diags_above, num_diags_above+1))
banded = sp.diags_array([np.random.randn(N-abs(offset)) for offset in offsets], offsets=offsets)
A = np.linalg.inv(banded.toarray())
A_tilde = random_access_greedy_alg(A, order, 2*num_diags_above)
print(np.linalg.norm(A - A_tilde.toarray(), np.inf))
A_tilde_matvec = matvec_alg_unified_sketch(A, order, 2*num_diags_above, 3*(2*num_diags_above))
print(np.linalg.norm(A - A_tilde_matvec.toarray(), np.inf))
A_tilde_resketch = matvec_alg_resketch(A, order, 2*num_diags_above, 3*(2*num_diags_above))
print(np.linalg.norm(A - A_tilde_resketch.toarray(), np.inf))


A_tilde = random_access_greedy_alg(A, order, 1)
print(np.linalg.norm(A - A_tilde.toarray(), np.inf))
A_tilde_matvec = matvec_alg_unified_sketch(A, order, 1, 3*(2*num_diags_above))
print(np.linalg.norm(A - A_tilde_matvec.toarray(), np.inf))
A_tilde_resketch = matvec_alg_resketch(A, order, 1, 3*(2*num_diags_above))
print(np.linalg.norm(A - A_tilde_resketch.toarray(), np.inf))