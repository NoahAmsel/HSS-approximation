from itertools import accumulate, product
from joblib import Parallel, delayed
import numpy as np
import operator
import scipy.sparse as sp
from scipy.sparse.linalg import aslinearoperator as alo, factorized, LinearOperator
from tqdm import tqdm


class HcatLinearOperator(LinearOperator):
    def __init__(self, A1: LinearOperator, A2: LinearOperator, dtype=None):
        assert A1.shape[0] == A2.shape[0]
        self.A1 = A1
        self.A2 = A2
        super().__init__(dtype, (A1.shape[0], A1.shape[1] + A2.shape[1]))
        self._init_dtype()

    def _matmat(self, other):
        return self.A1 @ other[:self.A1.shape[1]] + self.A2 @ other[self.A1.shape[1]:]

    def _adjoint(self):
        VcatLinearOperator(self.A1.T, self.A2.T)

    def toarray(self):
        return np.concatenate([self.A1.toarray(), self.A2.toarray()], axis=1)


class VcatLinearOperator(LinearOperator):
    def __init__(self, A1: LinearOperator, A2: LinearOperator, dtype=None):
        assert A1.shape[1] == A2.shape[1]
        self.A1 = A1
        self.A2 = A2
        super().__init__(dtype, (A1.shape[0] + A2.shape[0], A1.shape[1]))
        self._init_dtype()

    def _matmat(self, other):
        return np.concatenate((self.A1 @ other, self.A2 @ other))

    def _adjoint(self):
        HcatLinearOperator(self.A1.T, self.A2.T)

    def toarray(self):
        return np.concatenate([self.A1.toarray(), self.A2.toarray()], axis=0)


class Block22Matrix(VcatLinearOperator):
    def __init__(self, A11: LinearOperator, A12: LinearOperator, A21: LinearOperator, A22: LinearOperator):
        super().__init__(HcatLinearOperator(A11, A12), HcatLinearOperator(A21, A22))


class TiledMatrix(LinearOperator):
    def __init__(self, A: LinearOperator, block_rows: int, block_cols: int):
        self.A = A
        self.block_rows = block_rows
        self.block_cols = block_cols
        super().__init__(None, (A.shape[0] * block_rows, A.shape[1] * block_cols))
        self._init_dtype()

    def _matmat(self, other):
        # whoa this is dumb! TODO: improve this using einsum, etc
        return self.toarray() @ other

    def toarray(self):
        return np.tile(self.A, (self.block_rows, self.block_cols))



# class Block22Matrix(LinearOperator):
#     def __init__(self, A11: LinearOperator, A12: LinearOperator, A21: LinearOperator, A22: LinearOperator):
#         # assert A11.shape[0] + A21.shape[0] == A12.shape[0] + A22.shape[0]
#         # assert A11.shape[1] + A12.shape[1] == A21.shape[1] + A22.shape[1]
#         # Enforce that the block structure 
#         assert A11.shape[0] == A12.shape[0]
#         assert A21.shape[0] == A22.shape[0]
#         assert A11.shape[1] == A21.shape[1]
#         self.A11 = A11
#         self.A12 = A12
#         self.A21 = A21
#         self.A22 = A22

#     @property
#     def shape(self):
#         return (self.A11.shape[0] + self.A21.shape[0], self.A11.shape[1] + self.A12.shape[1])

#     def _matmat(self, other):
#         top_half = other[:self.A11.shape[1]]
#         bottom_half = other[self.A11.shape[1]:]
#         return self.A11 @ top_half + self

#     def _adjoint(self):
#         pass

# # WRONG! I don't think this is actually what we want
# class BranchMatrix:
#     def __init__(self, A11, B12, B21, A22):
#         """All inputs should implement LinearOperator interface"""
#         self.A11 = A11
#         self.B12 = B12
#         self.B21 = B21
#         self.A22 = A22
#     # block with arbitrary A11 and A22, but A12 = [[1,1], [1,1]] \kron B12 and likewise A21 = [[1,1], [1,1]] \kron B21

#     def _matmat(self, other):
#         top_half = other[:self.A11.shape[1]]
#         bottom_half = other[self.A11.shape[1]:]
#         out = np.concat([self.A11 @ top_half, self.A22 @ bottom_half])
#         self.B12 @ top_half.reshape(-1, 2).sum(axis=1)


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
        # if isinstance(self.A, FourPartLens):
        #     A_array = self.A.toarray()
        # else:
        #     A_array = self.A
        # return self.U @ A_array @ self.V.T + self.D
        return np.array(self.U @ (self.A @ self.V.T) + self.D)

    @property
    def size(self):
        return self.U.size + self.A.size + self.V.size + self.D.size

# TODO! block_diag is outputting COO format.
# we should use CSR for U and D, and CSC for V

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
    # TODO! I should use a fresh omega here too! \/
    Dpart1 = blockwise_right_pseudoinv(AOmega1 - U_l @ (U_l.T @ AOmega1), Omega1, level)
    Dpart2 = U_l @ (U_l.T @ blockwise_right_pseudoinv(ATOmega2 - V_l @ (V_l.T @ ATOmega2), Omega2, level).T)
    D_l = Dpart1 + Dpart2
    A_lminus1 = matvec_alg_resketch(alo(U_l).T @ (alo(A) - alo(D_l)) @ alo(V_l), level-1, r, num_sketches_per_level)
    return FourPartLens(U_l, A_lminus1, V_l, D_l)


def best_tile_approximation(A, level: int):
    rows_per_block = A.shape[0] // 2**level
    cols_per_block = A.shape[1] // 2**level
    mean_block = A.reshape(2**level, rows_per_block, 2**level, cols_per_block).mean((0, 2))
    return TiledMatrix(mean_block, 2**level, 2**level)


def recover_branch_matrix(A, level: int):
    if level == 0:
        return A
    A11 = A[:(A.shape[0]//2), :(A.shape[1]//2)]
    A12 = A[:(A.shape[0]//2), (A.shape[1]//2):]
    A21 = A[(A.shape[0]//2):, :(A.shape[1]//2)]
    A22 = A[(A.shape[0]//2):, (A.shape[1]//2):]
    upper_left = recover_branch_matrix(A11, level-1)
    upper_right = best_tile_approximation(A12, level-1)
    lower_left = best_tile_approximation(A21, level-1)
    lower_right = recover_branch_matrix(A22, level-1)
    return Block22Matrix(upper_left, upper_right, lower_left, lower_right)


def diana_exact_helper(A, U, V, level):
    if level == 0:
        r = U.shape[1]
        return sp.csr_matrix((r, r), dtype=A.dtype)
    n, m = A.shape
    A11 = A[:n//2, :m//2]
    A12 = A[:n//2, m//2:]
    A21 = A[n//2:, :m//2]
    A22 = A[n//2:, m//2:]
    upper_left = diana_exact_helper(A11, U[:n//2,:], V[:m//2,:], level-1)
    upper_right = TiledMatrix(U[:n//2,:].T @ A12 @ V[m//2:,:], 2**(level-1), 2**(level-1))
    lower_left = TiledMatrix(U[n//2:,:].T @ A21 @ V[:m//2,:], 2**(level-1), 2**(level-1))
    lower_right = diana_exact_helper(A22, U[n//2:,:], V[m//2:,:], level-1)
    return Block22Matrix(upper_left, upper_right, lower_left, lower_right)
    # Udiag = sp.block_diag([rowblock(U, level, block_i) for block_i in range(2**level)], format="csr")
    # Vdiag = sp.block_diag([rowblock(V, level, block_i) for block_i in range(2**level)], format="csr")
    # np.allclose(Udiag[:4,:8] @ upper_right.toarray() @ Vdiag[4:,8:].T, A12)
    # np.allclose(Udiag[4:,8:] @ lower_left.toarray() @ Vdiag[:4,:8].T, A21)
    # Udiag[:4,:8] @ upper_left.toarray() @ Vdiag[:4,:8].T
    # A11[:2,2:]

def diana_exact(A, level: int, r: int):
    if level == 0:  # is this necessary? I think it is, but only because rowblock_x_diag doesn't know how to handle level = 0
        return A
    D = sp.block_diag([diagblock(A, level, block_i) for block_i in range(2**level)])
    Aclean = np.array(A - D)
    n, m = Aclean.shape
    svd_upper_right = np.linalg.svd(Aclean[:n//2,m//2:])
    svd_lower_left = np.linalg.svd(Aclean[n//2:,:m//2])
    U = np.concat((svd_upper_right.U[:,:r], svd_lower_left.U[:,:r]))
    V = np.concat((svd_lower_left.Vh.T[:,:r], svd_upper_right.Vh.T[:,:r]))
    A_compressed = diana_exact_helper(Aclean, U, V, level)
    Udiag = sp.block_diag([rowblock(U, level, block_i) for block_i in range(2**level)], format="csr")
    Vdiag = sp.block_diag([rowblock(V, level, block_i) for block_i in range(2**level)], format="csr")
    return FourPartLens(Udiag, A_compressed, Vdiag, D)


# OLD AND MISGUIDED
def diana_random_access(A, level: int, r: int):
    if level == 0:  # is this necessary? I think it is, but only because rowblock_x_diag doesn't know how to handle level = 0
        return A
    # replace with sklearn TruncatedSVD or with scipy.sparse.svds
    U = sp.block_diag([np.linalg.svd(rowblock_x_diag(A, level, block_i), full_matrices=False).U[:, :r] for block_i in range(2**level)], format="csr")
    # TODO: take conjugate transpose, not plain transpose. (or better yet, just switch this to be identical to above line but with A^* instead of A)
    V = sp.block_diag([np.linalg.svd(colblock_x_diag(A, level, block_i), full_matrices=False).Vh.T[:, :r] for block_i in range(2**level)], format="csc")
    D = sp.block_diag([diagblock(A, level, block_i) for block_i in range(2**level)])
    A_compressed = recover_branch_matrix(np.asarray(U.T @ (A - D) @ V), level-2)
    # A_compressed = U.T @ (A - D) @ V
    return FourPartLens(U, A_compressed, V, D)

# def diana_random_access(A, level: int, r: int):
#     if level == 0:  # is this necessary? I think it is, but only because rowblock_x_diag doesn't know how to handle level = 0
#         return A
#     D = sp.block_diag([diagblock(A, level, block_i) for block_i in range(2**level)])
#     Aclean = np.array(A - D)
#     svd = np.linalg.svd(Aclean)
#     U = sp.block_diag([rowblock(svd.U[:, :r], level, block_i) for block_i in range(2**level)], format="csr")
#     V = sp.block_diag([rowblock(svd.Vh.T[:, :r], level, block_i) for block_i in range(2**level)], format="csr")
#     A_compressed = recover_branch_matrix(np.asarray(U.T @ (A - D) @ V), level)
#     # A_compressed = U.T @ (A - D) @ V
#     return FourPartLens(U, A_compressed, V, D)

def diana_fixed_up_OLD(A, level: int, r: int):
    if level == 0:  # is this necessary? I think it is, but only because rowblock_x_diag doesn't know how to handle level = 0
        return A
    D = sp.csr_matrix(A.shape, dtype=A.dtype)
    running_U = sp.linalg._interface.IdentityOperator(A.shape)
    running_V = sp.linalg._interface.IdentityOperator(A.shape)
    for l in range(level, 0, -1):
        # U = sp.block_diag([np.linalg.svd(rowblock_x_diag(A, l, block_i), full_matrices=False).U[:, :r] for block_i in range(2**l)], format="csr")
        # V = sp.block_diag([np.linalg.svd(colblock_x_diag(A, l, block_i), full_matrices=False).Vh.T[:, :r] for block_i in range(2**l)], format="csc")
        rowblock_Us = [np.linalg.svd(rowblock_x_diag(A, l, block_i), full_matrices=False).U[:, :r] for block_i in range(2**l)]
        # TODO: probably shouldn't just concatenate two identical matrices?
        U = sp.block_diag([np.hstack((bigU/np.sqrt(2), bigU/np.sqrt(2))) for bigU in rowblock_Us], format="csr")
        colblock_Vs = [np.linalg.svd(colblock_x_diag(A, l, block_i), full_matrices=False).Vh.T[:, :r] for block_i in range(2**l)]
        V = sp.block_diag([np.hstack((bigV/np.sqrt(2), bigV/np.sqrt(2))) for bigV in colblock_Vs], format="csr")
        A = np.asarray(U.T @ A @ V)
        running_U = running_U @ U
        running_V = running_V @ V
    A_compressed = recover_branch_matrix(A, level)
    return FourPartLens(running_U, A_compressed, running_V, D)


def antidiagonals(As):
    assert len(As) >= 2
    if len(As) == 2:
        A12 = As[0]
        A21 = As[1]
        return sp.block_array([
            [None, A12],
            [A21, None],
        ])
    else:
        A11 = antidiagonals(As[:len(As)//2])
        A22 = antidiagonals(As[len(As)//2:])
        return sp.block_array([
            [A11, None],
            [None, A22],
        ])


def extract_antidiagonals(A, levels):
    if levels == 1:
        A12 = A[:A.shape[0]//2, A.shape[1]//2:]
        A21 = A[A.shape[0]//2:, :A.shape[1]//2]
        return np.concat((A12.flatten(), A21.flatten()))
    else:
        A11 = A[:A.shape[0]//2, :A.shape[1]//2]
        A22 = A[A.shape[0]//2:, A.shape[1]//2:]
        return np.concat((extract_antidiagonals(A11, levels-1), extract_antidiagonals(A22, levels-1)))


def assembleHSS(cummulativeUs, a, cummulativeVs, level, r):
    blocksize = cummulativeUs[0].shape[0] // (2**level)
    on_diagonal_blocks = a[:(blocksize**2) * (2**level)].reshape(2**level, blocksize, blocksize)
    HSS = sp.block_diag([on_diagonal_blocks[i] for i in range(2**level)], format="csr")
    ix = (blocksize**2) * (2**level)
    for i, l in enumerate(range(level, 0, -1)):
        num_as = r**2 * 2**l
        HSS = HSS + cummulativeUs[i] @ antidiagonals(a[ix:(ix + num_as)].reshape(2**l, r, r)) @ cummulativeVs[i].T
        ix += num_as
    return HSS


def assembleTranspose(cummulativeUs, M, cummulativeVs, level):
    M = M.reshape(cummulativeUs[0].shape[0], cummulativeVs[0].shape[0])
    lll = [diagblock(M, level, block_i).flatten() for block_i in range(2**level)]
    for i, l in enumerate(range(level, 0, -1)):
        lll.append(extract_antidiagonals(cummulativeUs[i].T @ M @ cummulativeVs[i], l))
    return np.concat(lll)


def diana_fixed_up(A, level: int, r: int):
    if level == 0:  # is this necessary? I think it is, but only because rowblock_x_diag doesn't know how to handle level = 0
        return A
    blocksize = A.shape[0] // (2**level)
    assert blocksize >= r  # otherwise this makes no sense, we should just use fewer levels
    allUs = []
    allVs = []
    runningA = A
    for l in range(level, 0, -1):
        U = sp.block_diag([np.linalg.svd(rowblock_x_diag(runningA, l, block_i), full_matrices=False).U[:, :r] for block_i in range(2**l)], format="csr")
        V = sp.block_diag([np.linalg.svd(colblock_x_diag(runningA, l, block_i), full_matrices=False).Vh.T[:, :r] for block_i in range(2**l)], format="csc")
        runningA = U.T @ runningA @ V
        allUs.append(U)
        allVs.append(V)
    cummulativeUs = list(accumulate(allUs, operator.matmul, initial=sp.linalg._interface.IdentityOperator((A.shape[0], A.shape[0]))))[1:]
    cummulativeVs = list(accumulate(allVs, operator.matmul, initial=sp.linalg._interface.IdentityOperator((A.shape[1], A.shape[1]))))[1:]
    def matvec(a):
        return assembleHSS(cummulativeUs, a, cummulativeVs, level, r).toarray().flatten()
    def rmatvec(M):
        return assembleTranspose(cummulativeUs, M, cummulativeVs, level)
    degrees_of_freedom = sum(r**2 * 2**l for l in range(level, 0, -1)) + A.shape[0] * A.shape[1] // (2**level)
    a_star = sp.linalg.lsqr(LinearOperator(shape=(A.shape[0]*A.shape[1], degrees_of_freedom), matvec=matvec, rmatvec=rmatvec), A.flatten())[0]
    return assembleHSS(cummulativeUs, a_star, cummulativeVs, level, r)


def diana_matvecs_helper(Omega1, AOmega1, Omega2, ATOmega2, level: int, r: int):
    if level == 0:  # is this necessary? I think it is, but only because rowblock_x_diag doesn't know how to handle level = 0
        # NOTE this isn't symmetric in approximate case
        return right_pseudoinv(AOmega1, Omega1)
    blocksize = AOmega1.shape[0] // (2**level)
    assert blocksize >= r  # otherwise this makes no sense, we should just use fewer levels
    allUs = []
    allVs = []
    running_Omega1 = Omega1
    running_AOmega1 = AOmega1
    running_Omega2 = Omega2
    running_ATOmega2 = ATOmega2
    for l in range(level, 0, -1):
        U = sp.block_diag([np.linalg.svd(rowblock(running_AOmega1, l, block_i) @ row_nullifier(running_Omega1, l, block_i), full_matrices=False).U[:, :r] for block_i in range(2**l)], format="csr")
        V = sp.block_diag([np.linalg.svd(rowblock(running_ATOmega2, l, block_i) @ row_nullifier(running_Omega2, l, block_i), full_matrices=False).U[:, :r] for block_i in range(2**l)], format="csc")

        Dpart1 = blockwise_right_pseudoinv(running_AOmega1 - U @ (U.T @ running_AOmega1), running_Omega1, l)
        Dpart2 = U @ (U.T @ blockwise_right_pseudoinv(running_ATOmega2 - V @ (V.T @ running_ATOmega2), running_Omega2, l).T)
        D_l = Dpart1 + Dpart2

        running_AOmega1 = U.T @ (running_AOmega1 - D_l @ running_Omega1)
        running_Omega1 = V.T @ running_Omega1
        running_ATOmega2 = V.T @ (running_ATOmega2 - D_l.T @ running_Omega2)
        running_Omega2 = U.T @ running_Omega2
        allUs.append(U)
        allVs.append(V)
    cummulativeUs = list(accumulate(allUs, operator.matmul, initial=sp.linalg._interface.IdentityOperator((AOmega1.shape[0], AOmega1.shape[0]))))[1:]
    cummulativeVs = list(accumulate(allVs, operator.matmul, initial=sp.linalg._interface.IdentityOperator((ATOmega2.shape[0], ATOmega2.shape[0]))))[1:]
    def matvec(a):
        hss = assembleHSS(cummulativeUs, a, cummulativeVs, level, r)
        return np.concat(((hss @ Omega1).flatten(), (hss.T @ Omega2).flatten()))
    def rmatvec(M):
        ppp = AOmega1.shape[0]*AOmega1.shape[1]
        M1 = M[:ppp].reshape(AOmega1.shape)
        M2 = M[ppp:].reshape(ATOmega2.shape)
        return assembleTranspose(cummulativeUs, M1 @ Omega1.T + Omega2 @ M2.T, cummulativeVs, level)
                        # this term is for all the core matrices         # this one is for the diagonal blocks. blocksize^2 * num blocks
    degrees_of_freedom = sum(r**2 * 2**l for l in range(level, 0, -1)) + AOmega1.shape[0] * ATOmega2.shape[0] // (2**level)
    # NOTE: you can adjust the tolerance here
    # TODO: we should warm start by stitching smaller linear systems
    a_star = sp.linalg.lsmr(LinearOperator(shape=(AOmega1.shape[0]*AOmega1.shape[1] + ATOmega2.shape[0]*ATOmega2.shape[1], degrees_of_freedom), matvec=matvec, rmatvec=rmatvec), np.concat((AOmega1.flatten(), ATOmega2.flatten())), show=True, atol=1e-14, btol=1e-14)[0]  # atol=1e-9, btol=1e-9
    return assembleHSS(cummulativeUs, a_star, cummulativeVs, level, r)


def diana_matvecs(A, level: int, r: int, num_sketches: int):
    left_Omega = np.random.randn(A.shape[0], num_sketches)
    right_Omega = np.random.randn(A.shape[1], num_sketches)
    return diana_matvecs_helper(right_Omega, A @ right_Omega, left_Omega, A.T @ left_Omega, level, r)



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
    print(np.linalg.norm(A - A_tilde.toarray(), np.inf))
    A_diana = diana_fixed_up(A, 1, 3)
    print(np.linalg.norm(A - A_diana.toarray(), np.inf))
    A_diana_matvecs = diana_matvecs(A, 1, 3, 10000)
    print(np.linalg.norm(A - A_diana_matvecs.toarray(), np.inf))


if False:
    order = 3 # 8
    num_diags_above = 1
    A = SparseInverse(banded_gaussian(2**order, num_diags_above))
    A_tilde = random_access_greedy_alg(A.toarray(), order, 2*num_diags_above)
    print(np.linalg.norm(A.toarray() - A_tilde.toarray(), np.inf))
    A_tilde_matvec = matvec_alg_unified_sketch(A, order, 2*num_diags_above, 3*(2*num_diags_above))
    print(np.linalg.norm(A.toarray() - A_tilde_matvec.toarray(), np.inf))
    A_tilde_resketch = matvec_alg_resketch(A, order, 2*num_diags_above, 3*(2*num_diags_above))
    print(np.linalg.norm(A.toarray() - A_tilde_resketch.toarray(), np.inf))
    # A_diana = diana_exact(A.toarray(), order, 2) #2*num_diags_above)
    # print(np.linalg.norm(A.toarray() - A_diana.toarray(), np.inf))
    # B = recover_branch_matrix(A_tilde.A.toarray(), order)
    A_diana = diana_fixed_up(A.toarray(), order-1, 2*num_diags_above)
    print(np.linalg.norm(A.toarray() - A_diana.toarray(), np.inf))
    A_diana_matvec = diana_matvecs(A, order-1, 2*num_diags_above, 3*(2*num_diags_above))
    print(np.linalg.norm(A.toarray() - A_diana_matvec.toarray(), np.inf))

    # now with a too-small rank
    A_tilde = random_access_greedy_alg(A.toarray(), order, 1)
    print(np.linalg.norm(A.toarray() - A_tilde.toarray(), np.inf))
    A_tilde_matvec = matvec_alg_unified_sketch(A, order, 1, 3*(2*num_diags_above))
    print(np.linalg.norm(A.toarray() - A_tilde_matvec.toarray(), np.inf))
    A_tilde_resketch = matvec_alg_resketch(A, order, 1, 3*(2*num_diags_above))
    print(np.linalg.norm(A.toarray() - A_tilde_resketch.toarray(), np.inf))
    A_tilde_diana = diana_fixed_up(A.toarray(), order, 1)
    print(np.linalg.norm(A.toarray() - A_tilde_diana.toarray(), np.inf))
    A_tilde_diana_matvec = diana_matvecs(A, order, 1, 3*(2*num_diags_above))
    print(np.linalg.norm(A.toarray() - A_tilde_diana_matvec.toarray(), np.inf))


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


# if __name__ == "__main__":
if True:
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
    r = 5 #num_diags_above # true rank is 2*num_diags_above
    title = f"Banded:\nnum_diag={num_diags_above},level={levels},r={r}"

    # to ensure that blocksize >= r, use slightly larger blocksize when necessary
    recovery_levels = int(np.floor(np.log2(A.shape[0] / r)))

    methods = {
        "regression": lambda s: diana_matvecs(A, recovery_levels, r, s),
        "fresh sketches": lambda s: matvec_alg_resketch(A, recovery_levels, r, s),
        "one sketch": lambda s: matvec_alg_unified_sketch(A, recovery_levels, r, s),
    }
    def rel_error(A_tilde):
        return approx_Frob(alo(A_tilde) - A, int(3e2)) / approx_Frob(A, int(3e2))
    def datum(sketch_dim, method_name, method):
        return dict(
            sketch_dim=sketch_dim,
            method=method_name,
            relative_error=rel_error(method(sketch_dim)),
        )
    results = Parallel(n_jobs=1)(
        delayed(datum)(sketch_dim, method_name, method)
        for _, (method_name, method), sketch_dim in tqdm(list(product(range(1), methods.items(), [16, 32, 64, 128, 256])))  # 16, 
    )
    # random_err = rel_error(random_access_greedy_alg(A.toarray(), levels, r))
    # results += [dict(sketch_dim=sketch_dim, method="random", relative_error=random_err,)
    #     for sketch_dim in [16, 256]
    # ]

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame(results)
    df["total_sketching"] = df.apply(lambda x: x["sketch_dim"] * {"fresh sketches": levels, "one sketch": 1, "random": 1, "regression": 1}[x["method"]], axis=1)
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


# TODO
# change the tolerance?
# start from the levitt martinson solution to get Us and Vs, then do regression on the Ds? (warm starting from what's there already)
# rewrite in pytorch for speed
# right now, the first step of the algorithms are both basically the same except for the step of subtracting off D. can doing so help our version somehow?