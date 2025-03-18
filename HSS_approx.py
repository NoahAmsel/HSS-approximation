from itertools import accumulate
import numpy as np
import operator
import scipy.sparse as sp
from scipy.sparse.linalg import aslinearoperator as alo, LinearOperator
from scipy.sparse.linalg._interface import IdentityOperator

from structures import FourPartLens


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


# class AlgIter:
#     def rowblock_x_diag(self, block_i):
#         pass

#     def colblock_x_diag(self, block_i):
#         pass

#     def get_D(self, block_i, U_i, V_i):
#         pass

#     def setup_next_level(self, U_l, V_l, D_l):
#         pass

#     def main(self):
#         if self.level == self.top_level:
#             return self.recover_last_level()
#         # TODO: replace with sklearn TruncatedSVD or with scipy.sparse.svds
#         U_l_list = [np.linalg.svd(self.rowblock_x_diag(block_i), full_matrices=False).U[:, :self.rank] for block_i in range(2**self.level)]
#         U_l = sp.block_diag(U_l_list)
#         # TODO: take conjugate transpose, not plain transpose. (or better yet, just switch this to be identical to above line but with A^* instead of A)
#         V_l_list = [np.linalg.svd(self.colblock_x_diag(block_i), full_matrices=False).U[:, :self.rank] for block_i in range(2**self.level)]
#         V_l = sp.block_diag(V_l_list)
#         D_l = sp.block_diag([self.D(U_l_list[block_i], V_l_list[block_i], block_i) for block_i in range(2**self.level)])
#         A_lminus1 = self.setup_next_level(U_l, V_l, D_l).main()
#         return FourPartLens(U_l, A_lminus1, V_l, D_l)

# def RandomAccess(AlgIter):
#     def __init__(self, A, level, rank, top_level):
#         self.A = A
#         self.level
#         self.rank
#         self.top_level

#     def rowblock_x_diag(self, block_i):
#         rowblock_x_diag(self.A, self.level, block_i)

#     def colblock_x_diag(self, block_i):
#         self.rowblock_x_diag(self.A.T, block_i)

#     def get_D(self, block_i, U_i, V_i):
#         return diagblock(self.A, self.level, block_i)

#     def setup_next_level(self, U_l, V_l, D_l):
#         return type(self)(
#             np.asarray(U_l.T @ (self.A - D_l) @ V_l),
#             self.level-1,
#             self.rank,
#             self.top_level,
#         )

#     @classmethod
#     def __call__(cls, A, level, rank, top_level):
#         return cls(A, level, rank, top_level).main()


# TODO! block_diag is outputting COO format.
# we should use CSR for U and D, and CSC for V
def random_access_greedy_alg(A: np.ndarray, level: int, r: int, top_level: int = 0):
    if level == top_level:
        return A
    # TODO: replace with sklearn TruncatedSVD or with scipy.sparse.svds
    U_l = sp.block_diag([np.linalg.svd(rowblock_x_diag(A, level, block_i), full_matrices=False).U[:, :r] for block_i in range(2**level)])
    # TODO: take conjugate transpose, not plain transpose. (or better yet, just switch this to be identical to above line but with A^* instead of A)
    V_l = sp.block_diag([np.linalg.svd(colblock_x_diag(A, level, block_i), full_matrices=False).Vh.T[:, :r] for block_i in range(2**level)])
    D_l = sp.block_diag([diagblock(A, level, block_i) for block_i in range(2**level)])
    A_lminus1 = random_access_greedy_alg(np.asarray(U_l.T @ (A - D_l) @ V_l), level-1, r, top_level=top_level)
    return FourPartLens(U_l, A_lminus1, V_l, D_l)


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
    out = nullspace_basis(rowblock(Omega, level, block_i))
    if out.shape[1] == 0:
        rows, samples = rowblock(Omega, level, block_i).shape
        raise ValueError(f"Unable to nullify {rows} rows when there are only {samples} columns. Try increasing the sketch size.")
    return out


def right_pseudoinv(X, Y):
    """Computes XY^+"""
    return np.linalg.lstsq(Y.T, X.T, rcond=None)[0].T


def two_sided_lstsq(Omega1, AOmega1, Omega2, ATOmega2):
    # NOTE: This can get quite slow
    def vec(X): return X.T.reshape(-1)
    def unvec(x, num_rows): return x.reshape(len(x) // num_rows, num_rows).T
    LHS = np.vstack((
        np.kron(Omega1.T, np.eye(AOmega1.shape[0])),
        np.kron(np.eye(ATOmega2.shape[0]), Omega2.T),
    ))
    RHS = np.concat((
        vec(AOmega1),
        vec(ATOmega2.T),
    ))
    vecA = np.linalg.lstsq(LHS, RHS, rcond=None)[0]
    return unvec(vecA, AOmega1.shape[0])


def two_sided_iterative(Omega1, AOmega1, Omega2, ATOmega2):
    # TODO: precondition me somehow
    Arows, Acols = AOmega1.shape[0], ATOmega2.shape[0]
    s1, s2 = Omega1.shape[1], Omega2.shape[1]
    def vec(X): return X.T.reshape(-1)
    def unvec(x, num_rows): return x.reshape(len(x) // num_rows, num_rows).T
    def matvec(vec_a):
        A = unvec(vec_a, AOmega1.shape[0])
        return np.concat((
            vec(A @ Omega1),
            vec(Omega2.T @ A),
        ))
    def rmatvec(v):
        mat1 = unvec(v[:Arows * s1], Arows)
        mat2 = unvec(v[Arows * s1:], s2)
        return vec(mat1 @ Omega1.T) + vec(Omega2 @ mat2)
    LHS = LinearOperator(shape=(Arows * s1 + Acols * s2, Arows * Acols), matvec=matvec, rmatvec=rmatvec)
    RHS = np.concat((
        vec(AOmega1),
        vec(ATOmega2.T),
    ))
    warm_start = vec(right_pseudoinv(AOmega1, Omega1))
    result = sp.linalg.lsqr(LHS, RHS, x0=warm_start, atol=1e-16, btol=1e-16)
    # assert result[1] < 7, result
    return unvec(result[0], Arows)


def blockwise_right_pseudoinv(X, Y, level):
    """X has row blocks X_i, Y has row blocks Y_i.
    Returns block diagonal matrix with X_i Y_i^+ blocks.
    """
    return sp.block_diag([right_pseudoinv(rowblock(X, level, block_i), rowblock(Y, level, block_i)) for block_i in range(2**level)])


# def matvec_alg_unified_sketch_helper(Omega1, AOmega1, Omega2, ATOmega2, level: int, r: int, top_level: int):
#     if level == top_level:
#         # NOTE this isn't symmetric in approximate case
#         return right_pseudoinv(AOmega1, Omega1)
#     U_l = sp.block_diag([np.linalg.svd(rowblock(AOmega1, level, block_i) @ row_nullifier(Omega1, level, block_i), full_matrices=False).U[:, :r] for block_i in range(2**level)])
#     V_l = sp.block_diag([np.linalg.svd(rowblock(ATOmega2, level, block_i) @ row_nullifier(Omega2, level, block_i), full_matrices=False).U[:, :r] for block_i in range(2**level)])
#     # TODO! instead of this diagonal recovery, do a two sided least squares solve to recover the diagonal blocks.
#     # each block is a pretty small least squares problem
#     # then just find D - UU^T D VV^T explicitly
#     # to see the effect in the experiments, just try adding some huge random entries on the diagonal blocks
#     Dpart1 = blockwise_right_pseudoinv(AOmega1 - U_l @ (U_l.T @ AOmega1), Omega1, level)
#     Dpart2 = U_l @ (U_l.T @ blockwise_right_pseudoinv(ATOmega2 - V_l @ (V_l.T @ ATOmega2), Omega2, level).T)
#     D_l = Dpart1 + Dpart2
#     newOmega1 = V_l.T @ Omega1
#     newAOmega1 = U_l.T @ (AOmega1 - D_l @ Omega1)
#     newOmega2 = U_l.T @ Omega2
#     newATOmega2 = V_l.T @ (ATOmega2 - D_l.T @ Omega2)
#     A_lminus1 = matvec_alg_unified_sketch_helper(newOmega1, newAOmega1, newOmega2, newATOmega2, level-1, r, top_level)
#     return FourPartLens(U_l, A_lminus1, V_l, D_l)


def matvec_alg_unified_sketch(A, level: int, r: int, num_sketches:int, top_level: int = 0, two_sided_pseudoinverse: bool = False):
    left_Omega = np.random.randn(A.shape[0], num_sketches)
    right_Omega = np.random.randn(A.shape[1], num_sketches)
    A_right_Omega = A @ right_Omega
    AT_left_Omega = A.T @ left_Omega
    return matvec_alg_double_unified_sketch_helper(
        right_Omega, A_right_Omega, left_Omega, AT_left_Omega,
        right_Omega, A_right_Omega, left_Omega, AT_left_Omega,
        level, r, top_level=top_level, two_sided_pseudoinverse=two_sided_pseudoinverse
    )


def matvec_alg_double_unified_sketch_helper(Omega1, AOmega1, Omega2, ATOmega2, tilde_Omega1, tilde_AOmega1, tilde_Omega2, tilde_ATOmega2, level: int, r: int, top_level: int, two_sided_pseudoinverse: bool = False):
    if level == top_level:
        if two_sided_pseudoinverse:
            return two_sided_iterative(tilde_Omega1, tilde_AOmega1, tilde_Omega2, tilde_ATOmega2)
        else:
            return right_pseudoinv(tilde_AOmega1, tilde_Omega1)
    U_l_list = [np.linalg.svd(rowblock(AOmega1, level, block_i) @ row_nullifier(Omega1, level, block_i), full_matrices=False).U[:, :r] for block_i in range(2**level)]
    U_l = sp.block_diag(U_l_list)
    V_l_list = [np.linalg.svd(rowblock(ATOmega2, level, block_i) @ row_nullifier(Omega2, level, block_i), full_matrices=False).U[:, :r] for block_i in range(2**level)]
    V_l = sp.block_diag(V_l_list)
    if two_sided_pseudoinverse:
        # TODO! instead of this diagonal recovery, do a two sided least squares solve to recover the diagonal blocks.
        # each block is a pretty small least squares problem
        # then just find D - UU^T D VV^T explicitly
        # to see the effect in the experiments, just try adding some huge random entries on the diagonal blocks
        Uorth_list = [IdentityOperator((U_l_i.shape[0], U_l_i.shape[0])) - alo(U_l_i) @ alo(U_l_i).T for U_l_i in U_l_list]
        Vorth_list = [IdentityOperator((V_l_i.shape[0], V_l_i.shape[0])) - alo(V_l_i) @ alo(V_l_i).T for V_l_i in V_l_list]
        # approx_A_ll = sp.block_diag([
        #     two_sided_iterative(rowblock(tilde_Omega1, level, block_i), rowblock(tilde_AOmega1, level, block_i), rowblock(tilde_Omega2, level, block_i), rowblock(tilde_ATOmega2, level, block_i))
        #     for block_i in range(2**level)
        # ])
        approx_A_ll = sp.block_diag([
            two_sided_lstsq(rowblock(tilde_Omega1, level, block_i), Uorth_list[block_i] @ rowblock(tilde_AOmega1, level, block_i), rowblock(tilde_Omega2, level, block_i), Vorth_list[block_i] @ rowblock(tilde_ATOmega2, level, block_i))
            for block_i in range(2**level)
        ])
        D_l = approx_A_ll - U_l @ ((U_l.T @ approx_A_ll) @ V_l) @ V_l.T
    else:
        Dpart1 = blockwise_right_pseudoinv(tilde_AOmega1 - U_l @ (U_l.T @ tilde_AOmega1), tilde_Omega1, level)
        Dpart2 = U_l @ (U_l.T @ blockwise_right_pseudoinv(tilde_ATOmega2 - V_l @ (V_l.T @ tilde_ATOmega2), tilde_Omega2, level).T)
        D_l = Dpart1 + Dpart2
    newOmega1 = V_l.T @ Omega1
    newAOmega1 = U_l.T @ (AOmega1 - D_l @ Omega1)
    newOmega2 = U_l.T @ Omega2
    newATOmega2 = V_l.T @ (ATOmega2 - D_l.T @ Omega2)
    new_tilde_Omega1 = V_l.T @ tilde_Omega1
    new_tilde_AOmega1 = U_l.T @ (tilde_AOmega1 - D_l @ tilde_Omega1)
    new_tilde_Omega2 = U_l.T @ tilde_Omega2
    new_tilde_ATOmega2 = V_l.T @ (tilde_ATOmega2 - D_l.T @ tilde_Omega2)
    A_lminus1 = matvec_alg_double_unified_sketch_helper(newOmega1, newAOmega1, newOmega2, newATOmega2, new_tilde_Omega1, new_tilde_AOmega1, new_tilde_Omega2, new_tilde_ATOmega2, level-1, r, top_level)
    return FourPartLens(U_l, A_lminus1, V_l, D_l)


def matvec_alg_double_unified_sketch(A, level: int, r: int, num_sketches:int, top_level: int = 0, two_sided_pseudoinverse: bool = False):
    left_Omega = np.random.randn(A.shape[0], num_sketches)
    right_Omega = np.random.randn(A.shape[1], num_sketches)
    tilde_left_Omega = np.random.randn(A.shape[0], num_sketches)
    tilde_right_Omega = np.random.randn(A.shape[1], num_sketches)
    return matvec_alg_double_unified_sketch_helper(
        right_Omega, A @ right_Omega, left_Omega, A.T @ left_Omega,
        tilde_right_Omega, A @ tilde_right_Omega, tilde_left_Omega, A.T @ tilde_left_Omega,
        level, r, top_level=top_level, two_sided_pseudoinverse=two_sided_pseudoinverse
    )


def matvec_alg_resketch(A, level: int, r: int, num_sketches_per_level: int, top_level: int = 0, second_sketch_for_D: bool = True):
    if level == top_level:
        if num_sketches_per_level >= A.shape[1]:
            return A @ IdentityOperator((A.shape[1], A.shape[1]))
        else:
            Omega1 = np.random.randn(A.shape[1], num_sketches_per_level)
            AOmega1 = A @ Omega1
            # NOTE this isn't symmetric in approximate case
            return right_pseudoinv(AOmega1, Omega1)
    Omega1 = np.random.randn(A.shape[1], num_sketches_per_level)
    AOmega1 = A @ Omega1
    Omega2 = np.random.randn(A.shape[0], num_sketches_per_level)
    ATOmega2 = A.T @ Omega2
    U_l = sp.block_diag([np.linalg.svd(rowblock(AOmega1, level, block_i) @ row_nullifier(Omega1, level, block_i), full_matrices=False).U[:, :r] for block_i in range(2**level)])
    V_l = sp.block_diag([np.linalg.svd(rowblock(ATOmega2, level, block_i) @ row_nullifier(Omega2, level, block_i), full_matrices=False).U[:, :r] for block_i in range(2**level)])
    if second_sketch_for_D:
        tilde_Omega1 = np.random.randn(A.shape[1], num_sketches_per_level)
        tilde_AOmega1 = A @ tilde_Omega1
        tilde_Omega2 = np.random.randn(A.shape[0], num_sketches_per_level)
        tilde_ATOmega2 = A.T @ tilde_Omega2
        Dpart1 = blockwise_right_pseudoinv(tilde_AOmega1 - U_l @ (U_l.T @ tilde_AOmega1), tilde_Omega1, level)
        Dpart2 = U_l @ (U_l.T @ blockwise_right_pseudoinv(tilde_ATOmega2 - V_l @ (V_l.T @ tilde_ATOmega2), tilde_Omega2, level).T)
    else:
        Dpart1 = blockwise_right_pseudoinv(AOmega1 - U_l @ (U_l.T @ AOmega1), Omega1, level)
        Dpart2 = U_l @ (U_l.T @ blockwise_right_pseudoinv(ATOmega2 - V_l @ (V_l.T @ ATOmega2), Omega2, level).T)
    D_l = Dpart1 + Dpart2
    A_lminus1 = matvec_alg_resketch(alo(U_l).T @ (alo(A) - alo(D_l)) @ alo(V_l), level-1, r, num_sketches_per_level, top_level=top_level)
    return FourPartLens(U_l, A_lminus1, V_l, D_l)


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

# TODO! rather than stepping with respect to all of the diagonals from all levels simultaneously
# just do one level at a time. Then the least squares problem is much easier to solve.
def random_access_optimal_core(A, level: int, r: int):
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


def matvecs_optimal_core_helper(Omega1, AOmega1, Omega2, ATOmega2, level: int, r: int, recover_D: bool):
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

        if recover_D:
            Dpart1 = blockwise_right_pseudoinv(running_AOmega1 - U @ (U.T @ running_AOmega1), running_Omega1, l)
            Dpart2 = U @ (U.T @ blockwise_right_pseudoinv(running_ATOmega2 - V @ (V.T @ running_ATOmega2), running_Omega2, l).T)
            D_l = Dpart1 + Dpart2
        else:
            # all zeros
            D_l = sp.csr_array((running_AOmega1.shape[0], running_ATOmega2.shape[0]))
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
    # TODO: consider lsmr instead
    a_star = sp.linalg.lsqr(LinearOperator(shape=(AOmega1.shape[0]*AOmega1.shape[1] + ATOmega2.shape[0]*ATOmega2.shape[1], degrees_of_freedom), matvec=matvec, rmatvec=rmatvec), np.concat((AOmega1.flatten(), ATOmega2.flatten())), show=False, atol=1e-9, btol=1e-9)[0]
    return assembleHSS(cummulativeUs, a_star, cummulativeVs, level, r)


def matvecs_optimal_core(A, level: int, r: int, num_sketches: int, recover_D: bool):
    left_Omega = np.random.randn(A.shape[0], num_sketches)
    right_Omega = np.random.randn(A.shape[1], num_sketches)
    return matvecs_optimal_core_helper(right_Omega, A @ right_Omega, left_Omega, A.T @ left_Omega, level, r, recover_D)


if False:
    N = 2**4; r = 3
    A = np.random.randn(N, r) @ np.random.randn(r, N)
    A_tilde = random_access_greedy_alg(A, 1, 3)
    print(np.linalg.norm(A - A_tilde.toarray(), np.inf))
    A_diana = random_access_optimal_core(A, 1, 3)
    print(np.linalg.norm(A - A_diana.toarray(), np.inf))
    A_diana_matvecs = matvecs_optimal_core(A, 1, 3, 10000, False)
    print(np.linalg.norm(A - A_diana_matvecs.toarray(), np.inf))


if False:
    from problems import banded_gaussian, SparseInverse
    order = 3 # 8
    num_diags_above = 1
    A = SparseInverse(banded_gaussian(2*num_diags_above * 2**(order+1), num_diags_above))
    print("orig", A.shape)
    print("expected", 2 * 2*num_diags_above)
    A_tilde = random_access_greedy_alg(A.toarray(), order, 2*num_diags_above)
    print("1.\t", np.linalg.norm(A.toarray() - A_tilde.toarray(), np.inf))
    A_tilde_matvec = matvec_alg_unified_sketch(A, order, 2*num_diags_above, 3*(2*num_diags_above))
    print("2.\t", np.linalg.norm(A.toarray() - A_tilde_matvec.toarray(), np.inf))
    A_tilde_matvec_double = matvec_alg_double_unified_sketch(A, order, 2*num_diags_above, 3*(2*num_diags_above))
    print("3.\t", np.linalg.norm(A.toarray() - A_tilde_matvec_double.toarray(), np.inf))
    A_tilde_resketch = matvec_alg_resketch(A, order, 2*num_diags_above, 3*(2*num_diags_above))
    print("4.\t", np.linalg.norm(A.toarray() - A_tilde_resketch.toarray(), np.inf))
    A_diana = random_access_optimal_core(A.toarray(), order-1, 2*num_diags_above)
    print("5.\t", np.linalg.norm(A.toarray() - A_diana.toarray(), np.inf))
    A_diana_matvec = matvecs_optimal_core(A, order-1, 2*num_diags_above, 3*(2*num_diags_above), False)
    print("6.\t", np.linalg.norm(A.toarray() - A_diana_matvec.toarray(), np.inf))

    # now with a too-small rank
    A_tilde = random_access_greedy_alg(A.toarray(), order, 1)
    print("1.\t", np.linalg.norm(A.toarray() - A_tilde.toarray(), np.inf))
    A_tilde_matvec = matvec_alg_unified_sketch(A, order, 1, 3*(2*num_diags_above))
    print("2.\t", np.linalg.norm(A.toarray() - A_tilde_matvec.toarray(), np.inf))
    A_tilde_matvec_double = matvec_alg_double_unified_sketch(A, order, 1, 3*(2*num_diags_above))
    print("3.\t", np.linalg.norm(A.toarray() - A_tilde_matvec_double.toarray(), np.inf))
    A_tilde_resketch = matvec_alg_resketch(A, order, 1, 3*(2*num_diags_above))
    print("4.\t", np.linalg.norm(A.toarray() - A_tilde_resketch.toarray(), np.inf))
    A_tilde_diana = random_access_optimal_core(A.toarray(), order, 1)
    print("5.\t", np.linalg.norm(A.toarray() - A_tilde_diana.toarray(), np.inf))
    A_tilde_diana_matvec = matvecs_optimal_core(A, order, 1, 3*(2*num_diags_above), True)
    print("6.\t", np.linalg.norm(A.toarray() - A_tilde_diana_matvec.toarray(), np.inf))

# TODO
# change the tolerance?
# start from the levitt martinson solution to get Us and Vs, then do regression on the Ds? (warm starting from what's there already)
# rewrite in pytorch for speed
# right now, the first step of the algorithms are both basically the same except for the step of subtracting off D. can doing so help our version somehow?
# TODO: correct the fresh sketch option by doing fresh sketches for the diagonal part too? actually check the paper
