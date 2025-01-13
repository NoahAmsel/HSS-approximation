import numpy as np
import scipy.sparse as sp

A = np.array([[1,0,1.01,0],[0,1,0,1],[1,0,0,1.02],[0,1.01,1,0]])

Uopt = [np.array([[1],[1]]), np.array([[1],[1]])]
Vopt = [np.array([[1],[1]]), np.array([[1],[1]])]
Gopt = 0.5 * np.array([[1, 1],[1,1]])
print("Opt error", np.linalg.norm(A - sp.block_diag(Uopt) @ Gopt @ sp.block_diag(Vopt).T)**2)

Ugreedy = [np.linalg.svd(A[:2, :])[0][:, [0]], np.linalg.svd(A[2:, :])[0][:, [0]]]
Vgreedy = [np.linalg.svd(A.T[:2, :])[0][:, [0]], np.linalg.svd(A.T[2:, :])[0][:, [0]]]
Gr = np.array([[0,0],[0,1.02]])
print("Greedy error", np.linalg.norm(A - sp.block_diag(Ugreedy) @ Gr @ sp.block_diag(Vgreedy).T)**2)

def best_core(A, U, V):
    """min_G ||UGV^T - A||_F = min_G ||G - U^T A V||"""
    return U.T @ A @ V

def left_pseudoinv(X, Y):
    """Computes X^+ Y"""
    return np.linalg.lstsq(X, Y, rcond=None)[0]

def right_pseudoinv(X, Y):
    """Computes XY^+"""
    return np.linalg.lstsq(Y.T, X.T, rcond=None)[0].T

def update_left_factor(A, right_factor):
    # TODO! renormalize. output should be ortho*normal*, which requires rescaling G too.
    short, long = right_factor.shape
    return sp.block_diag([
        right_pseudoinv(A[:long//2, :], right_factor[:short//2, :]),
        right_pseudoinv(A[long//2:, :], right_factor[short//2:, :])
    ])

def update_right_factor(A, left_factor):
    return update_left_factor(A.T, left_factor.T)


def alternating_minimization(A, U, V, G, iters):
    for _ in range(iters):
        G = best_core(A, U, V)
        U = update_left_factor(A, G @ V.T)
        V = update_right_factor(A, U @ G)
    return U, G, V


U0 = [np.random.randn(2,1), np.random.randn(2,1)]
V0 = [np.random.randn(2,1), np.random.randn(2,1)]
G0 = np.random.randn(2,2)

U, G, V = alternating_minimization(A, sp.block_diag(U0), sp.block_diag(V0), G0, 5)
print("AM error", np.linalg.norm(A - U @ G @ V.T)**2)

pertub_size = 0
U, G, V = alternating_minimization(A, sp.block_diag(Ugreedy), sp.block_diag(Vgreedy) + np.random.randn(4, 2) * pertub_size, Gr, 30)
print("AM error", np.linalg.norm(A - U @ G @ V.T)**2)
