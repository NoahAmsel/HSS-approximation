from itertools import product

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse.linalg import aslinearoperator as alo, LinearOperator
import seaborn as sns

from HSS_approx import matvec_alg_resketch, matvec_alg_unified_sketch, matvecs_optimal_core, random_access_greedy_alg, matvec_alg_double_unified_sketch
from problems import banded_gaussian, factor2_example, factor2_optimal_solution, grid_schur_complement, SparseInverse, star_matrix, random_hss


global_show_title = True


def approx_Frob(A, sketch_size):
    return np.linalg.norm(A @ np.random.randn(A.shape[1], sketch_size), ord='fro') / np.sqrt(sketch_size)


class QueryTracker(LinearOperator):
    def __init__(self, A):
        super().__init__(A.dtype, A.shape)
        self.A = A
        self.queries = 0
        self.transpose_queries = 0
        self._init_dtype()

    def _matmat(self, x):
        self.queries += x.shape[1]
        return self.A @ x

    def _rmatmat(self, x):
        self.transpose_queries += x.shape[1]
        return self.A.T @ x

    def total_queries(self):
        return self.queries + self.transpose_queries


def theorem_4_1_optimality_ratio(sketches_per_level, rank, levels):
    s = sketches_per_level
    k = rank
    gamma_r = (1 + 2*np.e*(s - 2*k)/np.sqrt((s - 3*k)**2 - 1))**2
    gamma_d = 2*k/(s - 2*k - 1)
    return 2 * gamma_r * (1 + gamma_d) * levels


def plot_sketch_size_vs_error(A, title, methods, num_sketches, non_sketching_methods={}, repeats=1, approx_frobenius=None, savedir=".", xscale="log"):
    Anorm = np.linalg.norm(A) if approx_frobenius is None else approx_Frob(A, approx_frobenius)
    def rel_error(A_tilde):
        error = np.linalg.norm(A_tilde.toarray() - A) if approx_frobenius is None else approx_Frob(alo(A_tilde) - A, approx_frobenius)
        return error / Anorm


    def datum(sketch_dim, method_name, method):
        wrapped_A = QueryTracker(A)
        error = rel_error(method(wrapped_A, sketch_dim))
        return {
            "Method": method_name,
            "Relative\nFrobenius Error": error,
            "Sketch Size ($s$)": sketch_dim,
            "Total Queries": wrapped_A.total_queries(),
        }

    # Unfortunately I can't figure out how to parallelize this because I can't pickle A if it's an LU factorization object
        # from joblib import load, dump, wrap_non_picklable_objects
        # filename = os.path.join('joblib_test.mmap')
        # _ = dump(wrap_non_picklable_objects(A), filename)
        # from scipy.sparse.linalg import SuperLU
        # SuperLU
    sketching_results = Parallel(n_jobs=1)(
        delayed(datum)(sketch_dim, method_name, method)
        for _, (method_name, method), sketch_dim in tqdm(list(product(range(repeats), methods.items(), num_sketches)))
    )
    non_sketching_results = {method_name: rel_error(method(A)) for method_name, method in non_sketching_methods.items()}

    df = pd.DataFrame(sketching_results)
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 18, "legend.fontsize": 13})
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 3.5))
    plt.yscale("log")
    if global_show_title:
        plt.suptitle(title + "\n")
        bbox_to_anchor=(0,0,1,.825)
    else:
        plt.suptitle("\n")
        bbox_to_anchor=(0,0,1,.85)
    for i, (ax, x_col) in enumerate(zip(axs, ["Sketch Size ($s$)", "Total Queries"])):
        # NOTE: This assumes that these non sketching methods are deterministic
        for (method_name, method_result), style in zip(non_sketching_results.items(), ['-.', ':']):
            ax.axhline(method_result, label=method_name, color='black', linestyle=style)
        p = sns.lineplot(df, x=x_col, y="Relative\nFrobenius Error", hue="Method", style="Method", errorbar=("ci", 100), marker='o', ax=ax, legend=(i == 1))
        ax.set_xscale(xscale)
        if i == 1:
            # fuller_grid = np.geomspace(df["Sketch Size ($s$)"].min(), df["Sketch Size ($s$)"].max(), num=100)
            # ax.plot(fuller_grid, method_result * theorem_4_1_optimality_ratio(fuller_grid, rank=RANK, levels=LEVELS), color='black', linestyle=style)
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            plt.figlegend(handles, labels, loc='outside upper center', bbox_to_anchor=bbox_to_anchor, ncol=len(methods)+len(non_sketching_methods), labelspacing=0.)
    Path(savedir).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.get_figure().savefig(Path(savedir) / f"{title.translate(str.maketrans({'\n': '', '$': '', '\\': ''}))}.pdf", dpi=600)


def min_queries(A, recovery_levels):
    return max(A.shape) // 2**recovery_levels + 1


def schur_gunnar():
    r = 30  # rank. given in fig 7
    m = 2*r  # leaf size. given in fig 7 and beginning of experiments section
    s = max(r+m, 3*r)  # sketching dimension. given in beginning of experiments section and beginning of sec. 4
    levels = 6
    n = 51  # the dimension that we split on. Groups are 25 x N, 25 x N, and 1 x N
    N = (2**levels) * m
    A = grid_schur_complement(n, N)
    recovery_rank = r
    title = f"Grid Schur Complement:\nn={n},N={N},m={m},k={recovery_rank}"
    methods = {
        "Fresh Sketches (Alg 4.1)": lambda A, s: matvec_alg_resketch(A, levels, recovery_rank, s),
        "Half Fresh Sketches": lambda A, s: matvec_alg_resketch(A, levels, recovery_rank, s, second_sketch_for_D=False),
        "Reused Sketch [LM24a]": lambda A, s: matvec_alg_unified_sketch(A, levels, recovery_rank, s),
    }
    plot_sketch_size_vs_error(
        A,
        title,
        methods,
        np.geomspace(s // 2, s * 2, num=16, endpoint=True, dtype=int),
        repeats=1,
        approx_frobenius=int(1e3),
        savedir="out/schur",
    )


def schur_smaller():
    r = 10
    m = r  # leaf size
    n = 51  # the dimension that we split on. Groups are 25 x N, 25 x N, and 1 x N
    levels = 6
    N = m * (2**(levels+1))
    A = grid_schur_complement(n, N)

    title = "Grid Schur Complement:\n" + rf"$N={N},L={levels},k={r}$"
    methods = {
        "Fresh Sketches (Alg 4.1)": lambda A, s: matvec_alg_resketch(A, levels, r, s),
        # "Half Fresh Sketches": lambda A, s: matvec_alg_resketch(A, levels, recovery_rank, s, second_sketch_for_D=False),
        "Reused Sketch QR [LM24a]": lambda A, s: matvec_alg_unified_sketch(A, levels, r, s, use_qr=True),
        "Reused Sketch SVD": lambda A, s: matvec_alg_unified_sketch(A, levels, r, s, use_qr=False),
    }
    A = A @ np.eye(*A.shape)
    non_sketching_methods = {"Entrywise Access": lambda A: random_access_greedy_alg(A, levels, r)}
    min_q = max(3*r+2, min_queries(A, levels))
    plot_sketch_size_vs_error(
        A,
        title,
        methods,
        np.geomspace(min_q, 20 * min_q, 20, endpoint=True, dtype=int),
        non_sketching_methods=non_sketching_methods,
        repeats=10,
        approx_frobenius=None,
        savedir="out/schur",
    )


def star():
    A = star_matrix()
    levels = 5
    r = 30
    title = "Boundary Integral Operator\n" + rf"$N = {A.shape[0]}, L={levels}, k={r}$"
    methods = {
        "Fresh Sketches (Alg 4.1)": lambda A, s: matvec_alg_resketch(A, levels, r, s),
        # "Half Fresh Sketches": lambda A, s: matvec_alg_resketch(A, levels, r, s, second_sketch_for_D=False),
        "Reused Sketch QR [LM24a]": lambda A, s: matvec_alg_unified_sketch(A, levels, r, s, use_qr=True),
        "Reused Sketch SVD": lambda A, s: matvec_alg_unified_sketch(A, levels, r, s, use_qr=False),
    }
    non_sketching_methods = {"Entrywise Access": lambda A: random_access_greedy_alg(A, levels, r)}
    min_sketches = max(3*r+2, min_queries(A, levels))
    plot_sketch_size_vs_error(
        A,
        title,
        methods,
        np.geomspace(min_sketches, 20 * min_sketches, 20, endpoint=True, dtype=int),
        non_sketching_methods=non_sketching_methods,
        repeats=10,
        savedir="out/star",
    )


def banded_inverse(levels, num_diags_above, r=None):
    if r is None:
        # this is the true rank of the matrix
        r = 2 * num_diags_above
    # banded inverse
    A = SparseInverse(banded_gaussian(2**levels, num_diags_above))
    # to ensure that blocksize >= r, use slightly larger blocksize when necessary
    recovery_levels = int(np.floor(np.log2(A.shape[0] / r)))
    title = "Banded Matrix\n" + rf"$N = {A.shape[0]}, L = {recovery_levels}, b = {2*num_diags_above + 1}, k = {r}$"

    methods = {
        "Fresh Sketches (Alg 4.1)": lambda A, s: matvec_alg_resketch(A, recovery_levels, r, s),
        # "Half Fresh Sketches": lambda A, s: matvec_alg_resketch(A, recovery_levels, r, s, second_sketch_for_D=False),
        "Reused Sketch QR [LM24a]": lambda A, s: matvec_alg_unified_sketch(A, recovery_levels, r, s, use_qr=True),
        "Reused Sketch SVD": lambda A, s: matvec_alg_unified_sketch(A, recovery_levels, r, s, use_qr=False),
    }
    non_sketching_methods = {"Entrywise Access": lambda A: random_access_greedy_alg(A.toarray(), levels, r)}
    min_q = max(3*r+2, min_queries(A, levels))
    plot_sketch_size_vs_error(
        A,
        title,
        methods,
        np.geomspace(min_q, 20 * min_q, 20, endpoint=True, dtype=int),
        non_sketching_methods=non_sketching_methods,
        repeats=10,
        approx_frobenius=int(1e3),
        savedir="out/banded_inverse",
    )


def two_factor(level, eps):
    blocks = 2 ** level
    A = factor2_example(blocks, eps)
    rank = 1
    top_level = 0
    methods = {
        "Fresh Sketches (Alg 4.1)": lambda A, s: matvec_alg_resketch(A, level, rank, s, top_level=top_level),
        # "Half Fresh Sketches": lambda A, s: matvec_alg_resketch(A, level, rank, s, top_level=top_level, second_sketch_for_D=False),
        "Reused Sketch QR [LM24a]": lambda A, s: matvec_alg_unified_sketch(A, level, rank, s, top_level=top_level, use_qr=True),
        "Reused Sketch SVD": lambda A, s: matvec_alg_unified_sketch(A, level, rank, s, top_level=top_level, use_qr=False),
        # "Double Recycled Sketch": lambda A, s: matvec_alg_double_unified_sketch(A, level, rank, s, top_level=top_level),
    }
    non_sketching_methods = {"Entrywise Access": lambda A: random_access_greedy_alg(A, level, rank, top_level=top_level), "Optimal": lambda _: factor2_optimal_solution(blocks)}
    title = "Hard Construction\n" + rf"$N={A.shape[0]},L={level},k={rank},\delta={eps}$"
    min_q = max(3*rank+2, min_queries(A, level))
    plot_sketch_size_vs_error(
        A,
        title,
        methods,
        np.geomspace(min_q, 10 * 96 * min_q, 20, endpoint=True, dtype=int),
        non_sketching_methods=non_sketching_methods,
        repeats=10,
        savedir="out/two_factor",
    )


def spike(A, level: int, rank: int, top_level: int):
    # NOTE: Two sided least squares at the final iteration helps the spike
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] == 2 ** (level+1) * rank
    critical_num_sketches = A.shape[1] // 2**(level - top_level)
    methods = {
        "Reused Sketch [LM24a]": lambda A, s: matvec_alg_unified_sketch(A, level, rank, s, top_level=top_level),
        "Double Recycled Sketch": lambda A, s: matvec_alg_double_unified_sketch(A, level, rank, s, top_level=top_level),
    }
    title = f"Spike Due to Pseudoinverse\nN={A.shape[0]},rank={rank},top_level={top_level}"
    all_num_sketches = np.concatenate((
        np.geomspace(critical_num_sketches//3, critical_num_sketches, 20, endpoint=True, dtype=int),
        np.geomspace(critical_num_sketches, critical_num_sketches*3, 20, endpoint=True, dtype=int),
    ))
    plot_sketch_size_vs_error(
        A,
        title,
        methods,
        all_num_sketches,
        repeats=5,
        savedir="out/spike",
    )


def gaussian_spike():
    level = 5
    N = 2 ** level
    top_level = level - 3
    rank = 3
    A = np.random.randn(rank * 2*N, rank * 2*N)
    spike(A, level, rank, top_level)


# TODO!
# convex combo of random HSS with s.v.'s of all blocks = 1 and random gaussian. check for the spike
# also, try adding a large diagonal element instead of random gaussian and see if that throws us off. maybe that will motivate double sided diagonal recovery
# in cases where A really isn't close to U_l U_l^T A V_l V_l^T, we expect recycling sketches to hurt. but haven't found a good example yet
# diana: take an exact telescoping factorization and just perturb the top level U and V


def noisy():
    level = 7
    rank = 5
    HSS = random_hss(level, r=rank, diag_weight=1)
    # HSS = HSS + 2 * sp.block_diag([np.random.randn(2*rank, 2*rank) for _ in range(2**level)]).toarray()
    noise = (np.eye(*HSS.shape) - HSS.U @ HSS.U.T) @ np.random.randn(*HSS.shape)
    noise_level = 1e-2
    noise *= noise_level * np.linalg.norm(HSS.toarray()) / np.linalg.norm(noise)
    A = np.array(HSS.toarray() + noise)
    top_level = 0
    methods = {
        # "Fresh 2 side": lambda A, s: matvec_alg_resketch(A, level, rank, s, top_level=top_level),
        "Recycled 2 side": lambda A, s: matvec_alg_unified_sketch(A, level, rank, s, top_level=top_level, two_sided_pseudoinverse=True),
        "Fresh": lambda A, s: matvec_alg_resketch(A, level, rank, s, top_level=top_level),
        "Recycled": lambda A, s: matvec_alg_unified_sketch(A, level, rank, s, top_level=top_level),
    }
    title = f"Noisy gaussian"
    plot_sketch_size_vs_error(
        A,
        title,
        methods,
        np.geomspace(min_queries(A, level), 4*min_queries(A, level), 10, dtype=int),
        non_sketching_methods={"Entrywise Access": lambda A: random_access_greedy_alg(A, level, rank, top_level=top_level)},
        repeats=1,
        savedir="out/noisy",
    )


def error():
    # TODO! this shows that there's an error in "Recycled 2 side"
    level = 7
    rank = 2
    A = sp.block_diag([np.random.randn(2*rank, 2*rank) for _ in range(2**level)]).toarray()
    top_level = 0
    methods = {
        # "Fresh 2 side": lambda A, s: matvec_alg_resketch(A, level, rank, s, top_level=top_level),
        "Recycled 2 side": lambda A, s: matvec_alg_unified_sketch(A, level, rank, s, top_level=top_level, two_sided_pseudoinverse=True),
        "Fresh": lambda A, s: matvec_alg_resketch(A, level, rank, s, top_level=top_level),
        "Recycled": lambda A, s: matvec_alg_unified_sketch(A, level, rank, s, top_level=top_level),
    }
    title = f"error??"
    plot_sketch_size_vs_error(
        A,
        title,
        methods,
        np.array([30, 300]),
        non_sketching_methods={"Entrywise Access": lambda A: random_access_greedy_alg(A, level, rank, top_level=top_level)},
        repeats=1,
        savedir="out/noisy",
    )


if __name__ == "__main__":
    global_show_title = False
    two_factor(4, 0.1)
    schur_smaller()
    banded_inverse(levels=12, num_diags_above=8, r=8)
    star()

    # # newer experiments
    # noisy()
    # error()

    # two_factor(9, 0.1)
    # two_factor(9, 0.01)
    # two_factor(9, 0.001)  # SPIKE
    # two_factor(9, 1)
    # two_factor(4, .01)
    # schur_gunnar()
    # schur_smaller(2)  # this is weird for m = 2, 3, 4
    # schur_smaller(3)
    # schur_smaller(4)
    # banded_inverse(levels=12, num_diags_above=5)
    # banded_inverse(levels=12, num_diags_above=5, r=5)
    # gaussian_spike()
