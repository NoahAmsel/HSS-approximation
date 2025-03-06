from itertools import product

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse.linalg import aslinearoperator as alo
import seaborn as sns

from HSS_approx import matvec_alg_resketch, matvec_alg_unified_sketch, matvecs_optimal_core, random_access_greedy_alg, matvec_alg_double_unified_sketch
from problems import banded_gaussian, factor2_example, factor2_optimal_solution, grid_schur_complement, SparseInverse, star_matrix


def approx_Frob(A, sketch_size):
    return np.linalg.norm(A @ np.random.randn(A.shape[1], sketch_size), ord='fro') / np.sqrt(sketch_size)


# class QueryTracker:
#     class Counts:

#     def __init__(self, A):
#         self.A = A
#         self.queries = 0
#         self.transpose_queries = 0

#     def _matmat(self, x):
#         self.queries += x.shape[1]
#         return self.A @ x

#     def _rmatmat(self, x):
#         self.transpose_queries += x.shape[0]
#         return x @ self.A

#     def _adjoint(self):


def theorem_4_1_optimality_ratio(sketches_per_level, rank, levels):
    s = sketches_per_level
    k = rank
    gamma_r = (1 + 2*np.e*(s - 2*k)/np.sqrt((s - 3*k)**2 - 1))**2
    gamma_d = 2*k/(s - 2*k - 1)
    return 2 * gamma_r * (1 + gamma_d) * levels


def plot_sketch_size_vs_error(A, title, methods, num_sketches, total_sketches_multiplier, non_sketching_methods={}, repeats=1, approx_frobenius=None, savedir="."):
    Anorm = np.linalg.norm(A) if approx_frobenius is None else approx_Frob(A, approx_frobenius)
    def rel_error(A_tilde):
        error = np.linalg.norm(A_tilde.toarray() - A) if approx_frobenius is None else approx_Frob(alo(A_tilde) - A, approx_frobenius)
        return error / Anorm


    def datum(sketch_dim, method_name, method):
        # TODO: track total queries by wrapping A in something
        return {
            "Queries per Sketch": sketch_dim,
            "Method": method_name,
            "Relative Frobenius Error": rel_error(method(A, sketch_dim)),
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
    df["Total Queries"] = df.apply(lambda x: x["Queries per Sketch"] * total_sketches_multiplier.get(x["Method"], 1), axis=1)
    plt.rcParams.update({"font.size": 14, "font.family": "serif"})
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    plt.yscale("log")
    plt.suptitle(title + "\n")
    for i, (ax, x_col) in enumerate(zip(axs, ["Queries per Sketch", "Total Queries"])):
        # NOTE: This assumes that these non sketching methods are deterministic
        for (method_name, method_result), style in zip(non_sketching_results.items(), ['-.', ':']):
            ax.axhline(method_result, label=method_name, color='black', linestyle=style)
        p = sns.lineplot(df, x=x_col, y="Relative Frobenius Error", hue="Method", style="Method", errorbar=("ci", 95), marker='o', ax=ax, legend=(i == 1))
        ax.set_xscale("log")
        if i == 1:
            # fuller_grid = np.geomspace(df["Queries per Sketch"].min(), df["Queries per Sketch"].max(), num=100)
            # ax.plot(fuller_grid, method_result * theorem_4_1_optimality_ratio(fuller_grid, rank=RANK, levels=LEVELS), color='black', linestyle=style)
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            plt.figlegend(handles, labels, loc='outside upper center', bbox_to_anchor=(0,0,1,.825), ncol=len(methods)+len(non_sketching_methods), labelspacing=0., prop={'size': 10})
    Path(savedir).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.get_figure().savefig(Path(savedir) / f"{title.replace("\n", "")}.pdf", dpi=600)


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
        "Fresh Sketches": lambda A, s: matvec_alg_resketch(A, levels, recovery_rank, s),
        "Half Fresh Sketches": lambda A, s: matvec_alg_resketch(A, levels, recovery_rank, s, second_sketch_for_D=False),
        "Recycled Sketch": lambda A, s: matvec_alg_unified_sketch(A, levels, recovery_rank, s),
    }
    plot_sketch_size_vs_error(
        A,
        title,
        methods,
        np.geomspace(s // 2, s * 2, num=16, endpoint=True, dtype=int),
        {"Half Fresh Sketches": levels, "Fresh Sketches": 2 * levels},
        repeats=1,
        approx_frobenius=int(1e3),
        savedir="out",
    )


def schur_smaller(m):
    n = 51  # the dimension that we split on. Groups are 25 x N, 25 x N, and 1 x N
    levels = 4
    N = m * (2**levels)
    A = grid_schur_complement(n, N)
    r = 10
    s = max(r+m, 3*r)  # sketching dimension. given in beginning of experiments section and beginning of sec. 4

    recovery_rank = r
    title = f"Grid Schur Complement:\nn={n},N={N},m={m},k={recovery_rank}"
    methods = {
        "Fresh Sketches": lambda A, s: matvec_alg_resketch(A, levels, recovery_rank, s),
        "Half Fresh Sketches": lambda A, s: matvec_alg_resketch(A, levels, recovery_rank, s, second_sketch_for_D=False),
        "Recycled Sketch": lambda A, s: matvec_alg_unified_sketch(A, levels, recovery_rank, s),
    }
    A = A @ np.eye(*A.shape)
    non_sketching_methods = {"Random Access": lambda A: random_access_greedy_alg(A, levels, r)}
    plot_sketch_size_vs_error(
        A,
        title,
        methods,
        np.geomspace(s // 2, s * 3 // 2, 30, endpoint=True, dtype=int),
        {"Half Fresh Sketches": levels, "Fresh Sketches": 2 * levels},
        non_sketching_methods=non_sketching_methods,
        repeats=10,
        approx_frobenius=None,
        savedir="out",
    )


def star():
    A = star_matrix()
    title = "Boundary Integral Equation"
    levels = 5
    r = 30
    methods = {
        "Fresh Sketches": lambda A, s: matvec_alg_resketch(A, levels, r, s),
        "Half Fresh Sketches": lambda A, s: matvec_alg_resketch(A, levels, r, s, second_sketch_for_D=False),
        "Recycled Sketch": lambda A, s: matvec_alg_unified_sketch(A, levels, r, s),
    }
    non_sketching_methods = {"Random Access": lambda A: random_access_greedy_alg(A, levels, r)}
    plot_sketch_size_vs_error(
        A,
        title,
        methods,
        np.geomspace(2 * r, 8 * r, 20, endpoint=True, dtype=int),
        {"Half Fresh Sketches": levels, "Fresh Sketches": 2 * levels},
        non_sketching_methods=non_sketching_methods,
        repeats=3,
        savedir="out",
    )


def banded_inverse(levels, num_diags_above, r=None):
    if r is None:
        # this is the true rank of the matrix
        r = 2 * num_diags_above
    # banded inverse
    A = SparseInverse(banded_gaussian(2**levels, num_diags_above))
    title = f"Banded:\nnum_diag={num_diags_above},level={levels},k={r}"

    # to ensure that blocksize >= r, use slightly larger blocksize when necessary
    recovery_levels = int(np.floor(np.log2(A.shape[0] / r)))

    methods = {
        "Fresh Sketches": lambda A, s: matvec_alg_resketch(A, recovery_levels, r, s),
        "Half Fresh Sketches": lambda A, s: matvec_alg_resketch(A, recovery_levels, r, s, second_sketch_for_D=False),
        "Recycled Sketch": lambda A, s: matvec_alg_unified_sketch(A, recovery_levels, r, s),
    }
    non_sketching_methods = {"Random Access": lambda A: random_access_greedy_alg(A.toarray(), levels, r)}
    plot_sketch_size_vs_error(
        A,
        title,
        methods,
        np.geomspace(min_queries(A, recovery_levels), 8 * r, 20, endpoint=True, dtype=int),
        {"Half Fresh Sketches": levels, "Fresh Sketches": 2 * levels},
        non_sketching_methods=non_sketching_methods,
        repeats=10,
        approx_frobenius=int(1e3),
        savedir="out",
    )


def two_factor(logN, eps):
    N = 2 ** logN
    A = factor2_example(N, eps)
    level = logN
    rank = 1
    top_level = 0
    methods = {
        "Fresh Sketches": lambda A, s: matvec_alg_resketch(A, level, rank, s, top_level=top_level),
        "Half Fresh Sketches": lambda A, s: matvec_alg_resketch(A, level, rank, s, top_level=top_level, second_sketch_for_D=False),
        "Recycled Sketch": lambda A, s: matvec_alg_unified_sketch(A, level, rank, s, top_level=top_level),
        "Double Recycled Sketch": lambda A, s: matvec_alg_double_unified_sketch(A, level, rank, s, top_level=top_level),
    }
    non_sketching_methods = {"Greedy": lambda A: random_access_greedy_alg(A, level, rank, top_level=top_level), "Optimal": lambda _: factor2_optimal_solution(N)}
    title = f"Hard Construction\nN={N},eps={eps}"
    plot_sketch_size_vs_error(
        A,
        title,
        methods,
        np.geomspace(min_queries(A, level), N*20, 30, endpoint=True, dtype=int),
        {"Half Fresh Sketches": (level - top_level), "Fresh Sketches": 2 * (level - top_level), "Double Recycled Sketch": 2},
        non_sketching_methods=non_sketching_methods,
        repeats=5,
        savedir="out",
    )


def spike(A, level: int, rank: int, top_level: int):
    # NOTE: Two sided least squares at the final iteration helps the spike
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] == 2 ** (level+1) * rank
    critical_num_sketches = A.shape[1] // 2**(level - top_level)
    methods = {
        "Recycled Sketch": lambda A, s: matvec_alg_unified_sketch(A, level, rank, s, top_level=top_level),
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
        {"Half Fresh Sketches": (level - top_level), "Fresh Sketches": 2 * (level - top_level), "Double Recycled Sketch": 2},
        repeats=5,
        savedir="out",
    )


def gaussian_spike():
    level = 5
    N = 2 ** level
    top_level = level - 3
    rank = 3
    A = np.random.randn(rank * 2*N, rank * 2*N)
    spike(A, level, rank, top_level)


if __name__ == "__main__":
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
    # star()
    gaussian_spike()

# TODO! Redo everything with a leaf size > 1.
# The problem is that currently, our algs assume we're going all the way to the diagonal
# So for the hard case, we want level=1, block size = 2. but it's implicitly assuming level=1 => block size = 4.
