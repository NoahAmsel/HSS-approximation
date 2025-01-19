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

from HSS_approx import matvec_alg_resketch, matvec_alg_unified_sketch, matvecs_optimal_core, random_access_greedy_alg
from problems import banded_gaussian, factor2_example, factor2_optimal_solution, grid_schur_complement, SparseInverse


def approx_Frob(A, sketch_size):
    return np.linalg.norm(A @ np.random.randn(A.shape[1], sketch_size), ord='fro') / np.sqrt(sketch_size)


def plot_sketch_size_vs_error(A, title, methods, num_sketches, total_sketches_multiplier, non_sketching_methods={}, repeats=1, approx_frobenius=None, savedir="."):
    Anorm = np.linalg.norm(A) if approx_frobenius is None else approx_Frob(A, approx_frobenius * 10)
    def rel_error(A_tilde):
        error = np.linalg.norm(A_tilde.toarray() - A) if approx_frobenius is None else approx_Frob(alo(A_tilde) - A, approx_frobenius)
        return error / Anorm


    def datum(sketch_dim, method_name, method):
        return {
            "Queries per Level": sketch_dim,
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
    df["Total Queries"] = df.apply(lambda x: x["Queries per Level"] * total_sketches_multiplier.get(x["Method"], 1), axis=1)
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
    plt.yscale("log")
    plt.suptitle(title)
    for ax, x_col in zip(axs, ["Queries per Level", "Total Queries"]):
        # NOTE: This assumes that these non sketching methods are deterministic
        for (method_name, method_result), style in zip(non_sketching_results.items(), ['-.', ':']):
            ax.axhline(method_result, label=method_name, color='black', linestyle=style)
        sns.lineplot(df, x=x_col, y="Relative Frobenius Error", hue="Method", style="Method", errorbar=("ci", 95), marker='o', ax=ax)
        ax.set_xscale("log")
    Path(savedir).mkdir(parents=True, exist_ok=True)
    fig.get_figure().savefig(Path(savedir) / f"{title.replace("\n", "")}.png", dpi=400)


def schur_gunnar():
    r = 30  # rank. given in fig 7
    m = 2*r  # leaf size. given in beginning of experiments section
    s = max(r+m, 3*r)  # sketching dimension. given in beginning of experiments section and beginning of sec. 4
    levels = 4
    n = 51  # the dimension that we split on. Groups are 25 x N, 25 x N, and 1 x N
    N = (2**levels) * m
    A = grid_schur_complement(n, N)
    recovery_rank = r
    title = f"Grid Schur Complement:\nn={n},N={N},m={m},r={recovery_rank}"
    methods = {
        "Fresh Sketches": lambda A, s: matvec_alg_resketch(A, levels, recovery_rank, s),
        "Recycled Sketch": lambda A, s: matvec_alg_unified_sketch(A, levels, recovery_rank, s),
    }
    plot_sketch_size_vs_error(
        A,
        title,
        methods,
        [(s * 6) // 9, (s * 7) // 9, (s * 8) // 9, s],
        {"Fresh Sketches": levels},
        repeats=10,
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
    title = f"Grid Schur Complement:\nn={n},N={N},m={m},r={recovery_rank}"
    methods = {
        "Fresh Sketches": lambda A, s: matvec_alg_resketch(A, levels, recovery_rank, s),
        "Recycled Sketch": lambda A, s: matvec_alg_unified_sketch(A, levels, recovery_rank, s),
    }
    A = A @ np.eye(*A.shape)
    non_sketching_methods = {"Random Access": lambda A: random_access_greedy_alg(A, levels, r)}
    plot_sketch_size_vs_error(
        A,
        title,
        methods,
        np.linspace(s // 2, s * 3 // 2, 8, endpoint=True, dtype=int),
        {"Fresh Sketches": levels},
        non_sketching_methods=non_sketching_methods,
        repeats=10,
        approx_frobenius=None,
        savedir="out",
    )


def banded_inverse(levels, num_diags_above, r=None):
    if r is None:
        # this is the true rank of the matrix
        r = 2 * num_diags_above
    # banded inverse
    A = SparseInverse(banded_gaussian(2**levels, num_diags_above))
    title = f"Banded:\nnum_diag={num_diags_above},level={levels},r={r}"

    # to ensure that blocksize >= r, use slightly larger blocksize when necessary
    recovery_levels = int(np.floor(np.log2(A.shape[0] / r)))

    methods = {
        "Fresh Sketches": lambda A, s: matvec_alg_resketch(A, recovery_levels, r, s),
        "Recycled Sketch": lambda A, s: matvec_alg_unified_sketch(A, recovery_levels, r, s),
    }
    non_sketching_methods = {"Random Access": lambda A: random_access_greedy_alg(A.toarray(), levels, r)}
    plot_sketch_size_vs_error(A, title, methods, [16, 32, 64, 128, 256], {"Fresh Sketches": levels}, non_sketching_methods=non_sketching_methods, repeats=10, approx_frobenius=int(1e3), savedir="out")


def two_factor(N, eps):
    A = factor2_example(N, eps)
    methods = {
        "Fresh Sketches": lambda A, s: matvec_alg_resketch(A, 1, 1, s),
        "Recycled Sketch": lambda A, s: matvec_alg_unified_sketch(A, 1, 1, s),
    }
    non_sketching_methods = {"Greedy": lambda A: random_access_greedy_alg(A, 1, 1), "Optimal": lambda _: factor2_optimal_solution(N)}
    title = f"Hard Construction\nN={N},eps={eps}"
    plot_sketch_size_vs_error(A, title, methods, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], {"Fresh Sketches": 1}, non_sketching_methods=non_sketching_methods, repeats=10, savedir="out")


if __name__ == "__main__":
    # schur_gunnar()
    # schur_smaller(2)  # this is weird for m = 2, 3, 4
    # schur_smaller(3)
    # schur_smaller(4)
    # banded_inverse(levels=12, num_diags_above=5)
    # banded_inverse(levels=12, num_diags_above=5, r=5)
    two_factor(100, 0.1)

# TODO! Redo everything with a leaf size > 1.
# The problem is that currently, our algs assume we're going all the way to the diagonal
# So for the hard case, we want level=1, block size = 2. but it's implicitly assuming level=1 => block size = 4.