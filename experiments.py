from itertools import product
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from scipy.sparse.linalg import aslinearoperator as alo

from HBS import matvecs_optimal_core, matvec_alg_resketch, matvec_alg_unified_sketch
from problems import banded_gaussian, SparseInverse


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
        # "regression": lambda s: matvecs_optimal_core(A, recovery_levels, r, s, False),
        # "regression_recover_diagonal": lambda s: matvecs_optimal_core(A, recovery_levels, r, s, True),
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
