import os

import matplotlib.pyplot as plt
import seaborn as sns


colors = sns.color_palette()[:3] + ["black"] * 2
markers = ["o", "x", "s"] + [
    "",
] * 2
linestyles = [(0, d) for d in sns._base.unique_dashes(3)] + ["-.", ":"]

styles = [
    dict(color=color, marker=marker, linestyle=linestyle, lw=1, ms=3)
    for color, marker, linestyle in zip(colors, markers, linestyles)
]


dir_path = "legend_pictures"

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

for ix, style in enumerate(styles):
    fig = plt.figure(figsize=(0.4, 0.08))
    ax = fig.add_axes([0.1, 0, 0.8, 1])
    ax.plot([0.05, 0.2, 0.35], [0.04, 0.04, 0.04], **style, markevery=[1])
    ax.set_xlim(0, 0.4)
    plt.axis("off")
    plt.savefig(f"{dir_path}/legend_{ix}.pdf")
