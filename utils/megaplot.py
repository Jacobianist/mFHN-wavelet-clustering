# %% Cell
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# %% Cell
plt.rcParams.update(
    {
        "axes.linewidth": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)
# %% Cell
root1d = Path("../data/snapshots_1D_0830/")
root2d = Path("../data/snapshots_2D_0829/")
npy_files_1d = list(root1d.rglob("*.npy"))
npy_files_2d = list(root2d.rglob("*.npy"))
# %% Cell
n_rows, n_cols = 12, 21
n_points = 1024
total_plots = n_rows * n_cols

x = np.arange(n_points)
# %% Cell
df_paths_1d = pd.DataFrame({"filepath": npy_files_1d})
df_paths_1d["filename"] = df_paths_1d["filepath"].apply(lambda p: p.name)
df_paths_1d["a"] = df_paths_1d["filename"].str[:-4].str.split("_").str[1].astype(float)
df_paths_1d["e3"] = df_paths_1d["filename"].str[:-4].str.split("_").str[3].astype(float)
df_paths_1d.head()
# %% Cell
load_npy = lambda path: np.load(path)
# %% Cell

df_paths_1d["data"] = df_paths_1d["filepath"].apply(load_npy)
# %% Cell
df = df_paths_1d.sort_values(by=["a", "e3"]).reset_index(drop=True)
arr_a = np.sort(df["a"].unique())[2:]
arr_e3 = np.sort(df["e3"].unique())
df = df.drop(columns=["filepath", "filename"])
vmax = df["data"].apply(np.max).max()
vmin = df["data"].apply(np.min).min()
# %% Cell
fig = plt.figure(figsize=(36, 16))
gs = fig.add_gridspec(
    n_rows,
    n_cols,
    hspace=0.05,
    wspace=0.1,
    left=0.06,
    right=0.94,
    top=0.96,
    bottom=0.04,
)

for i in range(n_rows):
    for j in range(n_cols):
        ax = fig.add_subplot(gs[i, j])
        ax.plot(
            x,
            df.query(f"a == {arr_a[j]} and e3 == {arr_e3[::-1][i]}")["data"].iloc[0],
            ".-",
            ms=4,
            lw=0.1,
            color="silver",
            mfc="#0C2B4E",
            mec="none",
        )
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        if j == 0:
            ax.set_ylabel(rf"$\varepsilon_3 = {arr_e3[::-1][i]}$")
        if i == n_rows - 1:
            ax.set_xlabel(rf"$a = {arr_a[j]}$")
        # else:
        # ax.set_ylabel("")
        # ax.set_xlabel("")
    # if i<n_rows-1:
    # ax.axes.get_xaxis().set_ticks([])
for ax in fig.get_axes():
    ax.set_ylim(vmin, vmax + 0.1)
    ax.label_outer(remove_inner_ticks=True)
    ax.spines[:].set_visible(False)

    ax.set_xlim(0, 1024)
    ax.set_xticks([512, 1024])
# plt.savefig("mosaic_1D.pdf", bbox_inches="tight", pad_inches=0.05)
plt.savefig("mosaic_1D.png", bbox_inches="tight", pad_inches=0.05)
plt.close()
# plt.show()

# %% Cell

df_paths_2d = pd.DataFrame({"filepath": npy_files_2d})
df_paths_2d["filename"] = df_paths_2d["filepath"].apply(lambda p: p.name)
df_paths_2d["a"] = df_paths_2d["filename"].str[:-4].str.split("_").str[1].astype(float)
df_paths_2d["e3"] = df_paths_2d["filename"].str[:-4].str.split("_").str[3].astype(float)
df_paths_2d.head()
# %% Cell
df_paths_2d["data"] = df_paths_2d["filepath"].apply(load_npy)
# %% Cell
df2 = df_paths_2d.sort_values(by=["a", "e3"]).reset_index(drop=True)
arr_a2 = np.sort(df2["a"].unique())[2:]
arr_e32 = np.sort(df2["e3"].unique())
df2 = df2.drop(columns=["filepath", "filename"])

# %% Cell

fig = plt.figure(figsize=(30, 16))
gs = fig.add_gridspec(
    n_rows,
    n_cols,
    hspace=0.05,
    wspace=0.05,
    left=0.06,
    right=0.94,
    top=0.96,
    bottom=0.04,
)

for i in range(n_rows):
    for j in range(n_cols):
        ax = fig.add_subplot(gs[i, j])
        ax.imshow(
            df2.query(f"a == {arr_a2[j]} and e3 == {arr_e32[::-1][i]}")["data"].iloc[0],
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
            origin="lower",
            rasterized=True,
        )
        if j == 0:
            ax.set_ylabel(rf"$\varepsilon_3 = {arr_e3[::-1][i]}$")
            ax.set_yticks([512, 1024])
        if i == n_rows - 1:
            ax.set_xlabel(rf"$a = {arr_a[j]}$")
            ax.set_xticks([512, 1024])

for ax in fig.get_axes():
    ax.label_outer(True)
    ax.spines[:].set_visible(False)
    # ax.set_xlim(0, 1024)
    # ax.set_xticks([512, 1024])
# plt.savefig("mosaic_2D.pdf", bbox_inches="tight", pad_inches=0.05)
plt.savefig("mosaic_2D.png", bbox_inches="tight", pad_inches=0.05)
plt.close()
# plt.show()
# %% Cell
