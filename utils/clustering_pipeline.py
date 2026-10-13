# %% Cell
import sqlite3
from pathlib import Path

import matplotlib.gridspec as gridspec

# %% Cell
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.cluster.hierarchy import (
    # set_link_color_palette,
    dendrogram,
    fcluster,
    linkage,
)
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# %% Cell
from sklearn.utils import resample
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm

# %% Cell
from umap import UMAP

# %% Cell
plt.rcParams.update(
    {
        "font.size": 16,
        # 'font.family': 'sans-serif',
        "font.sans-serif": ["DejaVu Sans"],
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "STIXGeneral"
# %% Cell
def load_data_from_testdb(path_to_db, idx=0):
    with sqlite3.connect(path_to_db) as con:
        df_en = pd.read_sql(
            "SELECT simulation_id, snapshot_index, "
            "energy_d1, energy_d2, energy_d3, energy_d4, energy_d5, energy_d6, energy_d7, energy_an, energy_total "
            # "FROM snapshot_energies WHERE snapshot_index BETWEEN 100 AND ?",
            "FROM snapshot_energies WHERE snapshot_index > ?",
            con,
            params=(idx,),
        )
        df_sim = pd.read_sql(
            "SELECT run_id, model_a, model_eps3, "
            # "metric_L, metric_SI, metric_SI3220, metric_SI1620, metric_SI3218, metric_SI1618, metric_SI3222, metric_SI1622 "
            "metric_L, metric_SI3220 "
            "FROM common",
            con,
        )
    energy_cols = [f"energy_d{i}" for i in range(1, 8)] + ["energy_an", "energy_total"]
    en_cols = [f"e_{i}" for i in range(9)]  # e_1..e_8, e_0
    e_0 = df_en["energy_total"].replace(0, np.nan)
    normalized = df_en[energy_cols[:-1]].div(e_0, axis=0).fillna(0)
    normalized.columns = en_cols[1:]  # e_1..e_8
    filtered_df = pd.concat([df_en[["simulation_id"]], normalized, pd.Series(e_0, name="e_0")], axis=1).dropna()
    grouped = filtered_df.groupby("simulation_id", as_index=False).mean(numeric_only=True)
    df = pd.merge(grouped, df_sim, left_on="simulation_id", right_on="run_id", how="left")
    final_renames = {
        "energy_total": "e_0",
        "model_a": "a",
        "model_eps3": "eps3",
    }
    # rename energy_d1..d7, energy_an -> e_1..e_8
    for i in range(1, 8):
        final_renames[f"energy_d{i}"] = f"e_{i}"
    final_renames["energy_an"] = "e_8"
    df = df.rename(columns=final_renames)
    df = (
        df.drop(columns=["run_id", "simulation_id"], errors="ignore")
        .query("a >= 1.5")
        .round({"eps3": 3, "a": 3})
        .sort_values(["a", "eps3"])
        .reset_index(drop=True)
    )
    # raw dataframe
    raw_df = df_en[df_en["snapshot_index"] < idx].copy()
    return df, raw_df
# %% Cell
root_mfhn = Path("../")
root1d = Path(root_mfhn / "./data/snapshots_1D_0830")
root2d = Path(root_mfhn / "./data/snapshots_2D_0829")
df1d, _ = load_data_from_testdb(f"{root_mfhn}/utils/db-0830-1d.db", idx=750)
df1d.Name = "1D"
df2d, _ = load_data_from_testdb(f"{root_mfhn}/utils/db-0829-2d.db", idx=750)
df2d.Name = "2D"
df2d.head().round(3)
# %% Cell
en_cols = ["e_0", "e_1", "e_2", "e_3", "e_4", "e_5", "e_6", "e_7", "e_8"]
E_cols = [r"$E_1$", r"$E_2$", r"$E_3$", r"$E_4$", r"$E_5$", r"$E_6$", r"$E_7$", r"$E_8$"]
p_cols = [r"$p_1$", r"$p_2$", r"$p_3$", r"$p_4$", r"$p_5$", r"$p_6$", r"$p_7$", r"$p_8$"]
# %% Cell
def check_component_to_delete(df):
    P = df[en_cols[1:]].values
    R8 = np.corrcoef(P, rowvar=False)
    print("det R8 =", np.linalg.det(R8))  # ~1e-16
    for drop in range(8):
        keep = [i for i in range(8) if i != drop]
        R = np.corrcoef(P[:, keep], rowvar=False)
        kappa = np.linalg.cond(R)
        vif_max = np.diag(np.linalg.inv(R)).max()
        print(f"drop p_{drop + 1}: kappa={kappa:8.1f}, max VIF={vif_max:8.2f}, R={np.linalg.det(R):.1e}")

check_component_to_delete(df2d)

# %% Cell
random_state = 42
cols = [
    # "e_0",
    "e_1",
    "e_2",
    "e_3",
    "e_4",
    "e_5",
    "e_6",
    "e_7",
    # "e_8",
]
X1d_scaled = StandardScaler().fit_transform(df1d[cols].values)
X2d_scaled = StandardScaler().fit_transform(df2d[cols].values)

# %% Cell
def plot_corr_matrix():
    fig = plt.figure(figsize=(9, 3))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1, 1], hspace=0.5, wspace=0.3, left=0.04, right=0.95, top=0.96, bottom=0.04
    )  # , height_ratios=[1, 2]
    ax1 = fig.add_subplot(gs[0])
    sns.heatmap(
        df1d[cols].corr(),
        annot=True,
        # annot_kws={"fontsize": 12,},
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        linecolor="k",
        linewidths=0.01,
        square=True,
        ax=ax1,
        cbar_kws={"ticks": [-1, 0, 1]},
    )
    ax1.set_xticklabels(p_cols[:-1])
    ax1.set_yticklabels(p_cols[:-1])
    ax1.set_title(f"Det = {np.linalg.det(df1d[cols].corr()):.2e} ({df1d.Name} Data)")
    ax2 = fig.add_subplot(gs[1])
    sns.heatmap(
        df2d[cols].corr(),
        annot=True,
        # annot_kws={"fontsize": 12},
        fmt=".2f",
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        linecolor="k",
        linewidths=0.01,
        square=True,
        ax=ax2,
        cbar_kws={"ticks": [-1, 0, 1]},
    )
    ax2.set_title(f"Det = {np.linalg.det(df2d[cols].corr()):.2e} ({df2d.Name} Data)")
    ax2.set_xticklabels(p_cols[:-1])
    ax2.set_yticklabels(p_cols[:-1])
    return plt.gca()

plot_corr_matrix()
# %% Cell
print(f"Condition Number: 1D={np.linalg.cond(X1d_scaled):.2f}, 2D={np.linalg.cond(X2d_scaled):.2f}")
vif_data = []
for i in range(len(cols)):
    vif_data.append(
        {
            "feature": cols[i],
            "VIF1D": variance_inflation_factor(X1d_scaled, i),
            "VIF2D": variance_inflation_factor(X2d_scaled, i),
        }
    )
vif_df = pd.DataFrame(vif_data).sort_values("VIF2D", ascending=False)
print("VIF (Variance Inflation Factor):")
print(vif_df.to_string(index=False))
max_vif = vif_df["VIF2D"].max()
print(f"max VIF = {max_vif:.1f}")

# %% Cell

# %% Cell
pca_2d_1 = PCA(n_components=2, random_state=random_state)
X1d_pca = pca_2d_1.fit_transform(X1d_scaled)
explained_var1 = pca_2d_1.explained_variance_ratio_

pca_2d_2 = PCA(n_components=2, random_state=random_state)
X2d_pca = pca_2d_2.fit_transform(X2d_scaled)
explained_var2 = pca_2d_2.explained_variance_ratio_

X1d_umap = UMAP(n_components=2, random_state=random_state).fit_transform(X1d_scaled)
X2d_umap = UMAP(n_components=2, random_state=random_state).fit_transform(X2d_scaled)
# %% Cell
lnkg = "ward"
Z1 = linkage(X1d_scaled, method=lnkg, metric="euclidean")
Z2 = linkage(X2d_scaled, method=lnkg, metric="euclidean")
treshold = 10
labels1 = fcluster(Z1, t=treshold, criterion="distance")
labels2 = fcluster(Z2, t=treshold, criterion="distance")
# %% Cell
def consensus_k(X, Z):
    results = []
    for k in range(2, 13):
        # kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = fcluster(Z, k, criterion="maxclust")
        silhouette = silhouette_score(X, labels)
        davies = davies_bouldin_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        unique_labels = np.unique(labels)
        n_clusters_actual = len(unique_labels)

        intra_vars = []
        for cluster_id in unique_labels:
            mask = labels == cluster_id
            n_points = mask.sum()
            if n_points > 1:
                points = X[mask]
                centroid = points.mean(axis=0)
                intra_vars.append(2 * np.mean((points - centroid) ** 2))
        mean_intra_var = np.mean(intra_vars) if intra_vars else np.nan
        results.append(
            {
                "k": k,
                "silhouette": silhouette,
                "davies": davies,
                "calinski": calinski,
                "intra_cluster_variance": mean_intra_var,
                "n_clusters_actual": n_clusters_actual,
            }
        )
    return pd.DataFrame(results)
# %% Cell
indexes_X1 = consensus_k(X1d_scaled, Z1)
indexes_X2 = consensus_k(X2d_scaled, Z2)
# %% Cell
def plot_consensus(Z1, Z2, indexes_X1, indexes_X2, treshold, save_on=False, filename="fig3"):
    fig = plt.figure(figsize=(14, 5), constrained_layout=True)
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[1, 1.55],
        hspace=0.05,
        wspace=0.05,
        left=0.04,
        right=0.95,
        top=0.96,
        bottom=0.04,
    )
    ax1, ax2, ax3, ax4 = [fig.add_subplot(gs[0, i]) for i in range(4)]
    d_ax1 = fig.add_subplot(gs[1, 0:2])
    d_ax2 = fig.add_subplot(gs[1, 2:])

    dendro_kwargs = {
        "no_labels": True,
        "color_threshold": 0,
        "above_threshold_color": "royalblue",
    }
    dendrogram(Z1, ax=d_ax1, **dendro_kwargs)
    dendrogram(Z2, ax=d_ax2, **dendro_kwargs)

    d_ax1.set_title("Dendrogram (1D data)")
    d_ax2.set_title("Dendrogram (2D data)")

    n_merges = len(Z1)
    for ax, Z, title in ((ax1, Z1, "1D"), (ax2, Z2, "2D")):
        ax.plot(range(1, n_merges + 1), Z[:, 2][:], ".:", ms=6, c="royalblue", alpha=0.8)
        ax.set(
            title=f"Scree plot ({title} data)",
            ylabel="Ward distance",
            # xlim=(-1, 40),
            xlim=(200, 253),
            yticks=np.arange(0, 35, 10),
            xlabel="Merge step",
        )
        ax.tick_params(axis="both", which="major")

    metrics = [
        (ax3, "silhouette", "Silhouette Score", None),
        (ax4, "davies", "Davies-Bouldin Index", ["1D data", "2D data"]),
    ]

    for ax, col, title, _ in metrics:
        ax.plot(indexes_X1["k"], indexes_X1[col], "o:", ms=6, color="orangered", label="1D data")
        ax.plot(indexes_X2["k"], indexes_X2[col], "s:", ms=6, color="royalblue", label="2D data")
        ax.legend(loc="best", fontsize=10)
        ax.set(title=title, xticks=np.arange(2, 13, 1), xlabel="k")
        ax.tick_params(axis="both", which="major")

    ax3.axvspan(6, 9, color="lightgray", alpha=0.3, ec=None)
    ax4.axvspan(6, 8, color="lightgray", alpha=0.3, ec=None)
    ax3.set_ylim(0.35, 0.65)
    ax4.set_ylim(0.45, None)
    for ax in (d_ax1, d_ax2, ax1, ax2):
        ax.axhline(treshold, c="r", ls="--", alpha=0.7)
        if ax == d_ax2:
            d_ax2.text(
                0.95,
                treshold,
                f"cut={treshold:.1f}",
                transform=d_ax2.get_yaxis_transform(),
                ha="right",
                va="bottom",
                color="r",
                fontsize=14,
            )
        else:
            ax.text(
                0.65,
                treshold,
                f"cut={treshold:.1f}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                color="r",
                fontsize=14,
            )
        ax.set_ylabel("Ward distance")

    if save_on:
        plt.savefig(f"{root_mfhn}/images/{filename}.pdf", format="pdf", pad_inches=0.05, bbox_inches="tight")
        plt.savefig(f"{root_mfhn}/images/{filename}.png", format="png", pad_inches=0.05, bbox_inches="tight")
    return plt.gca()

plot_consensus(Z1, Z2, indexes_X1, indexes_X2, treshold)
# %% Cell
df1d["cluster"] = labels1
df2d["cluster"] = labels2

#
map_1d_cluster = {5: 1, 4: 2, 3: 3, 1: 4, 2: 5, 7: 6, 6: 7}
map_2d_cluster = {7: 1, 4: 2, 3: 3, 2: 4, 1: 5, 5: 6, 6: 7}

df1d["cluster"] = np.array([map_1d_cluster[x] for x in labels1])
df2d["cluster"] = np.array([map_2d_cluster[x] for x in labels2])

unique_labels = sorted(set(labels1))
n_clusters = len(unique_labels)
# %% Cell
dots = np.array(
    [
        [3.0, 5.12],
        [3.5, 1.28],
        [6.0, 2.56],
        [6.0, 0.16],
        [5.0, 0.02],
        [2.25, 0.02],
        [4.00, 0.04],
    ]
)
y_positions = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48])
x_positions = np.array([1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5])

# %% Cell
def get_edges(vals):
    """edges helper"""
    mid = (vals[:-1] + vals[1:]) / 2
    return np.r_[2 * vals[0] - mid[0], mid, 2 * vals[-1] - mid[-1]]

def get_log_edges(vals):
    """log-scale edges"""
    log_v = np.log10(vals)
    d = np.diff(log_v) / 2
    return 10 ** np.r_[log_v[0] - d[0], log_v[:-1] + d, log_v[-1] + d[-1]]

def pco(ax, df, val="cluster", cmap="Set2", norm=None):
    grid = df.pivot_table(index="eps3", columns="a", values=val)
    a, eps3 = grid.columns.values, grid.index.values
    im = ax.pcolormesh(get_edges(a), get_log_edges(eps3), grid.values, cmap=cmap, norm=norm, shading="auto", edgecolors="black", lw=0.01)
    ax.set_yscale("log", base=2)
    ax.set(
        yticks=eps3,
        yticklabels=np.round(eps3, 2),
        xticks=a[::2],
        xlabel=r"$a$",
        ylabel=r"$\varepsilon_3$",
        ylim=(eps3.min() / 1.2, eps3.max() * 1.2),
    )
    cbar = plt.colorbar(im, ax=ax, label="Cluster", ticks=range(1, n_clusters + 1))
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, which="major", labelsize=20)
    ax.scatter(dots[:, 0], dots[:, 1], edgecolors="darkblue", facecolors="none", s=80, linewidths=2, alpha=0.9)

    return ax

def plot_cluster_dashboard(cmap, cmap2, cmapc, vmin=0, save_on=False, filename="cluster_dashboard"):
    metrics = {
        "L_1D": df1d.pivot_table("metric_L", "eps3", "a").sort_index(axis=0).sort_index(axis=1),
        "SI_1D": df1d.pivot_table("metric_SI3220", "eps3", "a").sort_index(axis=0).sort_index(axis=1),
        "L_2D": df2d.pivot_table("metric_L", "eps3", "a").sort_index(axis=0).sort_index(axis=1),
        "SI_2D": df2d.pivot_table("metric_SI3220", "eps3", "a").sort_index(axis=0).sort_index(axis=1),
    }

    fig = plt.figure(figsize=(15, 6))
    gs0 = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=[1, 1.9], hspace=0.3, left=0.04, right=0.95, top=0.96, bottom=0.04
    )

    gs00 = gridspec.GridSpecFromSubplotSpec(
        1,
        4,
        subplot_spec=gs0[0],
        hspace=0.15,
        wspace=0.15,
    )
    ax1, ax2, ax3, ax4 = [fig.add_subplot(gs00[i]) for i in range(4)]
    ax1.set_title("Metric L (1D)")
    ax2.set_title("Metric L (2D)")
    ax3.set_title("Metric SI (1D)")
    ax4.set_title("Metric SI (2D)")

    gs01 = gs0[1].subgridspec(1, 2, wspace=0.15)
    ax5 = fig.add_subplot(gs01[0])
    ax6 = fig.add_subplot(gs01[1])

    im1 = ax1.pcolormesh(
        metrics["L_1D"].columns,
        metrics["L_1D"].index,
        metrics["L_1D"].values,
        cmap=cmap,
        vmax=1,
        vmin=vmin,
        shading="auto",
        rasterized=True,
        edgecolor=None,
    )
    ax1.set_yscale("log", base=2)
    cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02)

    im2 = ax2.pcolormesh(
        metrics["L_2D"].columns,
        metrics["L_2D"].index,
        metrics["L_2D"].values,
        cmap=cmap,
        vmax=1,
        vmin=vmin,
        shading="auto",
        rasterized=True,
        edgecolor=None,
    )
    ax2.set_yscale("log", base=2)
    cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02)

    im3 = ax3.pcolormesh(
        metrics["SI_1D"].columns,
        metrics["SI_1D"].index,
        metrics["SI_1D"].values,
        cmap=cmap2,
        vmax=1,
        vmin=vmin,
        shading="auto",
        rasterized=True,
        edgecolor=None,
    )
    ax3.set_yscale("log", base=2)
    cbar3 = plt.colorbar(im1, ax=ax3, pad=0.02)

    im4 = ax4.pcolormesh(
        metrics["SI_2D"].columns,
        metrics["SI_2D"].index,
        metrics["SI_2D"].values,
        cmap=cmap2,
        vmax=1,
        vmin=vmin,
        shading="auto",
        rasterized=True,
        edgecolor=None,
    )
    ax4.set_yscale("log", base=2)
    cbar4 = plt.colorbar(im4, ax=ax4, pad=0.02)

    for cax in [cbar1, cbar2, cbar3, cbar4]:
        cax.ax.tick_params(labelsize=8)
        cax.outline.set_visible(False)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.scatter(dots[:, 0], dots[:, 1], edgecolors="w", facecolors="none", s=40, linewidths=2.2)
        ax.set(
            yticks=y_positions[::2],
            yticklabels=np.round(metrics["L_1D"].index, 2)[::2],
            xticks=x_positions[::4],
            xticklabels=np.round(metrics["L_1D"].columns, 2)[::4],
        )
        ax.tick_params(labelsize=10)
    ax1.set(ylabel=r"$\varepsilon_3$")

    palette = sns.color_palette(cmapc, n_clusters)
    colors_list = palette.as_hex()

    bounds = np.arange(0.5, len(colors_list) + 1.5)
    norm = BoundaryNorm(bounds, ListedColormap(colors_list).N)

    pco(ax5, df1d, "cluster", cmap=cmapc, norm=norm)
    ax5.set_title("Clustered 1D data")

    pco(ax6, df2d, "cluster", cmap=cmapc, norm=norm)
    ax6.set_title("Clustered 2D data")

    if save_on:
        plt.savefig(f"{root_mfhn}/images/{filename}.pdf", format="pdf", pad_inches=0.05, bbox_inches="tight")
        plt.savefig(f"{root_mfhn}/images/{filename}.png", format="png", pad_inches=0.05, bbox_inches="tight")
    return plt.gca()

plot_cluster_dashboard(cmap="brg", cmap2="brg_r", cmapc="Set3", vmin=0.0)
# %% Cell
def plot_projections(cmap, save_on=False, filename="projections"):
    colors_list = sns.color_palette(cmap, n_clusters).as_hex()

    fig = plt.figure(figsize=(15, 2.25))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1], hspace=0.4, wspace=0.3, left=0.04, right=0.95, top=0.96, bottom=0.04)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    ax0.set_xlabel(rf"PC1 $({explained_var1[0] * 100:.1f}\%)$" if explained_var1.any() else "Component 1")
    ax0.set_ylabel(rf"PC2 $({explained_var1[1] * 100:.1f}\%)$" if explained_var1.any() else "Component 2")
    ax0.set_title(r"PCA (1D data)")
    ax1.set_xlabel(rf"PC1 $({explained_var2[0] * 100:.1f}\%)$" if explained_var2.any() else "Component 1")
    ax1.set_ylabel(rf"PC2 $({explained_var2[1] * 100:.1f}\%)$" if explained_var2.any() else "Component 2")
    ax1.set_title(r"PCA (2D data)")

    ax2 = fig.add_subplot(gs[2])
    ax2.set_xlabel("UMAP1")
    ax2.set_ylabel("UMAP2")
    ax2.set_title(r"UMAP (1D data)")

    ax3 = fig.add_subplot(gs[3])
    ax3.set_xlabel("UMAP1")
    ax3.set_ylabel("UMAP2")
    ax3.set_title(r"UMAP (2D data)")
    for cluster_id in unique_labels:
        mask = df1d["cluster"] == cluster_id
        mask2 = df2d["cluster"] == cluster_id
        color_index = cluster_id - 1
        ax0.scatter(
            X1d_pca[mask, 0], X1d_pca[mask, 1], c=colors_list[color_index], alpha=0.8, s=100, edgecolors="k", linewidths=0.5
        )
        ax1.scatter(
            X2d_pca[mask2, 0],
            X2d_pca[mask2, 1],
            c=colors_list[color_index],
            alpha=0.8,
            s=120,
            edgecolors="k",
            linewidths=0.5,
        )

        ax2.scatter(
            X1d_umap[mask, 0],
            X1d_umap[mask, 1],
            c=colors_list[color_index],
            alpha=0.8,
            s=140,
            edgecolors="k",
            linewidths=0.5,
        )
        ax3.scatter(
            X2d_umap[mask2, 0],
            X2d_umap[mask2, 1],
            c=colors_list[color_index],
            alpha=0.8,
            s=160,
            edgecolors="k",
            linewidths=0.5,
        )
    if save_on:
        plt.savefig(f"{root_mfhn}/images/{filename}.pdf", format="pdf", pad_inches=0.05, bbox_inches="tight")
        plt.savefig(f"{root_mfhn}/images/{filename}.png", format="png", pad_inches=0.05, bbox_inches="tight")
    return plt.gca()

plot_projections(cmap="Set3")
# %% Cell
def plot_mean_energy_ari(cmap, save_on=False, filename="mean_energy_ari"):
    mean_data1 = df1d.groupby("cluster")[en_cols[1:]].mean()
    mean_data2 = df2d.groupby("cluster")[en_cols[1:]].mean()
    k_values = range(2, 13)
    ari_values = []
    for k in k_values:
        labels_1d = fcluster(Z1, t=k, criterion="maxclust")
        labels_2d = fcluster(Z2, t=k, criterion="maxclust")
        ari = adjusted_rand_score(labels_1d, labels_2d)
        ari_values.append(ari)

    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1, 1],
        height_ratios=[1, 0.4],
        hspace=0.2,
        wspace=0.25,
        left=0.04,
        right=0.95,
        top=0.96,
        bottom=0.04,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    hm1 = sns.heatmap(
        mean_data1,
        annot=True,
        annot_kws={"fontsize": 13},
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        vmin=0.0,
        vmax=1.0,
        square=True,
        ax=ax1,
        cbar_kws={"ticks": [0, 0.5, 1], "shrink": 0.7},
    )
    ax1.set_xlabel("Energy level")
    ax1.set_ylabel("Cluster number")
    ax1.set_title(r"Energy profile (1D data)")

    ax2 = fig.add_subplot(gs[0, 1])
    hm2 = sns.heatmap(
        mean_data2,
        annot=True,
        annot_kws={"fontsize": 13},
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        vmin=0.0,
        vmax=1.0,
        square=True,
        ax=ax2,
        cbar_kws={"ticks": [0, 0.5, 1], "shrink": 0.7},
    )
    ax2.set_xlabel("Energy level")
    ax2.set_ylabel("Cluster number")
    ax2.set_title(r"Energy profile (2D data)")
    ax1.set_xticklabels(p_cols[: len(en_cols[1:])])
    hm1.set_yticklabels(hm1.get_yticklabels(), rotation=0)
    ax2.set_xticklabels(p_cols[: len(en_cols[1:])])
    hm2.set_yticklabels(hm2.get_yticklabels(), rotation=0)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(
        k_values,
        ari_values,
        "o:",
        ms=10,
        color="royalblue",
    )
    ax3.scatter(k_values[5], ari_values[5], c="orangered", s=100, zorder=10, label=f"ARI = {ari_values[5]:.3f}")
    ax3.legend(loc="best")
    ax3.grid(alpha=0.2)
    ax3.set(
        xlabel="k",
        ylabel="ARI(1D, 2D)",
        title="Dimensional invariance",
        ylim=(0.47, 0.85),
        xticks=np.arange(2, 13, 1),
        yticks=[0.5, 0.6, 0.7, 0.8],
    )
    pos = ax3.get_position()
    new_pos = [
        pos.x0 + pos.width * 0.15,  # left
        pos.y0,  # bottom
        pos.width * 0.7,  # width
        pos.height,  # height
    ]
    ax3.set_position(new_pos)

    if save_on:
        plt.savefig(f"{root_mfhn}/images/{filename}.pdf", format="pdf", pad_inches=0.05, bbox_inches="tight")
        plt.savefig(f"{root_mfhn}/images/{filename}.png", format="png", pad_inches=0.05, bbox_inches="tight")
    return plt.gca()

plot_mean_energy_ari(cmap="afmhot_r")
# %% Cell
def preload_data():
    last_1d = []
    last_2d = []
    for key in range(7):
        a, eps3 = dots[key]
        pattern = f"a_{a}*eps3_{eps3}*.npy"
        matches1 = list(root1d.glob(pattern))
        matches2 = list(root2d.glob(pattern))
        if matches1:
            last_1d.append(np.load(matches1[0]))
        if matches2:
            last_2d.append(np.load(matches2[0]))
    return last_1d, last_2d

last_1d, last_2d = preload_data()
# %% Cell
def plot_all_clusters(barpalette, cmap, save_on=False, filename="all_clusters"):
    fig = plt.figure(figsize=(15, 14))
    outer = fig.add_gridspec(
        7,
        3,
        width_ratios=[1.3, 0.24, 1],
        wspace=0.2,
        hspace=0.25,
        left=0.08,
        right=0.90,
        top=0.96,
        bottom=0.04,
    )
    cluster_name = [
        "Shadowed wave I",
        "Shadowed wave II",
        "Oscillation death",
        "Oscillation death",
        "Oscillation death",
        "Chimera state",
        "Coherent oscillations",
    ]
    x = np.arange(0, 1024)
    vmin_global = min(arr.min() for arr in last_2d)
    vmax_global = max(arr.max() for arr in last_2d)
    for r in range(7):
        a, eps3 = dots[r]

        gs_left = outer[r, 0].subgridspec(1, 2, wspace=0.2, width_ratios=[1.65, 1])
        ax0 = fig.add_subplot(gs_left[0, 0])  # scatterline
        ax1 = fig.add_subplot(gs_left[0, 1])  # bar 1D
        gs_right = outer[r, 2].subgridspec(1, 2, wspace=0.08, width_ratios=[1, 1.25])
        ax2 = fig.add_subplot(gs_right[0, 0])  # bar 2D
        ax3 = fig.add_subplot(gs_right[0, 1])  # heatmap
        ax_row = [ax0, ax1, ax2, ax3]

        # --- Scatter plot ---
        ax0.plot(x, last_1d[r], ".-", ms=4, lw=0.1, color="silver", mfc="#0C2B4E", mec="none")
        ax0.set_xlim(0, 1024)
        ax0.set_xticks([0, 512, 1024])

        # --- Grouped Bar Chart (Bar 1)
        data_b1 = df1d[(df1d["a"] == a) & (df1d["eps3"] == eps3)][en_cols[1:]]
        sns.barplot(data=data_b1, legend=False, ax=ax1, palette=barpalette, edgecolor="black", linewidth=0.5)

        # --- Grouped Bar Chart (Bar 2)
        data_b2 = df2d[(df2d["a"] == a) & (df2d["eps3"] == eps3)][en_cols[1:]]
        sns.barplot(data=data_b2, legend=False, ax=ax2, palette=barpalette, saturation=0.75, edgecolor="black", linewidth=0.5)

        # --- Heatmap ---
        im = ax3.imshow(
            last_2d[r],
            # vmin=vmin_global,
            # vmax=vmax_global,
            cmap=cmap,
            aspect="equal",
            origin="lower",
        )
        ax3.set_xticks([0, 1024])
        ax3.set_yticks([0, 1024])

        ax_ann = fig.add_subplot(outer[r, 1])
        ax_ann.text(
            0.4,
            0.6,
            f"Cluster {r + 1}\n``{cluster_name[r]}''\n\n$a$={a:.2f}, $\\varepsilon_3$={eps3:.2f}",
            transform=ax_ann.transAxes,
            ha="center",
            va="center",
            # fontsize=20,
            # fontweight='heavy',
        )
        ax_ann.axis("off")

        for ax in ax_row:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xticklabels([])
            ax.tick_params(axis="y", labelsize=12)
        y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1]) * 1.08
        ax1.set_ylim(0, y_max)
        ax2.set_ylim(0, y_max)

        if r == 6:
            ax1.set_xticks(range(len(p_cols)))
            ax2.set_xticks(range(len(p_cols)))
            ax1.set_xticklabels(p_cols, ha="center")
            ax2.set_xticklabels(p_cols, ha="center")
            ax0.set_xticklabels([0, 512, 1024])
            ax3.set_xticklabels([0, 1024], size=12)
            ax0.set_xlabel("Index")
            ax3.set_xlabel("Index")
            ax1.set_xlabel("Energy level")
            ax2.set_xlabel("Energy level")
    if save_on:
        plt.savefig(f"{root_mfhn}/images/{filename}.pdf", format="pdf", pad_inches=0.05, bbox_inches="tight")
        plt.savefig(f"{root_mfhn}/images/{filename}.png", format="png", pad_inches=0.05, bbox_inches="tight")
    return plt.gca()

plot_all_clusters(barpalette="Set2", cmap="seismic")

# %% Cell

plot_cluster_dashboard(cmap='brg', cmap2='brg_r', cmapc = "Set3", vmin=0.0, save_on=True, filename="Figure2-clustermap")
plot_projections(cmap='Set3', save_on=True, filename="Figure3-projection")
plot_consensus(Z1, Z2, indexes_X1, indexes_X2, treshold, save_on=True, filename="Figure4-consensus")
plot_mean_energy_ari(cmap='afmhot_r', save_on=True, filename="Figure5-energyprofile")
plot_all_clusters(barpalette="Set2", cmap='seismic', save_on=True, filename="Figure6-interpretation")



# %% Cell
def bootstrap_stability(X, linked, k, n_iterations=50, sample_fraction=0.8):
    """
    Bootstrap: subsample 80%, clustering repeat, ARI-to-total.
    """
    np.random.seed(42)
    aris = []
    for i in range(n_iterations):
        idx = resample(range(len(X)), n_samples=int(sample_fraction * len(X)))
        X_boot = X[idx]

        linked_boot = linkage(X_boot, method="ward", metric="euclidean")
        labels_boot = fcluster(linked_boot, t=k, criterion="maxclust")

        labels_full_subset = fcluster(linked, t=k, criterion="maxclust")[idx]

        ari = adjusted_rand_score(labels_full_subset, labels_boot)
        aris.append(ari)
    return np.mean(aris), np.std(aris)

mean_ari_1d, std_ari_1d = bootstrap_stability(X1d_scaled, Z1, k=7)
mean_ari_2d, std_ari_2d = bootstrap_stability(X2d_scaled, Z2, k=7)

print(f"Simple bootstrap stability (1D): ARI = {mean_ari_1d:.3f} ± {std_ari_1d:.3f}")
print(f"Simple bootstrap stability (2D): ARI = {mean_ari_2d:.3f} ± {std_ari_2d:.3f}")
# %% Cell
def compare_methods(X, k):
    """
    Compare Ward, Complete, Spectral Embedding + Ward.
    """
    results = {}

    # 1. Ward raw
    linked_ward = linkage(X, method="ward", metric="euclidean")
    labels_ward = fcluster(linked_ward, t=k, criterion="maxclust")
    sil_ward = silhouette_score(X, labels_ward)
    db_ward = davies_bouldin_score(X, labels_ward)
    results["Ward (raw)"] = {"labels": labels_ward, "silhouette": sil_ward, "davies": db_ward}

    # 2. Complete raw
    linked_comp = linkage(X, method="complete", metric="euclidean")
    labels_comp = fcluster(linked_comp, t=k, criterion="maxclust")
    sil_comp = silhouette_score(X, labels_comp)
    db_comp = davies_bouldin_score(X, labels_comp)
    results["Complete (raw)"] = {"labels": labels_comp, "silhouette": sil_comp, "davies": db_comp}

    # 3. Spectral Embedding + Ward
    se = SpectralEmbedding(n_components=3, affinity="nearest_neighbors", n_neighbors=15)
    X_emb = se.fit_transform(X)
    linked_se = linkage(X_emb, method="ward", metric="euclidean")
    labels_se = fcluster(linked_se, t=k, criterion="maxclust")

    sil_se = silhouette_score(X, labels_se)
    db_se = davies_bouldin_score(X, labels_se)
    results["Spectral + Ward"] = {"labels": labels_se, "silhouette": sil_se, "davies": db_se}

    # 4. Average raw
    linked_av = linkage(X, method="average", metric="euclidean")
    labels_av = fcluster(linked_av, t=k, criterion="maxclust")
    sil_av = silhouette_score(X, labels_av)
    db_av = davies_bouldin_score(X, labels_av)
    results["Average (raw)"] = {"labels": labels_av, "silhouette": sil_av, "davies": db_av}

    # # 5. Single raw
    linked_sin = linkage(X, method="single", metric="euclidean")
    labels_sin = fcluster(linked_sin, t=k, criterion="maxclust")
    sil_sin = silhouette_score(X, labels_sin)
    db_sin = davies_bouldin_score(X, labels_sin)
    results["Single (raw)"] = {"labels": labels_sin, "silhouette": sil_sin, "davies": db_sin}

    return results

results = compare_methods(X1d_scaled, k=7)
for method, data in results.items():
    print(f"1D:: {method}: silhouette = {data['silhouette']:.3f}, davies_bouldin = {data['davies']:.3f}")

results = compare_methods(X2d_scaled, k=7)
for method, data in results.items():
    print(f"2D:: {method}: silhouette = {data['silhouette']:.3f}, davies_bouldin = {data['davies']:.3f}")
# %% Cell
def centroid_bootstrap_stability(df, cols, n_clusters=7, n_bootstrap=1000, random_seed=42):
    rng = np.random.default_rng(random_seed)

    # 1. original clustering
    scaler = StandardScaler()
    X_full = scaler.fit_transform(df[cols].values)
    Z_full = linkage(X_full, method="ward")
    labels_orig = fcluster(Z_full, t=n_clusters, criterion="maxclust")

    ari_scores = []
    cluster_jaccards = {i: [] for i in np.unique(labels_orig)}

    # 2. bootstrap cycle
    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping process: "):
        # 3. index generator and inner clustering
        indices = rng.choice(len(df), size=len(df), replace=True)
        X_boot = X_full[indices]

        Z_boot = linkage(X_boot, method="ward")
        labels_boot = fcluster(Z_boot, t=n_clusters, criterion="maxclust")

        # 4. looking for centroids
        unique_boot_labels = np.unique(labels_boot)
        centroids = []
        for label in unique_boot_labels:
            mask = labels_boot == label
            centroids.append(np.mean(X_boot[mask], axis=0))
        centroids = np.array(centroids)

        # 5. adjusting indexes to centroids
        dists = cdist(X_full, centroids)
        labels_boot_pred = np.argmin(dists, axis=1) + 1  # index starts with 0, fcluster --- with 1

        # 6. now can compare
        ari_scores.append(adjusted_rand_score(labels_orig, labels_boot_pred))

        # 7. per-cluster Jaccard
        for cluster_id in np.unique(labels_orig):
            orig_mask = labels_orig == cluster_id
            best_jaccard = 0
            for boot_id in np.unique(labels_boot_pred):
                boot_mask = labels_boot_pred == boot_id
                intersection = np.sum(orig_mask & boot_mask)
                union = np.sum(orig_mask | boot_mask)
                if union > 0:
                    jaccard = intersection / union
                    if jaccard > best_jaccard:
                        best_jaccard = jaccard
            cluster_jaccards[cluster_id].append(best_jaccard)

    return {
        "Mean ARI": np.mean(ari_scores),
        "Std ARI": np.std(ari_scores),
        "Cluster Jaccards": {k: {"Mean": np.mean(v), "Min": np.min(v)} for k, v in cluster_jaccards.items()},
    }

# %% Cell
results_1d = centroid_bootstrap_stability(df1d, cols, n_clusters=7, n_bootstrap=1000)
print(f"\nCentroid bootstrap stability (1D): ARI = {results_1d["Mean ARI"]:.3f} ± {results_1d["Std ARI"]:.3f}")
results_2d = centroid_bootstrap_stability(df2d, cols, n_clusters=7, n_bootstrap=1000)
print(f"\nCentroid bootstrap stability (2D): ARI = {results_2d["Mean ARI"]:.3f} ± {results_2d["Std ARI"]:.3f}")

# %% Cell
# rearrange Jaccard dict according physical labels according used map_1d_cluster/map_2d_cluster
def remap_jaccard(jaccard_dict, mapping):
    remapped = {}
    for algo_label, stats in jaccard_dict.items():
        phys_label = mapping[int(algo_label)]
        remapped[phys_label] = stats
    return dict(sorted(remapped.items()))
jaccard_1d_remapped = remap_jaccard(results_1d["Cluster Jaccards"], map_1d_cluster)
jaccard_2d_remapped = remap_jaccard(results_2d["Cluster Jaccards"], map_2d_cluster)

print("\nPhys Cluster | 1D Jaccard (Mean) | 1D Jaccard (Min) | 2D Jaccard (Mean) | 2D Jaccard (Min)")
for i in range(1, 8):
    j1 = jaccard_1d_remapped[i]
    j2 = jaccard_2d_remapped[i]
    print(
        f"  {i}         |       {j1['Mean']:.3f}        |      {j1['Min']:.3f}      |       {j2['Mean']:.3f}        |      {j2['Min']:.3f}"
    )
# %% Cell
def bootstrap_hc_noise(X_scaled, Z_orig, labels_orig, treshold, n_bootstrap=1000, noise_frac=0.1, random_seed=42):
    rng = np.random.default_rng(random_seed)
    n_samples, n_features = X_scaled.shape

    feature_stds = np.std(X_scaled, axis=0)

    ari_scores = []
    coassoc = np.zeros((n_samples, n_samples))

    for _ in tqdm(range(n_bootstrap), desc="Noise bootstrap"):
        noise = rng.normal(0, noise_frac * feature_stds, size=X_scaled.shape)
        X_noisy = X_scaled + noise

        Z_boot = linkage(X_noisy, method="ward", metric="euclidean")
        labels_boot = fcluster(Z_boot, t=treshold, criterion="distance")

        ari_scores.append(adjusted_rand_score(labels_orig, labels_boot))

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if labels_boot[i] == labels_boot[j]:
                    coassoc[i, j] += 1
                    coassoc[j, i] += 1

    coassoc /= n_bootstrap

    return {
        "ari_mean": np.mean(ari_scores),
        "ari_std": np.std(ari_scores),
        "consensus_matrix": coassoc,
        "ari_scores": ari_scores,
    }

noise_boot_1d = bootstrap_hc_noise(X1d_scaled, Z1, labels1, 10, n_bootstrap=1000, noise_frac=0.1, random_seed=42)
noise_boot_2d = bootstrap_hc_noise(X2d_scaled, Z2, labels2, 10, n_bootstrap=1000, noise_frac=0.1, random_seed=42)
print(f"\n1D ARI(noise):: {noise_boot_1d['ari_mean']:.3f} ± {noise_boot_1d['ari_std']:.3f}")
print(f"\n2D ARI(noise):: {noise_boot_2d['ari_mean']:.3f} ± {noise_boot_2d['ari_std']:.3f}")
# %% Cell
def subsampling_bootstrap2(X_scaled, n_bootstrap=100, sample_frac=0.8, random_seed=42):
    rng = np.random.default_rng(random_seed)
    n = len(X_scaled)
    cooccurrence = np.zeros((n, n))
    pair_counts = np.zeros((n, n))

    for _ in tqdm(range(n_bootstrap)):
        idx = rng.choice(n, int(n * sample_frac), replace=False)
        X_sub = X_scaled[idx]

        Z_sub = linkage(X_sub, method="ward", metric="euclidean")
        labels_sub = fcluster(Z_sub, t=treshold, criterion="distance")

        # co-occurrence update
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                if labels_sub[i] == labels_sub[j]:
                    cooccurrence[idx[i], idx[j]] += 1
                    cooccurrence[idx[j], idx[i]] += 1
                pair_counts[idx[i], idx[j]] += 1
                pair_counts[idx[j], idx[i]] += 1

    pair_counts[pair_counts == 0] = 1
    cooccurrence /= pair_counts

    return cooccurrence

def plot_cooc(ss_cooc, labels):
    order = np.argsort(labels)
    coocc_sorted = ss_cooc[order][:, order]

    # boundaries = np.where(np.diff(labels1[order]))[0] + 0.5

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        coocc_sorted,
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Co-occurrence probability"},
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )

    # for b in boundaries:
    #     ax.axhline(b, color='white', lw=1)
    #     ax.axvline(b, color='white', lw=1)

    ax.set_title("Consensus matrix (ordered by Ward clusters)")
    plt.gca()
# %% Cell
ss_cooc_1d = subsampling_bootstrap2(X1d_scaled)
ss_cooc_2d = subsampling_bootstrap2(X2d_scaled)

# %% Cell
plot_cooc(ss_cooc_1d, labels1)

# %% Cell
plot_cooc(ss_cooc_2d, labels2)

# %% Cell

# %% Cell

# %% Cell
