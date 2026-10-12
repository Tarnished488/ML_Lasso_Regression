from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 16, "axes.titlesize": 16,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.25,
})

DEFAULT_ELEV, DEFAULT_AZIM = 20, -60

_HERE = Path(__file__).resolve().parent
DATA_FILE = next(
    (p for p in (_HERE / "week3.php.csv", _HERE.parent / "week3.php.csv") if p.exists()),
    _HERE.parent / "week3.php.csv",
)
FIG_DIR = _HERE.parent / "Figures"
FIG_DIR.mkdir(exist_ok=True)

Z_LABEL = "target  (y)"
AXIS_LABELS = ("feature 1  (x_1)", "feature 2  (x_2)")


def load_dataframe(path=DATA_FILE):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"data file not found: {path}")

    df = pd.read_csv(path, comment="#", header=None, sep=",", skipinitialspace=True,
                     skip_blank_lines=True, dtype=float, encoding="utf-8")
    df = df.dropna(how="all")
    if df.shape[1] < 3:
        raise ValueError(f"expected 3 columns (x1, x2, y), got {df.shape[1]}: {path}")
    df = df.iloc[:, :3].copy()
    df.columns = ["x1", "x2", "y"]
    return df


def load_data(path=DATA_FILE):
    df = load_dataframe(path)
    X1, X2, y = df["x1"], df["x2"], df["y"]
    return np.column_stack((X1.to_numpy(float), X2.to_numpy(float))), y, X1, X2


def summarize_data(path=DATA_FILE):
    with open(path, encoding="utf-8") as f:
        print(f"comment   : {f.readline().strip()}")
    df = load_dataframe(path)
    print(f"shape     : {df.shape}")
    with pd.option_context("display.precision", 4, "display.width", 100):
        print(df.describe().T.to_string())
    return df


def make_grid(lo=-1.5, hi=1.5, n=41):
    g = np.linspace(lo, hi, n)
    G1, G2 = np.meshgrid(g, g)
    return G1, G2, np.column_stack((G1.ravel(), G2.ravel()))


def new_figure(nrows=1, ncols=1, figsize=None, **kwargs):
    if figsize is None:
        figsize = (7 * ncols, 5.5 * nrows)
    return plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, **kwargs)


def save_figure(fig, filename, dpi=150):
    out = FIG_DIR / filename
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"figure saved -> {out}")
    plt.show()
    return out


def _set_axes_3d(ax, title, zlim=None, elev=DEFAULT_ELEV, azim=DEFAULT_AZIM):
    ax.set_xlabel(AXIS_LABELS[0], labelpad=10)
    ax.set_ylabel(AXIS_LABELS[1], labelpad=10)
    ax.set_zlabel(Z_LABEL, labelpad=10)
    ax.set_title(title, pad=15)
    ax.tick_params(labelsize=11)
    if zlim is not None:
        ax.set_zlim(*zlim)
    ax.view_init(elev=elev, azim=azim)


def plot_data_3d(X, y, filename="data_3d.png", elev=DEFAULT_ELEV, azim=DEFAULT_AZIM,
                 figsize=(9, 7.5), show=True, title=None):
    X, y = np.asarray(X, float), np.asarray(y, float)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap="viridis", s=32, alpha=0.9,
               edgecolors="k", linewidths=0.3, depthshade=False, label="training data")

    _set_axes_3d(ax, title or "(i)(a)  training data:  feature 1 vs feature 2 vs target",
                 elev=elev, azim=azim)
    ax.legend(loc="upper left")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(y.min(), y.max()))
    fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.1, label=Z_LABEL)

    return save_figure(fig, filename) if show else (fig, ax)


def plot_fit_surface(X, y, surfaces, filename="fit_surface.png", elev=DEFAULT_ELEV,
                     azim=DEFAULT_AZIM, figsize=(10, 8), data_color="black",
                     alpha=0.75, zlim=None, show=True, title=None):
    X, y = np.asarray(X, float), np.asarray(y, float)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(X[:, 0], X[:, 1], y, c=data_color, s=26, alpha=0.85,
               edgecolors="none", depthshade=False, label="training data")

    colors = plt.cm.plasma(np.linspace(0.05, 0.8, max(len(surfaces), 1)))
    for (label, G1, G2, Z), color in zip(surfaces, colors):
        ax.plot_surface(G1, G2, np.asarray(Z, float).reshape(np.asarray(G1).shape),
                        color=color, alpha=alpha, linewidth=0, antialiased=True)

    _set_axes_3d(ax, title or "(i)(c)  model predictions on a grid of feature values",
                 zlim, elev, azim)
    handles = [Patch(facecolor=c, label=lab) for (lab, *_), c in zip(surfaces, colors)]
    handles.append(plt.Line2D([], [], marker="o", linestyle="none", color=data_color,
                              markersize=7, label="training data"))
    ax.legend(handles=handles, loc="upper left", framealpha=0.9)

    return save_figure(fig, filename) if show else (fig, ax)


def plot_cv_errorbar(C_values, mean_errors, std_errors=None, filename="cv_error.png",
                     log_x=True, figsize=(9, 6.5), show=True, title=None,
                     ylabel="prediction error (mean squared error)", xlabel=None,
                     mark_best=False, baseline=None):
    C_values, mean_errors = np.asarray(C_values, float), np.asarray(mean_errors, float)
    fig, axes = new_figure(figsize=figsize)
    ax = axes[0][0]

    if baseline is not None:
        ax.axhline(baseline, color="gray", linestyle=":", linewidth=1.5,
                   label=f"baseline  (MSE = {baseline:.4f})")

    if mark_best:
        k = int(np.argmin(mean_errors))
        ax.axvline(C_values[k], color="crimson", linestyle="--", linewidth=1.8,
                   label=f"best C = {C_values[k]:g}  (error = {mean_errors[k]:.4f})")

    ax.errorbar(C_values, mean_errors, yerr=std_errors, marker="o", markersize=6,
                linewidth=1.8, capsize=4, capthick=1.4, color="tab:blue",
                ecolor="tab:blue", label="5-fold CV error (mean $\\pm$ std)")
    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel(xlabel or ("penalty weight  C" + ("  (log scale)" if log_x else "")),
                  fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title or "(ii)(a)  cross-validation error vs C", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    return save_figure(fig, filename) if show else (fig, ax)


def plot_coef_vs_C(C_values, coefs, feature_names=None, filename="coef_vs_C.png",
                   log_x=True, figsize=(11, 7), show=True, title=None,
                   highlight_zero=True):
    C_values, coefs = np.asarray(C_values, float), np.asarray(coefs, float)
    fig, axes = new_figure(figsize=figsize)
    ax = axes[0][0]
    n_feat = coefs.shape[1]
    cmap = plt.cm.turbo(np.linspace(0.05, 0.95, n_feat))

    for j in range(n_feat):
        ax.plot(C_values, coefs[:, j], marker="o", markersize=3.5, linewidth=1.6,
                color=cmap[j],
                label=feature_names[j] if feature_names is not None else f"feature {j+1}")
    if highlight_zero:
        ax.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel("penalty weight  C" + ("  (log scale)" if log_x else ""), fontsize=16)
    ax.set_ylabel("fitted coefficient value", fontsize=16)
    ax.set_title(title or "(i)(b)  model coefficients vs C", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2 if n_feat <= 12 else 3, fontsize=11 if n_feat <= 12 else 9)

    return save_figure(fig, filename) if show else (fig, ax)


if __name__ == "__main__":
    summarize_data()
    X, y, *rest = load_data()
    print(f"\nload_data() -> X {X.shape}, y {y.shape}")
    plot_data_3d(X, y)

    G1, G2, Xgrid = make_grid()
    Z = (0.9 * Xgrid[:, 1]).reshape(G1.shape)
    plot_fit_surface(X, y, [("demo", G1, G2, Z)], filename="_smoke_surface.png")
    Cs = np.logspace(-3, 3, 8)
    plot_cv_errorbar(Cs, 0.1 + 1 / Cs, 0.02 + 0.1 / Cs, filename="_smoke_cv.png", mark_best=True)
    plot_coef_vs_C(Cs, np.column_stack([np.zeros_like(Cs), np.log10(Cs) * 0.2]),
                   ["x_1", "x_2"], filename="_smoke_coef.png")
    print("smoke tests passed")
