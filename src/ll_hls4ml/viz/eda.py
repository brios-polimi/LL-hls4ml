"""EDA visualizations: PCA, t-SNE, LDA, vocab plots."""

from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

DEFAULT_FIGSIZE = (10, 8)
DEFAULT_POINT_SIZE = 20
DEFAULT_ALPHA = 0.8


def compute_pca(X, pca_components, feature_names=None, top_k=10):
    pca = PCA(n_components=pca_components)
    embedding = pca.fit_transform(X)

    print("\nPCA explained variance ratio:")
    print(pca.explained_variance_ratio_)

    components = pca.components_
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    for pc_idx, comp in enumerate(components):
        weights = np.abs(comp)
        top_idx = np.argsort(weights)[::-1][:top_k]
        print(f"\nPC{pc_idx + 1} top contributors:")
        for i in top_idx:
            print(f"  {feature_names[i]}: {comp[i]:.4f}")

    energy = np.sum(
        (pca.explained_variance_ratio_[:, None] * np.abs(components)),
        axis=0,
    )
    energy_df = pd.DataFrame({
        "feature": feature_names,
        "energy": energy,
    }).sort_values("energy", ascending=False)

    print("\nPCA Energy of Features:")
    print(energy_df)

    return embedding, pca, energy_df


def plot_tsne(
    X,
    labels,
    title="t-SNE Projection of Kernels",
    perplexity=30,
    cmap_name="tab10",
    figsize=DEFAULT_FIGSIZE,
):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    embedding = tsne.fit_transform(X)

    plt.figure(figsize=figsize)
    unique_labels = sorted(set(labels))
    cmap = cm.get_cmap(cmap_name, len(unique_labels))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=DEFAULT_POINT_SIZE,
            alpha=DEFAULT_ALPHA,
            color=cmap(idx),
            label=label,
        )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_tsne_continuous(
    X,
    df,
    label_name,
    title="t-SNE Projection of Kernels",
    log_y=False,
    perplexity=30,
    cmap_name="viridis",
    figsize=DEFAULT_FIGSIZE,
):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    embedding = tsne.fit_transform(X)

    label_values = df[label_name].values
    if log_y:
        label_values = np.where(label_values == 0, 0, np.log(label_values))

    cmap = cm.get_cmap(cmap_name)
    norm = colors.Normalize(vmin=np.min(label_values), vmax=np.max(label_values))

    plt.figure(figsize=figsize)
    sc = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=DEFAULT_POINT_SIZE,
        alpha=DEFAULT_ALPHA,
        c=label_values,
        cmap=cmap,
        norm=norm,
    )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(title)
    plt.colorbar(sc, label=label_name)
    plt.tight_layout()
    plt.show()


def between_class_variance_ratio(X, labels, feature_names):
    """Per-feature ratio of between-class to total variance."""
    unique_labels = np.unique(labels)
    global_mean = X.mean(axis=0)
    total_var = X.var(axis=0)

    class_means = np.array([X[labels == k].mean(axis=0) for k in unique_labels])
    class_counts = np.array([np.sum(labels == k) for k in unique_labels])

    between_var = np.average(
        (class_means - global_mean) ** 2,
        axis=0,
        weights=class_counts / len(labels),
    )
    ratio = np.where(total_var > 0, between_var / total_var, 0.0)

    return pd.DataFrame({
        "feature": feature_names,
        "between_class_var_ratio": ratio,
    }).sort_values("between_class_var_ratio", ascending=False)


def plot_lda_weights(X_scaled, labels, feature_cols):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_scaled, labels)
    return pd.DataFrame({
        "feature": feature_cols,
        "LD1": np.abs(lda.coef_[0]),
    }).sort_values("LD1", ascending=False)


def plot_vocab_counts(vocab_counts: dict, figsize=(16, 10)):
    """Bar plots of vocabulary token frequencies by node category."""
    df = pd.DataFrame([
        {"category": cat, "token": tok, "count": cnt}
        for cat, table in vocab_counts.items()
        for tok, cnt in table.items()
    ])

    totals = df.groupby("category")["count"].sum().sort_values()
    top_instruction = (
        df[df["category"] == "instruction"]
        .nlargest(15, "count")
        .sort_values("count", ascending=False)
    )
    top_variable = (
        df[df["category"] == "variable"]
        .nlargest(10, "count")
        .sort_values("count", ascending=False)
    )
    top_constant = (
        df[df["category"] == "constant"]
        .nlargest(10, "count")
        .sort_values("count", ascending=False)
    )

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax = axes[0, 0]
    sns.barplot(x=totals.values, y=totals.index, ax=ax, palette="muted")
    ax.set_title("Total vocab counts by category (logscale)")
    ax.set_xlabel("Count")
    ax.set_xscale("log")
    ax.set_xlim(1e4, totals.max() * 3)
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", label_type="edge", padding=3)

    sns.barplot(x="count", y="token", data=top_instruction, ax=axes[0, 1], palette="rocket")
    axes[0, 1].set_title("Top 15 instructions")

    sns.barplot(x="count", y="token", data=top_variable, ax=axes[1, 0], palette="viridis")
    axes[1, 0].set_title("Top variable types (logscale)")
    axes[1, 0].set_xscale("log")

    sns.barplot(x="count", y="token", data=top_constant, ax=axes[1, 1], palette="cubehelix")
    axes[1, 1].set_title("Top constant types (logscale)")
    axes[1, 1].set_xscale("log")

    plt.tight_layout()
    plt.show()
