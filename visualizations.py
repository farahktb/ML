"""
visualizations.py
-----------------
All matplotlib/seaborn chart functions for the Streamlit dashboard.
Each function returns a matplotlib Figure object ready for st.pyplot().
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams.update({
    "figure.facecolor": "#0a0a0a",
    "axes.facecolor": "#111111",
    "axes.edgecolor": "#2a2a2a",
    "axes.labelcolor": "#cccccc",
    "xtick.color": "#888888",
    "ytick.color": "#888888",
    "text.color": "#cccccc",
    "grid.color": "#1e1e1e",
    "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
})

FRAUD_COLOR  = "#cc2222"
LEGIT_COLOR  = "#22aa55"
ACCENT_COLOR = "#aaaaaa"
BG_COLOR     = "#0a0a0a"
CARD_COLOR   = "#111111"


# ── 1. Class distribution ──────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame) -> plt.Figure:
    counts = df["Class"].value_counts().sort_index()
    labels = ["Légitime", "Frauduleux"]
    colors = [LEGIT_COLOR, FRAUD_COLOR]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor(BG_COLOR)

    # Bar chart
    bars = ax1.bar(labels, counts.values, color=colors, width=0.5, edgecolor="#0d1117", linewidth=1.5)
    ax1.set_title("Distribution des Transactions", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Nombre de transactions", fontsize=10)
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold",
                 color=bar.get_facecolor())
    ax1.set_facecolor(CARD_COLOR)
    ax1.grid(axis="y", alpha=0.4)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Pie chart
    wedges, texts, autotexts = ax2.pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.2f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="#0d1117", linewidth=2),
        textprops=dict(color="#c9d1d9", fontsize=10),
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax2.set_title("Proportion Fraude / Légitime", fontsize=13, fontweight="bold", pad=12)
    ax2.set_facecolor(BG_COLOR)

    fig.tight_layout(pad=2)
    return fig


# ── 2. Transaction amount distribution ────────────────────────────────────────

def plot_amount_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(CARD_COLOR)

    legit = df[df["Class"] == 0]["Amount"]
    fraud = df[df["Class"] == 1]["Amount"]

    ax.hist(legit.clip(upper=5000), bins=60, color=LEGIT_COLOR, alpha=0.7,
            label=f"Légitime (n={len(legit):,})", edgecolor="none")
    ax.hist(fraud.clip(upper=5000), bins=60, color=FRAUD_COLOR, alpha=0.9,
            label=f"Frauduleux (n={len(fraud):,})", edgecolor="none")

    ax.set_title("Distribution des Montants (€) par Classe", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Montant (€)  [tronqué à 5 000]", fontsize=10)
    ax.set_ylabel("Fréquence", fontsize=10)
    ax.legend(framealpha=0.3, fontsize=10)
    ax.grid(axis="y", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=2)
    return fig


# ── 3. Fraud probability distribution ────────────────────────────────────────

def plot_probability_distribution(y_proba: np.ndarray, y_true) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(CARD_COLOR)

    y_true = np.array(y_true)
    ax.hist(y_proba[y_true == 0], bins=50, color=LEGIT_COLOR, alpha=0.7,
            label="Légitime", edgecolor="none")
    ax.hist(y_proba[y_true == 1], bins=50, color=FRAUD_COLOR, alpha=0.9,
            label="Frauduleux", edgecolor="none")

    ax.axvline(0.5, color="#facc15", linestyle="--", linewidth=1.5, label="Seuil = 0.5")
    ax.set_title("Distribution des Probabilités de Fraude (Modèle)", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_xlabel("P(Fraude)", fontsize=10)
    ax.set_ylabel("Fréquence", fontsize=10)
    ax.legend(framealpha=0.3, fontsize=10)
    ax.grid(axis="y", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=2)
    return fig


# ── 4. Confusion matrix heatmap ───────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(CARD_COLOR)

    cmap = LinearSegmentedColormap.from_list("fraud_cm", ["#161b22", "#3b82f6"])
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    labels = ["Légitime (0)", "Frauduleux (1)"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Prédiction", fontsize=11, labelpad=10)
    ax.set_ylabel("Réalité", fontsize=11, labelpad=10)
    ax.set_title("Matrice de Confusion", fontsize=13, fontweight="bold", pad=12)

    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    cell_colors = [[LEGIT_COLOR, FRAUD_COLOR], [FRAUD_COLOR, LEGIT_COLOR]]

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.text(j, i,
                    f"{cell_labels[i][j]}\n{val:,}",
                    ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color=cell_colors[i][j])

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=2)
    return fig


# ── 5. Loss / error curve ─────────────────────────────────────────────────────

def plot_loss_curve(loss_history: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(CARD_COLOR)

    epochs = np.arange(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, color=ACCENT_COLOR, linewidth=2, label="Coût J(θ)")
    ax.fill_between(epochs, loss_history, alpha=0.15, color=ACCENT_COLOR)

    # Annotate key points
    ax.scatter([1], [loss_history[0]], color="#facc15", zorder=5, s=60, label=f"Début : {loss_history[0]:.3f}")
    ax.scatter([len(epochs)], [loss_history[-1]], color=LEGIT_COLOR, zorder=5, s=60,
               label=f"Final : {loss_history[-1]:.3f}")

    ax.set_title("Courbe d'Apprentissage — Descente de Gradient", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Époque", fontsize=10)
    ax.set_ylabel("Coût J(θ)  [Binary Cross-Entropy]", fontsize=10)
    ax.legend(framealpha=0.3, fontsize=10)
    ax.grid(alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=2)
    return fig


# ── 6. Fraud over time ────────────────────────────────────────────────────────

def plot_fraud_over_time(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(CARD_COLOR)

    # Bin by hour (Time is in seconds)
    df = df.copy()
    df["Hour"] = (df["Time"] // 3600).astype(int) % 48  # dataset spans ~48h
    grouped = df.groupby(["Hour", "Class"]).size().unstack(fill_value=0)

    if 0 in grouped.columns:
        ax.fill_between(grouped.index, grouped[0], alpha=0.4, color=LEGIT_COLOR, label="Légitime")
        ax.plot(grouped.index, grouped[0], color=LEGIT_COLOR, linewidth=1.5)
    if 1 in grouped.columns:
        ax.fill_between(grouped.index, grouped[1], alpha=0.6, color=FRAUD_COLOR, label="Frauduleux")
        ax.plot(grouped.index, grouped[1], color=FRAUD_COLOR, linewidth=1.5)

    ax.set_title("Évolution des Transactions dans le Temps", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Heure (depuis le début du dataset)", fontsize=10)
    ax.set_ylabel("Nombre de transactions", fontsize=10)
    ax.legend(framealpha=0.3, fontsize=10)
    ax.grid(alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=2)
    return fig


# ── 7. Sigmoid function plot ──────────────────────────────────────────────────

def plot_sigmoid() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(CARD_COLOR)

    z = np.linspace(-10, 10, 400)
    sigma = 1 / (1 + np.exp(-z))

    ax.plot(z, sigma, color=ACCENT_COLOR, linewidth=2.5, label="σ(z) = 1 / (1 + e⁻ᶻ)")
    ax.axhline(0.5, color="#facc15", linestyle="--", linewidth=1.2, alpha=0.8, label="Seuil de décision = 0.5")
    ax.axvline(0, color="#8b949e", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.fill_between(z, sigma, 0.5, where=(sigma >= 0.5), alpha=0.15, color=FRAUD_COLOR, label="→ Fraude")
    ax.fill_between(z, sigma, 0.5, where=(sigma < 0.5), alpha=0.15, color=LEGIT_COLOR, label="→ Légitime")

    ax.set_title("Fonction Sigmoïde — Activation du Perceptron", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("z  =  θᵀx  (entrée pondérée)", fontsize=10)
    ax.set_ylabel("σ(z)  — Probabilité de fraude", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(framealpha=0.3, fontsize=9)
    ax.grid(alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=2)
    return fig


# ── 8. Gradient descent illustration ─────────────────────────────────────────

def plot_gradient_descent_illustration() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(CARD_COLOR)

    theta = np.linspace(-3, 3, 300)
    J = theta ** 2 + 0.5  # Simple convex cost for illustration

    ax.plot(theta, J, color=ACCENT_COLOR, linewidth=2.5, label="J(θ)  [Coût]")

    # Gradient steps
    start = 2.5
    steps = [start]
    lr = 0.4
    for _ in range(6):
        start = start - lr * 2 * start
        steps.append(start)

    step_J = [s ** 2 + 0.5 for s in steps]
    for i in range(len(steps) - 1):
        ax.annotate("", xy=(steps[i + 1], step_J[i + 1]), xytext=(steps[i], step_J[i]),
                    arrowprops=dict(arrowstyle="->", color="#facc15", lw=1.5))
    ax.scatter(steps, step_J, color="#facc15", zorder=5, s=40, label="Étapes de descente")
    ax.scatter([0], [0.5], color=LEGIT_COLOR, zorder=6, s=80, label="Minimum global", marker="*")

    ax.set_title("Descente de Gradient — Convergence vers le Minimum", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Paramètre θ", fontsize=10)
    ax.set_ylabel("J(θ)  [Coût]", fontsize=10)
    ax.legend(framealpha=0.3, fontsize=9)
    ax.grid(alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=2)
    return fig
