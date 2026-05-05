"""
Customer Churn Prediction - Exploratory Data Analysis
======================================================
Author: CodSoft ML Internship - Task 3
Purpose: Understand the dataset, spot patterns, and prepare for modelling.

I've structured this as a standalone script so you can run it independently
before touching the modelling pipeline. Good EDA = fewer surprises later.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "Churn_Modelling.csv")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#f9f9f9",
    "axes.facecolor":   "#f9f9f9",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.family":      "DejaVu Sans",
})
PALETTE = ["#2196F3", "#F44336"]  # blue = stayed, red = churned


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop non-predictive ID column if present
    df.drop(columns=["CustomerId", "Surname"], errors="ignore", inplace=True)
    print(f"\n✓ Data loaded  →  {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
    return df


def basic_overview(df: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print("  DATASET OVERVIEW")
    print("="*60)
    print(df.dtypes.to_string())
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDuplicates: {df.duplicated().sum()}")
    print("\nNumeric summary:")
    print(df.describe().round(2).to_string())


def plot_churn_distribution(df: pd.DataFrame) -> None:
    """Simple bar chart – how many customers actually churned?"""
    counts = df["Exited"].value_counts()
    labels = ["Stayed", "Churned"]
    pct    = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=PALETTE, edgecolor="white", width=0.5)

    for bar, p in zip(bars, pct.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 80,
                f"{p:.1f}%", ha="center", fontsize=11, fontweight="bold")

    ax.set_title("Customer Churn Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Customers")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_churn_distribution.png"), dpi=150)
    plt.close()
    print("✓ Saved: 01_churn_distribution.png")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """
    KDE plots for every numeric feature, split by churn label.
    This quickly shows which features 'separate' churners from non-churners.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Exited"]

    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        for label, color in zip([0, 1], PALETTE):
            subset = df[df["Exited"] == label][col]
            subset.plot.kde(ax=ax, color=color, linewidth=2,
                            label=("Stayed" if label == 0 else "Churned"))
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=8)

    # Hide leftover subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Churn Status", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_numeric_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Saved: 02_numeric_distributions.png")


def plot_categorical_churn(df: pd.DataFrame) -> None:
    """Churn rate broken down by categorical features."""
    cat_cols = ["Geography", "Gender", "NumOfProducts", "HasCrCard", "IsActiveMember"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    fig, axes = plt.subplots(1, len(cat_cols), figsize=(18, 4))

    for ax, col in zip(axes, cat_cols):
        churn_rate = df.groupby(col)["Exited"].mean().reset_index()
        churn_rate.columns = [col, "ChurnRate"]
        churn_rate["ChurnRate"] *= 100

        bars = ax.bar(churn_rate[col].astype(str), churn_rate["ChurnRate"],
                      color="#e57373", edgecolor="white")
        ax.axhline(df["Exited"].mean() * 100, color="#1565C0", linestyle="--",
                   linewidth=1.4, label="Overall avg")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%", ha="center", fontsize=9)

        ax.set_title(col, fontweight="bold")
        ax.set_ylabel("Churn Rate (%)" if col == cat_cols[0] else "")
        ax.set_ylim(0, churn_rate["ChurnRate"].max() * 1.3)
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Churn Rate by Category", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_categorical_churn_rates.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Saved: 03_categorical_churn_rates.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation matrix — helps spot multicollinearity before modelling."""
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_correlation_heatmap.png"), dpi=150)
    plt.close()
    print("✓ Saved: 04_correlation_heatmap.png")


def plot_age_balance_scatter(df: pd.DataFrame) -> None:
    """Age vs Balance, coloured by churn — a really intuitive view."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color, name in zip([0, 1], PALETTE, ["Stayed", "Churned"]):
        sub = df[df["Exited"] == label]
        ax.scatter(sub["Age"], sub["Balance"], alpha=0.3, s=12,
                   c=color, label=name)
    ax.set_xlabel("Age")
    ax.set_ylabel("Account Balance (€)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1000:.0f}k"))
    ax.set_title("Age vs Balance — coloured by Churn", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_age_balance_scatter.png"), dpi=150)
    plt.close()
    print("✓ Saved: 05_age_balance_scatter.png")


def run_eda():
    df = load_data(DATA_PATH)
    basic_overview(df)
    plot_churn_distribution(df)
    plot_numeric_distributions(df)
    plot_categorical_churn(df)
    plot_correlation_heatmap(df)
    plot_age_balance_scatter(df)
    print("\n✅  EDA complete — check the plots/ folder for all charts.\n")


if __name__ == "__main__":
    run_eda()
