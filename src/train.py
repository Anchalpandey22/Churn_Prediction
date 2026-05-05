"""
Customer Churn Prediction - Model Training & Evaluation
=========================================================
Trains three classifiers (Logistic Regression, Random Forest, Gradient Boosting),
compares them, and saves the best one.

Why three models?
  - Logistic Regression: interpretable baseline — if it works well, great.
  - Random Forest: handles non-linear patterns and feature interactions.
  - Gradient Boosting: usually strongest but slower; uses GradientBoostingClassifier
    from sklearn (no xgboost dependency needed).
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics       import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import get_preprocessed_data

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#f9f9f9",
    "axes.facecolor":   "#f9f9f9",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ── Model definitions ──────────────────────────────────────────────────────────

MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=4,
        subsample=0.8, random_state=42
    ),
}


# ── Training helpers ───────────────────────────────────────────────────────────

def train_all_models(X_train, y_train) -> dict:
    """
    Train each model and return a dict of fitted models.
    Also runs 5-fold CV so we can compare on training data before touching the test set.
    """
    fitted = {}
    print("\n" + "="*60)
    print("  TRAINING MODELS")
    print("="*60)

    for name, model in MODELS.items():
        print(f"\n→ {name}")
        model.fit(X_train, y_train)

        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=5, scoring="roc_auc", n_jobs=-1)
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        fitted[name] = model

    return fitted


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Full evaluation of a single model on the held-out test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)

    print(f"\n{'─'*50}")
    print(f"  {name}  —  Test set results")
    print(f"{'─'*50}")
    print(f"  ROC-AUC:           {roc_auc:.4f}")
    print(f"  Avg Precision:     {avg_prec:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Stayed','Churned'])}")

    return {
        "name":    name,
        "model":   model,
        "roc_auc": roc_auc,
        "avg_prec": avg_prec,
        "y_pred":  y_pred,
        "y_prob":  y_prob,
    }


def evaluate_all(fitted_models: dict, X_test, y_test) -> list:
    results = []
    for name, model in fitted_models.items():
        results.append(evaluate_model(name, model, X_test, y_test))
    return results


# ── Visualisation helpers ──────────────────────────────────────────────────────

def plot_confusion_matrices(results: list, y_test) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = confusion_matrix(y_test, res["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(res["name"], fontweight="bold", fontsize=11)

    fig.suptitle("Confusion Matrices — Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "06_confusion_matrices.png"), dpi=150)
    plt.close()
    print("✓ Saved: 06_confusion_matrices.png")


def plot_roc_curves(results: list, y_test) -> None:
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    fig, ax = plt.subplots(figsize=(7, 5))

    for res, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{res['name']}  (AUC = {res['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "07_roc_curves.png"), dpi=150)
    plt.close()
    print("✓ Saved: 07_roc_curves.png")


def plot_precision_recall(results: list, y_test) -> None:
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    fig, ax = plt.subplots(figsize=(7, 5))

    for res, color in zip(results, colors):
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        ax.plot(rec, prec, color=color, linewidth=2,
                label=f"{res['name']}  (AP = {res['avg_prec']:.3f})")

    baseline = y_test.mean()
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Baseline (prevalence {baseline:.2%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "08_precision_recall.png"), dpi=150)
    plt.close()
    print("✓ Saved: 08_precision_recall.png")


def plot_feature_importance(best_result: dict, feature_names: list) -> None:
    """
    Random Forest and GBM both expose feature_importances_; LR uses coefficients.
    We handle both so this function works regardless of which model 'won'.
    """
    model = best_result["model"]
    name  = best_result["name"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        label = "Feature Importance (Gini)"
    else:
        importances = np.abs(model.coef_[0])
        label = "|Coefficient| (Logistic Regression)"

    idx = np.argsort(importances)[-15:]  # top 15
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color="#42A5F5", edgecolor="white"
    )
    ax.set_xlabel(label)
    ax.set_title(f"Top 15 Features  —  {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "09_feature_importance.png"), dpi=150)
    plt.close()
    print("✓ Saved: 09_feature_importance.png")


def plot_model_comparison(results: list) -> None:
    names    = [r["name"] for r in results]
    roc_vals = [r["roc_auc"] for r in results]
    ap_vals  = [r["avg_prec"] for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, roc_vals, width, label="ROC-AUC",  color="#1E88E5")
    bars2 = ax.bar(x + width/2, ap_vals,  width, label="Avg Precision", color="#43A047")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "10_model_comparison.png"), dpi=150)
    plt.close()
    print("✓ Saved: 10_model_comparison.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_training():
    X_train, X_test, y_train, y_test, feat_names, scaler = get_preprocessed_data()

    fitted_models = train_all_models(X_train, y_train)

    print("\n" + "="*60)
    print("  EVALUATION ON TEST SET")
    print("="*60)
    results = evaluate_all(fitted_models, X_test, y_test)

    # Pick best by ROC-AUC
    best = max(results, key=lambda r: r["roc_auc"])
    print(f"\n🏆  Best model: {best['name']}  (ROC-AUC = {best['roc_auc']:.4f})")

    # Save best model
    joblib.dump(best["model"], os.path.join(MODELS_DIR, "best_model.pkl"))
    print(f"✓  Best model saved → models/best_model.pkl")

    # Save all models too (useful for the prediction script)
    for res in results:
        safe_name = res["name"].lower().replace(" ", "_")
        joblib.dump(res["model"], os.path.join(MODELS_DIR, f"{safe_name}.pkl"))

    # Plots
    print("\n" + "="*60)
    print("  GENERATING PLOTS")
    print("="*60)
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_precision_recall(results, y_test)
    plot_feature_importance(best, feat_names)
    plot_model_comparison(results)

    print("\n✅  Training complete.\n")
    return best, feat_names


if __name__ == "__main__":
    run_training()
