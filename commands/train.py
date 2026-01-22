import click
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize


# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_embeddings(path):
    """Supports TF-IDF (dict) and BERT (array)"""
    data = joblib.load(path)
    if isinstance(data, dict):
        return data["matrix"], data.get("vectorizer")
    return data, None


# ---------------------------
# Plot ROC Curve
# ---------------------------
def plot_roc(y_test, y_score, model_name, out_dir):
    try:
        classes = sorted(set(y_test))
        y_bin = label_binarize(y_test, classes=classes)

        if y_score.ndim == 2 and y_score.shape[1] > 1:
            y_score = y_score[:, 1:]

        plt.figure(figsize=(7, 6))
        for i in range(y_bin.shape[1]):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        path = os.path.join(out_dir, f"roc_{model_name}.png")
        plt.savefig(path)
        plt.close()
        return path
    except Exception as e:
        print(f"ROC skipped: {e}")
        return None


# ---------------------------
# Feature Importance
# ---------------------------
def plot_features(model, vectorizer, model_name, out_dir):
    if vectorizer is None:
        return None

    try:
        features = vectorizer.get_feature_names_out()

        if hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            return None

        idx = np.argsort(importances)[::-1][:20]
        top_feats = features[idx]
        top_vals = importances[idx]

        plt.figure(figsize=(8, 6))
        sns.barplot(x=top_vals, y=top_feats)
        plt.title(f"Top 20 Features - {model_name}")
        path = os.path.join(out_dir, f"features_{model_name}.png")
        plt.savefig(path)
        plt.close()
        return path
    except Exception as e:
        print(f"Feature plot skipped: {e}")
        return None


# ---------------------------
# CLI
# ---------------------------
@click.group()
def train():
    pass


@train.command()
@click.option("--csv_path", required=True)
@click.option("--input_col", required=True, help="Embeddings .pkl")
@click.option("--output_col", required=True, help="Label column")
@click.option("--models", default="lr", help="knn,lr,rf,all")
@click.option("--save_model", is_flag=True)
def train_models(csv_path, input_col, output_col, models, save_model):
    ensure_dir("outputs/models")
    ensure_dir("outputs/reports")
    ensure_dir("outputs/visualizations")

    df = pd.read_csv(csv_path)
    y = df[output_col]

    print(f"Loading embeddings: {input_col}")
    X, vectorizer = load_embeddings(input_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_map = {
        "knn": KNeighborsClassifier(),
        "lr": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=100)
    }

    selected = model_map.keys() if models == "all" else models.split(",")

    report = [f"# Training Report\n\nDate: {datetime.now()}\n"]
    best_acc = 0
    best_model = None
    best_name = ""

    for m in selected:
        m = m.strip().lower()
        if m not in model_map:
            continue

        print(f"\nTraining {m.upper()}")
        model = model_map[m]
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # ✅ PRINT ACCURACY
        print(f"{m.upper()} Accuracy: {acc:.4f}")

        report.append(f"## Model: {m.upper()}")
        report.append(f"**Accuracy:** {acc:.4f}\n")
        report.append("```")
        report.append(classification_report(y_test, preds))
        report.append("```")

        # Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {m.upper()}")
        cm_path = f"outputs/visualizations/cm_{m}.png"
        plt.savefig(cm_path)
        plt.close()
        report.append(f"![CM](../visualizations/cm_{m}.png)\n")

        # ROC
        if hasattr(model, "predict_proba"):
            roc_path = plot_roc(y_test, model.predict_proba(X_test), m.upper(), "outputs/visualizations")
            if roc_path:
                report.append(f"![ROC](../visualizations/{os.path.basename(roc_path)})\n")

        # Feature Importance
        feat_path = plot_features(model, vectorizer, m.upper(), "outputs/visualizations")
        if feat_path:
            report.append(f"![Features](../visualizations/{os.path.basename(feat_path)})\n")

        report.append("---\n")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = m

    # Save report
    report_path = f"outputs/reports/report_{datetime.now().strftime('%H%M%S')}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"\nFull report saved to {report_path}")

    # Save best model
    if save_model and best_model:
        model_path = f"outputs/models/best_model_{best_name}.pkl"
        joblib.dump(best_model, model_path)
        print(f"Best model saved to {model_path}")
