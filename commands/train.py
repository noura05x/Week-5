import click
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_roc_curve(y_test, y_score, n_classes, model_name, output_dir):
    try:
        y_test_bin = label_binarize(y_test, classes=sorted(set(y_test)))
        if n_classes == 2:
            if y_score.ndim == 1:
                y_score = y_score.reshape(-1, 1)
            if y_score.shape[1] == 2:
                y_score = y_score[:, 1].reshape(-1, 1)
            y_test_bin = y_test_bin.reshape(-1, 1)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        plt.figure(figsize=(8, 6))
        
        for i in range(y_test_bin.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (area = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        
        save_path = os.path.join(output_dir, f"roc_{model_name}.png")
        plt.savefig(save_path)
        plt.close()
        return f"roc_{model_name}.png"
    except Exception as e:
        print(f"Could not plot ROC for {model_name}: {e}")
        return None

def plot_feature_importance(model, vectorizer, model_name, output_dir):
    try:
        feature_names = vectorizer.get_feature_names_out()
        importances = None
        
      
        if hasattr(model, 'feature_importances_'): 
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'): 
            importances = np.abs(model.coef_[0]) 
            
        if importances is not None:
            indices = np.argsort(importances)[::-1][:20] 
            top_features = [feature_names[i] for i in indices]
            top_importances = importances[indices]
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x=top_importances, y=top_features, palette="viridis")
            plt.title(f'Top 20 Important Features - {model_name}')
            plt.xlabel('Importance Score')
            
            save_path = os.path.join(output_dir, f"features_{model_name}.png")
            plt.savefig(save_path)
            plt.close()
            return f"features_{model_name}.png"
    except Exception as e:
        print(f"Could not plot Feature Importance for {model_name}: {e}")
    return None

@click.group()
def train():
    pass

@train.command()
@click.option('--csv_path', required=True)
@click.option('--input_col', required=True, help='Path to embeddings (.pkl)')
@click.option('--output_col', required=True, help='Label column name')
@click.option('--models', default='lr', help='Comma separated: knn, lr, rf, all')
@click.option('--save_model', is_flag=True)

def train_models(csv_path, input_col, output_col, models, save_model):
    df = pd.read_csv(csv_path)
    y = df[output_col]
    print(f"Loading embeddings from {input_col}")
    embed_data = joblib.load(input_col)
    X = embed_data['matrix']
    vectorizer = embed_data.get('vectorizer') 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    classes = sorted(list(set(y)))
    n_classes = len(classes)
    model_map = {
        'knn': KNeighborsClassifier(n_neighbors=5),
        'lr': LogisticRegression(max_iter=1000), 
        'rf': RandomForestClassifier(n_estimators=100) }
    selected = ['knn', 'lr', 'rf'] if models == 'all' else models.split(',')
    ensure_dir('outputs/reports')
    ensure_dir('outputs/models')
    ensure_dir('outputs/visualizations')

    report_lines = [f"# Training Report - {datetime.now().strftime('%Y-%m-%d')}\n"]
    best_acc = 0
    best_model = None
    best_name = ""
    
    for m in selected:
        m = m.strip().lower()
        if m not in model_map: continue
        
        print(f"Training {m.upper()}")
        clf = model_map[m]
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        report_lines.append(f"Model: {m.upper()}")
        report_lines.append(f"Accuracy: {acc:.4f}")
        report_lines.append(classification_report(y_test, preds))
      
        plt.figure(figsize=(6,5))
        sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {m.upper()}')
        cm_filename = f"cm_{m}.png"
        plt.savefig(f"outputs/visualizations/{cm_filename}")
        plt.close()
        report_lines.append(f"![Confusion Matrix](../visualizations/{cm_filename})\n")
        
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)
            roc_file = plot_roc_curve(y_test, y_score, n_classes, m.upper(), 'outputs/visualizations')
            if roc_file:
                 report_lines.append(f"![ROC Curve](../visualizations/{roc_file})\n")

        if vectorizer:
            feat_file = plot_feature_importance(clf, vectorizer, m.upper(), 'outputs/visualizations')
            if feat_file:
                report_lines.append(f"![Feature Importance](../visualizations/{feat_file})\n")
        
        report_lines.append("---\n")

        if acc > best_acc:
            best_acc = acc
            best_model = clf
            best_name = m

    report_path = f"outputs/reports/report_{datetime.now().strftime('%H%M%S')}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"Full report saved to {report_path}")
    

    if save_model and best_model:
        m_path = f"outputs/models/best_model_{best_name}.pkl"
        joblib.dump(best_model, m_path)
        print(f"Best model saved to {m_path}")