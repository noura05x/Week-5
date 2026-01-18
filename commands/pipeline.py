import click
import pandas as pd
import os
import re
import joblib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_stopwords(filepath='list.txt'):
    if not os.path.exists(filepath):
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'list.txt')
    
    if not os.path.exists(filepath):
        return set()
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def clean_text(text, mode='all', stopwords_set=None):
    if not isinstance(text, str): return str(text)
    if mode in ['remove', 'all'] or 'remove' in mode:
        text = re.sub(r'[\u064B-\u065F\u0640]', '', text)
        text = re.sub(r'(http\S+)|(www\S+)', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'[0-9]+', ' ', text)

    if (mode in ['stopwords', 'all'] or 'stopwords' in mode) and stopwords_set:
        words = text.split()
        text = " ".join([w for w in words if w not in stopwords_set])

    if mode in ['replace', 'all'] or 'replace' in mode:
        text = re.sub("[إأآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("گ", "ك", text)
        
    return text.strip()

@click.command()
@click.option('--csv_path', required=True, help='Input CSV file')
@click.option('--text_col', required=True, help='Text column name')
@click.option('--label_col', required=True, help='Label column name')
@click.option('--preprocessing', default='all', help='Steps: remove,stopwords,replace,all')
@click.option('--embedding', default='tfidf', help='Embedding type (currently tfidf)')
@click.option('--training', default='all', help='Models: knn,lr,rf,all')
@click.option('--output', default='results/', help='Output directory')
@click.option('--save_report', is_flag=True, help='Save markdown report')
@click.option('--save_models', is_flag=True, help='Save trained models')
def pipeline(csv_path, text_col, label_col, preprocessing, embedding, training, output, save_report, save_models):
    start_time = datetime.now()
    click.echo(f"Starting Pipeline at {start_time.strftime('%H:%M:%S')}")

    models_dir = os.path.join(output, 'models')
    viz_dir = os.path.join(output, 'visualizations')
    report_dir = os.path.join(output, 'reports')
    ensure_dir(models_dir)
    ensure_dir(viz_dir)
    ensure_dir(report_dir)

    click.echo(f"\n1️Loading and Cleaning Data ({preprocessing})...")
    try:
        df = pd.read_csv(csv_path)
        original_len = len(df)
        
        stops = None
        if 'stopwords' in preprocessing or preprocessing == 'all':
            stops = load_stopwords()
            click.echo(f"   - Loaded {len(stops)} stopwords.")

        df['clean_text'] = df[text_col].apply(lambda x: clean_text(x, preprocessing, stops))

        df = df[df['clean_text'].str.strip().astype(bool)]
        click.echo(f"   - Rows: {original_len} -> {len(df)} (Dropped empty)")
        
        clean_path = os.path.join(output, 'processed_data.csv')
        df.to_csv(clean_path, index=False)
        
    except Exception as e:
        click.echo(f"Error in preprocessing: {e}")
        return

    click.echo(f"\n2️Generating Embeddings ({embedding})")
    
    X = None
    vectorizer = None
    
    if embedding == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['clean_text'])
        click.echo(f"Vector Shape: {X.shape}")
        
        if save_models:
            joblib.dump(vectorizer, os.path.join(models_dir, 'tfidf_vectorizer.pkl'))

    click.echo(f"\n3️Training Models ({training})")
    
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_map = {
        'knn': KNeighborsClassifier(n_neighbors=5),
        'lr': LogisticRegression(max_iter=1000),
        'rf': RandomForestClassifier(n_estimators=100)}
    
    selected_models = ['knn', 'lr', 'rf'] if training == 'all' else training.split(',')
    report_content = [f"# Pipeline Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
    report_content.append(f"Data: {csv_path} | Preprocessing: {preprocessing}\n")
    
    best_acc = 0
    best_model_name = ""
    
    for m in selected_models:
        m_key = m.strip().lower()
        if m_key not in model_map: continue
        
        click.echo(f"Training {m_key.upper()}")
        clf = model_map[m_key]
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
    
        report_content.append(f"{m_key.upper()} Model")
        report_content.append(f"Accuracy: {acc:.4f}")
        report_content.append("\n" + classification_report(y_test, preds) + "\n")
        
        cm_filename = f"cm_{m_key}.png"
        plt.figure(figsize=(5,4))
        sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Greens')
        plt.title(f'CM - {m_key.upper()}')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, cm_filename))
        plt.close()
        
        report_content.append(f"![Confusion Matrix](../visualizations/{cm_filename})\n")
        
        if save_models:
            joblib.dump(clf, os.path.join(models_dir, f"model_{m_key}.pkl"))
            
        if acc > best_acc:
            best_acc = acc
            best_model_name = m_key.upper()

    report_content.append(f"Best Model: {best_model_name} ({best_acc:.4f})")
    
    if save_report:
        report_file = os.path.join(report_dir, 'pipeline_report.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_content))
        click.echo(f"\nReport saved to: {report_file}")

    click.echo(f"\nPipeline Completed in {datetime.now() - start_time}!")