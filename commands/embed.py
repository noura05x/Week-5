import click
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer

def ensure_dir(path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

@click.group()
def embed():
    pass

@embed.command()
@click.option('--csv_path', required=True)
@click.option('--text_col', required=True)
@click.option('--max_features', default=5000)
@click.option('--output', required=True)
def tfidf(csv_path, text_col, max_features, output):
    ensure_dir(output)
    print("Loading data")
    df = pd.read_csv(csv_path)
    df[text_col] = df[text_col].fillna('')
    
    print("Vectorizing (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df[text_col])
    
    data = {'matrix': X, 'vectorizer': vectorizer, 'type': 'tfidf'}
    joblib.dump(data, output)
    print(f"Saved TF-IDF embeddings to {output} shape:{X.shape}")

@embed.command()
@click.option('--csv_path', required=True)
@click.option('--text_col', required=True)
@click.option('--model_name', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
@click.option('--batch_size', default=32)
@click.option('--output', required=True)
def bert(csv_path, text_col, model_name, batch_size, output):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: Please run 'pip install sentence-transformers'")
        return

    ensure_dir(output)
    print("Loading data")
    df = pd.read_csv(csv_path)
    sentences = df[text_col].fillna('').astype(str).tolist()
    
    print(f"Loading BERT Model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("Encoding sentences with BERT (This may take a while)...")
    X = model.encode(sentences, batch_size=batch_size, show_progress_bar=True)
    
    data = {'matrix': X, 'model_name': model_name, 'type': 'bert'}
    joblib.dump(data, output)
    print(f"Saved BERT embeddings to {output} shape:{X.shape}")