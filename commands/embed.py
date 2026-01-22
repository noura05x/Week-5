import click
import pandas as pd
import joblib
import os
import numpy as np
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

    df = pd.read_csv(csv_path)
    texts = df[text_col].fillna('').astype(str)

    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)

    joblib.dump({'matrix': X, 'vectorizer': vectorizer, 'type': 'tfidf'},output)
    print(f"TF-IDF saved → {output}, shape={X.shape}")
@embed.command()
@click.option('--csv_path', required=True)
@click.option('--text_col', required=True)
@click.option('--model_name', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
@click.option('--batch_size', default=32)
@click.option('--output', required=True)
def bert(csv_path, text_col, model_name, batch_size, output):
    from sentence_transformers import SentenceTransformer
    ensure_dir(output)
    df = pd.read_csv(csv_path)
    texts = df[text_col].fillna('').astype(str).tolist()

    model = SentenceTransformer(model_name)
    X = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    joblib.dump({'matrix': X, 'model_name': model_name, 'type': 'bert'},output)
    print(f"BERT embeddings saved → {output}, shape={X.shape}")
@embed.command()
@click.option('--csv_path', required=True)
@click.option('--text_col', required=True)
@click.option('--model_name',default='JadwalAlmaa/model2vec-ARBERTv2')
@click.option('--batch_size', default=32)
@click.option('--output', required=True)
def model2vec(csv_path, text_col, model_name, batch_size, output):
    from sentence_transformers import SentenceTransformer
    ensure_dir(output)

    df = pd.read_csv(csv_path)
    texts = df[text_col].fillna('').astype(str).tolist()

    print(f"Loading Model2Vec model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Encoding with Model2Vec (ARBERTv2)...")
    X = model.encode(texts,batch_size=batch_size,show_progress_bar=True)
    joblib.dump({'matrix': X, 'model_name': model_name, 'type': 'model2vec'},output)
    print(f"Model2Vec embeddings saved → {output}, shape={X.shape}")
