import click
import pandas as pd
import re
import os

def load_stopwords(filepath='list.txt'):
    if not os.path.exists(filepath):
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'list.txt')
    if not os.path.exists(filepath):
        return set()
    with open(filepath, 'r', encoding='utf-8') as f:
        words = set(line.strip() for line in f if line.strip())
    return words

def normalize_arabic(text):
    if not isinstance(text, str): return str(text)
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

def remove_diacritics(text):
    if not isinstance(text, str): return str(text)
    return re.sub(r'[\u064B-\u065F\u0640]', '', text)

def remove_noise(text):
    if not isinstance(text, str): return str(text)
    text = re.sub(r'(http\S+)|(www\S+)', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text) 
    text = re.sub(r'[0-9]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords_from_text(text, stopwords_set):
    if not isinstance(text, str): return ""
    words = text.split()
    filtered = [w for w in words if w not in stopwords_set]
    return " ".join(filtered)

def _apply_cleaning(df, text_col, mode):
    if mode in ['remove', 'all']:
        print("... Removing noise & diacritics")
        df[text_col] = df[text_col].apply(remove_diacritics).apply(remove_noise)
    
    if mode in ['stopwords', 'all']:
        print("... Removing stopwords")
        stops = load_stopwords('list.txt')
        df[text_col] = df[text_col].apply(lambda x: remove_stopwords_from_text(x, stops))
            
    if mode in ['replace', 'all']:
        print("... Normalizing Arabic")
        df[text_col] = df[text_col].apply(normalize_arabic)
    return df

@click.group()
def preprocess():
    pass

@preprocess.command()
@click.option('--csv_path', required=True)
@click.option('--text_col', required=True)
@click.option('--output', required=True)
def remove(csv_path, text_col, output):
    df = pd.read_csv(csv_path)
    df = _apply_cleaning(df, text_col, 'remove')
    df.to_csv(output, index=False)
    print(f"Saved clean data to {output}")

@preprocess.command()
@click.option('--csv_path', required=True)
@click.option('--text_col', required=True)
@click.option('--output', required=True)
def stopwords(csv_path, text_col, output):
    df = pd.read_csv(csv_path)
    df = _apply_cleaning(df, text_col, 'stopwords')
    df.to_csv(output, index=False)
    print(f"Saved data without stopwords to {output}")

@preprocess.command()
@click.option('--csv_path', required=True)
@click.option('--text_col', required=True)
@click.option('--output', required=True)
def replace(csv_path, text_col, output):
    df = pd.read_csv(csv_path)
    df = _apply_cleaning(df, text_col, 'replace')
    df.to_csv(output, index=False)
    print(f"Saved normalized data to {output}")

@preprocess.command()
@click.option('--csv_path', required=True)
@click.option('--text_col', required=True)
@click.option('--output', required=True)
def all(csv_path, text_col, output):
    df = pd.read_csv(csv_path)
    df = _apply_cleaning(df, text_col, 'all')
    df = df[df[text_col].str.strip().astype(bool)]
    df.to_csv(output, index=False)
    print(f"Saved fully processed data to {output}")