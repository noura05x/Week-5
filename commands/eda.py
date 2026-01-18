import click
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

@click.group()
def eda():
    pass

@eda.command()
@click.option('--csv_path', required=True, help='Path to CSV file')
@click.option('--label_col', required=True, help='Column name for classes')
@click.option('--plot_type', default='pie', type=click.Choice(['pie', 'bar']))
def distribution(csv_path, label_col, plot_type):
    try:
        df = pd.read_csv(csv_path)
        output_dir = 'outputs/visualizations'
        ensure_dir(output_dir)
        plt.figure(figsize=(10, 6))
        counts = df[label_col].value_counts()
        
        if plot_type == 'pie':
            plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
            plt.title(f'Distribution: {label_col}')
        else:
            sns.barplot(x=counts.index, y=counts.values)
            plt.title(f'Distribution: {label_col}')
            
        save_path = os.path.join(output_dir, f'dist_{plot_type}.png')
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
        print("\nCounts", counts)
        plt.close()
        
    except Exception as e:
        print(f"Error: {e}")

@eda.command()
@click.option('--csv_path', required=True)
@click.option('--text_col', required=True)
@click.option('--unit', default='words', type=click.Choice(['words', 'chars']))
def histogram(csv_path, text_col, unit):
    df = pd.read_csv(csv_path)
    output_dir = 'outputs/visualizations'
    ensure_dir(output_dir)
    
    plt.figure(figsize=(10, 6))
    if unit == 'words':
        lengths = df[text_col].astype(str).apply(lambda x: len(x.split()))
    else:
        lengths = df[text_col].astype(str).apply(len)
        
    sns.histplot(lengths, bins=50, kde=True)
    plt.title(f'Text Length Distribution ({unit})')
    
    save_path = os.path.join(output_dir, f'hist_{unit}.png')
    plt.savefig(save_path)
    print(f"Histogram saved to: {save_path}")
    plt.close()



@eda.command()
@click.option('--csv_path', required=True, help='Path to input CSV')
@click.option('--text_col', required=True, help='Column containing text')
@click.option('--method', default='iqr', type=click.Choice(['iqr']), help='Method to detect outliers')
@click.option('--output', required=True, help='Path to save cleaned CSV')
def remove_outliers(csv_path, text_col, method, output):
    try:
        df = pd.read_csv(csv_path)
        original_len = len(df)
        print(f"Original Data Size: {original_len} rows")
        
        df['word_count'] = df[text_col].fillna('').astype(str).apply(lambda x: len(x.split()))
        
        output_dir = 'outputs/visualizations'
        ensure_dir(output_dir)
        
        plt.figure(figsize=(10, 2))
        sns.boxplot(x=df['word_count'], color='black')
        plt.title('Text Length Distribution (Before Removing Outliers)')
        plt.xlabel('Word Count')
        plt.savefig(os.path.join(output_dir, 'outliers_boxplot_before.png'))
        plt.close()
        
        if method == 'iqr':
            Q1 = df['word_count'].quantile(0.25)
            Q3 = df['word_count'].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            if lower_bound < 1:
                lower_bound = 1
                
            print(f"IQR Thresholds: Lower={lower_bound:.1f}, Upper={upper_bound:.1f} words")
            

            clean_df = df[(df['word_count'] >= lower_bound) & (df['word_count'] <= upper_bound)].copy()
            
            removed_count = original_len - len(clean_df)
            print(f"Removed {removed_count} outliers ({(removed_count/original_len)*100:.1f}%)")
            
            clean_df.drop(columns=['word_count'], inplace=True)
            clean_df.to_csv(output, index=False)
            print(f"Saved cleaned data to {output}")
            
    except Exception as e:
        print(f"Error: {e}")
