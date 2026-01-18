
# NLP CLI WEEKLY PROJECT
# Features

* **Modular Design**: Organized structure using distinct subcommands for each stage.
* **Arabic Preprocessing**: Specialized cleaning for Arabic text (Removing Tashkeel, normalizing Hamza/Alef, removing stopwords).
* **EDA & Visualization**: Generate distribution plots and text length histograms.
* **Statistical Outlier Detection**: Detect and remove statistical outliers (text that is too short or too long) using the IQR method.
* **End-to-End Pipeline**: A single command to run the entire workflow from raw data to trained model.
* **Advanced Evaluation**: Comprehensive reports including **ROC Curves** and **Feature Importance** plots.

## setup
```bash
uv venv -p 3.11
```
Activate the Virtual Environment
```bash
.venv\Scripts\activate
```

install dependencies

```bash
uv pip install -r requirements.txt
```

## EDA Commands
class distribution
```bash
uv run main.py eda distribution --csv_path "dataset.csv" --label_col "label_col"
```
Text Lengths
```bash
uv run main.py eda histogram --csv_path "dataset.csv" --text_col "text_col" --unit words
```

Outlier Removal
```bash
uv run main.py eda remove-outliers --csv_path "dataset.csv" --text_col "text_col" --output "data_no_outliers.csv"
```

## Preprocessing
Clean Arabic text (remove diacritics, stopwords, normalize).
```bash
uv run main.py preprocess all --csv_path "data_no_outliers.csv" --text_col "text_col" --output "cleaned_data.csv"
```

## Embedding
TF-IDF vectors
```bash
uv run main.py embed tfidf --csv_path "cleaned_data.csv" --text_col "text_col" --output "vectors.pkl"
```
BERT
```bash
uv run main.py embed bert --csv_path "cleaned_data.csv" --text_col "text_col" --output "vectors.pkl"
```

## Training & Evaluation
Train models and generate reports.
```bash
uv run main.py train train-models --csv_path "cleaned_data.csv" --input_col "vectors.pkl" --output_col "label_col" --models all --save_model
```


## One-Line Pipeline
 Run the full workflow (Cleaning → Embedding → Training) in a single command.
```bash
uv run main.py pipeline \
  --csv_path "dataset.csv" \
  --text_col "text_col" \
  --label_col "label_col" \
  --preprocessing all \
  --training all \
  --save_report \
  --save_models \
  --output final_results/

