
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
##
### Dataset

This project uses the Arabic Company Reviews dataset for demonstration.

🔗 **[Download Dataset: Arabic Company Reviews (عربي)](https://www.kaggle.com/datasets/fahdseddik/arabic-company-reviews)**




## EDA Commands
class distribution (pie chart):
```bash
uv run main.py eda distribution --csv_path CompanyReviews.csv --label_col rating 
```
class distribution(bar chart):
```bash
uv run main.py eda distribution --csv_path CompanyReviews.csv --label_col rating --plot_type bar
```

text length histogram (word count):
```bash
uv run main.py eda histogram  --csv_path CompanyReviews.csv --text_col review_description --unit words
```
text length histogram (character count):
```bash
uv run main.py eda histogram --csv_path CompanyReviews.csv --text_col review_description --unit chars 
```

Outlier Removal
```bash
uv run main.py eda remove-outliers --csv_path CompanyReviews.csv --text_col review_description --method iqr --output clean_data.csv
```

## Preprocessing
Remove Arabic-specific characters (tashkeel, tatweel, tarqeem, links, etc.)
```bash
uv run main.py preprocess remove --csv_path CompanyReviews.csv --text_col review_description --output cleaned.csv
```

Remove stopwords using Arabic stopwords list
```bash
uv run main.py preprocess stopwords --csv_path cleaned.csv --text_col review_description --output no_stops.csv
```

Normalize Arabic text (hamza, alef maqsoura, taa marbouta)
```bash
uv run main.py preprocess replace --csv_path no_stops.csv --text_col review_description --output normalized.csv
```

Chain all preprocessing steps
```bash
uv run main.py preprocess all --csv_path CompanyReviews.csv --text_col review_description --output final.csv
```

## Embedding
TF-IDF vectors
```bash
uv run main.py embed tfidf --csv_path cleaned.csv --text_col review_description --max_features 5000 --output tfidf_vectors.pkl
```
Model2Vec Embedding
```bash
uv run main.py embed model2vec --csv_path cleaned.csv --text_col review_description --output model2vec_vectors.pkl
```

BERT
```bash
uv run main.py embed bert --csv_path cleaned.csv --text_col review_description --model_name aubmindlab/bert-base-arabertv2 --output bert_vectors.pkl
```

## Training & Evaluation
### Available Models

| Code  | Model                  |
| ----- | ---------------------- |
| `knn` | K-Nearest Neighbors    |
| `lr`  | Logistic Regression    |
| `rf`  | Random Forest          | 


Train models and generate reports.
```bash
uv run main.py train train-models --csv_path clean_final.csv --input_col vec1.pkl --output_col rating --models all --save_model
```
Training KNN
 Accuracy: 0.6575


Training LR
Accuracy: 0.8268


Training
RF Accuracy: 0.8128
##

## One-Line Pipeline
Run all steps in sequence
```bash
uv run main.py pipeline --csv_path CompanyReviews.csv --text_col review_description --label_col rating --preprocessing "remove,stopwords,replace" --embedding tfidf --training "knn,lr,rf" --output results/
```
thank you <3