# News Topic Modeling and Trend Detection

Analysis of news headlines through unsupervised topic modeling, supervised category classification, and trend detection over time. The project uses **BERTopic** to discover latent topics in a collection of news headlines, then builds a classifier to predict categories and analyzes how topics evolve across the years.

Dataset: [News Category Dataset (HuffPost)](https://www.kaggle.com/datasets/rmisra/news-category-dataset).

## Contents

- [Overview](#overview)
- [Data](#data)
- [Architecture / Pipeline](#architecture--pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Visualizations](#visualizations)
- [Repository Structure](#repository-structure)
- [Technologies](#technologies)

## Overview

The project consists of three main parts:

1. **Topic Modeling (unsupervised)** — using BERTopic to automatically group headlines into topics via sentence embeddings, dimensionality reduction, and clustering.
2. **Supervised Learning** — classification of news categories with logistic regression on embeddings, both with and without fine-tuning the embedding model.
3. **Trend Detection** — analysis of how topic prevalence changes over time (topics over time).

## Data

The project uses `News_Category_Dataset_v3.json` with roughly **209,527** headlines. For the analysis:

- The data is shuffled (`sample(frac=1, random_state=42)`) and a subset of **35,000** records is taken for computational efficiency.
- The `link` column is dropped.
- A new `text` field is created by concatenating `headline` and `short_description`.
- Similar categories are merged (e.g. `ARTS`, `CULTURE & ARTS` → `ARTS & CULTURE`; `PARENTING`, `PARENTS` → `PARENTS & FAMILY`; `THE WORLDPOST`, `WORLDPOST` → `WORLD NEWS`).

Dataset fields: `link`, `headline`, `category`, `short_description`, `authors`, `date`.

## Architecture / Pipeline

### Topic Modeling

| Step | Tool | Configuration |
|------|------|---------------|
| Embeddings | SentenceTransformer | `all-MiniLM-L6-v2` |
| Dimensionality reduction | UMAP | `n_neighbors=15`, `n_components=5`, `min_dist=0.0`, `metric=cosine` |
| Clustering | HDBSCAN | `min_cluster_size=15`, `metric=euclidean`, `cluster_selection_method=eom` |
| Tokenization | CountVectorizer | `stop_words=english`, `min_df=2`, `ngram_range=(1,2)` |

Additional operations: updating n-grams, merging specific topics (`merge_topics`), and reducing the number of topics (`reduce_topics`).

### Supervised Learning

- Embeddings from `all-MiniLM-L6-v2`.
- Train/test split (80/20, stratified).
- Classifier: `LogisticRegression(max_iter=1000)`.
- **Fine-tuning**: further training of the sentence transformer with `SoftmaxLoss` (7 epochs, `lr=2e-5`, `batch_size=32`), then retraining the classifier with `class_weight='balanced'`.

### Trend Detection

- Conversion of `date` to datetime.
- `topics_over_time` with `nr_bins=100`.
- Visualization of trends across the full period and filtering by a specific year (e.g. 2017).

## Installation

```bash
pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn pandas plotly
```

> A GPU-enabled environment (e.g. Google Colab) is recommended due to embedding generation and fine-tuning.

## Usage

1. Place `News_Category_Dataset_v3.json` in an accessible location (e.g. Google Drive if using Colab).
2. Open the notebook and run the cells in order:
   - Loading and preprocessing the data
   - Generating embeddings
   - Training the BERTopic model
   - Classification and fine-tuning
   - Trend analysis

```python
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

data = pd.read_json('News_Category_Dataset_v3.json', lines=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)[:35000]
data['text'] = data['headline'] + '. ' + data['short_description']

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(data['text'], show_progress_bar=True)

topic_model = BERTopic(embedding_model=embedding_model, verbose=True)
topics, probs = topic_model.fit_transform(data['text'], embeddings)
```

## Results

Category classification based on embeddings and logistic regression:

| Model | Accuracy |
|-------|----------|
| Base embeddings + Logistic Regression | **62.63%** |
| Fine-tuned embeddings + Logistic Regression | **78.13%** |

Fine-tuning the embedding model led to a substantial improvement (≈ +15.5 percentage points), especially for categories such as `STYLE & BEAUTY`, `TRAVEL`, `HOME & LIVING`, and `SPORTS`, where F1 scores exceed 0.85.

BERTopic discovered **over 230 topics** (including the `-1` outlier topic), with clearly interpretable groups such as:

- Travel / hotels / destinations
- Recipes / food / cooking
- Hillary Clinton / Sanders / elections
- LGBT / transgender rights

## Visualizations

The project includes several interactive visualizations via BERTopic and Plotly:

- `visualize_topics()` — topic map
- `visualize_barchart()` — top keywords per topic
- `visualize_heatmap()` — topic similarity
- `visualize_documents()` — documents in 2D space (UMAP reduction)
- `visualize_distribution()` — topic distribution within a document
- `visualize_topics_over_time()` — trends over time

## Repository Structure

```
news-topic-modeling-and-trend-detection/
├── README.md
├── News_Topic_Modeling_and_Trend_Detection.ipynb
└── data/
    └── News_Category_Dataset_v3.json
```

## Technologies

- **Python**
- **BERTopic** — topic modeling
- **Sentence Transformers** (`all-MiniLM-L6-v2`) — embeddings and fine-tuning
- **UMAP** — dimensionality reduction
- **HDBSCAN** — clustering
- **scikit-learn** — logistic regression, evaluation
- **pandas** — data processing
- **Plotly** — interactive visualizations
