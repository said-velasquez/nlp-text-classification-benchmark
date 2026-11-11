# NLP Text Classification Benchmark

Classic NLP classification pipelines compared on three real datasets: IMDb movie reviews (sentiment), HuffPost news headlines (topic) and COVID-19 tweets (fine-grained sentiment). Every pipeline uses Bag-of-Words or TF-IDF features with scikit-learn models, and every model is evaluated with accuracy, precision, recall, F1 and a confusion matrix.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

## Results

| Dataset | Task | Model | Features | Test accuracy |
|---|---|---|---|---|
| IMDb (50,000 reviews) | Binary sentiment | Logistic Regression | Bag-of-Words | **88.4%** |
| IMDb (50,000 reviews) | Binary sentiment | Multinomial Naive Bayes | Bag-of-Words | 84.7% |
| HuffPost News (~13,000 headlines) | 3 topics | Logistic Regression | TF-IDF | **85.9%** |
| HuffPost News | 3 topics | Random Forest | TF-IDF | 83.5% |
| HuffPost News | 3 topics | K-Nearest Neighbors | TF-IDF | 48.3% |
| COVID-19 tweets (~45,000) | 5-class sentiment | Logistic Regression | TF-IDF | **56.0%** |
| COVID-19 tweets | 5-class sentiment | Random Forest | TF-IDF | 51.5% |
| COVID-19 tweets | 5-class sentiment | Multinomial Naive Bayes | TF-IDF | 45.6% |

**Takeaways**

- Logistic Regression was the strongest model on all three datasets.
- KNN collapses on sparse TF-IDF vectors and predicts almost everything as the majority class.
- Fine-grained sentiment (5 classes) is much harder than binary sentiment: most confusion is between an extreme class and its milder neighbour, such as *Extremely Positive* and *Positive*.

Trained models and vectorizers are saved with `pickle` so they can be reused without retraining.

## Notebooks

| Notebook | Content |
|---|---|
| [`01_imdb_sentiment_logistic_regression`](notebooks/01_imdb_sentiment_logistic_regression.ipynb) | HTML cleaning, Bag-of-Words, Logistic Regression, confusion matrix |
| [`02_imdb_sentiment_naive_bayes`](notebooks/02_imdb_sentiment_naive_bayes.ipynb) | Bag-of-Words with 10,000 features, Multinomial Naive Bayes, model saved with pickle |
| [`03_news_topic_classification`](notebooks/03_news_topic_classification.ipynb) | SPORTS / TECH / BUSINESS headlines, 3 models compared in a bar chart |
| [`04_covid_tweet_sentiment`](notebooks/04_covid_tweet_sentiment.ipynb) | 5 sentiment classes, Naive Bayes vs Logistic Regression vs Random Forest |

## Run it

```bash
git clone https://github.com/said-velasquez/nlp-text-classification-benchmark.git
cd nlp-text-classification-benchmark
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Download the datasets first and put them in `data/` (see [`data/README.md`](data/README.md)).

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, pickle, Jupyter.

## Author

**Said Velásquez**, Computer Engineering student based in Madrid.
[LinkedIn](https://www.linkedin.com/in/said-velasquez) · [GitHub](https://github.com/said-velasquez)
