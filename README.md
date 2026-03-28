# NLP Preprocessing and Word Frequencies for Sentiment Analysis

This project provides a comprehensive guide and implementation for the foundational steps of Natural Language Processing (NLP) in the context of sentiment analysis, specifically targeting Twitter data. It covers the entire pipeline from raw text preprocessing to building feature-rich word frequency dictionaries.

## Project Structure

- `M1_L1_lecture_nb_01_preprocessing.ipynb`: Detailed walkthrough of tweet preprocessing.
- `M1_L1_lecture_nb_02_word frequencies.ipynb`: Guide on building and visualizing word frequency dictionaries.
- `utils.py`: Contains the core helper functions `process_tweet` and `build_freqs`.

## Getting Started

### Prerequisites

Ensure you have the following Python libraries installed:

- `nltk`
- `numpy`
- `matplotlib`
- `spacy`

You can install them via pip:
```bash
pip install nltk numpy matplotlib
```

### Setup NLTK Data

The notebooks require specific NLTK datasets. Run the following in your Python environment:
```python
import nltk
nltk.download('twitter_samples')
nltk.download('stopwords')
```

## Key Components

### 1. Tweet Preprocessing
The preprocessing pipeline is crucial for cleaning noise from social media data. Key steps include:
- **Regex Cleaning**: Removing URLs, stock market tickers (e.g., `$GE`), retweet marks (`RT`), and hashtag signs.
- **Tokenization**: Converting strings into lists of individual words using `TweetTokenizer`.
- **Lowercasing**: Standardizing text to lower case.
- **Stopword Removal**: Eliminating common words that carry little sentimental weight.
- **Stemming**: Reducing words to their root forms using the Porter Stemmer (e.g., "happier" -> "happi").

### 2. Word Frequency Dictionary
To extract features for machine learning models, we build a dictionary that maps `(word, label)` pairs to their occurrence counts in the corpus.
- **Labeling**: `1` for positive tweets, `0` for negative tweets.
- **Efficiency**: Leveraging Python dictionaries for $O(1)$ lookup times.

## Utils Module

The `utils.py` file provides production-ready implementations of the logic discussed in the notebooks:

- **`process_tweet(tweet)`**: A single function that performs the entire preprocessing pipeline.
- **`build_freqs(tweets, labels)`**: Efficiently constructs the word frequency dictionary from a list of tweets and their corresponding labels.

