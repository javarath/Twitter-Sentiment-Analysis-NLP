# NLP on Twitter Sentiment Analysis

This repository contains code and resources related to **Twitter Sentiment Analysis** using natural language processing (NLP) techniques. The goal is to predict the sentiment (positive, negative, or neutral) of tweets related to major U.S. airlines.

## Table of Contents

1. [Dataset Description](#dataset-description)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Building](#model-building)
4. [Dependencies](#dependencies)
5. [References](#references)

## Dataset Description

- **Dataset Source**: The dataset was scraped from Twitter in February 2015 and consists of over 14,000 tweets related to six major U.S. airlines: United, American, Southwest, Delta, US Airways, and JetBlue.
- Each tweet is labeled with one of the following sentiments:
    - **Positive**: Indicates a positive sentiment.
    - **Negative**: Indicates a negative sentiment.
    - **Neutral**: Indicates a neutral sentiment.
- Contributors also categorized negative reasons (such as "late flight" or "rude service") for negative tweets.
- Certainly! Here's a concise table summarizing the key features of the **Twitter US Airline Sentiment Analysis dataset**:

| **Feature**                  | **Description**                                                                                   |
|--------------------------|-----------------------------------------------------------------------------------------------|
| `tweet_id`                      | Unique identifier for each tweet.                                                            |
| `airline_sentiment`       | Sentiment label: "positive," "negative," or "neutral."                           |
| `airline_sentiment_confidence` | Confidence score for sentiment label. Ranges from 0 to 1.                        |
| `negativereason`            | Reason for negative sentiment (if applicable).                                       |
| `negativereason_confidence` | Confidence score for negative reason (if provided).                                  |
| `airline`                         | Mentioned airline (e.g., United, American, Southwest).                           |
| `name`                           | Twitter username of the tweet author.                                                    |
| `retweet_count`              | Number of retweets for the tweet.                                                          |
| `text`                             | Actual content of the tweet.                                                                      |
| `tweet_created`              | Timestamp when the tweet was created.                                                    |

- After removing the un-important features, our dataset looks like:
![image](https://github.com/javarath/Twitter-Sentiment-Analysis-NLP/assets/102171533/e57a6d20-cc02-4e20-bc7d-cd2d5d75abe9)
Now, we will start actual pre-processing.

## Data Preprocessing

1. **Tokenization**:
    - Tokenization breaks down the text into individual words or subword units (tokens). It's essential because most NLP algorithms operate at the word level.

2. **Lowercasing**:
    - Converting all text to lowercase ensures consistency in word representations.

3. **Stop Word Removal**:
    - Removing common words (stop words) that don't carry significant meaning.

4. **Lemmatization**:
In natural language processing (NLP), **lemmatization** is a technique where a possibly inflected word form is transformed to yield a lemma. The **WordNet Lemmatizer** is one such method that plays a crucial role in NLP tasks. Let's explore what it does and how it works:

## What is Lemmatization?

- **Lemmatization** groups together different inflected forms of a word to analyze them as a single item.
- Unlike stemming (which simply chops off word endings), lemmatization considers a language's full vocabulary to apply morphological analysis to words.
- The goal is to remove inflectional endings while returning the base or dictionary form of a word, known as the **lemma**.

## WordNet Lemmatizer:

1. **WordNet**:
    - WordNet is a publicly available lexical database that provides semantic relationships between words in over 200 languages.
    - It links words into semantic relations, such as synonyms.
    - WordNet groups synonyms into **synsets** (groups of semantically equivalent words).

2. **How WordNet Lemmatizer Works**:
    - Given an input word, the WordNet Lemmatizer:
        - Looks up the word in WordNet.
        - Determines the corresponding lemma (base form) based on the word's semantic relationships.
        - Considers the context and part of speech (POS) to choose the appropriate lemma.
    - For example:
        - Original Word: "meeting" → Root Word (Lemma): "meet" (core-word extraction)
        - Original Word: "was" → Root Word (Lemma): "be" (tense conversion to present tense)
        - Original Word: "mice" → Root Word (Lemma): "mouse" (plural to singular)

3. **Using WordNet Lemmatizer**:
    - To use the WordNet Lemmatizer in Python:
        - Download the NLTK package (if not already installed).
        - Import the `WordNetLemmatizer` class from NLTK.
        - Apply lemmatization to your text data.

# Why Lemmatization Matters:
- Lemmatization is more powerful than stemming because it considers the full vocabulary and aims to retain meaningful base forms.
- It helps maintain context and semantic relationships between words.
- Lemmatized text is often more interpretable and useful for downstream NLP tasks.
- Sample Text post-lemmatization:
- ![image](https://github.com/javarath/Twitter-Sentiment-Analysis-NLP/assets/102171533/3e4e39c4-e72d-44e1-916f-914fc1e93f99)


5. **Cleaning**:
    - Removing special characters, URLs, and other noise ensures meaningful content.

## Model Building

1. **Train-Test Split**:
    - The preprocessed data was split into training and testing sets.

2. **SMOTE (Synthetic Minority Oversampling Technique)**:
    - To address class imbalance (especially for the negative class), SMOTE was applied to oversample the minority class (positive and negative sentiments).
    - Before Applying SMOTE, our training data looked like:
    - Certainly! Here's a structured table summarizing the distribution of sentiments in the dataset:

| **Sentiment** | **Count (airline_sentiment)** | 
|---------------|-------------------------------|
| Negative      | 6851                          |
| Neutral       | 2327                          |
| Positive      | 1802                          |     

It clearly shows that our classes are highly imbalanced due to which, our model, if fit on this, it will definitely be biased towards negative sentiments. To prevent that, we usually use SMOTE.

## What is SMOTE?

- **SMOTE** is an oversampling technique designed to address the class imbalance problem.
- It aims to balance the class distribution by creating synthetic examples for the minority class.
- Unlike simple duplication (which doesn't add new information), SMOTE generates new examples by interpolating between existing minority class samples.

## How SMOTE Works:

- **Synthesizing New Examples**:
    - Given a minority class example, SMOTE selects its k nearest neighbors.
    - It creates new synthetic examples by interpolating between the original example and its neighbors.
    - These synthetic examples lie along the line connecting the original example and its neighbors.

- **Benefits of SMOTE**:
    - Provides additional information to the model by introducing new data points.
    - Helps the model learn the decision boundary more effectively for the minority class.

## Using SMOTE:

- **Implementation**:
    - In Python, you can use libraries like `imbalanced-learn` to apply SMOTE.
    - It's essential to apply SMOTE only to the training data (not the testing data).

- **Impact on Model Performance**:
    - SMOTE improves the model's ability to correctly classify the minority class.
    - However, it may lead to overfitting if not used carefully.

3. **Using TF-IDF Vectorizer as the values of the features for each tweet**:
# Understanding TF-IDF (Term Frequency-Inverse Document Frequency)

In natural language processing (NLP), **TF-IDF** (Term Frequency-Inverse Document Frequency) is a powerful technique for measuring the importance of a word in a document or a collection (corpus). 

## What is TF-IDF?

- **TF-IDF** stands for **Term Frequency-Inverse Document Frequency**.
- It quantifies how relevant a word is to a specific text within a series or a corpus.
- The importance of a word increases proportionally to its frequency in the text but is compensated by the word's frequency in the entire corpus.

## Key Components:

1. **Term Frequency (TF)**:
    - Represents the number of instances of a given word in a document.
    - Calculated as: `TF(t, d) = (count of t in d) / (total words in d)`
    - It becomes more relevant when a word appears frequently in the text.

2. **Document Frequency (DF)**:
    - Measures how important a word is across the entire corpus.
    - Calculated as: `DF(t) = number of documents containing term t`
    - It depends on the entire collection of documents.

3. **Inverse Document Frequency (IDF)**:
    - Reflects how relevant the word is.
    - Calculated as: `IDF(t) = log(N / DF(t))`
        - `N`: Total number of documents in the corpus.
        - `DF(t)`: Document frequency of term t (number of documents containing the term).

## Computation:

- TF-IDF assigns a weight to each word in a document based on its TF and IDF scores.
- Words with higher TF-IDF scores are considered more significant.


4.. **Multinomial Naive Bayes Classifier**:
    - A Multinomial Naive Bayes model was trained on the resampled training data.
    - The model achieved an accuracy of approximately 73.7% on the testing set.
## Metrics
![image](https://github.com/javarath/Twitter-Sentiment-Analysis-NLP/assets/102171533/17de2c43-ba61-4ef8-9433-24eebfb3851b)
![image](https://github.com/javarath/Twitter-Sentiment-Analysis-NLP/assets/102171533/d72c1e82-b795-472b-931b-31cc7d46122c)


## Dependencies

- Python 3.6+
- Libraries: NLTK, scikit-learn, pandas, numpy

## References

1. [Twitter US Airline Sentiment Analysis Dataset](https://paperswithcode.com/dataset/twitter-us-airline-sentiment)
2. [NLTK Documentation](https://www.nltk.org/)
3. [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

Feel free to explore the code and experiment further with different models and features! 📊🔍
