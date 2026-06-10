# spam-detector-nlp
Email Spam Detection NLP Project

GitHub Repository:
https://github.com/addyyadiyy/spam-detector-nlp

Streamlit Application:
https://spam-detector-nlp.streamlit.app/

# Email Spam Detector

## Project Overview

This project develops an Email Spam Detection System using Natural Language Processing (NLP) and Machine Learning techniques. The system automatically classifies messages as either Spam or Ham (Not Spam) based on their textual content.

The project demonstrates the complete NLP workflow, including text preprocessing, feature extraction, model training, evaluation, and comparison of multiple machine learning approaches.

## Dataset

SMS Spam Collection Dataset

SMS Spam Collection Dataset
Total Messages: 5,572
Ham Messages: 4,825
Spam Messages: 747

The dataset contains real SMS messages labeled as either spam or ham and is widely used for text classification research.

## NLP Techniques

### Text Preprocessing

- Lowercase conversion
- Removal of punctuation and special characters
- Tokenization
- Stopword removal
- Stemming using Porter Stemmer

### Feature Extraction

1. Bag of Words (BoW)
2. TF-IDF

### Classification Models

1. Naive Bayes
2. K-Nearest Neighbors (KNN)

## Model Comparison

BoW + Naive Bayes
BoW + KNN
TF-IDF + Naive Bayes
TF-IDF + KNN

## Performance was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

## Visualizations

* Spam vs Ham Distribution
* Spam Word Cloud
* Ham Word Cloud
* Top 20 Spam Indicator Words
* Model Accuracy Comparison
* Confusion Matrix of Best Model
* Performance Metrics Comparison (Accuracy, Precision, Recall, F1-Score)
