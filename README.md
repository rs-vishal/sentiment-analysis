# Sentiment Analysis Project

## Overview

This project aims to build a sentiment analysis model that can automatically classify text as positive, negative, or neutral. The model uses natural language processing (NLP) techniques to understand and predict the sentiment of a given input, such as tweets, reviews, or any other form of text data.

**Demo**: [Sentiment Analysis](https://sentiment-analysis00.streamlit.app)

## Introduction

Sentiment analysis, also known as opinion mining, is a process of determining the emotional tone behind a series of words. It is widely used in applications such as customer feedback analysis, social media monitoring, and market research.

This project focuses on creating a robust model that can identify sentiment with high accuracy using state-of-the-art NLP techniques.

## Features

- **Preprocessing**: Cleans the text data by removing stopwords, punctuation, and performing tokenization.
- **Modeling**: Trains a machine learning model using various algorithms like Logistic Regression, Naive Bayes.
- **Evaluation**: Evaluates the model using various metrics such as accuracy, precision, recall, and F1-score.
- **Prediction**: Allows for real-time sentiment prediction on new text inputs.

## Dataset

The dataset used for training and testing the model consists of labeled text data with corresponding sentiment labels.

- **Source**: [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Description**: The dataset includes various text samples with annotations for sentiment (positive, negative).
- **Size**: The dataset contains approximately 1.6 million samples.

## Model

The model is built using Python and popular machine learning libraries such as Scikit-Learn.

## Installation

To run this project locally, follow these steps:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/rs-vishal/Sentimental-analysis.git
   cd sentiment-analysis
2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate
3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Start**:
  ```bash
 streamlit run app.py
 ```


    

