import json
import random
from typing import List
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Import custom modules
from data_preprocessing import preprocess_text, load_intents

def prepare_dataset(intents_path: str) -> pd.DataFrame:
    intents_data = load_intents(intents_path)
    records = []
    for intent in intents_data['intents']:
        tag = intent['tag']
        for pattern in intent['patterns']:
            preprocessed = preprocess_text(pattern)
            records.append({
                'text': pattern,
                'cleaned_text': preprocessed['cleaned_text'],
                'lemmatized_tokens': ' '.join(preprocessed['lemmatized_tokens']),
                'tag': tag
            })
    return pd.DataFrame(records)

def text_pipeline(texts: List[str]) -> List[str]:
    outputs = []
    for text in texts:
        preprocessed = preprocess_text(text)
        outputs.append(' '.join(preprocessed['lemmatized_tokens']))
    return outputs

def train_model(intents_path: str, out_dir: str = 'models'):
    df = prepare_dataset(intents_path)
    X_raw = text_pipeline(df['text'].tolist())
    y = df['tag'].tolist()

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    decisiontree_model = DecisionTreeClassifier()
    decisiontree_model.fit(X_train, y_train)

    naivebayes_model = MultinomialNB()
    naivebayes_model.fit(X_train, y_train)

    ypred_decisiontree = decisiontree_model.predict(X_test)
    ypred_naivebayes = naivebayes_model.predict(X_test)

    print("Naive Bayes Accuracy:", accuracy_score(y_test, ypred_naivebayes))
    print(classification_report(y_test, ypred_naivebayes))
    print("Decision Tree Accuracy:", accuracy_score(y_test, ypred_decisiontree))
    print(classification_report(y_test, ypred_decisiontree))

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(decisiontree_model, os.path.join(out_dir, 'decisiontree_model.joblib'))
    joblib.dump(naivebayes_model, os.path.join(out_dir, 'naivebayes_model.joblib'))
    joblib.dump(vectorizer, os.path.join(out_dir, 'vectorizer.joblib'))

if __name__ == '__main__':
    intents_file = 'intents.json'  # Path to your intents JSON file
    train_model(intents_file)
