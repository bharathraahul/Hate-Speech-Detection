import pandas as pd
import numpy as np
import requests
from io import StringIO
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix, 
    precision_recall_fscore_support
)

# Local imports
from config import logger
import global_state as state


def load_twitter_davidson_dataset(url: str):
    """Load the Twitter hate speech dataset"""
    try:
        logger.info(f"Downloading dataset from {url}")
        df = pd.read_csv(url)
        
        # Map classes: 0,1 -> hate speech (1), 2 -> normal (0)
        df['binary_label'] = df['class'].apply(lambda x: 1 if x in [0, 1] else 0)
        
        texts = df['tweet'].tolist()
        labels = df['binary_label'].tolist()
        
        logger.info(f"Loaded {len(texts)} samples from Twitter dataset")
        return texts, labels
    except Exception as e:
        logger.error(f"Failed to load Twitter dataset: {e}")
        return None, None

def load_csv_from_url(url: str, text_column: str, label_column: str):
    """Load a CSV dataset from any URL"""
    try:
        logger.info(f"Downloading CSV from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        
        if text_column not in df.columns or label_column not in df.columns:
            raise ValueError(f"Columns {text_column} or {label_column} not found")
        
        texts = df[text_column].astype(str).tolist()
        labels = df[label_column].tolist()
        
        logger.info(f"Loaded {len(texts)} samples from CSV")
        return texts, labels
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return None, None

def train_model(texts, labels, algorithm="naive_bayes", test_size=0.2):
    """Train the model and store test data"""
    try:
        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
        
        # Create pipeline
        if algorithm == "logistic_regression":
            classifier = LogisticRegression(max_iter=1000, random_state=42)
        else:
            classifier = MultinomialNB()
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
            ('classifier', classifier)
        ])
        
        # Train
        logger.info(f"Training {algorithm} model...")
        pipeline.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', zero_division=0
        )
        
        # Store test data and predictions
        state.test_data = {
            "texts": X_test,
            "labels": y_test,
            "predictions": y_pred.tolist(),
            "probabilities": y_pred_proba[:, 1].tolist()
        }
        
        # Update metrics
        state.model_metrics = {
            "status": "trained",
            "algorithm": algorithm,
            "total_samples": len(texts),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Update the global model
        state.model = pipeline
        
        logger.info(f"Model trained successfully!")
        logger.info(f"Accuracy: {accuracy:.2%}")
        logger.info(f"Precision: {precision:.2%}")
        logger.info(f"Recall: {recall:.2%}")
        logger.info(f"F1 Score: {f1:.2%}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
        
        return True
    except Exception as e:
        logger.error(f"Training failed: {e}")
        state.model_metrics["status"] = "training_failed"
        state.model = None
        return False

def predict_single(text: str):
    """Make a prediction on a single text string"""
    if state.model is None:
        raise ValueError("Model is not trained")
        
    probabilities = state.model.predict_proba([text])[0]
    prediction = state.model.predict([text])[0]
    
    # probabilities[1] is the probability of class 1 (hate speech)
    return int(prediction), float(probabilities[1])

def predict_batch(texts: List[str]):
    """Make predictions on a batch of text strings"""
    if state.model is None:
        raise ValueError("Model is not trained")
    
    results = []
    for text in texts:
        try:
            prediction, confidence = predict_single(text)
            results.append({
                "text": text,
                "is_hate_speech": bool(prediction == 1),
                "confidence": confidence,
                "label": "hate_speech" if prediction == 1 else "normal"
            })
        except Exception as e:
            results.append({"text": text, "error": str(e)})
    return results