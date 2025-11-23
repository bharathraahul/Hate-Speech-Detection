import pandas as pd
import numpy as np
import requests
import re
from io import StringIO
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix, 
    precision_recall_fscore_support
)

# Local imports
from config import logger
import global_state as state
import hate_speech_patterns


def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text for better feature extraction.
    This improves model accuracy by normalizing the input.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags (keep the word part)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def preprocess_texts(texts: List[str]) -> List[str]:
    """Preprocess a list of texts"""
    return [preprocess_text(text) for text in texts]


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

def train_model(texts, labels, algorithm="naive_bayes", test_size=0.2, use_preprocessing=True, tune_hyperparameters=False):
    """
    Train the model with improved preprocessing and feature engineering.
    
    Args:
        texts: List of text strings
        labels: List of labels (0 or 1)
        algorithm: Model algorithm ('naive_bayes', 'logistic_regression', 'svm', 'random_forest')
        test_size: Proportion of data for testing
        use_preprocessing: Whether to use text preprocessing (recommended)
        tune_hyperparameters: Whether to perform grid search for hyperparameter tuning
    """
    try:
        # Split data into train and test (before preprocessing to avoid data leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
        
        # Create classifier based on algorithm
        if algorithm == "logistic_regression":
            classifier = LogisticRegression(
                max_iter=2000, 
                random_state=42, 
                class_weight="balanced",
                C=1.0,
                solver='lbfgs'
            )
        elif algorithm == "svm":
            classifier = LinearSVC(
                class_weight="balanced",
                random_state=42,
                max_iter=2000,
                C=1.0
            )
        elif algorithm == "random_forest":
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        else:  # naive_bayes
            classifier = MultinomialNB(alpha=1.0)
        
        # Improved TF-IDF vectorizer with better parameters
        tfidf_params = {
            'max_features': 10000,  # Increased from 5000
            'ngram_range': (1, 3),  # Unigrams, bigrams, trigrams
            'min_df': 2,  # Ignore terms that appear in less than 2 documents
            'max_df': 0.95,  # Ignore terms that appear in more than 95% of documents
            'sublinear_tf': True,  # Apply sublinear tf scaling
            'norm': 'l2',  # L2 normalization
            'smooth_idf': True  # Smooth idf weights
        }
        
        # Create pipeline with preprocessing transformer
        # Preprocessing is done in the pipeline so new texts are automatically preprocessed
        steps = []
        if use_preprocessing:
            # Create a transformer that applies preprocessing to each text
            def preprocess_transformer(X):
                if isinstance(X, list):
                    return [preprocess_text(text) for text in X]
                else:
                    return preprocess_text(X)
            
            steps.append(('preprocessor', FunctionTransformer(
                func=lambda X: [preprocess_text(text) if isinstance(text, str) else preprocess_text(str(text)) for text in X],
                validate=False
            )))
        
        steps.extend([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('classifier', classifier)
        ])
        
        pipeline = Pipeline(steps)
        
        # Hyperparameter tuning if requested
        if tune_hyperparameters and algorithm in ["logistic_regression", "svm"]:
            logger.info("Performing hyperparameter tuning...")
            param_grid = {}
            if algorithm == "logistic_regression":
                param_grid = {
                    'classifier__C': [0.1, 1.0, 10.0],
                    'classifier__solver': ['lbfgs', 'liblinear']
                }
            elif algorithm == "svm":
                param_grid = {
                    'classifier__C': [0.1, 1.0, 10.0]
                }
            
            grid_search = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=3, 
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            pipeline = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
        
        # Train
        logger.info(f"Training {algorithm} model...")
        pipeline.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = pipeline.predict(X_test)
        
        # Handle models without predict_proba (like LinearSVC)
        try:
            y_pred_proba = pipeline.predict_proba(X_test)
        except AttributeError:
            # For SVM, use decision_function and convert to probabilities
            try:
                decision_scores = pipeline.decision_function(X_test)
                # Convert to probability-like scores using sigmoid
                y_pred_proba = np.column_stack([
                    1 / (1 + np.exp(decision_scores)),  # Probability of class 0
                    1 / (1 + np.exp(-decision_scores))  # Probability of class 1
                ])
            except:
                # Fallback: use predictions as probabilities
                y_pred_proba = np.column_stack([1 - y_pred, y_pred])
        
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

def predict_single(text: str, preprocess=False, use_pattern_matching=True):
    """
    Make a prediction on a single text string with optional pattern matching.
    
    Args:
        text: Input text to classify
        preprocess: Whether to manually preprocess (usually not needed as pipeline handles it)
        use_pattern_matching: Whether to use rule-based pattern matching as fallback
    """
    if state.model is None:
        raise ValueError("Model is not trained")
    
    # Manual preprocessing only if explicitly requested (pipeline usually handles it)
    if preprocess:
        text = preprocess_text(text)
    
    # Check if model supports predict_proba (SVM doesn't by default)
    try:
        probabilities = state.model.predict_proba([text])[0]
        prediction = state.model.predict([text])[0]
        ml_confidence = float(probabilities[1])
    except AttributeError:
        # For models without predict_proba (like LinearSVC), use decision_function
        prediction = state.model.predict([text])[0]
        try:
            decision_scores = state.model.decision_function([text])[0]
            # Convert decision function to probability-like score using sigmoid
            ml_confidence = 1 / (1 + np.exp(-decision_scores)) if prediction == 1 else 1 / (1 + np.exp(decision_scores))
        except:
            ml_confidence = 0.5  # Default confidence if we can't calculate
    
    # Hybrid approach: Combine ML prediction with pattern matching
    if use_pattern_matching:
        pattern_result = hate_speech_patterns.check_hate_speech_patterns(text)
        pattern_match = pattern_result["pattern_match"]
        pattern_confidence = pattern_result["confidence"]
        
        # If pattern matching detects hate speech, override ML if ML confidence is low
        if pattern_match and pattern_confidence > 0.7:
            if prediction == 0 or ml_confidence < 0.6:
                # Pattern matching is confident, ML is not - use pattern matching
                logger.info(f"Pattern matching overrode ML: pattern_conf={pattern_confidence:.2f}, ml_conf={ml_confidence:.2f}")
                prediction = 1
                # Use weighted average, but favor pattern matching when it's confident
                confidence = max(ml_confidence, pattern_confidence * 0.8)
            else:
                # Both agree or ML is confident - use weighted average
                confidence = (ml_confidence * 0.6 + pattern_confidence * 0.4)
        else:
            # Pattern matching doesn't detect or is uncertain - use ML
            confidence = ml_confidence
    else:
        confidence = ml_confidence
    
    return int(prediction), float(confidence)

def predict_batch(texts: List[str], use_pattern_matching=True):
    """Make predictions on a batch of text strings"""
    if state.model is None:
        raise ValueError("Model is not trained")
    
    results = []
    for text in texts:
        try:
            prediction, confidence = predict_single(text, use_pattern_matching=use_pattern_matching)
            results.append({
                "text": text,
                "is_hate_speech": bool(prediction == 1),
                "confidence": confidence,
                "label": "hate_speech" if prediction == 1 else "normal"
            })
        except Exception as e:
            results.append({"text": text, "error": str(e)})
    return results