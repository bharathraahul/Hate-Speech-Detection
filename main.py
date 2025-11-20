import uvicorn
import numpy as np
from typing import List
from fastapi import FastAPI, HTTPException

# Local imports
from models import TextInput, PredictionOutput, TrainRequest
from config import logger, DATASET_URLS, SAMPLE_DATA
import global_state as state
import ml_service

app = FastAPI(title="Hate Speech Detection API with Train/Test Split")

@app.on_event("startup")
async def startup_event():
    """Train model on startup"""
    logger.info("Loading dataset and training model on startup...")
    texts, labels = ml_service.load_twitter_davidson_dataset(DATASET_URLS["twitter_hate"])
    
    if texts is None:
        logger.warning("Failed to load web data, using sample data")
        texts, labels = SAMPLE_DATA["texts"], SAMPLE_DATA["labels"]
    
    ml_service.train_model(texts, labels, algorithm="logistic_regression", use_preprocessing=True)

@app.get("/")
def root():
    return {
        "message": "Hate Speech Detection API with Train/Test Split",
        "model_metrics": state.model_metrics,
        "endpoints": {
            "/predict": "POST - Predict hate speech for new text",
            "/batch-predict": "POST - Predict multiple texts",
            "/train": "POST - Train model with dataset (algorithms: naive_bayes, logistic_regression, svm, random_forest)",
            "/test-results": "GET - View test set predictions",
            "/test-sample": "GET - View random test samples",
            "/evaluate": "GET - Get detailed evaluation metrics",
            "/model-info": "GET - Get model information",
            "/datasets": "GET - List available datasets",
            "/health": "GET - Health check"
        },
        "improvements": {
            "text_preprocessing": "Enabled by default - cleans URLs, mentions, special chars",
            "better_features": "Improved TF-IDF with 10k features, sublinear scaling",
            "algorithms": "Support for logistic_regression, svm, random_forest, naive_bayes",
            "class_balance": "Automatic class weight balancing for imbalanced datasets",
            "hyperparameter_tuning": "Optional grid search for optimal parameters"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_status": state.model_metrics["status"],
        "model_ready": state.model is not None
    }

@app.get("/model-info")
def get_model_info():
    """Get detailed model information and metrics"""
    return state.model_metrics

@app.get("/evaluate")
def evaluate_model():
    """Get detailed evaluation metrics from test set"""
    if state.model is None or not state.test_data["texts"]:
        raise HTTPException(status_code=503, detail="Model not trained or no test data")
    
    cm = state.model_metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    
    return {
        "metrics": {
            "accuracy": state.model_metrics["accuracy"],
            "precision": state.model_metrics["precision"],
            "recall": state.model_metrics["recall"],
            "f1_score": state.model_metrics["f1_score"]
        },
        "confusion_matrix": {
            "true_negative": cm[0][0],
            "false_positive": cm[0][1],
            "false_negative": cm[1][0],
            "true_positive": cm[1][1]
        },
        "dataset_info": {
            "total_samples": state.model_metrics["total_samples"],
            "train_size": state.model_metrics["train_size"],
            "test_size": state.model_metrics["test_size"]
        }
    }

@app.get("/test-results")
def get_test_results(limit: int = 50, show_incorrect_only: bool = False):
    """View predictions on test set"""
    if not state.test_data["texts"]:
        raise HTTPException(status_code=503, detail="No test data available")
    
    results = []
    for i, (text, true_label, pred_label, prob) in enumerate(
        zip(state.test_data["texts"], state.test_data["labels"], 
            state.test_data["predictions"], state.test_data["probabilities"])
    ):
        is_correct = true_label == pred_label
        
        if show_incorrect_only and is_correct:
            continue
        
        results.append({
            "index": i,
            "text": text,
            "true_label": int(true_label),
            "predicted_label": int(pred_label),
            "true_label_name": "hate_speech" if true_label == 1 else "normal",
            "predicted_label_name": "hate_speech" if pred_label == 1 else "normal",
            "is_correct": is_correct,
            "confidence": float(prob)
        })
        
        if len(results) >= limit:
            break
    
    correct_count = sum(1 for r in results if r["is_correct"])
    
    return {
        "results": results,
        "summary": {
            "total_shown": len(results),
            "correct": correct_count,
            "incorrect": len(results) - correct_count,
            "accuracy": correct_count / len(results) if results else 0
        }
    }

@app.get("/test-sample")
def get_test_sample(sample_size: int = 10):
    """Get random samples from test set"""
    if not state.test_data["texts"]:
        raise HTTPException(status_code=503, detail="No test data available")
    
    total_samples = len(state.test_data["texts"])
    indices = np.random.choice(
        total_samples, 
        min(sample_size, total_samples), 
        replace=False
    )
    
    samples = []
    for i in indices:
        samples.append({
            "text": state.test_data["texts"][i],
            "true_label": "hate_speech" if state.test_data["labels"][i] == 1 else "normal",
            "predicted_label": "hate_speech" if state.test_data["predictions"][i] == 1 else "normal",
            "is_correct": state.test_data["labels"][i] == state.test_data["predictions"][i],
            "confidence": float(state.test_data["probabilities"][i])
        })
    
    return {"samples": samples, "count": len(samples)}

@app.get("/datasets")
def list_datasets():
    """List available datasets"""
    return {
        "available_datasets": {
            "twitter_hate": {
                "description": "Twitter hate speech dataset by Davidson et al.",
                "size": "~24k tweets",
                "url": DATASET_URLS["twitter_hate"]
            },
            "sample": {
                "description": "Sample dataset for testing",
                "size": "30 samples"
            },
            "custom_url": {
                "description": "Load your own CSV from URL",
                "example": "https://example.com/dataset.csv"
            }
        }
    }

@app.post("/train")
async def train_with_dataset(request: TrainRequest):
    """Train the model with specified dataset"""
    state.model_metrics["status"] = "training"
    
    # Load dataset
    if request.dataset == "twitter_hate":
        texts, labels = ml_service.load_twitter_davidson_dataset(DATASET_URLS["twitter_hate"])
    elif request.dataset == "sample":
        texts, labels = SAMPLE_DATA["texts"], SAMPLE_DATA["labels"]
    elif request.dataset == "custom_url" and request.custom_url:
        texts, labels = ml_service.load_csv_from_url(
            request.custom_url, request.text_column, request.label_column
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid dataset selection")
    
    if texts is None or labels is None:
        state.model_metrics["status"] = "training_failed"
        raise HTTPException(status_code=500, detail="Failed to load dataset")
    
    # Train model
    success = ml_service.train_model(
        texts, 
        labels, 
        algorithm=request.algorithm, 
        test_size=request.test_size,
        use_preprocessing=request.use_preprocessing,
        tune_hyperparameters=request.tune_hyperparameters
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Model training failed")
    
    return {
        "message": "Model trained successfully",
        "metrics": state.model_metrics
    }

@app.post("/predict", response_model=PredictionOutput)
def predict_hate_speech(input_data: TextInput):
    """Predict hate speech for new text"""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not trained")
    
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        prediction, confidence = ml_service.predict_single(input_data.text)
        
        return PredictionOutput(
            text=input_data.text,
            is_hate_speech=bool(prediction == 1),
            confidence=confidence,
            label="hate_speech" if prediction == 1 else "normal"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict")
def batch_predict(texts: List[TextInput]):
    """Predict multiple texts"""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not trained")
    
    if len(texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")
    
    input_texts = [t.text for t in texts]
    results = ml_service.predict_batch(input_texts)
    
    return {"results": results, "total": len(results)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)