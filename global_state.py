# --- Global Application State ---

# The trained machine learning model
model = None

# Results from the test set for evaluation
test_data = {
    "texts": [], 
    "labels": [], 
    "predictions": [],
    "probabilities": []
}

# Metrics about the currently trained model
model_metrics = {
    "status": "not_trained",
    "train_size": 0,
    "test_size": 0,
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1_score": 0.0
}