from pydantic import BaseModel
from typing import Optional, List

# --- Request/Response Models ---

class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    text: str
    is_hate_speech: bool
    confidence: float
    label: str

class TrainRequest(BaseModel):
    dataset: str = "twitter_hate"
    custom_url: Optional[str] = None
    text_column: Optional[str] = "text"
    label_column: Optional[str] = "label"
    algorithm: str = "logistic_regression"  # Changed default to better performing model
    test_size: float = 0.2
    use_preprocessing: bool = True  # Enable text preprocessing by default
    tune_hyperparameters: bool = False  # Grid search (slower but better results)

class TestResult(BaseModel):
    text: str
    true_label: int
    predicted_label: int
    is_correct: bool
    confidence: float