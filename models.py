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
    algorithm: str = "naive_bayes"
    test_size: float = 0.2

class TestResult(BaseModel):
    text: str
    true_label: int
    predicted_label: int
    is_correct: bool
    confidence: float