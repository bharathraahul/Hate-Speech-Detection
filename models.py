from pydantic import BaseModel
from typing import Optional, List, Dict

# --- Request/Response Models ---

class TextInput(BaseModel):
    text: str
    use_pattern_matching: Optional[bool] = True  # Use rule-based pattern matching

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

class MaskRequest(BaseModel):
    text: str
    mask_char: Optional[str] = "[REDACTED]"  # Character/string to use for masking

class MaskResponse(BaseModel):
    original_text: str
    masked_text: Optional[str] = None  # Masked text (None if no hate speech detected)
    is_hate_speech: bool  # Whether hate speech was detected
    hate_speech_confidence: float  # Confidence of hate speech detection
    was_masked: bool  # Whether masking was performed
    masked_words: List[Dict] = []  # List of words that were masked
    words_masked: int = 0  # Number of words masked
    message: str  # Explanation message

# Keep old models for backward compatibility (deprecated)
class ParaphraseRequest(BaseModel):
    text: str
    num_paraphrases: int = 1
    method: Optional[str] = None

class ParaphraseResponse(BaseModel):
    original_text: str
    paraphrases: Optional[List[str]] = None
    method_used: Optional[str] = None
    count: int = 0
    is_hate_speech: bool
    hate_speech_confidence: float
    was_paraphrased: bool
    message: str