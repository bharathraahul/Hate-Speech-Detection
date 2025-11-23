# Hate Speech Detection API

A FastAPI-based hate speech detection system with intelligent text masking capabilities.

## Features

- **Hate Speech Detection**: ML-based classification using Logistic Regression with TF-IDF features
- **Hybrid Detection**: Combines ML model with rule-based pattern matching for improved accuracy
- **Smart Text Masking**: Masks offensive words while preserving context (reader-safe)
- **Multiple ML Algorithms**: Supports Logistic Regression, Naive Bayes, SVM, and Random Forest
- **RESTful API**: Easy-to-use endpoints for detection and masking

## Installation

```bash
pip install -r requirement.txt
```

## Usage

### Start the Server

```bash
python main.py
```

The server will start on `http://localhost:8000`

### API Endpoints

#### Detect Hate Speech
```bash
POST /predict
{
  "text": "your text here"
}
```

#### Mask Hate Speech
```bash
POST /mask
{
  "text": "your text here",
  "mask_char": "[REDACTED]"  # optional
}
```

**Response:**
```json
{
  "original_text": "...",
  "masked_text": "...",
  "is_hate_speech": true,
  "hate_speech_confidence": 0.95,
  "was_masked": true,
  "masked_words": [...],
  "words_masked": 2
}
```

### Smart Masking Rules

The masking system uses intelligent rules to preserve context:

- **≤4 letters**: Mask last 2 letters (e.g., "hoes" → "ho**")
- **5 letters**: Show first 2, mask last 3 (e.g., "trash" → "tr***")
- **>5 letters**: Show first 2 and last letter, mask middle (e.g., "bitches" → "bi****s")

## Project Structure

```
Hate-Speech-Detection/
├── main.py                 # FastAPI application
├── ml_service.py           # ML training and prediction
├── masking_service.py     # Text masking functionality
├── hate_speech_patterns.py # Rule-based pattern matching
├── models.py              # Pydantic request/response models
├── config.py              # Configuration settings
├── global_state.py        # Global application state
└── requirement.txt        # Dependencies
```

## Features in Detail

### Hate Speech Detection
- Uses scikit-learn with TF-IDF vectorization
- Supports multiple algorithms (Logistic Regression, Naive Bayes, SVM, Random Forest)
- Includes rule-based pattern matching as fallback
- Trained on Twitter hate speech dataset

### Text Masking
- Only masks text when hate speech is detected
- Preserves context by showing partial word structure
- Customizable mask characters
- Returns detailed information about masked words

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing

Test the masking feature:
```bash
python test_mask.py
```

Or use curl:
```bash
curl -X POST "http://localhost:8000/mask" \
  -H "Content-Type: application/json" \
  -d '{"text": "test text"}'
```

## License

This project is for educational purposes.

