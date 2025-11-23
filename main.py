import uvicorn
import numpy as np
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import os
import json

# Local imports
from models import TextInput, PredictionOutput, TrainRequest, ParaphraseRequest, ParaphraseResponse, MaskRequest, MaskResponse, HateWordRequest, HateWordResponse, HateWordUpdateRequest
from config import logger, DATASET_URLS, SAMPLE_DATA, PARAPHRASER_CONFIG
import global_state as state
import ml_service
import paraphraser_service

app = FastAPI(title="Hate Speech Detection API with Train/Test Split")

# Hate words dictionary file path
HATE_WORDS_FILE = os.path.join(os.path.dirname(__file__), "hate_words.json")

def load_hate_words() -> Dict[str, str]:
    """Load hate words dictionary from file"""
    if os.path.exists(HATE_WORDS_FILE):
        try:
            with open(HATE_WORDS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading hate words: {e}")
            return {}
    return {}

def save_hate_words(hate_words: Dict[str, str]):
    """Save hate words dictionary to file"""
    try:
        with open(HATE_WORDS_FILE, 'w', encoding='utf-8') as f:
            json.dump(hate_words, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving hate words: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save hate words: {str(e)}")

# Initialize hate words dictionary
hate_words_dict = load_hate_words()

# Initialize CORS middleware - MUST be added before routes
# Allow requests from all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Must be False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Mount static files directory
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.on_event("startup")
async def startup_event():
    """Train model on startup"""
    logger.info("Loading dataset and training model on startup...")
    texts, labels = ml_service.load_twitter_davidson_dataset(DATASET_URLS["twitter_hate"])
    
    if texts is None:
        logger.warning("Failed to load web data, using sample data")
        texts, labels = SAMPLE_DATA["texts"], SAMPLE_DATA["labels"]
    
    ml_service.train_model(texts, labels, algorithm="logistic_regression", use_preprocessing=True)

@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the frontend HTML file"""
    # Try multiple possible paths
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "index.html"),
        os.path.join(os.getcwd(), "frontend", "index.html"),
        "frontend/index.html",
    ]
    
    html_content = None
    for frontend_path in possible_paths:
        if os.path.exists(frontend_path):
            try:
                with open(frontend_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                logger.info(f"Successfully loaded HTML from {frontend_path}")
                break
            except Exception as e:
                logger.error(f"Error reading {frontend_path}: {e}")
                continue
    
    if html_content:
        return HTMLResponse(content=html_content)
    
    # Fallback to JSON if file doesn't exist
    logger.warning("Frontend HTML file not found, returning JSON")
    from fastapi.responses import JSONResponse
    return JSONResponse({
        "message": "Hate Speech Detection API with Train/Test Split",
        "model_metrics": state.model_metrics,
        "endpoints": {
            "/predict": "POST - Predict hate speech for new text",
            "/batch-predict": "POST - Predict multiple texts",
            "/train": "POST - Train model with dataset (algorithms: naive_bayes, logistic_regression, svm, random_forest)",
            "/paraphrase": "POST - Paraphrase text (supports transformer and rule-based methods)",
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
    })

@app.get("/api")
def api_info():
    """API information endpoint"""
    return {
        "message": "Hate Speech Detection API with Train/Test Split",
        "model_metrics": state.model_metrics,
        "endpoints": {
            "/predict": "POST - Predict hate speech for new text",
            "/batch-predict": "POST - Predict multiple texts",
            "/train": "POST - Train model with dataset (algorithms: naive_bayes, logistic_regression, svm, random_forest)",
            "/paraphrase": "POST - Paraphrase text (supports transformer and rule-based methods)",
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

@app.post("/paraphrase", response_model=ParaphraseResponse)
def paraphrase_text(request: ParaphraseRequest):
    """
    Paraphrase input text using transformer or rule-based methods.
    
    Args:
        request: ParaphraseRequest with text, num_paraphrases, and optional method
        
    Returns:
        ParaphraseResponse with original text, paraphrases, and method used
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Validate num_paraphrases
    num_paraphrases = max(1, min(request.num_paraphrases, 5))  # Limit to 1-5
    
    try:
        # Get paraphraser instance
        paraphraser = paraphraser_service.get_paraphraser(
            use_transformer=PARAPHRASER_CONFIG["use_transformer"],
            model_name=PARAPHRASER_CONFIG["model_name"]
        )
        
        # Paraphrase the text
        paraphrases = paraphraser.paraphrase(
            text=request.text,
            num_paraphrases=num_paraphrases,
            method=request.method
        )
        
        # Determine which method was actually used
        method_used = "transformer" if paraphraser.use_transformer and paraphraser._loaded else "rule_based"
        
        return ParaphraseResponse(
            original_text=request.text,
            paraphrases=paraphrases,
            method_used=method_used,
            count=len(paraphrases)
        )
        
    except Exception as e:
        logger.error(f"Paraphrasing error: {e}")
        raise HTTPException(status_code=500, detail=f"Paraphrasing error: {str(e)}")

@app.post("/extract-hate-words-ml")
def extract_hate_words_ml(request: TextInput):
    """
    Extract hate words using ML model and automatically add up to 20 to dictionary.
    Returns words that ML model identifies as contributing to hate speech.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        import re
        
        # Define offensive word lists (same as in mask endpoint)
        always_mask_words = [
            'fuck', 'fucking', 'fucked', 'fucker', 'fucks',
            'shit', 'shitting', 'shits', 'shitty',
            'damn', 'damned', 'damning',
            'hell', 'hells',
            'ass', 'asses', 'asshole', 'assholes',
            'bitch', 'bitches', 'bitching', 'bitched',
            'cunt', 'cunts',
            'piss', 'pissing', 'pissed', 'pisses',
            'crap', 'craps', 'crappy',
            'dick', 'dicks', 'dickhead',
            'bastard', 'bastards',
            'whore', 'whores', 'hoe', 'hoes',
            'retard', 'retards', 'retarded', 'retarding',
        ]
        
        context_dependent_words = [
            'kill', 'kills', 'killed', 'killing', 'killer',
            'murder', 'murders', 'murdered', 'murdering', 'murderer',
            'eliminate', 'eliminates', 'eliminated', 'eliminating', 'elimination',
            'destroy', 'destroys', 'destroyed', 'destroying', 'destruction',
            'annihilate', 'annihilates', 'annihilated', 'annihilating',
            'inferior', 'worthless', 'subhuman',
            'plague', 'scum',
            'despicable', 'repulsive',
            'threat', 'threats', 'threatened', 'threatening',
            'assault', 'assaults', 'assaulted', 'assaulting',
            'stab', 'stabs', 'stabbed', 'stabbing',
            'vermin', 'roach', 'roaches', 'parasite', 'parasites',
        ]
        
        # Check if model is trained
        if state.model is None:
            return {"words": [], "count": 0, "message": "ML model is not trained. Please train the model first."}
        
        # First check if it's hate speech
        prediction, confidence = ml_service.predict_single(request.text)
        is_hate_speech = bool(prediction == 1)
        
        if not is_hate_speech:
            return {"words": [], "count": 0, "message": "No hate speech detected"}
        
        # Get ML-identified words
        ml_candidates = ml_service.identify_hate_words_ml(request.text, top_n=30)
        logger.info(f"ML model identified {len(ml_candidates)} candidate words: {ml_candidates[:10]}")
        
        if not ml_candidates:
            logger.warning("ML model returned no candidate words")
            return {"words": [], "count": 0, "message": "ML model could not identify specific hate words", "ml_candidates": []}
        
        # Load current dictionary
        custom_hate_words = load_hate_words()
        all_offensive_words = set(always_mask_words + context_dependent_words + list(custom_hate_words.keys()))
        
        # Common words whitelist
        common_words_whitelist = {
            'the', 'they', 'them', 'their', 'there', 'these', 'those',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'a', 'an', 'all', 'and', 'or', 'but', 'if', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from',
            'i', 'you', 'he', 'she', 'it', 'we', 'us', 'our', 'my', 'your', 'his', 'her', 'its',
            'this', 'that', 'these', 'those',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might',
            'not', 'no', 'yes', 'so', 'very', 'too', 'just', 'only', 'also', 'more', 'most', 'much', 'many',
            'what', 'when', 'where', 'why', 'how', 'who', 'which', 'whom',
            'get', 'got', 'go', 'went', 'come', 'came', 'see', 'saw', 'know', 'knew', 'think', 'thought',
            'say', 'said', 'tell', 'told', 'ask', 'asked', 'give', 'gave', 'take', 'took',
            'make', 'made', 'use', 'used', 'want', 'wanted', 'need', 'needed', 'try', 'tried',
            'like', 'liked', 'love', 'loved', 'hate', 'hated',
            'good', 'bad', 'better', 'best', 'worse', 'worst', 'big', 'small', 'new', 'old',
            'one', 'two', 'three', 'first', 'second', 'last', 'next', 'previous',
            'time', 'times', 'day', 'days', 'year', 'years', 'way', 'ways', 'thing', 'things',
            'people', 'person', 'man', 'men', 'woman', 'women', 'child', 'children',
            'some', 'any', 'every', 'each', 'other', 'another', 'same', 'different',
        }
        
        # Filter words: exclude common words and words already in dictionary
        # We want to add ML-identified words that aren't already in the custom dictionary
        # Even if they're in hardcoded lists, we can still add them for consistency
        ml_words_for_dict = []
        for w in ml_candidates:
            w_lower = w.lower()
            # Skip if already in custom dictionary
            if w_lower in custom_hate_words:
                continue
            # Skip common words
            if w_lower in common_words_whitelist:
                continue
            # Skip very short words
            if len(w) <= 2:
                continue
            # Add this word
            ml_words_for_dict.append(w)
            if len(ml_words_for_dict) >= 20:
                break
        
        logger.info(f"Filtered {len(ml_words_for_dict)} words for dictionary from {len(ml_candidates)} candidates")
        logger.info(f"ML candidates were: {ml_candidates[:10]}")
        logger.info(f"Words to add: {[w.lower() for w in ml_words_for_dict]}")
        
        # Add to dictionary
        added_words = []
        if ml_words_for_dict:
            updated_dict = load_hate_words()
            for word in ml_words_for_dict:
                word_lower = word.lower()
                if word_lower not in updated_dict:
                    # Determine category.
                    # IMPORTANT: only add words that clearly match strong offensive patterns.
                    # We NEVER auto-add neutral words as "other" to avoid masking normal language.
                    category = None
                    if any(prof in word_lower for prof in ['fuck', 'shit', 'damn', 'hell', 'ass', 'bitch', 'cunt', 'piss', 'crap', 'dick', 'bastard', 'whore', 'hoe']):
                        category = "profanity"
                    elif any(viol in word_lower for viol in ['kill', 'murder', 'eliminate', 'destroy', 'annihilate']):
                        category = "violence"
                    elif any(deg in word_lower for deg in ['inferior', 'worthless', 'subhuman', 'scum', 'despicable', 'repulsive']):
                        category = "degrading"
                    
                    # If we couldn't confidently classify it as offensive,
                    # DO NOT add it (prevents filling dictionary with normal words).
                    if category is None:
                        continue

                    updated_dict[word_lower] = category
                    added_words.append({"word": word_lower, "category": category})
            
            if added_words:
                save_hate_words(updated_dict)
                # Verify the save worked
                verify_dict = load_hate_words()
                logger.info(f"Auto-added {len(added_words)} ML-identified words to dictionary: {[w['word'] for w in added_words]}")
                logger.info(f"Dictionary now contains {len(verify_dict)} words total")
            else:
                logger.info(f"No new words to add. ML candidates: {ml_candidates[:10]}, Filtered: {ml_words_for_dict[:10]}")
        
        return {
            "words": added_words,
            "count": len(added_words),
            "message": f"Added {len(added_words)} new words to dictionary from ML findings",
            "ml_candidates": ml_candidates[:10] if ml_candidates else [],  # For debugging
            "filtered": ml_words_for_dict[:10] if ml_words_for_dict else []  # For debugging
        }
        
    except Exception as e:
        logger.error(f"Error extracting hate words with ML: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting hate words: {str(e)}")

@app.post("/extract-hate-words")
def extract_hate_words(request: TextInput):
    """
    Extract potential hate words from text based on known patterns.
    Returns words that match hate word patterns.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        import re
        
        # Get all known hate words (hardcoded + custom dictionary)
        always_mask_words = [
            'fuck', 'fucking', 'fucked', 'fucker', 'fucks',
            'shit', 'shitting', 'shits', 'shitty',
            'damn', 'damned', 'damning',
            'hell', 'hells',
            'ass', 'asses', 'asshole', 'assholes',
            'bitch', 'bitches', 'bitching', 'bitched',
            'cunt', 'cunts',
            'piss', 'pissing', 'pissed', 'pisses',
            'crap', 'craps', 'crappy',
            'dick', 'dicks', 'dickhead',
            'bastard', 'bastards',
            'whore', 'whores', 'hoe', 'hoes',
            'retard', 'retards', 'retarded', 'retarding',
        ]
        
        context_dependent_words = [
            'kill', 'kills', 'killed', 'killing', 'killer',
            'murder', 'murders', 'murdered', 'murdering', 'murderer',
            'eliminate', 'eliminates', 'eliminated', 'eliminating', 'elimination',
            'destroy', 'destroys', 'destroyed', 'destroying', 'destruction',
            'annihilate', 'annihilates', 'annihilated', 'annihilating',
            'inferior', 'worthless', 'subhuman',
            'plague', 'scum',
            'despicable', 'repulsive',
            'threat', 'threats', 'threatened', 'threatening',
            'assault', 'assaults', 'assaulted', 'assaulting',
            'stab', 'stabs', 'stabbed', 'stabbing',
            'vermin', 'roach', 'roaches', 'parasite', 'parasites',
        ]
        
        # Load custom dictionary
        custom_hate_words = load_hate_words()
        all_hate_words = set(always_mask_words + context_dependent_words + list(custom_hate_words.keys()))
        
        # Tokenize text and find matching words
        words = re.findall(r'\b\w+\b', request.text.lower())
        found_words = []
        
        for word in words:
            if word in all_hate_words:
                # Determine category
                category = "profanity"
                if word in custom_hate_words:
                    category = custom_hate_words[word]
                elif word in always_mask_words:
                    if word in ['retard', 'retards', 'retarded', 'retarding']:
                        category = "slur"
                    else:
                        category = "profanity"
                elif word in context_dependent_words:
                    if word in ['kill', 'kills', 'killed', 'killing', 'killer', 'murder', 'murders', 'murdered', 'murdering', 'murderer']:
                        category = "violence"
                    elif word in ['threat', 'threats', 'threatened', 'threatening', 'assault', 'assaults', 'assaulted', 'assaulting']:
                        category = "threatening"
                    else:
                        category = "degrading"
                
                if word not in [w['word'] for w in found_words]:
                    found_words.append({"word": word, "category": category})
        
        return {"words": found_words, "count": len(found_words)}
        
    except Exception as e:
        logger.error(f"Error extracting hate words: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting hate words: {str(e)}")

@app.post("/mask", response_model=MaskResponse)
def mask_hate_speech(request: MaskRequest):
    """
    Mask hate speech words in text by replacing them with asterisks.
    
    Args:
        request: MaskRequest with text to mask
        
    Returns:
        MaskResponse with original text, masked text, and prediction info
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        import re
        
        # Always mask these words (profanity, slurs) regardless of context
        always_mask_words = [
            # Profanity
            'fuck', 'fucking', 'fucked', 'fucker', 'fucks',
            'shit', 'shitting', 'shits', 'shitty',
            'damn', 'damned', 'damning',
            'hell', 'hells',
            'ass', 'asses', 'asshole', 'assholes',
            'bitch', 'bitches', 'bitching', 'bitched',
            'cunt', 'cunts',
            'piss', 'pissing', 'pissed', 'pisses',
            'crap', 'craps', 'crappy',
            'dick', 'dicks', 'dickhead',
            'bastard', 'bastards',
            'whore', 'whores', 'hoe', 'hoes',
            
            # Slurs and always-offensive terms
            'retard', 'retards', 'retarded', 'retarding',
        ]
        
        # Context-dependent words - only mask if hate speech is detected
        context_dependent_words = [
            # Violence (only mask in hate speech context)
            'kill', 'kills', 'killed', 'killing', 'killer',
            'murder', 'murders', 'murdered', 'murdering', 'murderer',
            'eliminate', 'eliminates', 'eliminated', 'eliminating', 'elimination',
            'destroy', 'destroys', 'destroyed', 'destroying', 'destruction',
            'annihilate', 'annihilates', 'annihilated', 'annihilating',
            
            # Degrading terms (only mask in hate speech context)
            'inferior', 'worthless', 'subhuman',
            'plague', 'scum',
            'despicable', 'repulsive',
            
            # Threatening language (only mask in hate speech context)
            'threat', 'threats', 'threatened', 'threatening',
            'assault', 'assaults', 'assaulted', 'assaulting',
            'stab', 'stabs', 'stabbed', 'stabbing',
            
            # Dehumanizing terms (only mask in hate speech context)
            'vermin', 'roach', 'roaches', 'parasite', 'parasites',
        ]
        
        # First, check if it's hate speech
        prediction, confidence = ml_service.predict_single(request.text)
        is_hate_speech = bool(prediction == 1)
        
        def create_partial_mask(word: str) -> str:
            """
            Create a partial mask based on word length for reader-safe context:
            - If word has <= 4 letters: mask last 2 letters, leave first letters exposed
            - If word has >= 5 letters: mask middle, leave first 2 and last 1 letter exposed
            """
            word_len = len(word)
            
            if word_len <= 4:
                # Mask last 2 letters, leave first letters exposed
                if word_len <= 2:
                    # If word is 2 letters or less, mask all
                    return '*' * word_len
                else:
                    # Leave first letters, mask last 2
                    exposed = word[:word_len - 2]
                    masked = '*' * 2
                    return exposed + masked
            else:  # word_len >= 5
                # Mask middle, leave first 2 and last 1 letter exposed
                first_two = word[:2]
                last_one = word[-1]
                middle_masked = '*' * (word_len - 3)  # Total - first 2 - last 1
                return first_two + middle_masked + last_one
        
        # Common words that should NEVER be masked (whitelist)
        common_words_whitelist = {
            'the', 'they', 'them', 'their', 'there', 'these', 'those',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'a', 'an', 'all', 'and', 'or', 'but', 'if', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from',
            'i', 'you', 'he', 'she', 'it', 'we', 'us', 'our', 'my', 'your', 'his', 'her', 'its',
            'this', 'that', 'these', 'those',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might',
            'not', 'no', 'yes', 'so', 'very', 'too', 'just', 'only', 'also', 'more', 'most', 'much', 'many',
            'what', 'when', 'where', 'why', 'how', 'who', 'which', 'whom',
            'get', 'got', 'go', 'went', 'come', 'came', 'see', 'saw', 'know', 'knew', 'think', 'thought',
            'say', 'said', 'tell', 'told', 'ask', 'asked', 'give', 'gave', 'take', 'took',
            'make', 'made', 'use', 'used', 'want', 'wanted', 'need', 'needed', 'try', 'tried',
            'like', 'liked', 'love', 'loved', 'hate', 'hated',  # Note: 'hate' is in whitelist but context-dependent list
            'good', 'bad', 'better', 'best', 'worse', 'worst', 'big', 'small', 'new', 'old',
            'one', 'two', 'three', 'first', 'second', 'last', 'next', 'previous',
            'time', 'times', 'day', 'days', 'year', 'years', 'way', 'ways', 'thing', 'things',
            'people', 'person', 'man', 'men', 'woman', 'women', 'child', 'children',
            'some', 'any', 'every', 'each', 'other', 'another', 'same', 'different',
        }
        
        # Build a mapping of words to mask with their partial mask replacements
        words_to_mask = {}
        
        # Tokenize the text to find words
        words = re.findall(r'\b\w+\b', request.text)
        
        # Load custom hate words from dictionary
        custom_hate_words = load_hate_words()
        
        # Use ML model to identify hate words if hate speech is detected
        # NOTE: In the mask endpoint we ONLY use ML to help choose which words to mask.
        # Dictionary updates are handled by the /extract-hate-words-ml endpoint so that
        # the UI can show clear notifications whenever new words are added.
        ml_identified_words = []
        if is_hate_speech:
            try:
                ml_candidates = ml_service.identify_hate_words_ml(request.text, top_n=30)
                # Filter: only use ML-identified words that are in our offensive word lists for masking
                all_offensive_words = set(always_mask_words + context_dependent_words + list(custom_hate_words.keys()))
                ml_identified_words = [w for w in ml_candidates if w.lower() in all_offensive_words]
                
                logger.info(f"ML model identified hate words for masking: {ml_identified_words}")
            except Exception as e:
                logger.warning(f"Failed to identify words using ML model for masking: {e}")
        
        # Offensive substring stems for slang / variants (e.g. \"mofuckas\", \"motherfucker\")
        strong_offensive_stems = ['fuck', 'fuk']
        
        for word in words:
            word_lower = word.lower()
            
            # Skip common words - never mask them (unless they're explicitly in offensive lists)
            # Exception: if a word is in our offensive lists, mask it even if it's a common word
            is_offensive = (word_lower in custom_hate_words or 
                          word_lower in always_mask_words or 
                          (is_hate_speech and word_lower in context_dependent_words))
            
            if word_lower in common_words_whitelist and not is_offensive:
                continue
            
            # Priority 1: Check custom dictionary first
            if word_lower in custom_hate_words:
                words_to_mask[word] = create_partial_mask(word)
            
            # Priority 2: Always mask profanity and slurs (with partial masking)
            elif word_lower in always_mask_words:
                words_to_mask[word] = create_partial_mask(word)
            
            # Priority 2b: Mask strong offensive substrings within words
            # e.g. \"mofuckas\", \"motherfuckers\" should be treated as offensive due to \"fuck\" stem.
            elif any(stem in word_lower for stem in strong_offensive_stems):
                words_to_mask[word] = create_partial_mask(word)
            
            # Priority 3: Use ML model identified words (if hate speech detected and word is in offensive lists)
            elif is_hate_speech and word_lower in ml_identified_words:
                words_to_mask[word] = create_partial_mask(word)
            
            # Priority 4: Only mask context-dependent words if hate speech is detected (with partial masking)
            elif is_hate_speech and word_lower in context_dependent_words:
                words_to_mask[word] = create_partial_mask(word)
        
        # Apply masking: replace each word with partial mask
        masked_text = request.text
        for word, mask in words_to_mask.items():
            # Use word boundaries to ensure we match whole words only
            masked_text = re.sub(r'\b' + re.escape(word) + r'\b', mask, masked_text)
        
        return MaskResponse(
            original_text=request.text,
            masked_text=masked_text,
            is_hate_speech=is_hate_speech,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Masking error: {e}")
        raise HTTPException(status_code=500, detail=f"Masking error: {str(e)}")

# Hate Words Dictionary Management Endpoints
@app.get("/hate-words", response_model=Dict[str, str])
def get_hate_words():
    """Get all hate words from the dictionary"""
    global hate_words_dict
    hate_words_dict = load_hate_words()  # Reload from file
    return hate_words_dict

@app.post("/hate-words", response_model=HateWordResponse)
def add_hate_word(request: HateWordRequest):
    """Add a new hate word to the dictionary"""
    global hate_words_dict
    word_lower = request.word.lower().strip()
    
    if not word_lower:
        raise HTTPException(status_code=400, detail="Word cannot be empty")
    
    hate_words_dict = load_hate_words()
    hate_words_dict[word_lower] = request.category
    save_hate_words(hate_words_dict)
    
    logger.info(f"Added hate word: {word_lower} (category: {request.category})")
    return HateWordResponse(word=word_lower, category=request.category)

@app.put("/hate-words", response_model=HateWordResponse)
def update_hate_word(request: HateWordUpdateRequest):
    """Update an existing hate word in the dictionary"""
    global hate_words_dict
    old_word_lower = request.old_word.lower().strip()
    new_word_lower = request.new_word.lower().strip()
    
    if not old_word_lower or not new_word_lower:
        raise HTTPException(status_code=400, detail="Word cannot be empty")
    
    hate_words_dict = load_hate_words()
    
    if old_word_lower not in hate_words_dict:
        raise HTTPException(status_code=404, detail=f"Word '{request.old_word}' not found in dictionary")
    
    # Get existing category or use new one
    category = request.category if request.category else hate_words_dict[old_word_lower]
    
    # Remove old word and add new one
    del hate_words_dict[old_word_lower]
    hate_words_dict[new_word_lower] = category
    save_hate_words(hate_words_dict)
    
    logger.info(f"Updated hate word: {old_word_lower} -> {new_word_lower} (category: {category})")
    return HateWordResponse(word=new_word_lower, category=category)

@app.delete("/hate-words/{word}")
def delete_hate_word(word: str):
    """Delete a hate word from the dictionary"""
    global hate_words_dict
    word_lower = word.lower().strip()
    
    hate_words_dict = load_hate_words()
    
    if word_lower not in hate_words_dict:
        raise HTTPException(status_code=404, detail=f"Word '{word}' not found in dictionary")
    
    del hate_words_dict[word_lower]
    save_hate_words(hate_words_dict)
    
    logger.info(f"Deleted hate word: {word_lower}")
    return {"message": f"Word '{word}' deleted successfully", "deleted_word": word_lower}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)