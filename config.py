import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Dataset URLs ---
DATASET_URLS = {
    "twitter_hate": "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv",
}

# --- Paraphraser Configuration ---
PARAPHRASER_CONFIG = {
    "use_transformer": True,  # Use transformer model if available
    "model_name": "tuner007/pegasus_paraphrase",  # Hugging Face model
    "fallback_to_rule_based": True,  # Fall back to rule-based if transformer fails
    "max_text_length": 500,  # Maximum text length for paraphrasing
    "default_num_paraphrases": 1,  # Default number of paraphrases to generate
}

# --- Sample Data (Fallback) ---
SAMPLE_DATA = {
    "texts": [
        "I love spending time with my friends",
        "This is a great movie",
        "Have a wonderful day everyone",
        "I disagree with your opinion but respect it",
        "Let's work together to solve this",
        "That's an interesting perspective",
        "I enjoyed reading this article",
        "Thanks for your help today",
        "Looking forward to the weekend",
        "Great job on the presentation",
        "Nice weather we're having",
        "I appreciate your help",
        "This is wonderful news",
        "Happy to be here",
        "Thank you so much",
        "I hate all people from that group",
        "Those people are inferior and worthless",
        "Violence against them is justified",
        "They should all be eliminated",
        "Disgusting subhuman creatures",
        "They deserve to suffer",
        "We need to get rid of them all",
        "Those people are a plague",
        "They are animals and should be treated as such",
        "Kill all of them",
        "Worthless trash all of them",
        "They should be wiped out",
        "Subhuman filth",
        "Death to all of them",
        "Exterminate those vermin",
    ],
    "labels": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}