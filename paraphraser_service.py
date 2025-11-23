"""
Paraphraser Service - Provides text paraphrasing functionality
Supports both transformer-based and rule-based approaches
"""

import re
import random
from typing import List, Optional, Dict
from config import logger

# Try to import transformers, but make it optional
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. Using rule-based paraphrasing only.")


class ParaphraserService:
    """
    Service for paraphrasing text with multiple backends.
    """
    
    def __init__(self, use_transformer: bool = True, model_name: str = "tuner007/pegasus_paraphrase"):
        """
        Initialize the paraphraser service.
        
        Args:
            use_transformer: Whether to use transformer model (if available)
            model_name: Hugging Face model name for paraphrasing
        """
        self.use_transformer = use_transformer and TRANSFORMERS_AVAILABLE
        self.model_name = model_name
        self.paraphrase_pipeline = None
        self.tokenizer = None
        self.model = None
        self._loaded = False
        
        # Synonym dictionary for rule-based fallback
        self.synonym_dict = self._load_synonym_dict()
    
    def _load_synonym_dict(self) -> Dict[str, List[str]]:
        """Load a basic synonym dictionary for rule-based paraphrasing"""
        return {
            "good": ["great", "excellent", "wonderful", "fantastic", "nice"],
            "bad": ["terrible", "awful", "poor", "horrible", "lousy"],
            "big": ["large", "huge", "enormous", "massive", "gigantic"],
            "small": ["tiny", "little", "mini", "miniature", "petite"],
            "happy": ["joyful", "glad", "pleased", "delighted", "cheerful"],
            "sad": ["unhappy", "depressed", "down", "melancholy", "gloomy"],
            "important": ["significant", "crucial", "vital", "essential", "key"],
            "beautiful": ["pretty", "attractive", "gorgeous", "lovely", "stunning"],
            "smart": ["intelligent", "clever", "bright", "brilliant", "wise"],
            "fast": ["quick", "rapid", "swift", "speedy", "hasty"],
            "slow": ["sluggish", "leisurely", "gradual", "unhurried", "delayed"],
            "easy": ["simple", "straightforward", "effortless", "uncomplicated"],
            "difficult": ["hard", "challenging", "tough", "complex", "complicated"],
            "help": ["assist", "aid", "support", "guide", "facilitate"],
            "think": ["believe", "consider", "ponder", "reflect", "contemplate"],
            "say": ["state", "mention", "express", "declare", "articulate"],
            "get": ["obtain", "acquire", "receive", "fetch", "retrieve"],
            "make": ["create", "produce", "generate", "construct", "build"],
            "use": ["utilize", "employ", "apply", "leverage", "harness"],
            "show": ["display", "demonstrate", "exhibit", "present", "reveal"],
        }
    
    def _load_transformer_model(self):
        """Lazy load the transformer model"""
        if self._loaded:
            return
        
        try:
            logger.info(f"Loading paraphrase model: {self.model_name}")
            self.paraphrase_pipeline = pipeline(
                "text2text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=-1  # Use CPU (-1), set to 0 for GPU if available
            )
            self._loaded = True
            logger.info("Paraphrase model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load transformer model: {e}")
            logger.warning("Falling back to rule-based paraphrasing")
            self.use_transformer = False
            self._loaded = True
    
    def _paraphrase_with_transformer(self, text: str, num_paraphrases: int = 1) -> List[str]:
        """
        Paraphrase text using transformer model.
        
        Args:
            text: Input text to paraphrase
            num_paraphrases: Number of paraphrases to generate
            
        Returns:
            List of paraphrased texts
        """
        if not self._loaded:
            self._load_transformer_model()
        
        if not self.use_transformer or self.paraphrase_pipeline is None:
            return self._paraphrase_rule_based(text, num_paraphrases)
        
        try:
            # Prepare input for the model
            # Some models need specific prefixes
            if "pegasus" in self.model_name.lower():
                input_text = f"paraphrase: {text}"
            else:
                input_text = text
            
            # Generate paraphrases
            results = self.paraphrase_pipeline(
                input_text,
                num_return_sequences=num_paraphrases,
                num_beams=10,
                max_length=128,
                temperature=0.7,
                do_sample=True
            )
            
            paraphrases = []
            for result in results:
                paraphrased_text = result.get('generated_text', '').strip()
                # Clean up the output
                if paraphrased_text.startswith("paraphrase:"):
                    paraphrased_text = paraphrased_text.replace("paraphrase:", "").strip()
                if paraphrased_text:
                    paraphrases.append(paraphrased_text)
            
            # If we got fewer than requested, fill with rule-based
            while len(paraphrases) < num_paraphrases:
                rule_based = self._paraphrase_rule_based(text, 1)
                if rule_based and rule_based[0] not in paraphrases:
                    paraphrases.append(rule_based[0])
                else:
                    break
            
            return paraphrases[:num_paraphrases] if paraphrases else [text]
            
        except Exception as e:
            logger.error(f"Transformer paraphrasing failed: {e}")
            logger.warning("Falling back to rule-based paraphrasing")
            return self._paraphrase_rule_based(text, num_paraphrases)
    
    def _paraphrase_rule_based(self, text: str, num_paraphrases: int = 1) -> List[str]:
        """
        Paraphrase text using rule-based synonym replacement.
        This is a fallback when transformers are not available.
        
        Args:
            text: Input text to paraphrase
            num_paraphrases: Number of paraphrases to generate
            
        Returns:
            List of paraphrased texts
        """
        paraphrases = []
        words = text.split()
        
        for _ in range(num_paraphrases):
            paraphrased_words = []
            replacements_made = 0
            max_replacements = min(3, len(words) // 3)  # Replace up to 3 words or 1/3 of text
            
            for word in words:
                # Clean word for lookup (remove punctuation)
                clean_word = re.sub(r'[^\w]', '', word.lower())
                
                # Check if we can replace this word
                if clean_word in self.synonym_dict and replacements_made < max_replacements:
                    # Randomly decide whether to replace (70% chance)
                    if random.random() < 0.7:
                        synonyms = self.synonym_dict[clean_word]
                        replacement = random.choice(synonyms)
                        
                        # Preserve original capitalization
                        if word[0].isupper():
                            replacement = replacement.capitalize()
                        
                        # Preserve punctuation
                        punctuation = re.findall(r'[^\w]', word)
                        if punctuation:
                            replacement += ''.join(punctuation)
                        
                        paraphrased_words.append(replacement)
                        replacements_made += 1
                    else:
                        paraphrased_words.append(word)
                else:
                    paraphrased_words.append(word)
            
            paraphrased_text = ' '.join(paraphrased_words)
            
            # Only add if it's different from original
            if paraphrased_text != text and paraphrased_text not in paraphrases:
                paraphrases.append(paraphrased_text)
            elif len(paraphrases) == 0:
                # If no changes made, return original
                paraphrases.append(text)
        
        # If we need more paraphrases, try different strategies
        while len(paraphrases) < num_paraphrases:
            # Try with different replacement rate
            paraphrased_words = []
            for word in words:
                clean_word = re.sub(r'[^\w]', '', word.lower())
                if clean_word in self.synonym_dict and random.random() < 0.5:
                    synonyms = self.synonym_dict[clean_word]
                    replacement = random.choice(synonyms)
                    if word[0].isupper():
                        replacement = replacement.capitalize()
                    punctuation = re.findall(r'[^\w]', word)
                    if punctuation:
                        replacement += ''.join(punctuation)
                    paraphrased_words.append(replacement)
                else:
                    paraphrased_words.append(word)
            
            new_paraphrase = ' '.join(paraphrased_words)
            if new_paraphrase != text and new_paraphrase not in paraphrases:
                paraphrases.append(new_paraphrase)
            else:
                break
        
        return paraphrases[:num_paraphrases] if paraphrases else [text]
    
    def paraphrase(self, text: str, num_paraphrases: int = 1, method: Optional[str] = None) -> List[str]:
        """
        Main method to paraphrase text.
        
        Args:
            text: Input text to paraphrase
            num_paraphrases: Number of paraphrases to generate (default: 1)
            method: 'transformer' or 'rule_based'. If None, uses default based on availability
            
        Returns:
            List of paraphrased texts
        """
        if not text or not text.strip():
            return [text]
        
        # Limit text length for transformer models
        if len(text) > 500:
            text = text[:500]
            logger.warning("Text truncated to 500 characters for paraphrasing")
        
        # Determine method
        if method == "rule_based" or (method is None and not self.use_transformer):
            return self._paraphrase_rule_based(text, num_paraphrases)
        elif method == "transformer" or (method is None and self.use_transformer):
            return self._paraphrase_with_transformer(text, num_paraphrases)
        else:
            # Fallback
            return self._paraphrase_rule_based(text, num_paraphrases)


# Global instance (lazy loaded)
_paraphraser_instance = None

def get_paraphraser(use_transformer: bool = True, model_name: str = "tuner007/pegasus_paraphrase") -> ParaphraserService:
    """
    Get or create the global paraphraser instance.
    
    Args:
        use_transformer: Whether to use transformer model
        model_name: Model name for transformer
        
    Returns:
        ParaphraserService instance
    """
    global _paraphraser_instance
    if _paraphraser_instance is None:
        _paraphraser_instance = ParaphraserService(use_transformer=use_transformer, model_name=model_name)
    return _paraphraser_instance

def paraphrase_text(text: str, num_paraphrases: int = 1, method: Optional[str] = None) -> List[str]:
    """
    Convenience function to paraphrase text.
    
    Args:
        text: Input text to paraphrase
        num_paraphrases: Number of paraphrases to generate
        method: 'transformer' or 'rule_based'
        
    Returns:
        List of paraphrased texts
    """
    paraphraser = get_paraphraser()
    return paraphraser.paraphrase(text, num_paraphrases, method)

