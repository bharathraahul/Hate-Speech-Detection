"""
Masking Service - Masks offensive/hate words in text when hate speech is detected
"""

import re
from typing import List, Dict, Tuple
from config import logger


class MaskingService:
    """
    Service for masking offensive/hate words in text.
    """
    
    def __init__(self):
        """Initialize the masking service"""
        self.offensive_words = self._load_offensive_words()
        self.hate_patterns = self._load_hate_patterns()
    
    def _load_offensive_words(self) -> List[str]:
        """Load list of offensive/hate words to mask"""
        return [
            # Slurs and offensive terms
            'hoe', 'hoes', 'ho', 'hos',
            'bitch', 'bitches',
            'trash',
            'nigga', 'nigger', 'niggas', 'niggers',
            'fag', 'faggot', 'fags', 'faggots',
            'retard', 'retarded',
            'whore', 'whores',
            'slut', 'sluts',
            'cunt', 'cunts',
            'pussy', 'pussies',
            'asshole', 'assholes',
            'bastard', 'bastards',
            'damn', 'damned',
            'hell',
            'shit', 'shits',
            'fuck', 'fucks', 'fucking', 'fucked',
            'crap', 'craps',
            'stupid', 'idiot', 'idiots',
            'dumb', 'dumbass',
            'moron', 'morons',
            'jerk', 'jerks',
            'loser', 'losers',
            # Hate speech patterns
            'jungle bunnies',
            'subhuman',
            'inferior',
            'vermin',
            'scum',
            'filth',
            'animals',
            'worthless',
            'disgusting',
        ]
    
    def _load_hate_patterns(self) -> List[Tuple[str, str]]:
        """Load regex patterns for hate speech phrases"""
        return [
            (r'\b(kill|death|die|exterminate|eliminate)\s+(all\s+)?(.*?)\b', '[violence threat]'),
            (r'\b(send|ship|deport)\s+(.*?)\s+back\b', '[deportation phrase]'),
            (r'\b(.*?)\s+out\s+of\s+(.*?)\b', '[exclusion phrase]'),
            (r'\b(.*?)\s+should\s+be\s+(banned|deported|removed|eliminated)\b', '[removal phrase]'),
        ]
    
    def mask_offensive_words(self, text: str, mask_char: str = "[REDACTED]") -> Tuple[str, List[Dict]]:
        """
        Mask offensive words in text.
        If mask_char is "[REDACTED]" (default), uses first letter + stars format.
        Otherwise uses the provided mask_char.
        
        Args:
            text: Input text to mask
            mask_char: Character/string to use for masking (default: "[REDACTED]")
                      If "[REDACTED]", uses first letter + stars (e.g., "b****")
            
        Returns:
            Tuple of (masked_text, masked_words_info)
        """
        if not text or not isinstance(text, str):
            return text, []
        
        masked_text = text
        masked_words = []
        text_lower = text.lower()
        matches_found = []
        
        # Find all offensive word matches
        for word in self.offensive_words:
            pattern = r'\b' + re.escape(word) + r'\b'
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                original_word = text[match.start():match.end()]
                matches_found.append({
                    "word": original_word,
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Sort by position (reverse order) to mask from end to start
        # This preserves positions for earlier matches
        matches_found.sort(key=lambda x: x["start"], reverse=True)
        
        # Mask from end to start to preserve positions
        for match_info in matches_found:
            start = match_info["start"]
            end = match_info["end"]
            original_word = match_info["word"]
            
            # Smart masking based on word length for reader-safe context:
            # - <= 4 letters: mask last 2 letters, show first letters
            # - 5 letters: show first 2, mask last 3
            # - > 5 letters: show first 2 and last letter, mask middle
            if mask_char == "[REDACTED]":
                word_length = len(original_word)
                if word_length <= 4:
                    # Mask last 2 letters, show first letters
                    if word_length <= 2:
                        masked_word = "*" * word_length
                    else:
                        masked_word = original_word[:word_length-2] + "**"
                elif word_length == 5:
                    # Show first 2, mask last 3
                    masked_word = original_word[:2] + "***"
                else:
                    # word_length > 5: show first 2 and last, mask middle
                    masked_word = original_word[:2] + "*" * (word_length - 3) + original_word[-1]
            else:
                masked_word = mask_char
            
            masked_text = masked_text[:start] + masked_word + masked_text[end:]
            masked_words.append({
                "word": original_word,
                "position": start,
                "masked": masked_word
            })
        
        # Also mask hate speech patterns (after word masking)
        # For patterns, use the mask_char as-is (or [REDACTED] if default)
        for pattern, replacement in self.hate_patterns:
            matches = list(re.finditer(pattern, masked_text, re.IGNORECASE))
            for match in reversed(matches):  # Process from end to start
                matched_text = match.group(0)
                # For patterns, use the original replacement or mask_char
                if mask_char != "[REDACTED]":
                    replacement = mask_char
                masked_text = masked_text[:match.start()] + replacement + masked_text[match.end():]
                masked_words.append({
                    "word": matched_text,
                    "position": match.start(),
                    "masked": replacement
                })
        
        return masked_text, masked_words
    
    def mask_text(self, text: str, mask_char: str = "[REDACTED]") -> Dict[str, any]:
        """
        Main method to mask text.
        
        Args:
            text: Input text to mask
            mask_char: Character/string to use for masking
            
        Returns:
            Dictionary with masked_text, original_text, and masked_words
        """
        if not text or not text.strip():
            return {
                "original_text": text,
                "masked_text": text,
                "masked_words": [],
                "words_masked": 0
            }
        
        masked_text, masked_words = self.mask_offensive_words(text, mask_char)
        
        return {
            "original_text": text,
            "masked_text": masked_text,
            "masked_words": masked_words,
            "words_masked": len(masked_words)
        }


# Global instance
_masking_instance = None

def get_masking_service() -> MaskingService:
    """Get or create global masking service instance"""
    global _masking_instance
    if _masking_instance is None:
        _masking_instance = MaskingService()
    return _masking_instance

def mask_hate_speech(text: str, mask_char: str = "[REDACTED]") -> Dict[str, any]:
    """Convenience function to mask hate speech"""
    service = get_masking_service()
    return service.mask_text(text, mask_char)

