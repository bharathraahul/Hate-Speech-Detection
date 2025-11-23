"""
Rule-based pattern matching for hate speech detection.
This complements the ML model to catch patterns it might miss.
"""

import re
from typing import List, Tuple, Dict
from config import logger


class HateSpeechPatternMatcher:
    """
    Rule-based pattern matcher for hate speech detection.
    Catches common hate speech patterns that ML models might miss.
    """
    
    def __init__(self):
        """Initialize pattern matchers"""
        self.patterns = self._load_patterns()
        self.religious_terms = self._load_religious_terms()
        self.ethnic_terms = self._load_ethnic_terms()
        self.exclusion_patterns = self._load_exclusion_patterns()
    
    def _load_patterns(self) -> List[Tuple[str, str, float]]:
        """
        Load hate speech patterns.
        Returns: List of (pattern_regex, description, weight) tuples
        """
        return [
            # "X out of Y" pattern (e.g., "Islam out of Britain")
            (r'\b(\w+)\s+out\s+of\s+(\w+)\b', 'exclusion_pattern', 0.9),
            
            # "Protect X from Y" pattern
            (r'protect\s+.*?\s+from\s+.*?', 'protection_pattern', 0.7),
            
            # "X should be banned/deported/removed"
            (r'\b(should|must|need)\s+(be\s+)?(banned|deported|removed|eliminated|expelled)\b', 'removal_pattern', 0.85),
            
            # "X are/is [negative]"
            (r'\b(\w+)\s+(are|is)\s+(inferior|subhuman|animals|vermin|trash|scum|filth)\b', 'dehumanization', 0.9),
            
            # "Kill/Death to X"
            (r'\b(kill|death|die|exterminate|eliminate)\s+(all\s+)?(.*?)\b', 'violence_pattern', 0.95),
            
            # "X don't belong here"
            (r'\b(.*?)\s+(don\'?t|doesn\'?t)\s+belong\s+(here|in|to)\b', 'exclusion_belonging', 0.8),
            
            # "X are taking over"
            (r'\b(.*?)\s+(are|is)\s+(taking\s+over|invading|infesting)\b', 'invasion_pattern', 0.75),
            
            # "Send X back"
            (r'\b(send|ship|deport)\s+(.*?)\s+back\b', 'deportation_pattern', 0.85),
        ]
    
    def _load_religious_terms(self) -> List[str]:
        """Load religious terms that are often targeted"""
        return [
            'islam', 'muslim', 'muslims', 'islamic', 'jew', 'jews', 'jewish', 'judaism',
            'christian', 'christians', 'christianity', 'hindu', 'hindus', 'hinduism',
            'sikh', 'sikhs', 'sikhism', 'buddhist', 'buddhists', 'buddhism',
            'catholic', 'catholics', 'protestant', 'protestants'
        ]
    
    def _load_ethnic_terms(self) -> List[str]:
        """Load ethnic/racial terms that are often targeted"""
        return [
            'black', 'white', 'asian', 'hispanic', 'latino', 'latina', 'arab', 'arabs',
            'african', 'indian', 'chinese', 'japanese', 'korean', 'mexican'
        ]
    
    def _load_exclusion_patterns(self) -> List[str]:
        """Patterns that indicate exclusion but might be legitimate"""
        return [
            r'\bout\s+of\s+(business|stock|order)\b',  # Legitimate uses
            r'\bprotect\s+(yourself|children|data|privacy)\b',  # Legitimate protection
        ]
    
    def check_patterns(self, text: str) -> Dict[str, any]:
        """
        Check text against hate speech patterns.
        
        Args:
            text: Input text to check
            
        Returns:
            Dictionary with pattern_match, confidence, matched_patterns
        """
        if not text or not isinstance(text, str):
            return {
                "pattern_match": False,
                "confidence": 0.0,
                "matched_patterns": [],
                "reason": "empty_text"
            }
        
        text_lower = text.lower()
        matched_patterns = []
        total_weight = 0.0
        
        # Check exclusion patterns first (legitimate uses)
        for exclusion_pattern in self.exclusion_patterns:
            if re.search(exclusion_pattern, text_lower, re.IGNORECASE):
                # This might be legitimate, reduce weight
                continue
        
        # Check all hate speech patterns
        for pattern, description, weight in self.patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                matched_text = match.group(0)
                matched_patterns.append({
                    "pattern": description,
                    "matched_text": matched_text,
                    "weight": weight
                })
                total_weight += weight
        
        # Check for religious/ethnic targeting
        religious_targeted = any(term in text_lower for term in self.religious_terms)
        ethnic_targeted = any(term in text_lower for term in self.ethnic_terms)
        
        # If religious/ethnic terms present AND exclusion patterns found, increase weight
        if (religious_targeted or ethnic_targeted) and matched_patterns:
            total_weight *= 1.3  # Boost for targeting specific groups
            for pattern_info in matched_patterns:
                pattern_info["group_targeted"] = True
        
        # Calculate confidence based on total weight
        # Threshold: 0.7 for pattern match
        pattern_match = total_weight >= 0.7
        confidence = min(1.0, total_weight)
        
        return {
            "pattern_match": pattern_match,
            "confidence": confidence,
            "matched_patterns": matched_patterns,
            "total_weight": total_weight,
            "religious_targeted": religious_targeted,
            "ethnic_targeted": ethnic_targeted
        }
    
    def is_hate_speech_by_pattern(self, text: str) -> Tuple[bool, float]:
        """
        Simple interface: returns (is_hate_speech, confidence)
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (is_hate_speech: bool, confidence: float)
        """
        result = self.check_patterns(text)
        return result["pattern_match"], result["confidence"]


# Global instance
_pattern_matcher = None

def get_pattern_matcher() -> HateSpeechPatternMatcher:
    """Get or create global pattern matcher instance"""
    global _pattern_matcher
    if _pattern_matcher is None:
        _pattern_matcher = HateSpeechPatternMatcher()
    return _pattern_matcher

def check_hate_speech_patterns(text: str) -> Dict[str, any]:
    """Convenience function to check patterns"""
    matcher = get_pattern_matcher()
    return matcher.check_patterns(text)

