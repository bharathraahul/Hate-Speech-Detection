#!/usr/bin/env python3
"""
Test script for the masking feature
Tests various texts to demonstrate masking functionality
"""

import requests
import json

API_URL = "http://localhost:8000/mask"

# Test texts
TEST_TEXTS = {
    "Hate Speech Examples": [
        "dont even entertain these hoes, got somebody too important",
        "Ugly hoes love to fight lmao",
        "My house is so piece full, thank god these little jungle bunnies are outta my face",
        "Islam out of Britain. Protect the British people.",
        "Muslims should be deported from our country",
        "Those people are trash and bitches who should be eliminated",
        "I hate all those bitches who think they're better",
        "Send them back to where they came from",
        "Kill all of them, they don't belong here",
        "Those subhuman creatures deserve to suffer",
    ],
    "Normal Text Examples": [
        "I love spending time with my friends",
        "This is a great idea and I think we should implement it",
        "The weather is beautiful today, perfect for a walk",
        "Please stop scoring, otherwise you won't have any runs for tomorrow",
        "I think this movie is excellent and worth watching",
        "Thank you for your help, I really appreciate it",
        "Let's work together to solve this problem",
        "I disagree with your opinion but I respect it",
        "This is an important decision that needs careful consideration",
        "Have a wonderful day everyone!",
    ]
}

def test_mask(text, mask_char="[REDACTED]"):
    """Test masking for a single text"""
    try:
        response = requests.post(
            API_URL,
            json={"text": text, "mask_char": mask_char},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def print_result(text, result):
    """Pretty print the result"""
    print("\n" + "="*70)
    print(f"Original: {text}")
    print("-"*70)
    
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return
    
    print(f"Hate Speech Detected: {result['is_hate_speech']}")
    print(f"Confidence: {result['hate_speech_confidence']:.2%}")
    print(f"Was Masked: {result['was_masked']}")
    
    if result['was_masked']:
        print(f"Masked Text: {result['masked_text']}")
        print(f"Words Masked: {result['words_masked']}")
        if result['masked_words']:
            print("Masked Words Details:")
            for word_info in result['masked_words']:
                print(f"  - '{word_info['word']}' → '{word_info['masked']}' (position: {word_info['position']})")
    else:
        print("Masked Text: (not masked)")
    
    print(f"Message: {result['message']}")
    print("="*70)

def main():
    print("="*70)
    print("MASKING FEATURE TEST SUITE")
    print("="*70)
    
    # Test hate speech examples
    print("\n\n" + "="*70)
    print("HATE SPEECH EXAMPLES (Should be masked)")
    print("="*70)
    
    for i, text in enumerate(TEST_TEXTS["Hate Speech Examples"], 1):
        print(f"\n[Test {i}]")
        result = test_mask(text)
        print_result(text, result)
    
    # Test normal text examples
    print("\n\n" + "="*70)
    print("NORMAL TEXT EXAMPLES (Should NOT be masked)")
    print("="*70)
    
    for i, text in enumerate(TEST_TEXTS["Normal Text Examples"], 1):
        print(f"\n[Test {i}]")
        result = test_mask(text)
        print_result(text, result)
    
    # Test custom mask character
    print("\n\n" + "="*70)
    print("CUSTOM MASK CHARACTER TEST")
    print("="*70)
    
    test_text = "Ugly hoes love to fight"
    print(f"\nOriginal: {test_text}")
    result = test_mask(test_text, mask_char="[CENSORED]")
    print_result(test_text, result)
    
    print("\n\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()

