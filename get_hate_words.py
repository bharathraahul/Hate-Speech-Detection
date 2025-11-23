#!/usr/bin/env python3
"""
Simple script to extract hate words from text using the ML model
Usage: python3 get_hate_words.py "your text here"
"""

import requests
import json
import sys

API_BASE_URL = "http://localhost:8000"

def extract_hate_words(text):
    """Extract hate words from given text"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/extract-hate-words-ml",
            json={"text": text},
            headers={"Content-Type": "application/json"}
        )
        
        if response.ok:
            data = response.json()
            return data
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server. Make sure it's running at http://localhost:8000")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 get_hate_words.py \"your text here\"")
        print("\nExample:")
        print('  python3 get_hate_words.py "I fucking hate you, you are a bitch"')
        sys.exit(1)
    
    text = " ".join(sys.argv[1:])
    
    print("=" * 80)
    print("EXTRACTING HATE WORDS FROM TEXT")
    print("=" * 80)
    print(f"\nInput text: {text}\n")
    
    result = extract_hate_words(text)
    
    if result:
        print("-" * 80)
        print("RESULTS")
        print("-" * 80)
        
        ml_candidates = result.get("ml_candidates", [])
        filtered = result.get("filtered", [])
        added_words = result.get("words", [])
        count = result.get("count", 0)
        
        print(f"\nML-identified candidate words ({len(ml_candidates)}):")
        if ml_candidates:
            print(f"  {', '.join(ml_candidates[:20])}")
            if len(ml_candidates) > 20:
                print(f"  ... and {len(ml_candidates) - 20} more")
        else:
            print("  None")
        
        print(f"\nFiltered words (after removing common words, {len(filtered)}):")
        if filtered:
            print(f"  {', '.join(filtered[:20])}")
            if len(filtered) > 20:
                print(f"  ... and {len(filtered) - 20} more")
        else:
            print("  None")
        
        print(f"\nWords added to dictionary ({count}):")
        if added_words:
            for word_data in added_words:
                word = word_data.get("word", word_data) if isinstance(word_data, dict) else word_data
                category = word_data.get("category", "unknown") if isinstance(word_data, dict) else "unknown"
                print(f"  - {word} ({category})")
        else:
            print("  None (all words were already in dictionary or are common words)")
        
        print(f"\nMessage: {result.get('message', 'N/A')}")
    else:
        print("Failed to extract hate words")

if __name__ == "__main__":
    main()


