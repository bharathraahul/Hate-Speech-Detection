#!/usr/bin/env python3
"""
Extract hate words from test set using ML model
"""

import requests
import json
from collections import Counter

API_BASE_URL = "http://localhost:8000"

def extract_hate_words_from_test_set(num_samples=100):
    """Extract hate words from test set samples"""
    print(f"Fetching {num_samples} test samples...")
    
    # Get test samples that are hate speech
    response = requests.get(f"{API_BASE_URL}/test-results?limit={num_samples}")
    if not response.ok:
        print(f"Error: {response.status_code}")
        return
    
    data = response.json()
    results = data["results"]
    
    # Filter for hate speech samples
    hate_speech_samples = [r for r in results if r["true_label"] == 1]
    print(f"Found {len(hate_speech_samples)} hate speech samples")
    
    # Extract hate words from each sample
    all_hate_words = []
    word_frequency = Counter()
    
    print("\nExtracting hate words from samples...")
    for i, sample in enumerate(hate_speech_samples[:50], 1):  # Limit to 50 to avoid too many API calls
        text = sample["text"]
        try:
            extract_response = requests.post(
                f"{API_BASE_URL}/extract-hate-words-ml",
                json={"text": text},
                headers={"Content-Type": "application/json"}
            )
            
            if extract_response.ok:
                extract_data = extract_response.json()
                ml_candidates = extract_data.get("ml_candidates", [])
                for word in ml_candidates:
                    word_lower = word.lower()
                    if len(word_lower) > 2:  # Filter very short words
                        all_hate_words.append(word_lower)
                        word_frequency[word_lower] += 1
                
                if i % 10 == 0:
                    print(f"  Processed {i}/{min(50, len(hate_speech_samples))} samples...")
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("EXTRACTED HATE WORDS (from ML model)")
    print(f"{'='*80}\n")
    
    # Show most frequent words
    print("Most frequently identified hate words:")
    print("-" * 80)
    for word, count in word_frequency.most_common(50):
        print(f"{word:20} (appeared {count} times)")
    
    print(f"\n\nTotal unique words found: {len(word_frequency)}")
    print(f"Total word occurrences: {sum(word_frequency.values())}")
    
    # Save to file
    output_file = "extracted_hate_words.json"
    output_data = {
        "total_unique_words": len(word_frequency),
        "total_occurrences": sum(word_frequency.values()),
        "words_by_frequency": dict(word_frequency.most_common()),
        "all_words": list(word_frequency.keys())
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    return word_frequency

if __name__ == "__main__":
    print("=" * 80)
    print("HATE WORD EXTRACTION FROM TEST SET")
    print("=" * 80)
    print()
    
    try:
        # Check if server is running
        health = requests.get(f"{API_BASE_URL}/health")
        if not health.ok:
            print("Error: Server is not responding. Make sure the server is running.")
            exit(1)
        
        # Extract hate words
        word_freq = extract_hate_words_from_test_set(num_samples=100)
        
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server. Make sure it's running at http://localhost:8000")
    except Exception as e:
        print(f"Error: {e}")


