#!/usr/bin/env python3
"""
Simple script to test the paraphraser API endpoint
"""

import requests
import json
import sys

API_URL = "http://localhost:8000/paraphrase"

def paraphrase_text(text, num_paraphrases=1, method=None):
    """Make a paraphrase request for given text"""
    try:
        response = requests.post(
            API_URL,
            json={
                "text": text,
                "num_paraphrases": num_paraphrases,
                "method": method
            },
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to API: {e}")
        print("Make sure the server is running at http://localhost:8000")
        return None

def print_result(result):
    """Pretty print the paraphrase result"""
    if result is None:
        return
    
    print("\n" + "="*60)
    print(f"Original Text: {result['original_text']}")
    print(f"Method Used: {result['method_used']}")
    print(f"Number of Paraphrases: {result['count']}")
    print("-"*60)
    for i, para in enumerate(result['paraphrases'], 1):
        print(f"Paraphrase {i}: {para}")
    print("="*60 + "\n")

def main():
    if len(sys.argv) > 1:
        # Test with command line argument
        text = " ".join(sys.argv[1:])
        result = paraphrase_text(text, num_paraphrases=2)
        print_result(result)
    else:
        # Interactive mode
        print("Paraphraser API Tester")
        print("Type 'quit' or 'exit' to stop\n")
        
        while True:
            text = input("Enter text to paraphrase: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                print("Please enter some text.\n")
                continue
            
            # Ask for method
            method_input = input("Method (transformer/rule_based/auto, default=auto): ").strip().lower()
            method = None if method_input in ['', 'auto'] else method_input
            
            # Ask for number of paraphrases
            num_input = input("Number of paraphrases (1-5, default=1): ").strip()
            num_paraphrases = int(num_input) if num_input.isdigit() and 1 <= int(num_input) <= 5 else 1
            
            result = paraphrase_text(text, num_paraphrases=num_paraphrases, method=method)
            print_result(result)

if __name__ == "__main__":
    main()

