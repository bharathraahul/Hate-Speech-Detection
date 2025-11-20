#!/usr/bin/env python3
"""
Simple script to test hate speech detection predictions
"""

import requests
import json
import sys

API_URL = "http://localhost:8000/predict"

def predict_text(text):
    """Make a prediction for a given text"""
    try:
        response = requests.post(
            API_URL,
            json={"text": text},
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
    """Pretty print the prediction result"""
    if result is None:
        return
    
    print("\n" + "="*60)
    print(f"Text: {result['text']}")
    print(f"Prediction: {result['label'].upper()}")
    print(f"Is Hate Speech: {result['is_hate_speech']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("="*60 + "\n")

def main():
    if len(sys.argv) > 1:
        # Test with command line argument
        text = " ".join(sys.argv[1:])
        result = predict_text(text)
        print_result(result)
    else:
        # Interactive mode
        print("Hate Speech Detection Tester")
        print("Type 'quit' or 'exit' to stop\n")
        
        while True:
            text = input("Enter text to analyze: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                print("Please enter some text.\n")
                continue
            
            result = predict_text(text)
            print_result(result)

if __name__ == "__main__":
    main()

