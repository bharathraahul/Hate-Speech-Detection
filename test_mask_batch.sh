#!/bin/bash
# Batch test script for masking feature

echo "=== Testing Hate Speech Examples (Should be masked) ==="
echo ""

echo "Test 1: Basic offensive word"
curl -s -X POST "http://localhost:8000/mask" \
  -H "Content-Type: application/json" \
  -d '{"text": "dont even entertain these hoes, got somebody too important"}' | python -m json.tool | grep -E "(masked_text|words_masked|message)"

echo ""
echo "Test 2: Multiple offensive words"
curl -s -X POST "http://localhost:8000/mask" \
  -H "Content-Type: application/json" \
  -d '{"text": "Those people are trash and bitches who should be eliminated"}' | python -m json.tool | grep -E "(masked_text|words_masked|message)"

echo ""
echo "Test 3: Religious hate speech"
curl -s -X POST "http://localhost:8000/mask" \
  -H "Content-Type: application/json" \
  -d '{"text": "Islam out of Britain. Protect the British people."}' | python -m json.tool | grep -E "(masked_text|words_masked|message)"

echo ""
echo "Test 4: Deportation pattern"
curl -s -X POST "http://localhost:8000/mask" \
  -H "Content-Type: application/json" \
  -d '{"text": "Muslims should be deported from our country"}' | python -m json.tool | grep -E "(masked_text|words_masked|message)"

echo ""
echo "Test 5: Custom mask character"
curl -s -X POST "http://localhost:8000/mask" \
  -H "Content-Type: application/json" \
  -d '{"text": "Ugly hoes love to fight", "mask_char": "[CENSORED]"}' | python -m json.tool | grep -E "(masked_text|words_masked|message)"

echo ""
echo "=== Testing Normal Text Examples (Should NOT be masked) ==="
echo ""

echo "Test 6: Normal friendly text"
curl -s -X POST "http://localhost:8000/mask" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love spending time with my friends"}' | python -m json.tool | grep -E "(masked_text|was_masked|message)"

echo ""
echo "Test 7: Normal discussion text"
curl -s -X POST "http://localhost:8000/mask" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a great idea and I think we should implement it"}' | python -m json.tool | grep -E "(masked_text|was_masked|message)"

echo ""
echo "Test 8: Normal appreciation text"
curl -s -X POST "http://localhost:8000/mask" \
  -H "Content-Type: application/json" \
  -d '{"text": "Thank you for your help, I really appreciate it"}' | python -m json.tool | grep -E "(masked_text|was_masked|message)"

