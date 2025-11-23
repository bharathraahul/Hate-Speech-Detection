# Paraphraser Feature - UI Integration Guide

## Overview
The paraphraser feature provides text paraphrasing functionality through a REST API endpoint. **IMPORTANT: The paraphraser ONLY paraphrases text if it contains hate speech.** Normal text will not be paraphrased. It supports both transformer-based (high quality) and rule-based (lightweight) paraphrasing methods.

## API Endpoint

### POST `/paraphrase`

Paraphrase input text using transformer or rule-based methods.

#### Request Body
```json
{
  "text": "The quick brown fox jumps over the lazy dog",
  "num_paraphrases": 1,
  "method": null
}
```

**Parameters:**
- `text` (required): The text to paraphrase
- `num_paraphrases` (optional, default: 1): Number of paraphrases to generate (1-5)
- `method` (optional): 
  - `"transformer"` - Use transformer model (requires transformers library)
  - `"rule_based"` - Use rule-based synonym replacement
  - `null` - Auto-select based on availability

#### Response (Hate Speech Detected)
```json
{
  "original_text": "Islam out of Britain. Protect the British people.",
  "paraphrases": [
    "Islam out of Britain. Protect the British people."
  ],
  "method_used": "rule_based",
  "count": 1,
  "is_hate_speech": true,
  "hate_speech_confidence": 0.8,
  "was_paraphrased": true,
  "message": "Hate speech detected (confidence: 80.00%). Text has been paraphrased."
}
```

#### Response (No Hate Speech)
```json
{
  "original_text": "I love spending time with my friends",
  "paraphrases": null,
  "method_used": null,
  "count": 0,
  "is_hate_speech": false,
  "hate_speech_confidence": 0.47,
  "was_paraphrased": false,
  "message": "No hate speech detected. Text was not paraphrased."
}
```

**Response Fields:**
- `original_text`: The input text
- `paraphrases`: List of paraphrased versions (null if no hate speech)
- `method_used`: Which method was used ("transformer" or "rule_based", null if not paraphrased)
- `count`: Number of paraphrases returned (0 if not paraphrased)
- `is_hate_speech`: Whether hate speech was detected
- `hate_speech_confidence`: Confidence score of hate speech detection (0.0-1.0)
- `was_paraphrased`: Whether paraphrasing was performed
- `message`: Explanation message

#### Example cURL Request
```bash
curl -X POST "http://localhost:8000/paraphrase" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I think this is a great idea",
    "num_paraphrases": 2
  }'
```

## Frontend Integration Examples

### JavaScript/React Example
```javascript
async function paraphraseText(text, numParaphrases = 1) {
  try {
    const response = await fetch('http://localhost:8000/paraphrase', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: text,
        num_paraphrases: numParaphrases,
        method: null  // Auto-select
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Paraphrasing error:', error);
    throw error;
  }
}

// Usage
paraphraseText("This is a test sentence", 2)
  .then(result => {
    console.log('Original:', result.original_text);
    console.log('Paraphrases:', result.paraphrases);
    console.log('Method used:', result.method_used);
  });
```

### React Component Example
```jsx
import React, { useState } from 'react';

function ParaphraserComponent() {
  const [inputText, setInputText] = useState('');
  const [paraphrases, setParaphrases] = useState([]);
  const [loading, setLoading] = useState(false);
  const [method, setMethod] = useState(null);

  const handleParaphrase = async () => {
    if (!inputText.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/paraphrase', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          num_paraphrases: 2,
          method: method
        })
      });
      
      const data = await response.json();
      setParaphrases(data.paraphrases);
      setMethod(data.method_used);
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to paraphrase text');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="paraphraser">
      <h2>Text Paraphraser</h2>
      
      <textarea
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        placeholder="Enter text to paraphrase..."
        rows={4}
        style={{ width: '100%', marginBottom: '10px' }}
      />
      
      <div style={{ marginBottom: '10px' }}>
        <label>
          Method:
          <select value={method || ''} onChange={(e) => setMethod(e.target.value || null)}>
            <option value="">Auto</option>
            <option value="transformer">Transformer</option>
            <option value="rule_based">Rule-based</option>
          </select>
        </label>
      </div>
      
      <button 
        onClick={handleParaphrase} 
        disabled={loading || !inputText.trim()}
      >
        {loading ? 'Paraphrasing...' : 'Paraphrase'}
      </button>
      
      {paraphrases.length > 0 && (
        <div style={{ marginTop: '20px' }}>
          <h3>Paraphrases ({method}):</h3>
          <ul>
            {paraphrases.map((para, idx) => (
              <li key={idx} style={{ marginBottom: '10px' }}>
                {para}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default ParaphraserComponent;
```

### Python Example (for testing)
```python
import requests

def paraphrase_text(text, num_paraphrases=1, method=None):
    """Call the paraphrase API"""
    url = "http://localhost:8000/paraphrase"
    payload = {
        "text": text,
        "num_paraphrases": num_paraphrases,
        "method": method
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

# Usage
result = paraphrase_text("I think this is a great idea", num_paraphrases=2)
print(f"Original: {result['original_text']}")
print(f"Paraphrases: {result['paraphrases']}")
print(f"Method: {result['method_used']}")
```

## Error Handling

The API may return the following HTTP status codes:

- `200 OK`: Success
- `400 Bad Request`: Invalid input (empty text, invalid parameters)
- `500 Internal Server Error`: Server error during paraphrasing

Example error response:
```json
{
  "detail": "Text cannot be empty"
}
```

## Configuration

The paraphraser can be configured in `config.py`:

```python
PARAPHRASER_CONFIG = {
    "use_transformer": True,  # Use transformer if available
    "model_name": "tuner007/pegasus_paraphrase",  # Model to use
    "fallback_to_rule_based": True,  # Fallback if transformer fails
    "max_text_length": 500,  # Max text length
    "default_num_paraphrases": 1,  # Default number of paraphrases
}
```

## Methods Comparison

### Transformer Method
- **Pros**: High quality, natural paraphrases, context-aware
- **Cons**: Requires transformers library, slower, more memory
- **Best for**: Production use when quality is important

### Rule-based Method
- **Pros**: Fast, lightweight, no dependencies, always available
- **Cons**: Lower quality, limited vocabulary, simple replacements
- **Best for**: Fallback, quick testing, when transformers unavailable

## Notes for UI Developers

1. **Loading States**: Paraphrasing can take 1-5 seconds with transformers. Show loading indicators.

2. **Text Length**: Maximum 500 characters. Truncate longer texts or show a warning.

3. **Multiple Paraphrases**: Users can request 1-5 paraphrases. Consider showing them in a list or dropdown.

4. **Method Selection**: Allow users to choose method, or use auto-select for best experience.

5. **Error Handling**: Always handle network errors and API errors gracefully.

6. **CORS**: If your UI is on a different domain, ensure CORS is configured in FastAPI:
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["http://localhost:3000"],  # Your UI URL
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

## Testing

Test the endpoint using the provided `test_predict.py` script or create a similar test:

```python
import requests

# Test basic paraphrasing
response = requests.post(
    "http://localhost:8000/paraphrase",
    json={"text": "This is a test sentence"}
)
print(response.json())
```

## Integration Checklist

- [ ] Add CORS middleware if UI is on different domain
- [ ] Implement loading states in UI
- [ ] Add error handling for API calls
- [ ] Test with various text lengths
- [ ] Test with different num_paraphrases values
- [ ] Test method selection (transformer vs rule_based)
- [ ] Add user feedback for long-running requests
- [ ] Consider caching paraphrases for repeated requests

