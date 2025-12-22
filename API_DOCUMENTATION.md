# API Documentation

## Customer Support Ticket Auto-Triage REST API

A RESTful API for real-time ticket classification and integration.

### Base URL

```
http://localhost:5000
```

### Authentication

Currently, no authentication is required. For production use, implement API keys or OAuth2.

---

## Endpoints

### 1. Health Check

Check API and model status.

**Endpoint:** `GET /api/health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "ensemble"
}
```

**Example:**
```bash
curl http://localhost:5000/api/health
```

---

### 2. Get Categories

Get list of available ticket categories and priorities.

**Endpoint:** `GET /api/categories`

**Response:**
```json
{
  "status": "success",
  "categories": [
    "Bug Report",
    "Feature Request",
    "Technical Issue",
    "Billing Inquiry",
    "Account Management"
  ],
  "priorities": ["Low", "Medium", "High", "Critical"]
}
```

**Example:**
```bash
curl http://localhost:5000/api/categories
```

---

### 3. Predict Single Ticket

Classify a single support ticket.

**Endpoint:** `POST /api/predict`

**Request Body:**
```json
{
  "subject": "Application crashes when opening reports",
  "description": "Every time I try to open the monthly report, the application crashes. This started happening after the last update."
}
```

**Response:**
```json
{
  "status": "success",
  "prediction": {
    "predicted_category": "Bug Report",
    "confidence": 0.95,
    "probabilities": {
      "Bug Report": 0.95,
      "Feature Request": 0.02,
      "Technical Issue": 0.01,
      "Billing Inquiry": 0.01,
      "Account Management": 0.01
    }
  },
  "input": {
    "subject": "Application crashes when opening reports",
    "description": "Every time I try to open the monthly report..."
  }
}
```

**Example with cURL:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Application crashes",
    "description": "The app crashes when opening reports"
  }'
```

**Example with Python:**
```python
import requests

url = "http://localhost:5000/api/predict"
data = {
    "subject": "Application crashes",
    "description": "The app crashes when opening reports"
}

response = requests.post(url, json=data)
result = response.json()
print(f"Predicted: {result['prediction']['predicted_category']}")
print(f"Confidence: {result['prediction']['confidence']:.2%}")
```

**Example with JavaScript:**
```javascript
fetch('http://localhost:5000/api/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    subject: 'Application crashes',
    description: 'The app crashes when opening reports'
  })
})
.then(response => response.json())
.then(data => {
  console.log('Predicted:', data.prediction.predicted_category);
  console.log('Confidence:', data.prediction.confidence);
});
```

---

### 4. Predict Batch Tickets

Classify multiple tickets in a single request.

**Endpoint:** `POST /api/predict/batch`

**Request Body:**
```json
{
  "tickets": [
    {
      "subject": "Application crashes",
      "description": "The app crashes when opening reports"
    },
    {
      "subject": "Add dark mode",
      "description": "Please add dark mode theme option"
    },
    {
      "subject": "Cannot connect to server",
      "description": "Unable to connect to the server"
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "total_tickets": 3,
  "predictions": [
    {
      "index": 0,
      "status": "success",
      "predicted_category": "Bug Report",
      "confidence": 0.95,
      "probabilities": {...}
    },
    {
      "index": 1,
      "status": "success",
      "predicted_category": "Feature Request",
      "confidence": 0.88,
      "probabilities": {...}
    },
    {
      "index": 2,
      "status": "success",
      "predicted_category": "Technical Issue",
      "confidence": 0.92,
      "probabilities": {...}
    }
  ]
}
```

**Example with cURL:**
```bash
curl -X POST http://localhost:5000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "tickets": [
      {"subject": "App crashes", "description": "The app crashes"},
      {"subject": "Add feature", "description": "Please add dark mode"}
    ]
  }'
```

**Example with Python:**
```python
import requests

url = "http://localhost:5000/api/predict/batch"
data = {
    "tickets": [
        {"subject": "App crashes", "description": "The app crashes"},
        {"subject": "Add feature", "description": "Please add dark mode"}
    ]
}

response = requests.post(url, json=data)
results = response.json()

for pred in results['predictions']:
    print(f"Ticket {pred['index']}: {pred['predicted_category']} ({pred['confidence']:.2%})")
```

---

## Error Responses

### 400 Bad Request
```json
{
  "status": "error",
  "message": "Missing required fields: subject and description"
}
```

### 404 Not Found
```json
{
  "status": "error",
  "message": "Endpoint not found",
  "available_endpoints": ["/api/predict", "/api/predict/batch", "/api/health", "/api/categories"]
}
```

### 500 Internal Server Error
```json
{
  "status": "error",
  "message": "Prediction error: [error details]"
}
```

---

## Running the API

### Start the Server

```bash
python api.py
```

The server will start on `http://localhost:5000` by default.

### Custom Port and Host

Set environment variables:
```bash
# Windows
set PORT=8080
set HOST=0.0.0.0
python api.py

# Linux/Mac
export PORT=8080
export HOST=0.0.0.0
python api.py
```

### Production Deployment

For production, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api:app
```

---

## Integration Examples

### Python Integration

```python
import requests
import json

class TicketClassifierAPI:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def predict(self, subject, description):
        """Predict category for a single ticket"""
        url = f"{self.base_url}/api/predict"
        response = requests.post(url, json={
            "subject": subject,
            "description": description
        })
        return response.json()
    
    def predict_batch(self, tickets):
        """Predict categories for multiple tickets"""
        url = f"{self.base_url}/api/predict/batch"
        response = requests.post(url, json={"tickets": tickets})
        return response.json()

# Usage
api = TicketClassifierAPI()
result = api.predict(
    subject="Application crashes",
    description="The app crashes when opening reports"
)
print(result['prediction']['predicted_category'])
```

### JavaScript/Node.js Integration

```javascript
class TicketClassifierAPI {
  constructor(baseUrl = 'http://localhost:5000') {
    this.baseUrl = baseUrl;
  }

  async predict(subject, description) {
    const response = await fetch(`${this.baseUrl}/api/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ subject, description })
    });
    return await response.json();
  }

  async predictBatch(tickets) {
    const response = await fetch(`${this.baseUrl}/api/predict/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ tickets })
    });
    return await response.json();
  }
}

// Usage
const api = new TicketClassifierAPI();
const result = await api.predict(
  'Application crashes',
  'The app crashes when opening reports'
);
console.log(result.prediction.predicted_category);
```

---

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider:
- Implementing rate limiting (e.g., using Flask-Limiter)
- Adding request throttling
- Setting up API keys for authentication

---

## Performance

- **Single Prediction**: ~50-100ms
- **Batch Prediction**: ~50-100ms per ticket
- **Model Load Time**: ~1-2 seconds on startup

---

## Troubleshooting

### Model Not Loaded Error
**Solution:** Ensure the model is trained first:
```bash
python train.py
```

### Port Already in Use
**Solution:** Change the port:
```bash
set PORT=8080
python api.py
```

### CORS Errors
**Solution:** CORS is enabled by default. If issues persist, check Flask-CORS configuration in `api.py`.

---

## API Version

Current API Version: **1.0.0**

For updates and changes, check the project repository.

