# 📖 CengBot API Documentation

## 🌐 Overview

The CengBot API provides RESTful endpoints for managing bot data, user interactions, and system administration. Built with FastAPI, it offers automatic OpenAPI documentation and high performance.

**Base URL**: `http://localhost:8001`  
**API Documentation**: `http://localhost:8001/docs` (Swagger UI)  
**Alternative Docs**: `http://localhost:8001/redoc` (ReDoc)

## 🔐 Authentication

Currently, the API uses IP-based access control. In production, implement proper authentication:

```python
# Example JWT authentication (future implementation)
@app.middleware("http")
async def verify_token(request: Request, call_next):
    token = request.headers.get("Authorization")
    if not token or not verify_jwt_token(token):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)
```

## 📊 Data Models

### RawData Model
```typescript
interface RawData {
  id: number;                    // Primary key
  telegram_id: number;           // Telegram user ID
  telegram_message_id?: number;  // Message ID from Telegram
  username?: string;             // User's display name
  question: string;              // Original question
  answer?: string;               // AI-generated response
  language?: string;             // Language code ('TR' or 'EN')
  like?: number;                 // User feedback (-1, 0, 1)
  admin_approved: number;        // Admin approval status (0 or 1)
  is_duplicate: boolean;         // Duplicate detection flag
  duplicate_of_id?: number;      // Reference to original question
  created_at: string;            // ISO timestamp
  answered_at?: string;          // Response timestamp
  message_thread_id?: number;    // Telegram topic ID
}
```

### TrainingData Model
```typescript
interface TrainingData {
  id: number;                    // Primary key
  source_id: number;             // Reference to raw_data.id
  question: string;              // Approved question
  answer: string;                // Approved answer
  language?: string;             // Language code
  created_at: string;            // Approval timestamp
  point?: number;                // User rating from source
}
```

### Statistics Model
```typescript
interface Statistics {
  total_questions: number;       // Total questions received
  answered_questions: number;    // Questions with responses
  liked_questions: number;       // Positively rated responses
  disliked_questions: number;    // Negatively rated responses
  approved_questions: number;    // Admin-approved questions
  training_data_count: number;   // Total training data entries
  duplicate_count: number;       // Number of duplicate questions
  languages: {                   // Language distribution
    TR: number;
    EN: number;
  };
  avg_response_time?: number;    // Average response time in seconds
  success_rate?: number;         // Percentage of liked responses
}
```

## 🔌 API Endpoints

### Raw Data Management

#### Get All Raw Data
```http
GET /raw-data
```

**Query Parameters:**
- `skip` (int, optional): Number of records to skip (default: 0)
- `limit` (int, optional): Maximum records to return (default: 100)
- `language` (string, optional): Filter by language ('TR' or 'EN')
- `admin_approved` (int, optional): Filter by approval status (0 or 1)
- `has_answer` (bool, optional): Filter by answer presence

**Response:**
```json
[
  {
    "id": 1,
    "telegram_id": 123456789,
    "username": "john_doe",
    "question": "What is computer science?",
    "answer": "Computer science is the study of algorithms...",
    "language": "EN",
    "like": 1,
    "admin_approved": 1,
    "is_duplicate": false,
    "created_at": "2024-01-15T10:30:00Z",
    "answered_at": "2024-01-15T10:30:15Z"
  }
]
```

**Example:**
```bash
curl -X GET "http://localhost:8001/raw-data?limit=50&language=TR"
```

#### Get Single Raw Data Entry
```http
GET /raw-data/{id}
```

**Path Parameters:**
- `id` (int): Raw data entry ID

**Response:**
```json
{
  "id": 1,
  "telegram_id": 123456789,
  "username": "john_doe",
  "question": "What is computer science?",
  "answer": "Computer science is...",
  "language": "EN",
  "like": 1,
  "admin_approved": 1,
  "is_duplicate": false,
  "created_at": "2024-01-15T10:30:00Z",
  "answered_at": "2024-01-15T10:30:15Z"
}
```

#### Update Raw Data Answer
```http
PUT /raw-data/{id}
```

**Path Parameters:**
- `id` (int): Raw data entry ID

**Request Body:**
```json
{
  "answer": "Updated answer text"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Answer updated successfully"
}
```

**Example:**
```bash
curl -X PUT "http://localhost:8001/raw-data/1" \
  -H "Content-Type: application/json" \
  -d '{"answer": "Updated answer"}'
```

#### Delete Raw Data Entry
```http
DELETE /raw-data/{id}
```

**Path Parameters:**
- `id` (int): Raw data entry ID

**Response:**
```json
{
  "success": true,
  "message": "Entry deleted successfully"
}
```

### Training Data Management

#### Get Training Data
```http
GET /training-data
```

**Query Parameters:**
- `skip` (int, optional): Number of records to skip
- `limit` (int, optional): Maximum records to return
- `language` (string, optional): Filter by language

**Response:**
```json
[
  {
    "id": 1,
    "source_id": 5,
    "question": "What is computer science?",
    "answer": "Computer science is the study of algorithms...",
    "language": "EN",
    "created_at": "2024-01-15T10:30:00Z",
    "point": 1
  }
]
```

#### Approve Question for Training
```http
POST /approve/{id}
```

**Path Parameters:**
- `id` (int): Raw data entry ID to approve

**Response:**
```json
{
  "success": true,
  "training_data_id": 15,
  "message": "Question approved for training"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8001/approve/1"
```

#### Bulk Approve Questions
```http
POST /bulk-approve
```

**Request Body:**
```json
{
  "ids": [1, 2, 3, 4, 5]
}
```

**Response:**
```json
{
  "success": true,
  "approved_count": 5,
  "training_data_ids": [15, 16, 17, 18, 19]
}
```

### Statistics and Analytics

#### Get System Statistics
```http
GET /stats
```

**Response:**
```json
{
  "total_questions": 1250,
  "answered_questions": 1180,
  "liked_questions": 950,
  "disliked_questions": 120,
  "approved_questions": 800,
  "training_data_count": 800,
  "duplicate_count": 45,
  "languages": {
    "TR": 800,
    "EN": 450
  },
  "avg_response_time": 2.5,
  "success_rate": 88.8
}
```

#### Get Duplicate Questions
```http
GET /duplicates
```

**Response:**
```json
[
  {
    "original_id": 1,
    "original_question": "What is computer science?",
    "duplicates": [
      {
        "id": 15,
        "question": "What is computer science?",
        "similarity": 1.0
      },
      {
        "id": 23,
        "question": "What's computer science?",
        "similarity": 0.95
      }
    ]
  }
]
```

#### Get Language Distribution
```http
GET /language-stats
```

**Response:**
```json
{
  "total_questions": 1250,
  "languages": {
    "TR": {
      "count": 800,
      "percentage": 64.0,
      "avg_likes": 0.7
    },
    "EN": {
      "count": 450,
      "percentage": 36.0,
      "avg_likes": 0.8
    }
  }
}
```

### Health and Monitoring

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "database": "connected",
  "model": "loaded",
  "rabbitmq": "connected"
}
```

#### System Status
```http
GET /status
```

**Response:**
```json
{
  "system": {
    "uptime": "2 days, 14 hours",
    "memory_usage": "2.5 GB",
    "cpu_usage": "15%",
    "disk_usage": "45%"
  },
  "services": {
    "telegram_bot": "running",
    "ai_worker": "running",
    "database": "connected",
    "rabbitmq": "connected"
  },
  "model": {
    "status": "loaded",
    "model_name": "LLaMA-3.2-3B",
    "adapter": "method1",
    "memory_usage": "1.8 GB"
  }
}
```

## 🔧 Advanced Features

### Filtering and Searching

#### Search Questions
```http
GET /search
```

**Query Parameters:**
- `q` (string): Search query
- `language` (string, optional): Filter by language
- `limit` (int, optional): Maximum results

**Response:**
```json
{
  "query": "computer science",
  "results": [
    {
      "id": 1,
      "question": "What is computer science?",
      "answer": "Computer science is...",
      "relevance": 0.95,
      "language": "EN"
    }
  ],
  "total": 1
}
```

#### Advanced Filtering
```http
GET /raw-data/filter
```

**Query Parameters:**
- `start_date` (string): Start date (ISO format)
- `end_date` (string): End date (ISO format)
- `min_likes` (int): Minimum like count
- `max_likes` (int): Maximum like count
- `username` (string): Filter by username

**Example:**
```bash
curl -X GET "http://localhost:8001/raw-data/filter?start_date=2024-01-01&min_likes=0"
```

### Export and Import

#### Export Data
```http
GET /export
```

**Query Parameters:**
- `format` (string): Export format ('json', 'csv', 'xlsx')
- `table` (string): Table to export ('raw_data', 'training_data', 'all')
- `language` (string, optional): Filter by language

**Response:**
- JSON: Returns data as JSON
- CSV: Returns CSV file
- XLSX: Returns Excel file

**Example:**
```bash
curl -X GET "http://localhost:8001/export?format=csv&table=training_data" \
  -o training_data.csv
```

#### Import Training Data
```http
POST /import
```

**Request Body (multipart/form-data):**
- `file`: CSV/JSON file with training data
- `format`: File format ('csv' or 'json')

**Response:**
```json
{
  "success": true,
  "imported_count": 100,
  "skipped_count": 5,
  "errors": []
}
```

## 🚫 Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "question",
      "issue": "Question cannot be empty"
    }
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "path": "/raw-data/1"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 400 | Invalid input data |
| `DUPLICATE_ENTRY` | 409 | Duplicate resource |
| `INTERNAL_ERROR` | 500 | Server error |
| `DATABASE_ERROR` | 500 | Database connection issue |
| `MODEL_ERROR` | 503 | AI model unavailable |

## 📡 WebSocket Endpoints

### Real-time Updates
```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8001/ws');

// Listen for real-time updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('New data:', data);
};

// Send message
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'raw_data'
}));
```

### Available Channels
- `raw_data`: New questions and answers
- `training_data`: New training data approvals
- `stats`: Real-time statistics updates
- `system`: System status changes

## 🔐 Rate Limiting

### Default Limits
- **General API**: 100 requests per minute per IP
- **Search API**: 20 requests per minute per IP
- **Export API**: 5 requests per minute per IP

### Rate Limit Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248600
```

## 📊 API Usage Examples

### Python Client Example
```python
import requests

class CengBotAPI:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
    
    def get_raw_data(self, limit=100, language=None):
        params = {"limit": limit}
        if language:
            params["language"] = language
        
        response = requests.get(f"{self.base_url}/raw-data", params=params)
        return response.json()
    
    def update_answer(self, id, answer):
        data = {"answer": answer}
        response = requests.put(f"{self.base_url}/raw-data/{id}", json=data)
        return response.json()
    
    def approve_question(self, id):
        response = requests.post(f"{self.base_url}/approve/{id}")
        return response.json()
    
    def get_stats(self):
        response = requests.get(f"{self.base_url}/stats")
        return response.json()

# Usage
api = CengBotAPI()
data = api.get_raw_data(limit=50, language="TR")
stats = api.get_stats()
```

### JavaScript Client Example
```javascript
class CengBotAPI {
  constructor(baseUrl = 'http://localhost:8001') {
    this.baseUrl = baseUrl;
  }
  
  async getRawData(limit = 100, language = null) {
    const params = new URLSearchParams({ limit });
    if (language) params.append('language', language);
    
    const response = await fetch(`${this.baseUrl}/raw-data?${params}`);
    return response.json();
  }
  
  async updateAnswer(id, answer) {
    const response = await fetch(`${this.baseUrl}/raw-data/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ answer })
    });
    return response.json();
  }
  
  async approveQuestion(id) {
    const response = await fetch(`${this.baseUrl}/approve/${id}`, {
      method: 'POST'
    });
    return response.json();
  }
  
  async getStats() {
    const response = await fetch(`${this.baseUrl}/stats`);
    return response.json();
  }
}

// Usage
const api = new CengBotAPI();
const data = await api.getRawData(50, 'TR');
const stats = await api.getStats();
```

## 📈 Performance Tips

### Pagination
Always use pagination for large datasets:
```bash
# Get first page
curl "http://localhost:8001/raw-data?limit=50&skip=0"

# Get second page
curl "http://localhost:8001/raw-data?limit=50&skip=50"
```

### Caching
The API supports ETags for caching:
```bash
# Initial request
curl -H "If-None-Match: \"etag-value\"" "http://localhost:8001/stats"

# Returns 304 Not Modified if unchanged
```

### Compression
Enable gzip compression:
```bash
curl -H "Accept-Encoding: gzip" "http://localhost:8001/raw-data"
```

This API documentation provides comprehensive information for integrating with the CengBot system and managing its data effectively.