# 📖 CengBot API Documentation

## 🌐 Overview

The CengBot API provides RESTful endpoints for managing bot data, user interactions, and system administration. Built with FastAPI, it offers automatic OpenAPI documentation and high performance.

**Base URL**: `http://localhost:8001`  
**API Documentation**: `http://localhost:8001/docs` (Swagger UI)  
**Alternative Docs**: `http://localhost:8001/redoc` (ReDoc)

## 🔐 Authentication

The API uses HTTP Basic Authentication. Credentials are configured via environment variables:

```bash
ADMIN_USERNAME=admin
ADMIN_PASSWORD_HASH=<sha256_hash_of_password>
```

To generate password hash:
```python
import hashlib
password_hash = hashlib.sha256('your_password'.encode()).hexdigest()
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
  // Vote statistics
  likes?: number;                // Total likes count
  dislikes?: number;             // Total dislikes count
  total_votes?: number;          // Total votes count
  vote_score?: number;           // Vote score calculation
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

### PaginatedResponse Model
```typescript
interface PaginatedResponse<T> {
  items: T[];                    // Array of items
  total: number;                 // Total count of items
  page: number;                  // Current page number
  page_size: number;             // Items per page
  total_pages: number;           // Total number of pages
  has_next: boolean;             // Whether next page exists
  has_prev: boolean;             // Whether previous page exists
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

### General Endpoints

#### Root Endpoint
```http
GET /
```
Returns API welcome message and version info.

#### Health Check
```http
GET /health
```
Returns system health status including database connectivity and model status.

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "model": "loaded",
  "timestamp": "2025-08-01T10:30:00Z"
}
```

### Authentication

#### Login
```http
POST /auth/login
```

**Request Body:**
```json
{
  "username": "admin",
  "password": "your_password"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Login successful"
}
```

### Raw Data Management

#### Get All Raw Data (Paginated)
```http
GET /raw-data
```

**Query Parameters:**
- `page` (int, optional): Page number (default: 1)
- `page_size` (int, optional): Items per page (default: 50, max: 100)
- `language` (string, optional): Filter by language ('TR' or 'EN')
- `admin_approved` (int, optional): Filter by approval status (0 or 1)
- `has_answer` (bool, optional): Filter by answer presence

**Response:**
```json
{
  "items": [
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
      "answered_at": "2024-01-15T10:30:15Z",
      "likes": 5,
      "dislikes": 1,
      "total_votes": 6,
      "vote_score": 0.83
    }
  ],
  "total": 150,
  "page": 1,
  "page_size": 50,
  "total_pages": 3,
  "has_next": true,
  "has_prev": false
}
```

#### Update Raw Data Answer
```http
PUT /raw-data/{item_id}
```

**Path Parameters:**
- `item_id` (int): Raw data entry ID

**Request Body:**
```json
{
  "answer": "Updated answer text here"
}
```

**Response:**
```json
{
  "id": 1,
  "answer": "Updated answer text here",
  "admin_approved": 0
}
```

#### Delete Raw Data Entry
```http
DELETE /raw-data/{item_id}
```

**Path Parameters:**
- `item_id` (int): Raw data entry ID

**Response:**
```json
{
  "message": "Raw data entry deleted successfully"
}
```

#### Approve for Training Data
```http
POST /approve/{item_id}
```

**Path Parameters:**
- `item_id` (int): Raw data entry ID to approve

**Response:**
```json
{
  "message": "Approved and added to training data",
  "training_data_id": 123
}
```

### Training Data Management

#### Get Training Data (Paginated)
```http
GET /training-data
```

**Query Parameters:**
- `page` (int, optional): Page number (default: 1)
- `page_size` (int, optional): Items per page (default: 50, max: 100)
- `language` (string, optional): Filter by language

**Response:**
```json
{
  "items": [
    {
      "id": 1,
      "source_id": 10,
      "question": "What is OOP?",
      "answer": "Object-Oriented Programming is...",
      "language": "EN",
      "created_at": "2024-01-16T14:20:00Z",
      "point": 1
    }
  ],
  "total": 500,
  "page": 1,
  "page_size": 50,
  "total_pages": 10,
  "has_next": true,
  "has_prev": false
}
```

#### Remove from Training Data
```http
DELETE /training-data/{item_id}
```

**Path Parameters:**
- `item_id` (int): Training data entry ID

**Response:**
```json
{
  "message": "Training data entry removed successfully"
}
```

### Analytics & Statistics

#### Get Statistics
```http
GET /stats
```

**Response:**
```json
{
  "total_questions": 1250,
  "answered_questions": 1200,
  "liked_questions": 850,
  "disliked_questions": 150,
  "approved_questions": 500,
  "training_data_count": 500,
  "duplicate_count": 75,
  "languages": {
    "TR": 700,
    "EN": 550
  },
  "avg_response_time": 2.5,
  "success_rate": 85.0
}
```

#### Get Duplicate Groups
```http
GET /duplicates
```

**Response:**
```json
[
  {
    "group_id": 1,
    "count": 3,
    "questions": [
      {
        "id": 10,
        "question": "What is Python?",
        "telegram_id": 12345,
        "created_at": "2024-01-15T10:00:00Z"
      },
      {
        "id": 25,
        "question": "What is Python programming?",
        "telegram_id": 67890,
        "created_at": "2024-01-16T11:00:00Z"
      }
    ]
  }
]
```

#### Detect Duplicates
```http
POST /detect-duplicates
```

Runs duplicate detection algorithm on existing data.

**Response:**
```json
{
  "message": "Duplicate detection completed",
  "duplicates_found": 25,
  "groups_created": 10
}
```

### Documentation

#### List Available Documents
```http
GET /docs/list
```

**Response:**
```json
{
  "documents": [
    {
      "name": "README.md",
      "path": "README.md",
      "type": "markdown",
      "size": 45678
    },
    {
      "name": "training_20240731_144939_3511.log",
      "path": "logs/train_logs_llama_v1.2/training_20240731_144939_3511.log",
      "type": "log",
      "size": 12345
    }
  ]
}
```

#### Read Document
```http
GET /docs/read
```

**Query Parameters:**
- `path` (string): Path to the document file

**Response:**
```json
{
  "content": "# Document Content\n\nFile content here...",
  "filename": "README.md",
  "type": "markdown"
}
```

## 🚨 Error Responses

All endpoints return consistent error responses:

```json
{
  "detail": "Error message here"
}
```

Common HTTP status codes:
- `400`: Bad Request - Invalid parameters
- `401`: Unauthorized - Authentication required
- `404`: Not Found - Resource doesn't exist
- `422`: Unprocessable Entity - Validation error
- `500`: Internal Server Error - Server error

## 📝 Example Usage

### Python Example
```python
import requests
from requests.auth import HTTPBasicAuth

# Configuration
BASE_URL = "http://localhost:8001"
auth = HTTPBasicAuth("admin", "your_password")

# Get statistics
response = requests.get(f"{BASE_URL}/stats", auth=auth)
stats = response.json()
print(f"Total questions: {stats['total_questions']}")

# Get raw data with pagination
response = requests.get(
    f"{BASE_URL}/raw-data",
    params={"page": 1, "page_size": 50, "language": "TR"},
    auth=auth
)
data = response.json()
for item in data["items"]:
    print(f"Q: {item['question']}")
    print(f"A: {item['answer']}")

# Update an answer
response = requests.put(
    f"{BASE_URL}/raw-data/123",
    json={"answer": "Updated answer"},
    auth=auth
)
```

### JavaScript/TypeScript Example
```typescript
const BASE_URL = 'http://localhost:8001';
const auth = btoa('admin:your_password');

// Get statistics
fetch(`${BASE_URL}/stats`, {
  headers: {
    'Authorization': `Basic ${auth}`
  }
})
.then(res => res.json())
.then(stats => {
  console.log(`Total questions: ${stats.total_questions}`);
});

// Update answer
fetch(`${BASE_URL}/raw-data/123`, {
  method: 'PUT',
  headers: {
    'Authorization': `Basic ${auth}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    answer: 'Updated answer'
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

## 🔄 Pagination

Paginated endpoints use consistent query parameters:
- `page`: Page number (starts from 1)
- `page_size`: Number of items per page

Response includes metadata:
- `total`: Total number of items
- `total_pages`: Total number of pages
- `has_next`: Boolean indicating if next page exists
- `has_prev`: Boolean indicating if previous page exists

## 📊 Rate Limiting

Currently no rate limiting is implemented. In production, consider adding rate limiting:
- 100 requests per minute for authenticated users
- 10 requests per minute for unauthenticated users

## 🔗 CORS Configuration

CORS is enabled for all origins in development. For production, configure specific allowed origins:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```