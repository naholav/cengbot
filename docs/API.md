# CengBot Admin API

## Overview

The admin API (`src/admin_rest_api.py`, FastAPI) serves the React admin panel:
it lists and edits the collected questions, manages the training data, reports
statistics and duplicates, and shows the project's documentation and logs.

- **Base URL**: `http://localhost:8001` (`API_HOST` / `API_PORT` in `.env`;
  default `127.0.0.1`; expose it through the nginx `/api/` proxy, not `0.0.0.0`)
- **Interactive docs**: `http://localhost:8001/docs` (Swagger UI) and
  `http://localhost:8001/redoc`
- **Start**: `.venv/bin/python src/admin_rest_api.py` (also started by
  `scripts/start_system.sh` and the `cengbot-admin` systemd unit). With
  `API_RELOAD=true` it runs with auto-reload (development only).

## Authentication

Every endpoint that reads or modifies data requires **HTTP Basic
authentication**. Only these endpoints are public:

- `GET /` (static banner)
- `GET /health` (minimal status; details only with valid credentials)
- FastAPI's own `/docs`, `/redoc` and `/openapi.json` (the schema, no data)

Credentials come from the environment (`.env`):

```bash
ADMIN_USERNAME=admin
# Preferred: SHA-256 hex digest (64 hex characters) of the password
ADMIN_PASSWORD_HASH=<sha256 hex digest>
# Alternative: plain password (used only when no valid hash is set)
ADMIN_PASSWORD=
```

> **There is no default password.** Until `ADMIN_PASSWORD_HASH` or
> `ADMIN_PASSWORD` is set, every request with credentials is rejected with
> `401` and the API logs an ERROR at startup, so nobody can log into the panel.
> An empty or malformed hash, the hash of the empty password and placeholder
> values (`your_...`, `changeme`, `<...>`, ...) count as "not set".
> Use an **ASCII-only** password: FastAPI decodes HTTP Basic credentials as
> ASCII, so a password containing `ç`, `ğ`, `ş` ... can never log in.

The values are read from the process environment on every request, but `.env`
is loaded only when the API starts, so **restart the API after editing `.env`**.
Username and password are compared in constant time. Wrong or missing
credentials return `401` with `WWW-Authenticate: Basic`.

**Failed logins are rate limited per client IP** (in memory, per API process):
after `ADMIN_LOGIN_MAX_FAILURES` (default 10) wrong passwords within
`ADMIN_LOGIN_WINDOW_SECONDS` (900) the address gets `429 Too Many Requests` with
a `Retry-After` header for `ADMIN_LOGIN_LOCKOUT_SECONDS` (900), even with the
right password; `ADMIN_LOGIN_MAX_FAILURES=0` disables it. Credentials sent to
`/health` count too. Behind nginx the client IP comes from `X-Forwarded-For`,
which uvicorn trusts only from `FORWARDED_ALLOW_IPS` (default `127.0.0.1`: set it
to the proxy's address if nginx runs on another host).

Generate the hash (the password is read without echo):

```bash
python3 -c "import hashlib,getpass; print(hashlib.sha256(getpass.getpass().encode()).hexdigest())"
```

There is no session or token: the client sends the `Authorization` header with
every request. The admin panel keeps it in memory and in `sessionStorage`
(cleared when the browser tab closes), never in `localStorage`.

## Data Models

### RawData
```typescript
interface RawData {
  id: number;                    // Primary key
  telegram_id: number;           // Telegram user ID of the asker
  telegram_message_id?: number;  // ID of the bot's reply message (votes are checked against it)
  username?: string;             // Telegram username or first name
  question: string;              // The question
  answer?: string;               // The answer (worker or admin)
  language?: string;             // 'TR' or 'EN'
  like?: number;                 // 1 / -1 / null: sign of (likes - dislikes), kept in sync with the votes
  admin_approved: number;        // 0 pending, 1 approved to training data
  is_duplicate: boolean;         // Marked as a duplicate question
  duplicate_of_id?: number;      // The original question
  similarity_score?: number;     // Similarity to the original question
  created_at?: string;           // ISO timestamp (UTC)
  answered_at?: string;          // ISO timestamp (UTC)
  message_thread_id?: number;    // Telegram forum topic
  likes?: number;                // Vote statistics from user_votes
  dislikes?: number;
  total_votes?: number;
  vote_score?: number;           // likes - dislikes
}
```

### TrainingData
```typescript
interface TrainingData {
  id: number;                       // Primary key
  source_id?: number;               // raw_data.id (null for standalone rows)
  question: string;
  answer: string;
  language?: string;
  created_at?: string;              // Approval timestamp (UTC)
  point?: number;                   // 1 liked / -1 disliked / null, from the source row
  is_answer_duplicate?: boolean;    // Marked as a duplicate answer
  duplicate_answer_of_id?: number;  // The original training row
  answer_similarity_score?: number; // Similarity to that answer
}
```

### PaginatedResponse
```typescript
interface PaginatedResponse<T> {
  data: T[];          // Items of this page
  total: number;      // Total number of items
  page: number;       // Current page (starts at 1)
  page_size: number;  // Items per page
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}
```

### Statistics
```typescript
interface Statistics {
  total_questions: number;
  approved_questions: number;
  liked_questions: number;       // like = 1, or no legacy value and vote score > 0
  disliked_questions: number;    // like = -1, or no legacy value and vote score < 0
  training_data_count: number;
  duplicate_count: number;       // raw_data rows marked as duplicates
  unanswered_questions: number;  // answer NULL or blank
  approval_rate: string;         // e.g. "42.0%"
}
```

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/` | public | Banner |
| GET | `/health` | public (details with auth) | Health status |
| POST | `/auth/login` | required | Check credentials |
| GET | `/raw-data` | required | Questions, paginated |
| PUT | `/raw-data/{item_id}` | required | Edit an answer |
| DELETE | `/raw-data/{item_id}` | required | Delete a question |
| POST | `/approve/{item_id}` | required | Copy a question to the training data |
| GET | `/training-data` | required | Training data, paginated |
| DELETE | `/training-data/{item_id}` | required | Remove from the training data |
| GET | `/stats` | required | Statistics |
| GET | `/duplicates` | required | Duplicate question groups (paged: `limit`, `offset`) |
| POST | `/detect-duplicates` | required | Recompute all duplicate markings |
| GET | `/docs/list` | required | Viewable documentation, log and script files |
| GET | `/docs/read?path=...` | required | Read one of those files |

`item_id` must be an integer from 1 to 2^63-1 (otherwise `422`).

### General (public)

#### `GET /`
Returns `{"message": "University Bot Admin API"}`.

#### `GET /health`
Without credentials (or with wrong ones, which count as failed logins) only the
minimal status is returned:

```json
{
  "status": "healthy",
  "timestamp": "2026-11-11T10:30:00.123456",
  "database": {"healthy": true},
  "model": {"available": true}
}
```

- `database.healthy`: a `SELECT 1` succeeded. If it fails, `status` is
  `"unhealthy"` and the HTTP status is `503`.
- `model.available`: the model path (`MODEL_PATH`, default
  `models/active-model`) exists. This does not mean the worker has loaded it.

With valid credentials the response also contains `system` (CPU, memory and
disk usage) and `gpu` (from `nvidia-smi`, cached for 30 s; the API process never
initialises CUDA).

### Authentication

#### `POST /auth/login`
```http
POST /auth/login
Authorization: Basic base64(username:password)
```
No body. Returns `{"success": true, "username": "admin"}`, or `401`.

### Raw data

#### `GET /raw-data`
Query parameters:
- `page` (int, 1 to 1,000,000, default 1)
- `page_size` (int, 1 to 100, default 20)
- `only_unapproved` (bool, default false): only rows not yet approved

Rows are returned newest first. Response: `PaginatedResponse<RawData>`.

```json
{
  "data": [
    {
      "id": 1,
      "telegram_id": 123456789,
      "telegram_message_id": 4567,
      "username": "john_doe",
      "question": "What is computer science?",
      "answer": "Computer science is the study of algorithms...",
      "language": "EN",
      "like": 1,
      "admin_approved": 1,
      "is_duplicate": false,
      "created_at": "2026-01-15T10:30:00",
      "answered_at": "2026-01-15T10:30:03",
      "likes": 5,
      "dislikes": 1,
      "total_votes": 6,
      "vote_score": 4
    }
  ],
  "total": 150,
  "page": 1,
  "page_size": 20,
  "total_pages": 8,
  "has_next": true,
  "has_prev": false
}
```

#### `PUT /raw-data/{item_id}`
Body: `{"answer": "Updated answer text"}`. The answer is stripped, must not be
blank and may be at most 20,000 characters (`422` otherwise).

- If the row was already approved, its training copy gets the new answer too,
  and the training copy's duplicate-answer markings are recomputed.
- Rows that the worker answered automatically (an apology after repeated
  failures, a prompt-guard refusal, or an answer the output guard replaced
  with the safe fallback) carry an `[auto]` note and cannot be approved; editing the answer turns the note into `[auto-edited]`, after which
  the row can be approved.

```json
{
  "success": true,
  "message": "Answer updated successfully",
  "training_data_updated": false
}
```
`404` if the row does not exist.

#### `DELETE /raw-data/{item_id}`
Deletes the row together with its training copy and its votes, and clears
duplicate references that point to it.
Returns `{"success": true, "message": "Raw data deleted successfully"}`, or `404`.

#### `POST /approve/{item_id}`
Copies the question and answer into `training_data` (`source_type` "original",
`review_status` "approved", `reviewer_id` = the admin user), sets
`admin_approved = 1`, then marks duplicate questions and answers.

```json
{
  "success": true,
  "message": "Successfully approved to training data with duplicate detection",
  "training_data_id": 123
}
```

Errors: `404` row not found; `400` `Cannot approve without answer`; `400` for an
automatic error/refusal answer that was not edited; `400` `Already in training data`.

### Training data

#### `GET /training-data`
Query parameters `page` and `page_size` as for `/raw-data`. Newest first.
Response: `PaginatedResponse<TrainingData>`.

#### `DELETE /training-data/{item_id}`
Removes the row from the training data; its source question goes back to
"pending" (`admin_approved = 0`). Duplicate references to it are cleared.
Returns `{"success": true, "message": "Training data removed successfully"}`, or `404`.

### Statistics and duplicates

#### `GET /stats`
Response: `Statistics`.
```json
{
  "total_questions": 1250,
  "approved_questions": 500,
  "liked_questions": 850,
  "disliked_questions": 150,
  "training_data_count": 500,
  "duplicate_count": 75,
  "unanswered_questions": 50,
  "approval_rate": "40.0%"
}
```

#### `GET /duplicates`
One entry per original question; both `original` and `duplicates` are `RawData`
objects (with vote statistics). Groups are ordered by the original's id and
paged with `limit` (1 to 1000, default 500) and `offset` (default 0); the
`X-Total-Count` response header holds the total number of groups.
```json
[
  {
    "original": {"id": 10, "question": "What is Python?", "...": "..."},
    "duplicates": [
      {"id": 25, "question": "What is Python programming?", "similarity_score": 0.82, "...": "..."}
    ]
  }
]
```

#### `POST /detect-duplicates`
Recomputes every duplicate marking with the same similarity, thresholds and
window the bot uses for new questions: a question is a duplicate when its
TF-IDF cosine similarity to one of the 100 questions before it is at least 0.45
(it then points to the oldest original); a training answer is a duplicate when
its similarity to an older training answer is at least 0.85. Unlike the bot,
the recheck only considers texts longer than 10 characters. Everything is computed first and written in one transaction, so a
failure leaves the previous markings untouched.

```json
{
  "success": true,
  "message": "Duplicate detection completed. Found 25 question duplicates and 4 answer duplicates",
  "question_duplicates": 25,
  "answer_duplicates": 4
}
```

### Documentation and log viewer

Only an allowlist of files can be listed or read:

- Markdown files (`*.md`) and files named `README`, outside hidden directories
  and outside `node_modules`, `__pycache__`, `model_cache`, `build`,
  `site-packages`, `venv`, `env`
- `logs/*.log` (directly in `logs/`)
- `scripts/*.sh` (directly in `scripts/`)

`.env`, the database, model files and everything else are never served, and
symlinks that point outside the project are ignored.

#### `GET /docs/list`
Documentation first, then logs, then scripts, each sorted by path:
```json
[
  {"path": "README.md", "name": "README.md", "type": "documentation", "size": 45678},
  {"path": "logs/bot.log", "name": "bot.log", "type": "log", "size": 12345},
  {"path": "scripts/start_system.sh", "name": "start_system.sh", "type": "script", "size": 6608}
]
```

#### `GET /docs/read?path=README.md`
`path` is relative to the project root (1 to 1024 characters; no absolute paths,
no `..`, no NUL bytes). The response is `text/plain` with at most 1 MiB of
content: the **last** 1 MiB of a log (starting at a line boundary), the
**first** 1 MiB of other files, with a note when the file was cut. The headers
`X-Content-Truncated: true|false` and `X-File-Size: <bytes>` describe it.

Errors: `400` invalid path or not a file, `403` file type not viewable,
`404` not found, `500` read error.

## Errors

Errors use FastAPI's format:

```json
{"detail": "Error message here"}
```

For `422` validation errors, `detail` is a list describing each invalid field.
Database errors return `500` with a generic message; details go to the server
log and `logs/error.log`, not to the client.

| Code | Meaning |
|---|---|
| `400` | Invalid request (see the endpoint) |
| `401` | Missing or wrong credentials, or no admin password configured |
| `403` | File type not viewable (`/docs/read`) |
| `404` | Resource not found |
| `422` | Validation error (query, path or body) |
| `429` | Too many failed logins from this client IP (`Retry-After` header) |
| `500` | Server or database error |
| `503` | `/health` when the database is unreachable |

## Examples

### Python
```python
import getpass

import requests
from requests.auth import HTTPBasicAuth

BASE_URL = "http://localhost:8001"
auth = HTTPBasicAuth("admin", getpass.getpass("Admin password: "))

stats = requests.get(f"{BASE_URL}/stats", auth=auth, timeout=30)
stats.raise_for_status()
print("Total questions:", stats.json()["total_questions"])

page = requests.get(
    f"{BASE_URL}/raw-data",
    params={"page": 1, "page_size": 50, "only_unapproved": True},
    auth=auth,
    timeout=30,
).json()
for item in page["data"]:
    print(item["id"], item["question"], "->", item["answer"])

requests.put(f"{BASE_URL}/raw-data/123", json={"answer": "Updated answer"}, auth=auth, timeout=30)
```

### TypeScript
```typescript
const BASE_URL = 'http://localhost:8001';
const auth = btoa(`${username}:${password}`);  // ASCII credentials; never hard-code the password

const stats = await fetch(`${BASE_URL}/stats`, {
  headers: { Authorization: `Basic ${auth}` },
}).then((res) => res.json());
console.log(`Total questions: ${stats.total_questions}`);

await fetch(`${BASE_URL}/raw-data/123`, {
  method: 'PUT',
  headers: { Authorization: `Basic ${auth}`, 'Content-Type': 'application/json' },
  body: JSON.stringify({ answer: 'Updated answer' }),
});
```

## Pagination

`/raw-data` and `/training-data` take `page` (1 to 1,000,000) and `page_size`
(1 to 100, default 20) and return the `PaginatedResponse` envelope.
`total_pages` is 0 when there are no rows; a page past the end returns an empty
`data` list.

## Rate Limiting

Only failed logins are limited (see [Authentication](#authentication): `429`
with `Retry-After` per client IP, `ADMIN_LOGIN_*`); other requests are not.
HTTP Basic sends the password with every request, so keep the API on
`127.0.0.1` and publish it only through an HTTPS reverse proxy (see
[DEPLOYMENT.md](DEPLOYMENT.md)); add general request limits there if needed.

## CORS

Allowed origins come from `CORS_ORIGINS` (comma separated; default
`http://localhost:3000`). Trailing slashes are removed. It must contain the
exact origin (`scheme://host:port`) the admin panel is served from. Credentials
are allowed for the listed origins; if `*` is configured, any origin is allowed
and credentials are disabled. Allowed methods: `GET`, `POST`, `PUT`, `DELETE`,
`OPTIONS`; allowed headers: `Authorization`, `Content-Type`, `Accept`.

```bash
CORS_ORIGINS=http://localhost:3000,https://admin.example.edu
```

The panel's API base URL (`REACT_APP_API_URL`, default `http://localhost:8001`)
is baked into the frontend by Vite when the dev server starts or the build
runs. `scripts/deploy_production.sh` builds the panel with `/api` (default of
`PANEL_API_URL`) and nginx
proxies `/api/` to `127.0.0.1:8001`, so in production panel and API share one
origin and CORS is not involved.
