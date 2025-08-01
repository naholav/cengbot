# 🔧 CengBot Development Guide

## 🚀 Quick Start for Developers

### Prerequisites
- Python 3.8+ installed
- Node.js 16+ and npm
- RabbitMQ server
- Git
- Code editor (VS Code recommended)

### Development Setup
```bash
# 1. Clone repository
git clone https://github.com/your-username/university-bot.git
cd university-bot

# 2. Setup Python environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r config/requirements.txt

# 4. Setup frontend
cd admin_frontend
npm install
cd ..

# 5. Initialize database
cd src
python3 -c "from database_models import init_db; init_db()"
cd ..

# 6. Start development environment
./scripts/start_system.sh
```

## 📁 Project Structure for Developers

```
src/
├── telegram_bot.py                 # 🤖 Main Telegram bot
├── telegram_bot_rabbitmq.py        # 🔄 RabbitMQ-enabled bot
├── telegram_bot_standalone.py      # 🏃 Standalone bot
├── ai_model_worker.py              # ⚡ AI processing worker
├── admin_rest_api.py               # 🌐 FastAPI backend
├── llama_model_handler.py          # 🧠 LLaMA model handler
└── database_models.py              # 💾 Database models
```

## 🔧 Development Workflow

### 1. Code Organization

#### Python Code Style
```python
# Follow PEP 8 conventions
# Use type hints
# Add detailed docstrings

def process_message(message: str, language: str) -> Optional[str]:
    """
    Process incoming message and generate response.
    
    Args:
        message: User message text
        language: Detected language (TR/EN)
        
    Returns:
        Generated response or None if error
    """
    try:
        # Implementation here
        return response
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return None
```

#### Database Models
```python
# SQLAlchemy models with proper relationships
class RawData(Base):
    __tablename__ = "raw_data"
    
    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text)
    language = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    training_data = relationship("TrainingData", back_populates="source")
```

### 2. API Development

#### FastAPI Endpoints
```python
@app.get("/raw-data", response_model=List[RawDataResponse])
async def get_raw_data(
    skip: int = 0,
    limit: int = 100,
    language: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get raw data with pagination and filtering."""
    query = db.query(RawData)
    
    if language:
        query = query.filter(RawData.language == language)
    
    return query.offset(skip).limit(limit).all()
```

#### Error Handling
```python
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging."""
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )
```

### 3. Frontend Development

#### React Components
```typescript
// Use TypeScript for better type safety
interface RawDataProps {
  data: RawData[];
  onUpdate: (id: number, answer: string) => void;
}

const RawDataTable: React.FC<RawDataProps> = ({ data, onUpdate }) => {
  const [loading, setLoading] = useState(false);
  
  const handleUpdate = async (id: number, answer: string) => {
    setLoading(true);
    try {
      await onUpdate(id, answer);
      message.success('Answer updated successfully');
    } catch (error) {
      message.error('Failed to update answer');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <Table
      dataSource={data}
      loading={loading}
      columns={columns}
      rowKey="id"
    />
  );
};
```

#### State Management
```typescript
// Use React hooks for state management
const useRawData = () => {
  const [data, setData] = useState<RawData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/raw-data');
      const result = await response.json();
      setData(result);
    } catch (err) {
      setError('Failed to fetch data');
    } finally {
      setLoading(false);
    }
  };
  
  return { data, loading, error, fetchData };
};
```

## 🧪 Testing

### Unit Tests
```python
# tests/test_model_handler.py
import pytest
from src.llama_model_handler import ModelHandler

class TestModelHandler:
    @pytest.fixture
    def model_handler(self):
        return ModelHandler()
    
    def test_model_initialization(self, model_handler):
        """Test model loads correctly."""
        assert model_handler.model is not None
        assert model_handler.tokenizer is not None
    
    def test_generate_response(self, model_handler):
        """Test response generation."""
        question = "Test question"
        response = model_handler.generate_response(question)
        assert response is not None
        assert len(response) > 0
```

### Integration Tests
```python
# tests/test_api.py
from fastapi.testclient import TestClient
from src.admin_rest_api import app

client = TestClient(app)

def test_get_raw_data():
    """Test raw data endpoint."""
    response = client.get("/raw-data")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_update_answer():
    """Test answer update endpoint."""
    response = client.put("/raw-data/1", json={"answer": "Test answer"})
    assert response.status_code == 200
```

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run all tests with coverage
pytest --cov=src tests/
```

## 🔍 Debugging

### Logging Configuration
```python
# Enhanced logging for development
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/debug.log'),
        logging.StreamHandler()
    ]
)

# Create logger for specific module
logger = logging.getLogger(__name__)
```

### Debug Mode
```bash
# Enable debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with debug output
python3 src/telegram_bot.py --debug
```

### Common Debug Commands
```bash
# Check model loading
python3 -c "from src.llama_model_handler import ModelHandler; m = ModelHandler(); print('Model loaded successfully')"

# Test database connection
python3 -c "from src.database_models import SessionLocal; db = SessionLocal(); print('Database connected')"

# Check RabbitMQ connection
python3 -c "import pika; conn = pika.BlockingConnection(pika.ConnectionParameters('localhost')); print('RabbitMQ connected')"
```

## 📊 Performance Monitoring

### Profiling
```python
# Profile model inference
import cProfile
import pstats

def profile_model_inference():
    pr = cProfile.Profile()
    pr.enable()
    
    # Your model inference code here
    model_handler.generate_response("Test question")
    
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('tottime')
    stats.print_stats()
```

### Memory Monitoring
```python
# Monitor memory usage
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Log memory usage
logger.info(f"Memory usage: {get_memory_usage():.2f} MB")
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Test and Deploy

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      rabbitmq:
        image: rabbitmq:3.11
        ports:
          - 5672:5672
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r config/requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest --cov=src tests/
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.950
    hooks:
      - id: mypy
```

## 🛠️ Development Tools

### VS Code Configuration
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/node_modules": true
    }
}
```

### Makefile for Common Tasks
```makefile
# Makefile
.PHONY: install test format lint clean

install:
	pip install -r config/requirements.txt
	cd admin_frontend && npm install

test:
	pytest tests/ -v

format:
	black src/
	cd admin_frontend && npm run format

lint:
	flake8 src/
	mypy src/
	cd admin_frontend && npm run lint

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

dev:
	./scripts/start_system.sh

prod:
	./scripts/start_system.sh
```

## 🔧 Configuration Management

### Environment Variables
```python
# config/env_loader.py
import os
from typing import Optional
from dataclasses import dataclass

@dataclass
class Config:
    telegram_bot_token: str
    database_url: str
    rabbitmq_url: str
    model_path: str
    debug: bool = False
    
    @classmethod
    def from_env(cls) -> 'Config':
        return cls(
            telegram_bot_token=os.getenv('TELEGRAM_BOT_TOKEN'),
            database_url=os.getenv('DATABASE_URL', 'sqlite:///university_bot.db'),
            rabbitmq_url=os.getenv('RABBITMQ_URL', 'amqp://localhost:5672'),
            model_path=os.getenv('LORA_MODEL_PATH', 'models/final-best-model-v1/method1'),
            debug=os.getenv('DEBUG', 'false').lower() == 'true'
        )
```

### Configuration Validation
```python
def validate_config(config: Config) -> None:
    """Validate configuration settings."""
    if not config.telegram_bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN is required")
    
    if not os.path.exists(config.model_path):
        raise ValueError(f"Model path does not exist: {config.model_path}")
    
    # Test database connection
    try:
        engine = create_engine(config.database_url)
        engine.connect()
    except Exception as e:
        raise ValueError(f"Database connection failed: {e}")
```

## 📈 Performance Optimization

### Database Optimization
```python
# Use connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True
)
```

### Async Processing
```python
# Use asyncio for concurrent processing
import asyncio
from typing import List

async def process_messages_batch(messages: List[str]) -> List[str]:
    """Process multiple messages concurrently."""
    tasks = [process_single_message(msg) for msg in messages]
    return await asyncio.gather(*tasks)

async def process_single_message(message: str) -> str:
    """Process a single message asynchronously."""
    # Implementation here
    return response
```

## 📝 Documentation

### Code Documentation
```python
def generate_response(self, question: str, language: str = "TR") -> Optional[str]:
    """
    Generate AI response for a given question.
    
    This method processes the input question through the LLaMA model with
    LoRA adapter to generate contextually appropriate responses.
    
    Args:
        question (str): The user's question
        language (str): Language code ('TR' or 'EN')
        
    Returns:
        Optional[str]: Generated response or None if processing fails
        
    Raises:
        ModelError: If model inference fails
        ValidationError: If input validation fails
        
    Example:
        >>> handler = ModelHandler()
        >>> response = handler.generate_response("What is computer science?")
        >>> print(response)
        "Computer science is the study of algorithms..."
    """
```

### API Documentation
```python
# Use OpenAPI/Swagger for API documentation
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="CengBot Admin API",
    description="API for managing CengBot data and operations",
    version="1.0.0"
)

class RawDataCreate(BaseModel):
    """Schema for creating raw data entries."""
    question: str
    answer: Optional[str] = None
    language: str = "TR"
    
    class Config:
        schema_extra = {
            "example": {
                "question": "What is computer science?",
                "answer": "Computer science is...",
                "language": "EN"
            }
        }
```

## 🎯 Best Practices

### 1. Error Handling
```python
# Use specific exception types
class ModelError(Exception):
    """Raised when model inference fails."""
    pass

class ValidationError(Exception):
    """Raised when input validation fails."""
    pass

# Handle errors gracefully
try:
    response = model_handler.generate_response(question)
except ModelError as e:
    logger.error(f"Model inference failed: {e}")
    return "I'm sorry, I'm having trouble processing your question."
except ValidationError as e:
    logger.error(f"Input validation failed: {e}")
    return "Please check your input and try again."
```

### 2. Resource Management
```python
# Use context managers for resource cleanup
from contextlib import contextmanager

@contextmanager
def get_db_session():
    """Context manager for database sessions."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

# Usage
with get_db_session() as db:
    # Database operations
    user = db.query(User).first()
```

### 3. Security
```python
# Input validation and sanitization
from pydantic import validator

class MessageInput(BaseModel):
    content: str
    
    @validator('content')
    def validate_content(cls, v):
        if len(v) > 1000:
            raise ValueError('Message too long')
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()
```

This development guide provides information for developers working on CengBot, covering setup to development practices.