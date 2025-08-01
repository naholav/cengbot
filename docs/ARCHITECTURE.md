# 🏗️ CengBot System Architecture

## 📋 Overview

CengBot is built using a microservices architecture with clear separation of concerns. The system consists of multiple independent services that communicate through well-defined interfaces.

## 🔧 Core Components

### 1. 🤖 Telegram Bot Layer
- **Primary**: `telegram_bot.py` - Main bot with full features
- **RabbitMQ**: `telegram_bot_rabbitmq.py` - Message queue integration
- **Standalone**: `telegram_bot_standalone.py` - Direct model integration

### 2. 🧠 AI Processing Layer
- **Model Handler**: `llama_model_handler.py` - LLaMA 3.2 3B inference
- **Worker Process**: `ai_model_worker.py` - Async AI processing
- **LoRA Adapter**: Fine-tuned model weights for university domain

### 3. 🌐 API Layer
- **REST API**: `admin_rest_api.py` - FastAPI backend
- **Admin Interface**: React frontend for data management
- **Authentication**: Session-based admin access

### 4. 💾 Data Layer
- **Database**: SQLite for production simplicity
- **Models**: `database_models.py` - SQLAlchemy ORM
- **Caching**: Model cache for performance

## 🔄 Data Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Telegram  │────│    Bot      │────│  RabbitMQ   │────│ AI Worker   │
│    User     │    │  Handler    │    │   Queue     │    │  Process    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                           │                                      │
                           │                                      │
                           ▼                                      ▼
                   ┌─────────────┐                        ┌─────────────┐
                   │  Database   │◄───────────────────────│   LLaMA     │
                   │  (SQLite)   │                        │   Model     │
                   └─────────────┘                        └─────────────┘
                           ▲
                           │
                           │
                   ┌─────────────┐
                   │    Admin    │
                   │   Panel     │
                   └─────────────┘
```

## 🚀 Message Processing Flow

1. **User Input**: User sends message to Telegram bot
2. **Message Validation**: Bot validates and processes message
3. **Language Detection**: Automatic Turkish/English detection
4. **Database Storage**: Message saved to raw_data table
5. **Duplicate Check**: System checks for similar questions
6. **Queue Processing**: Message sent to RabbitMQ queue
7. **AI Processing**: Worker processes message with LLaMA model
8. **Response Generation**: AI generates contextual response
9. **Database Update**: Response saved with timestamp
10. **User Notification**: Response sent back to user
11. **Feedback Collection**: User can rate response quality

## 🔧 Technology Stack

### Backend
- **Language**: Python 3.8+
- **Framework**: FastAPI (REST API)
- **Database**: SQLite with SQLAlchemy ORM
- **Message Queue**: RabbitMQ with Pika
- **AI Model**: LLaMA 3.2 3B + LoRA adapter
- **ML Framework**: Transformers, PEFT, PyTorch

### Frontend
- **Framework**: React 18+ with TypeScript
- **UI Library**: Ant Design
- **Build Tool**: Create React App
- **Styling**: CSS with responsive design

### Infrastructure
- **Deployment**: Linux-based production environment
- **Process Management**: Shell scripts with PID tracking
- **Monitoring**: Log-based monitoring system
- **Environment**: Docker-ready containerization support

## 📊 Database Architecture

### Raw Data Table
```sql
raw_data (
    id,              -- Primary key
    telegram_id,     -- User identifier
    username,        -- User display name
    question,        -- Original question
    answer,          -- AI-generated response
    language,        -- TR/EN language detection
    like,            -- User feedback (-1/1)
    admin_approved,  -- Admin approval status
    is_duplicate,    -- Duplicate detection flag
    duplicate_of_id, -- Reference to original question
    created_at,      -- Timestamp
    answered_at      -- Response timestamp
)
```

### Training Data Table
```sql
training_data (
    id,              -- Primary key
    source_id,       -- Reference to raw_data
    question,        -- Approved question
    answer,          -- Approved answer
    language,        -- Language identifier
    created_at       -- Approval timestamp
)
```

## 🔄 Service Communication

### RabbitMQ Queues
- **questions**: Incoming questions for AI processing
- **answers**: Generated responses for delivery

### REST API Endpoints
- **GET /raw-data**: Retrieve all question-answer pairs
- **PUT /raw-data/{id}**: Update specific answer
- **POST /approve/{id}**: Approve for training data
- **GET /stats**: System statistics
- **GET /duplicates**: Duplicate question analysis

## 🛡️ Security Architecture

### Authentication
- Environment-based configuration
- Secure token management
- Session-based admin access

### Data Protection
- SQLite with secure file permissions
- Input validation and sanitization
- SQL injection prevention through ORM

### Network Security
- CORS configuration for frontend
- Rate limiting considerations
- Secure communication protocols

## 🔍 Monitoring & Logging

### Log Files
- **worker.log**: AI model operations
- **bot.log**: Telegram bot interactions
- **admin.log**: Admin panel operations
- **frontend.log**: React application logs

### Health Checks
- Service availability monitoring
- Database connectivity checks
- Model loading verification
- Queue status monitoring

## 📈 Scalability Considerations

### Horizontal Scaling
- Multiple AI worker processes
- Load balancing for API endpoints
- Database connection pooling

### Performance Optimization
- Model quantization options
- Caching strategies
- Async processing patterns

### Resource Management
- Memory usage monitoring
- GPU utilization optimization
- Queue size management

## 🔄 Development Workflow

### Development Environment
```bash
./scripts/start_system.sh  # Start development services
./scripts/health_check.sh  # Monitor system health
```

### Production Deployment
```bash
./scripts/setup_system.sh  # Initial setup
./scripts/start_system.sh  # Start all services
./scripts/stop_system.sh   # Graceful shutdown
```

## 🚀 Future Enhancements

### Planned Features
- Multi-model support
- Analytics dashboard
- Automated training pipeline
- Real-time performance metrics

### Technical Improvements
- Kubernetes orchestration
- Redis caching layer
- Monitoring with Prometheus
- CI/CD pipeline integration

This architecture provides a scalable foundation for CengBot while maintaining simplicity and maintainability.