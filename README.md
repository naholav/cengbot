# 🤖 CengBot - AI-Powered University Assistant

<div align="center">

![CengBot Logo](https://img.shields.io/badge/CengBot-AI%20Assistant-blue?style=for-the-badge&logo=telegram)

**LLaMA 3.2 3B + LoRA** fine-tuned AI chatbot for Çukurova University Computer Engineering Department with dynamic learning capabilities.

🤖 **Try the Bot**: [CU_CengBOT Telegram Group](https://t.me/CU_CengBOT)  
🤗 **Model**: [Naholav/cengbot-lora-tr-en-cukurova](https://huggingface.co/Naholav/cengbot-lora-tr-en-cukurova)  
📊 **Dataset**: [Naholav/cukurova_university_chatbot](https://huggingface.co/datasets/Naholav/cukurova_university_chatbot)  
👨‍💻 **Developer**: naholav  
📅 **Date**: August 2025

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat&logo=react&logoColor=black)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![RabbitMQ](https://img.shields.io/badge/RabbitMQ-3.11+-FF6600?style=flat&logo=rabbitmq&logoColor=white)](https://rabbitmq.com)
[![SQLite](https://img.shields.io/badge/SQLite-3.40+-003B57?style=flat&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

## 🚀 Overview

CengBot is an AI-powered chatbot system for Çukurova University's Computer Engineering Department. Built with LLaMA 3.2 3B language model and LoRA fine-tuning, it provides intelligent responses to student inquiries about courses, programs, and department information.

> **⚠️ Important Note**: This bot only works in the specific Telegram group and does not respond to private messages. Join the official group to interact with the bot.

### 🌟 What Makes CengBot Special?

- **🔒 Secure Admin Panel**: Password-protected admin interface with authentication
- **📊 TanStack Query Integration**: Efficient data fetching with automatic caching and pagination
- **📁 Documentation Viewer**: Built-in viewer for README files, logs, and system documentation
- **🚀 Rate Limiting**: Anti-spam protection with intelligent user rate limiting
- **⚡ Optimized Database**: Enhanced with proper indexing and query optimization
- **🛡️ Production Security**: Hashed passwords and secure token management

**Dynamic Learning System**: CengBot continuously evolves through user interactions. Starting with ~600 English and ~600 Turkish question-answer pairs, the system expanded to over 22,000+ training pairs through AI-powered data augmentation. The system collects new interactions, allows admin review, and supports model retraining for continuous improvement.

### ✨ Key Features

- **🧠 AI Model**: LLaMA 3.2 3B model with custom LoRA fine-tuning
- **🌍 Multi-language Support**: Automatic Turkish/English detection and response
- **📱 Telegram Integration**: Bot with interactive feedback system
  - **/start** - Welcome message and bot introduction
  - **/stats** - Display bot statistics and performance metrics
- **🔧 Admin Panel**: React-based interface for question management and analytics
- **🔄 Message Queue Architecture**: RabbitMQ for scalable asynchronous processing
- **💾 Dual Database System**: Raw data collection + curated training data
- **🎯 Continuous Learning**: System for collecting and reviewing new training data
- **🛡️ Security**: Environment-based configuration with proper secret management
- **📊 Analytics**: Basic statistics and user feedback tracking
- **🔄 Automated Training**: Scripts for model retraining and deployment
- **📝 Training Log System**: Versioned training logs (v1.log, v2.log, etc.) with metadata
- **📊 Database Export**: Automatic export of training data from database to JSONL format
- **📋 Excel Export**: Complete database export to Excel format with timestamp and analytics
- **🔄 Model Versioning**: Automatic model versioning (v1, v2, v3, etc.) with easy switching
- **🤖 Data Augmentation**: AI-powered training data augmentation using Anthropic Claude API
- **⚠️ Error Handling**: Centralized error handling and logging system
- **📈 System Monitoring**: Resource usage monitoring and performance tracking

---

## 📁 Project Structure

```
cu_ceng_bot/
├── 📂 src/                                    # Core Application Source
│   ├── 🤖 telegram_bot.py                    # Main Telegram bot (standalone)
│   ├── 🔄 telegram_bot_rabbitmq.py           # Telegram bot with RabbitMQ
│   ├── 🚀 telegram_bot_standalone.py         # Standalone bot without queue
│   ├── ⚡ ai_model_worker.py                 # AI processing worker
│   ├── 🌐 admin_rest_api.py                  # FastAPI admin API
│   ├── 🧠 llama_model_handler.py             # LLaMA model inference
│   ├── 💾 database_models.py                 # SQLAlchemy models
│   ├── 🎯 train_model.py                     # Model training script
│   ├── 📊 database_to_training.py            # Database export to training format
│   ├── 🤖 data_augmentation.py               # AI-powered data augmentation
│   ├── 📋 export_to_excel.py                 # Database export to Excel format
│   ├── ⚠️ error_handler.py                   # Error handling utilities
│   └── 📈 system_monitor.py                  # System monitoring utilities
│
├── 📂 admin_frontend/                         # React Admin Panel
│   ├── 📦 package.json                       # Node dependencies
│   ├── ⚙️ tsconfig.json                      # TypeScript config
│   ├── 📂 public/                           # Public assets
│   │   └── 🌐 index.html                     # HTML template
│   └── 📂 src/
│       ├── 🎨 App.tsx                        # Main React app
│       ├── 🎯 index.tsx                      # Entry point
│       ├── 💅 App.css                        # Styles
│       ├── 📂 components/                   # React components
│       │   ├── 📖 DocumentViewer.tsx         # Document viewer component
│       │   └── 🔒 Login.tsx                 # Login component
│       ├── 📂 contexts/                     # React contexts
│       │   └── 🔐 AuthContext.tsx           # Authentication context
│       ├── 📂 hooks/                        # Custom React hooks
│       │   └── 🅰️ useApiQueries.ts          # API query hooks
│       ├── 📂 services/                     # Service layer
│       │   └── 🌐 api.ts                    # API client
│       └── 📂 types/                        # TypeScript types
│           └── 📑 api.ts                    # API type definitions
│
├── 📂 scripts/                               # Automation Scripts
│   ├── 🚀 train_model.sh                     # Automatic training
│   ├── 🤖 augment_training_data.sh           # AI-powered data augmentation
│   ├── 🧪 test_environment.sh                # Environment testing
│   ├── 📊 view_training_history.sh           # Training history viewer
│   ├── 📤 export_training_data.sh            # Database to training export
│   ├── 🔄 switch_model.sh                    # Model version switcher
│   ├── 📋 export_database.sh                 # Database export to Excel
│   ├── 🏗️ setup_system.sh                    # System setup script
│   ├── ▶️ start_system.sh                     # Start all services
│   ├── ⏹️ stop_system.sh                      # Stop all services
│   ├── 🏥 health_check.sh                    # System health monitoring
│   ├── 🧹 cleanup_system.sh                  # System cleanup utilities
│   ├── 🚢 deploy_production.sh               # Production deployment
│   ├── ⚡ optimize_linux.sh                  # Linux optimization script
│   ├── 💾 backup_data.sh                     # Data backup script
│   └── 📖 README.md                          # Scripts documentation
│
├── 📂 models/                                # Model Storage
│   ├── 🔗 active-model                       # Active model symlink
│   ├── 📂 final-best-model-v1/              # Model version 1
│   │   ├── 📂 method1/                       # LoRA adapter
│   │   └── 📊 training_info.json             # Training metadata
│   ├── 📂 final-best-model-v2/              # Model version 2 (auto-created)
│   └── 📂 final-best-model-v3/              # Model version 3 (auto-created)
│
├── 📂 config/                                # Configuration
│   ├── 📂 env/                               # Environment configs
│   ├── 🐍 env_loader.py                      # Environment loader
│   └── 📋 requirements.txt                   # Python dependencies
│
├── 📂 docs/                                  # Documentation
│   ├── 📄 API.md                             # API documentation
│   ├── 📄 ARCHITECTURE.md                    # Architecture guide
│   ├── 📄 DEPLOYMENT.md                      # Deployment guide
│   └── 📄 DEVELOPMENT.md                     # Development guide
│
├── 📂 logs/                                  # Application Logs
│   ├── 📄 admin.log                          # Admin API logs
│   ├── 📄 bot.log                            # Telegram bot logs
│   ├── 📄 frontend.log                       # Frontend logs
│   ├── 📄 worker.log                         # AI worker logs
│   ├── 📄 error.log                          # Error logs
│   ├── 📄 excel_export.log                   # Excel export logs
│   └── 📄 system_monitor.log                 # System monitoring logs
│
├── 📂 data/                                  # Training Data
│   └── 📄 merged_qa_data.jsonl              # Training dataset
│
├── 📂 excel/                                 # Excel Export Files
│   └── 📄 cengbot_database_export_*.xlsx    # Database exports (timestamp)
│
├── 💾 university_bot.db                      # SQLite database
├── 🔧 .env                                   # Environment variables
├── 📄 .env.example                           # Environment template
├── 📄 .pids                                  # Process ID tracking file
├── 📂 .venv/                                 # Python virtual environment
├── 🔐 PRODUCTION_SECURITY_CHECKLIST.md      # Production security checklist
└── 📚 README.md                              # This documentation
```

---

## 🔐 Security & Configuration

### Environment Variables

**IMPORTANT**: All sensitive information is stored in environment variables. Create a `.env` file in the project root:

```bash
# Required Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here
HUGGING_FACE_TOKEN=your_huggingface_token_here
ADMIN_PASSWORD=your_admin_password_here
SECRET_KEY=your_secret_key_here

# Data Augmentation (Optional)
# Note: CLAUDE_API_KEY is NOT stored here for security
# It will be requested when running augmentation script
CLAUDE_API_KEY=your_claude_api_key_here
```

**Security Features**:
- All sensitive data in `.env` (gitignored)
- No hardcoded credentials in source code
- Complete `.gitignore` for security
- Environment variable validation

**⚠️ IMPORTANT SECURITY NOTICE**:
- Replace ALL placeholder tokens in `.env` with real values before use
- Never commit `.env` file to version control
- See `PRODUCTION_SECURITY_CHECKLIST.md` for complete security guidelines
- API keys in this repo are examples only - not functional

---

## 🛠️ Installation & Setup

### 📁 Required Files & Directories Setup

After cloning the repository, you need to set up the following files and directories:

**📋 Essential Files to Create:**
```bash
# 1. Environment configuration (REQUIRED)
cp .env.example .env
# Edit .env file with your actual tokens:
# - TELEGRAM_BOT_TOKEN=your_actual_bot_token
# - HUGGING_FACE_TOKEN=your_actual_hf_token  
# - ADMIN_PASSWORD_HASH=your_hashed_password

# 2. Database file (will be created automatically)
# university_bot.db - SQLite database file

# 3. Process tracking (will be created automatically)  
# .pids - Process ID tracking file
```

**🤗 Model Files to Download:**
```bash
# 1. Download the trained LoRA model from HuggingFace:
# https://huggingface.co/Naholav/cengbot-lora-tr-en-cukurova
# 
# 2. Create directory structure:
mkdir -p models/final-best-model-v1/method1/

# 3. Extract downloaded model files to:
# models/final-best-model-v1/method1/
# - adapter_config.json
# - adapter_model.safetensors
# - tokenizer files
# - training_info.json

# 4. Create symlink for active model:
cd models/
ln -sf final-best-model-v1 active-model
cd ..

# 5. Verify structure:
# models/
# ├── active-model -> final-best-model-v1
# └── final-best-model-v1/
#     └── method1/
#         ├── adapter_config.json
#         ├── adapter_model.safetensors
#         └── tokenizer files...
```

**📊 Training Data Setup:**
```bash
# 1. Download dataset from HuggingFace:
# https://huggingface.co/datasets/Naholav/cukurova_university_chatbot
#
# 2. Place the downloaded JSONL file in:
# data/merged_qa_data.jsonl
#
# 3. Verify file format (each line should be a JSON object):
# {"instruction": "question here", "output": "answer here", "language": "turkish/english"}
#
# 4. File structure should look like:
# data/
# └── merged_qa_data.jsonl  (your training dataset)
```

**📂 Directory Structure (Auto-created but listed for reference):**
- `logs/` - Application logs (auto-created)
- `model_cache/` - HuggingFace model cache (auto-created)
- `excel/` - Database exports (auto-created)
- `admin_frontend/node_modules/` - NPM dependencies (after npm install)
- `admin_frontend/build/` - React build output (after npm run build)

### 🎯 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/naholav/cengbot.git
cd cengbot

# 2. Create environment file
cp .env.example .env
# Edit .env with your actual tokens

# 3. Download and setup the trained model
# Go to: https://huggingface.co/Naholav/cengbot-lora-tr-en-cukurova
# Download all files and place in: models/final-best-model-v1/method1/
mkdir -p models/final-best-model-v1/method1/
# Extract model files to the above directory
cd models/ && ln -sf final-best-model-v1 active-model && cd ..

# 4. Download training data (optional)
# Go to: https://huggingface.co/datasets/Naholav/cukurova_university_chatbot
# Download and place as: data/merged_qa_data.jsonl

# 5. Install Python dependencies
pip install -r config/requirements.txt

# 6. Install Node.js dependencies
cd admin_frontend
npm install
cd ..

# 7. Initialize database
python3 -c "from src.database_models import init_db; init_db()"

# 8. Start RabbitMQ
sudo systemctl start rabbitmq-server

# 9. Test environment
./scripts/test_environment.sh
```

### 🚀 Running the System

```bash
# Quick Start - All services at once
./scripts/start_system.sh

# Stop all services
./scripts/stop_system.sh

# Check system health
./scripts/health_check.sh

# Manual start (separate terminals):
# Terminal 1: AI Model Worker
python3 src/ai_model_worker.py

# Terminal 2: Telegram Bot
python3 src/telegram_bot_rabbitmq.py

# Terminal 3: Admin API
python3 src/admin_rest_api.py

# Terminal 4: Frontend (in admin_frontend/)
npm start
```

### 🎯 Model Training

```bash
# Export training data from database
./scripts/export_training_data.sh

# Data augmentation (optional - requires Anthropic API key)
./scripts/augment_training_data.sh

# Automatic training with environment setup (creates v2.log, v3.log, etc.)
./scripts/train_model.sh  # Creates final-best-model-v2, v3, etc.

# Manual training
python3 src/train_model.py

# View training logs (when available)
ls -la logs/

# Analyze training sessions
./scripts/view_training_history.sh list
./scripts/view_training_history.sh view v1
./scripts/view_training_history.sh compare v1 v2  # when v2 exists
./scripts/view_training_history.sh best
```

### 🔄 Model Version Management

```bash
# List available model versions
./scripts/switch_model.sh list

# Switch to specific model version
./scripts/switch_model.sh switch v2

# Show current active model
./scripts/switch_model.sh current

# Auto-switch to next available version
./scripts/switch_model.sh auto
```

### 📋 Database Export to Excel

```bash
# Export complete database to Excel format
./scripts/export_database.sh

# Export with verbose logging
./scripts/export_database.sh -v

# Export in quiet mode (only errors shown)
./scripts/export_database.sh -q

# View export help
./scripts/export_database.sh -h
```

#### 📊 Export Features
- **Complete Database Export**: All tables (raw_data, training_data, user_votes, user_analytics, system_metrics)
- **Timestamped Files**: `cengbot_database_export_YYYYMMDD_HHMMSS.xlsx`
- **Multiple Worksheets**: Each database table in separate worksheet
- **Summary Sheet**: Export metadata and statistics
- **Automatic Backup**: Previous exports backed up automatically
- **Cleanup System**: Keeps only last 10 export files
- **Complete Logging**: All export operations logged to `logs/excel_export.log`
- **Error Recovery**: Error handling and recovery
- **Export Statistics**: File count, database size, and timing information

---

## 🏗️ Architecture

### 🔄 Message Queue System

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Telegram Bot  │───▶│  Questions      │───▶│  AI Worker      │
│   (Producer)    │    │  Queue          │    │  (Consumer)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Telegram Bot  │◄───│  Answers        │◄───│  LLaMA Model    │
│   (Consumer)    │    │  Queue          │    │  (Inference)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 💾 Database Schema

**Raw Data Table** - Stores all user interactions:
```sql
CREATE TABLE raw_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id BIGINT NOT NULL,
    username VARCHAR(100),
    question TEXT NOT NULL,
    answer TEXT,
    language VARCHAR(10),
    like INTEGER,                    -- User feedback (-1/1)
    admin_approved INTEGER DEFAULT 0,
    is_duplicate BOOLEAN DEFAULT FALSE,
    duplicate_of_id INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    answered_at DATETIME,
    message_thread_id INTEGER,
    processing_time FLOAT,
    similarity_score FLOAT,
    model_version VARCHAR(50),
    context_length INTEGER,
    response_length INTEGER,
    quality_score FLOAT,
    sentiment_score FLOAT,
    complexity_score INTEGER,
    topic_category VARCHAR(100),
    keywords TEXT,
    admin_notes TEXT,
    last_updated DATETIME
);
```

**Training Data Table** - Stores approved training examples:
```sql
CREATE TABLE training_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER REFERENCES raw_data(id),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    language VARCHAR(10),
    quality_score INTEGER,
    usage_count INTEGER DEFAULT 0,
    effectiveness_score FLOAT,
    category VARCHAR(100),
    difficulty_level INTEGER,
    topic_tags TEXT,
    training_weight FLOAT,
    augmentation_count INTEGER,
    source_type VARCHAR(50),
    review_status VARCHAR(20),
    reviewer_id VARCHAR(100),
    review_date DATETIME,
    admin_notes TEXT,
    version INTEGER,
    is_active BOOLEAN,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_used_at DATETIME,
    last_updated DATETIME
);
```

---

## 🔧 Admin Panel Features

### 🔒 Security & Authentication
- **Password Protection**: Secure login with configurable credentials
- **Session Management**: Automatic logout and session handling
- **Production Ready**: Hashed passwords and secure token management

### 📊 Dashboard
- **Real-time Statistics**: Total questions, approved questions, user feedback, duplicates
- **TanStack Query**: Efficient data fetching with automatic caching
- **Server-side Pagination**: Optimized for large datasets
- **Auto-refresh**: Updates every 30 seconds

### 📝 Data Management
- **Raw Data Table**: View and manage all user interactions with extended fields
  - **Telegram Information**: User ID, username, message ID display
  - **User Identity**: Username with anonymous user indicators
  - **Message Tracking**: Full telegram message ID for reference
  - **Telegram User Info**: Shows telegram_id (T.ID) and telegram_message_id (M.ID)
- **Training Data Table**: Curated Q&A pairs for model training
- **Inline Editing**: Edit answers directly in the table
- **Double-click Content View**: Full question/answer viewing with modal dialogs
- **Individual Operations**: Approve and delete items one by one
- **Enhanced Confirmation**: "Are you sure?" dialogs for training data approval
- **Duplicate Detection**: Automatic detection with visual indicators

### 🔍 Duplicate Detection System
- **Automatic Duplicate Detection**: Real-time detection when new questions arrive
- **Cosine Similarity Analysis**: Optimized thresholds for better quality control
  - **Questions**: 45% similarity threshold (optimized for Turkish language)
  - **Answers**: 85% similarity threshold (higher precision)
- **Turkish Language Support**: TF-IDF vectorizer configured for Turkish characters
- **Hybrid Similarity Approach**: Combines TF-IDF with Jaccard fallback
- **Visual Duplicate Indicators**: Simple tags showing duplicate status
- **Reference Display**: Shows which original ID the duplicate references
- **Oldest-First Reference**: Always uses the oldest question as reference
- **Automatic Integration**: No manual intervention needed

### 📁 Documentation Viewer
- **README Files**: View all project documentation
- **System Logs**: Monitor system activity and errors
- **Script Documentation**: Access to all automation scripts
- **File Browser**: Navigate through project structure

### ⚡ Performance Features
- **Efficient Pagination**: Page-based navigation with configurable sizes
- **Smart Caching**: Reduced API calls with intelligent cache invalidation
- **Optimized Queries**: Database indexes for faster performance
- **Rate Limiting**: Anti-spam protection for system stability
- **Status Indicators**: Visual feedback for question status
- **User Feedback**: Display like/dislike ratings
- **Enhanced Filtering**: Filter by duplicate status and approval status

### 🎓 Training Data Management
- **View Training Data**: Browse approved training examples
- **Delete Entries**: Remove unwanted training data
- **Statistics**: Real-time count of training examples
- **Duplicate Answer Detection**: Identifies duplicate answers in training data
- **Quality Indicators**: Visual feedback for training data quality

### 🔐 Enhanced Security Features
- **Group-Only Access**: Bot configured to work ONLY in specific Telegram group (-1002630398173)
- **Private Message Blocking**: All private messages and DMs are completely blocked
- **Unauthorized Access Prevention**: Messages from other groups are ignored
- **Database Protection**: Only authorized group messages are saved to database
- **Security Logging**: All unauthorized access attempts are logged
- **Traffic Filtering**: No unwanted traffic reaches the system

### 🔌 API Endpoints
- `GET /raw-data` - Get all raw data with telegram information
- `PUT /raw-data/{id}` - Update answer
- `POST /approve/{id}` - Approve question with duplicate detection
- `GET /stats` - System statistics including duplicate counts
- `GET /training-data` - Training data with duplicate information
- `DELETE /training-data/{id}` - Delete training data
- `GET /duplicates` - Get duplicate question groups

---

## 🔄 Model Versioning System

### 🎯 Automatic Model Versioning
Each training session creates a new versioned model:

```
models/
├── active-model → final-best-model-v1    # Current active model
├── final-best-model-v1/                  # First trained model
├── final-best-model-v2/                  # Second trained model
└── final-best-model-v3/                  # Third trained model
```

### 🔄 Model Switching
The system uses a symbolic link (`active-model`) to point to the current production model:

```bash
# List all model versions with info
./scripts/switch_model.sh list

# Switch to a specific version
./scripts/switch_model.sh switch v2

# Auto-switch to next available version
./scripts/switch_model.sh auto

# Check current active model
./scripts/switch_model.sh current
```

### 📊 Model Information
Each model version includes:
- **Training metrics**: Loss, steps, dataset size
- **Configuration**: Learning rate, batch size, epochs
- **Timestamp**: When the model was trained
- **Version tracking**: Both training and model versions

### 🚀 Production Benefits
- **Zero-downtime switching**: Switch models without service restart
- **Easy rollback**: Revert to previous model version instantly
- **A/B testing**: Compare different model versions
- **Progressive deployment**: Test new models before full rollout

---

## 🤖 Data Augmentation System

### 🎯 AI-Powered Training Data Enhancement
The system includes data augmentation using Anthropic's Claude API to generate training variations:

#### 🧠 How It Works
1. **Database Integration**: Loads approved training data from database
2. **AI Generation**: Uses Claude API to create 15 variations per question
3. **Quality Control**: Maintains semantic accuracy and natural language patterns
4. **Security**: API key requested from user (never stored)
5. **Checkpoint System**: Safe interruption and resumption

#### ⚙️ Technical Features
- **Multi-language Support**: Separate prompts for Turkish and English
- **Student Profiles**: Generates variations reflecting different user types
- **Style Diversity**: 15 different question and answer styles
- **Link Preservation**: Maintains all URLs and technical details
- **Progress Tracking**: Checkpoint system for large datasets

#### 🔧 Usage
```bash
# Run data augmentation (requires Anthropic API key)
./scripts/augment_training_data.sh

# The script will:
# 1. Check system prerequisites
# 2. Validate database content
# 3. Request API key from user
# 4. Generate variations using Claude API
# 5. Export to training format
```

#### 🛡️ Security Features
- **API Key Safety**: Never stored, only used in session
- **User Confirmation**: Explicit consent required
- **Cost Awareness**: Clear warnings about API usage costs
- **Process Isolation**: Lock files prevent concurrent runs

#### 🤖 Model Selection
- **Default Model**: claude-3-sonnet-20240229
- **Custom Models**: Users can choose different Claude models
- **Model Reference**: https://docs.anthropic.com/claude/docs/models-overview
- **Interactive Selection**: Choice presented during execution

#### 📊 Benefits
- **Data Quality**: High-quality, natural language variations
- **Model Performance**: Improved training diversity
- **Automated Pipeline**: Integration with training workflow
- **Safe Operation**: Error handling and recovery

---

## 🔍 Duplicate Detection System

### 🎯 High-Precision Semantic Deduplication
The training system includes duplicate detection to ensure training data quality:

#### 🧠 How It Works
1. **Semantic Analysis**: Uses sentence-transformers to compute embeddings for questions and answers
2. **Multi-Stage Filtering**: Primary check on answer similarity (94% threshold), secondary check on question similarity (98% threshold)
3. **Language-Specific Processing**: Separate deduplication for Turkish and English content
4. **Automatic Integration**: Runs automatically during training data preparation

#### ⚙️ Technical Configuration
- **Answer Similarity Threshold**: 94% (catches near-identical answers)
- **Question Similarity Threshold**: 98% (prevents false positives from different questions with similar answers)
- **Model**: paraphrase-multilingual-MiniLM-L12-v2 (supports Turkish and English)
- **Processing**: Batch processing for efficiency

#### 📊 Benefits
- **Data Quality**: Removes redundant training examples
- **Training Efficiency**: Reduces dataset size while maintaining quality
- **Model Performance**: Prevents overfitting on duplicate examples
- **Automatic**: No manual intervention required

#### 🔧 Usage
```bash
# Duplicate detection runs automatically during training
python3 src/train_model.py

# Or during automatic training
./scripts/train_model.sh
```

**Note**: Requires `sentence-transformers` package. Install with:
```bash
pip install sentence-transformers>=2.2.0
```

---

## 🧠 Model Training

### 🎯 Training Configuration
- **Model**: LLaMA 3.2 3B with LoRA fine-tuning
- **Hardware**: Optimized for RTX 5090 (24-25GB VRAM)
- **Training Time**: ~3.5 hours
- **Final Loss**: ~0.7665
- **Dataset**: 18,000+ Turkish/English Q&A pairs

### 🔄 Training Process
1. **Data Preparation**: Load and validate training data
2. **Model Setup**: Load LLaMA 3.2 3B with LoRA adapter
3. **Training**: 3 epochs with gradient accumulation
4. **Validation**: Separate Turkish/English validation
5. **Model Saving**: Multiple save methods for reliability

### 📊 Training Metrics
- **Batch Size**: 4 (effective: 64 with gradient accumulation)
- **Learning Rate**: 2e-4
- **LoRA Rank**: 16
- **Validation Split**: 7% of data
- **Early Stopping**: 2 patience epochs

### 📝 Training Log System
- **Version Tracking**: Automatic v1.log, v2.log, v3.log creation
- **Metadata Storage**: JSON files with training configuration and results
- **Dual Logging**: Both versioned and timestamped logs
- **Training History**: Complete log of all training sessions
- **Progress Tracking**: Detailed training progress and metrics

---

## 🚀 Usage Guide

### 🤖 Telegram Bot Commands
- Send any question to get AI response
- Use 👍/👎 buttons for feedback
- Bot automatically detects Turkish/English

### 🔧 Admin Panel Access
1. **Start System**: `./scripts/start_system.sh`
2. **Access Panel**: `http://localhost:3000`
3. **Login**: Use configured admin credentials
4. **Default Password**: `cucengedutr` (change in production)

### 🔒 Production Security Setup
```bash
# Set secure admin password hash in .env
export ADMIN_USERNAME=admin
export ADMIN_PASSWORD_HASH=$(python -c "import hashlib; print(hashlib.sha256('your_secure_password'.encode()).hexdigest())")
```

### 📊 Monitoring
- View logs in `logs/` directory
- Check system health with `./scripts/test_environment.sh`
- Monitor RabbitMQ queues

### 📝 Training Log Management
```bash
# View all training versions
ls -la logs/training/ # if training has been done

# View specific training log
tail -f logs/training/v1.log

# View training metadata
cat logs/training/v1_info.json

# Compare training sessions (when v2 exists)
diff logs/training/v1_info.json logs/training/v2_info.json
```

---

## 🔄 Dynamic Learning System

### 📈 Data Collection Flow
```
User Question → Raw Database → AI Processing → Response Generation
       ↓
User Feedback → Admin Review → Training Data → Model Retraining
```

### 🎯 Quality Control
- **User Feedback**: Like/dislike system
- **Admin Review**: Manual answer editing
- **High-Precision Duplicate Detection**: Semantic similarity-based deduplication (94% answer + 98% question threshold)
- **Quality Scoring**: Response quality assessment

### 🔄 Continuous Improvement
1. **Real-time Collection**: All interactions stored
2. **Admin Curation**: Manual review and approval
3. **Data Augmentation**: AI-powered training data enhancement using Claude API
4. **Database Export**: Automatic conversion to training format
5. **Duplicate Detection**: High-precision semantic deduplication during training
6. **Model Retraining**: Periodic model updates
7. **Quality Monitoring**: Track improvement metrics

### 📊 Database to Training Export
The system automatically converts approved database entries to training format:

```bash
# Manual export from database
./scripts/export_training_data.sh

# Export with options
./scripts/export_training_data.sh --include-inactive --no-backup
```

**Export Process**:
1. **Data Validation**: Checks question/answer quality and length
2. **Language Mapping**: Converts TR→turkish, EN→english
3. **Format Conversion**: Database → JSON format for training
4. **Backup Creation**: Saves previous training data
5. **Statistics**: Provides export summary and metrics

**Auto-Export Features**:
- **Triggered**: When training file is missing or older than 1 hour
- **Automatic**: Runs before each training session
- **Validated**: Ensures data quality and completeness
- **Append Mode**: New database entries are added to existing training data
- **Duplicate Prevention**: Avoids duplicate entries based on ID
- **Logged**: Complete export process logging

---

## 🛡️ Security Features

### 🔐 Environment Security
- All tokens in `.env` file (gitignored)
- No hardcoded credentials
- Environment variable validation
- Secure secret management

### 🔒 API Security
- Input validation and sanitization
- Error handling and logging
- CORS configuration
- Request/response monitoring

---

## 📝 Training Log System

### 🎯 Version Tracking
The training system automatically creates versioned logs for each training session:

```
logs/
├── training/        # Training logs (created during training)
│   ├── v1.log       # First training session (when available)
│   ├── v1_info.json # Training metadata for v1
│   └── ...          # Additional training sessions
├── admin.log        # Admin API logs
├── bot.log          # Telegram bot logs
├── worker.log       # AI worker logs
└── error.log        # Error logs
```

### 📊 Log Content
Each training log contains:
- **Real-time Progress**: Step-by-step training progress
- **Loss Metrics**: Training and validation loss values
- **Performance Stats**: Memory usage, GPU utilization
- **Error Messages**: Any issues encountered during training
- **Completion Status**: Final training results

### 🔍 Metadata Files
Each `v{N}_info.json` file contains:
```json
{
  "version": "v2",
  "final_loss": 0.7665,
  "total_steps": 1250,
  "dataset_size": 18000,
  "model_name": "meta-llama/Llama-3.2-3B",
  "lora_r": 16,
  "lora_alpha": 32,
  "learning_rate": 2e-4,
  "num_epochs": 3,
  "batch_size": 4,
  "gradient_accumulation_steps": 16,
  "validation_split": 0.07,
  "timestamp": "2025-07-16T13:38:00",
  "log_file": "v2.log",
  "model_save_path": "/home/ceng/cu_ceng_bot/models/..."
}
```

### 🛠️ Usage Examples
```bash
# View all training sessions
ls -la logs/training/ # if training has been done

# Follow latest training session
tail -f logs/training/v$(ls logs/training/ | grep -E '^v[0-9]+\.log$' | sort -V | tail -1 | sed 's/v\|\.log//g').log

# Compare training configurations
diff logs/training/v1_info.json logs/training/v2_info.json

# View training summary
cat logs/training/v2_info.json | jq '.final_loss, .total_steps, .dataset_size'

# Find best performing model
python3 -c "
import json
import os
history_dir = 'logs/training_history'
best_loss = float('inf')
best_version = None
for file in os.listdir(history_dir):
    if file.endswith('_info.json'):
        with open(os.path.join(history_dir, file), 'r') as f:
            data = json.load(f)
            if data['final_loss'] < best_loss:
                best_loss = data['final_loss']
                best_version = data['version']
print(f'Best model: {best_version} with loss {best_loss}')
"
```

### 🚀 Automatic Integration
The training system automatically:
1. **Detects Next Version**: Scans existing logs to determine next version number
2. **Creates Dual Logs**: Both versioned (v2.log) and timestamped logs
3. **Saves Metadata**: Complete training configuration and results
4. **Updates Summary**: Shows training history in completion summary

---

## 📈 Performance

### ⚡ Current Metrics
- **Response Time**: ~2-3 seconds average
- **Model Loading**: ~30 seconds initial
- **Memory Usage**: ~8GB during inference
- **GPU Utilization**: Optimized for RTX 5090

### 🎯 Optimization Features
- **Model Caching**: Faster subsequent responses
- **Queue Processing**: Asynchronous handling
- **Batch Processing**: Efficient inference
- **Memory Management**: Optimized model loading

---

## 🐛 Troubleshooting

### Common Issues

**Model Setup Problems**:
```bash
# If model files are missing, download from HuggingFace:
# https://huggingface.co/Naholav/cengbot-lora-tr-en-cukurova

# Create proper directory structure
mkdir -p models/final-best-model-v1/method1/

# Create symlink if missing
cd models/
ln -sf final-best-model-v1 active-model
cd ..

# Verify symlink
ls -la models/active-model
```

**Model Loading Problems**:
```bash
# Check GPU availability
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"

# Verify model files exist
ls -la models/final-best-model-v1/method1/
ls -la models/active-model

# Check if required model files are present
# Should contain: adapter_config.json, adapter_model.safetensors, etc.
```

**RabbitMQ Connection Issues**:
```bash
# Check RabbitMQ service
sudo systemctl status rabbitmq-server
sudo systemctl restart rabbitmq-server

# Verify queue status
sudo rabbitmqctl list_queues
```

**Database Issues**:
```bash
# Check database file
ls -la university_bot.db

# Reinitialize if needed
python3 -c "from src.database_models import init_db; init_db()"
```

---

## 📞 Support

### 📧 Contact Information
- **Email**: arda.mulayim@outlook.com
- **Bot**: [@cu_ceng_v1_bot](https://t.me/cu_ceng_v1_bot)
- **HuggingFace**: [Naholav](https://huggingface.co/Naholav)

### 📚 Documentation
- **Training Scripts**: `scripts/README.md`
- **API Documentation**: Auto-generated at `http://localhost:8001/docs`
- **Model Information**: `models/final-best-model-v1/method1/README.md`

---

## 📝 Changelog

### August 2025 Updates
- **Dataset Expansion**: Training dataset expanded from 18,000 to 22,000+ examples with new Q&A pairs
- **Model Update**: Active model version set to v1.2 (latest trained model)
- **Training Parameters Enhanced**:
  - Training epochs increased from 3 to 4 for optimal performance with 22k dataset
  - Warmup ratio adjusted from 0.03 to 0.05 for better convergence
  - Early stopping patience increased from 2 to 3 epochs
  - Training logs now stored in version-specific directories (train_logs_llama_v1.2/)
- **Inference Improvements**:
  - System prompt updated to allow maximum 4 sentences (previously 3) for more detailed responses
  - Response quality enhanced while maintaining conciseness
- **Documentation Updates**: All documentation updated to reflect latest changes

---

## 📄 License

This project is licensed under the **Apache License 2.0**.

---

## 👨‍💻 Author

**naholav**
- **Email**: arda.mulayim@outlook.com
- **University**: Çukurova University Computer Engineering Department
- **GitHub**: [naholav](https://github.com/naholav)
- **Date**: August 2025

---

## 🙏 Acknowledgments

- **Meta AI** for the LLaMA 3.2 3B base model
- **Hugging Face** for transformers and PEFT libraries
- **Çukurova University** for domain expertise and support
- **Open Source Community** for ML tools and frameworks

---

<div align="center">

### 🎉 **CengBot - Dynamic AI Learning System**

**Built with ❤️ for Çukurova University Computer Engineering Department**

**🚀 Production Ready • 🔧 Easy to Deploy • 📚 Well Documented • 🌟 Continuously Learning**

---

**🔥 Key Innovation**: Dynamic learning system that improves through user interactions  
**🎯 Mission**: Provide intelligent assistance to computer engineering students  
**🌟 Vision**: Create a self-evolving AI system that grows with its users  

</div>