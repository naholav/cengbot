# CengBot Training Scripts

This directory contains automated scripts for training the CengBot model.

## Available Scripts

### `augment_training_data.sh`

AI-powered training data augmentation using Anthropic Claude API:

#### Features:
- **Database Integration**: Loads approved training data from database
- **AI Generation**: Creates 15 variations per question using Claude API
- **Multi-language Support**: Separate prompts for Turkish and English
- **Security**: API key requested from user (never stored)
- **Progress Tracking**: Checkpoint system for safe interruption
- **Cost Awareness**: Clear warnings about API usage costs
- **Quality Control**: Maintains semantic accuracy and link preservation

#### Usage:
```bash
cd /home/ceng/cu_ceng_bot
./scripts/augment_training_data.sh
```

#### Prerequisites:
1. **Anthropic API Key**: Will be requested during execution
2. **Database**: Approved training data must exist in database
3. **Dependencies**: Python virtual environment with anthropic package

#### Process:
1. Check system prerequisites and database content
2. Show cost warning and request user confirmation
3. Request Anthropic API key from user (not stored)
4. Generate variations using Claude API
5. Export augmented data to training format
6. Provide detailed progress reporting

#### Security:
- API key is NOT stored anywhere
- Requested fresh each time
- Cleaned from memory after use
- Lock files prevent concurrent runs

#### Model Selection:
- **Default Model**: claude-3-sonnet-20240229
- **Custom Models**: Users can choose different Claude models during execution
- **Model Reference**: https://docs.anthropic.com/claude/docs/models-overview
- **Interactive Selection**: Choice presented after API key input

### `train_model.sh`

Automatic training script that handles the complete training pipeline:

#### Features:
- **System Checks**: Verifies GPU availability, required files, and dependencies
- **Environment Setup**: Creates and manages Python virtual environment
- **Dataset Validation**: Checks dataset quality and format
- **Backup Creation**: Backs up previous models before training
- **Complete Logging**: Detailed logging throughout the process
- **Model Validation**: Verifies trained model after completion
- **Cleanup**: Removes temporary files and caches

#### Usage:
```bash
cd /home/ceng/cu_ceng_bot
./scripts/train_model.sh
```

#### Prerequisites:
1. **GPU**: NVIDIA GPU with CUDA support (RTX 5090 recommended)
2. **Dataset**: Training data must be available at `data/merged_qa_data.jsonl`
3. **Environment**: `.env` file with `HUGGING_FACE_TOKEN` in project root
4. **Python**: Python 3.8+ with venv module

#### What the script does:
1. Checks system requirements (GPU, Python, files)
2. Validates dataset format and quality
3. Sets up Python virtual environment and dependencies
4. Creates backup of existing models
5. Runs the training process (`src/train_model.py`)
6. Validates the trained model
7. Provides training summary with metrics

#### Output locations:
- **Trained Model**: `/home/ceng/cu_ceng_bot/models/cengbot-llama-3b-lora-v1/final-best-model-v1/`
- **Training Logs**: `/home/ceng/cu_ceng_bot/logs/training/auto_training_YYYYMMDD_HHMMSS.log`
- **Backup Models**: `/home/ceng/cu_ceng_bot/models_backup_YYYYMMDD_HHMMSS/`

#### Training specifications:
- **Model**: LLaMA 3.2 3B with LoRA fine-tuning
- **GPU**: Optimized for RTX 5090 (24-25GB VRAM)
- **Training Time**: Approximately 3.5 hours
- **Expected Final Loss**: ~0.7665

#### Error handling:
- Script exits on any error with descriptive messages
- Detailed error logs saved to training log file
- Backup creation prevents data loss
- GPU availability checks prevent runtime errors

## Notes

- The script is designed for Ubuntu 22.04 environment
- Requires sufficient disk space for model files (~10GB)
- Internet connection needed for downloading model weights
- Training process cannot be interrupted safely once started

### `view_training_history.sh`

Training history viewer and analyzer:

#### Features:
- **List Sessions**: View all training sessions with summary information
- **View Details**: Detailed information about specific training sessions
- **Compare Sessions**: Side-by-side comparison of two training sessions
- **Find Best Model**: Identify the best performing model by loss
- **Metadata Analysis**: Complete training configuration and results

#### Usage:
```bash
cd /home/ceng/cu_ceng_bot

# List all training sessions
./scripts/view_training_history.sh list

# View specific training session
./scripts/view_training_history.sh view v2

# Compare two training sessions
./scripts/view_training_history.sh compare v1 v2

# Find best performing model
./scripts/view_training_history.sh best

# Show help
./scripts/view_training_history.sh help
```

#### Output Examples:
```bash
# List output
Training Sessions
v1:      1245 lines    2025-07-16    Loss: 0.8259
v2:      1389 lines    2025-07-16    Loss: 0.7665
v3:      1156 lines    2025-07-16    Loss: 0.7421

# Compare output
Comparing v1 vs v2
                    v1          v2
Final Loss:         0.8259      0.7665
Total Steps:        1250        1389
Dataset Size:       18000       18000
Learning Rate:      0.0002      0.0002
Batch Size:         4           4
Epochs:             3           3

Improvement: 7.19% (better)
```

### `switch_model.sh`

Model version switcher for production deployments:

#### Features:
- **List Models**: View all available model versions with metrics
- **Switch Active Model**: Change production model without downtime
- **Model Information**: Display training metrics and configuration
- **Auto-Switch**: Automatically switch to next available version
- **Rollback Support**: Easy revert to previous model version

#### Usage:
```bash
cd /home/ceng/cu_ceng_bot

# List all available models
./scripts/switch_model.sh list

# Switch to specific model version
./scripts/switch_model.sh switch v2

# Show current active model
./scripts/switch_model.sh current

# Auto-switch to next version
./scripts/switch_model.sh auto

# Show help
./scripts/switch_model.sh help
```

#### Output Examples:
```bash
# List output
Available Model Versions
v1 (ACTIVE)
    Loss: 0.8259 | Steps: 1250 | Date: 2025-07-16
v2
    Loss: 0.7665 | Steps: 1389 | Date: 2025-07-16
v3
    Loss: 0.7421 | Steps: 1156 | Date: 2025-07-16

# Switch output
Successfully switched to model v2
Model Information:
  Version: v2
  Final Loss: 0.7665
  Total Steps: 1389
  Dataset Size: 18000
  Training Date: 2025-07-16
```

#### Production Benefits:
- **Zero Downtime**: Switch models without service restart
- **Easy Rollback**: Instant revert to previous versions
- **A/B Testing**: Compare model performance in production
- **Progressive Deployment**: Test new models safely