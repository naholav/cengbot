import os
import json
import torch
import numpy as np
import math
import csv
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

from error_handler import handle_training_error, handle_error, ErrorLevel

# Duplicate detection dependencies
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    DUPLICATE_DETECTION_AVAILABLE = True
except ImportError:
    DUPLICATE_DETECTION_AVAILABLE = False

from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers.trainer_callback import TrainerCallback

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
@dataclass
class Config:
    # Paths
    base_path: str = "/home/ceng/cu_ceng_bot_backup_20250718_112640"
    dataset_path: str = "/home/ceng/cu_ceng_bot_backup_20250718_112640/data/merged_qa_data.jsonl"
    output_dir: str = "/home/ceng/cu_ceng_bot_backup_20250718_112640/models"
    log_dir: str = "/home/ceng/cu_ceng_bot_backup_20250718_112640/logs"
    
    # Model
    model_name: str = "meta-llama/Llama-3.2-3B"
    
    # Training - 3B model for RTX 5090 optimized
    num_epochs: int = 4              # 22k data for optimal
    per_device_train_batch_size: int = 4   # RTX 5090 32GB for safe
    per_device_eval_batch_size: int = 16    # Increased for eval
    gradient_accumulation_steps: int = 16   # Effective batch size = 2 * 32 = 64
    learning_rate: float = 2e-4            # Conservative LR
    warmup_ratio: float = 0.05             # Optimal warmup for 22k data
    weight_decay: float = 0.01             # Standard weight decay
    max_length: int = 512                  # Reduced for VRAM
    
    # LoRA - Optimized for 22K dataset
    lora_r: int = 16             # 16 to 24 - prevent underfit
    lora_alpha: int = 32          # 2*r (standard)
    lora_dropout: float = 0.1     # Standard dropout
    
    # Validation
    val_split_ratio: float = 0.07
    early_stopping_patience: int = 3
    evals_per_epoch: int = 5
    
    # Prompts
    turkish_template: str = """Sen Çukurova Üniversitesi Bilgisayar Mühendisliği bölümünün deneyimli dijital asistanısın. Öğrencilere samimi, yardımsever ve doğru bilgiler vererek destek oluyorsun.

Öğrenci: {question}
Asistan: {answer}"""
    
    english_template: str = """You are an experienced digital assistant for Cukurova University Computer Engineering Department. You provide friendly, helpful, and accurate information to support students.

Student: {question}
Assistant: {answer}"""

config = Config()

# Setup logging with version tracking
def get_next_version_number():
    """Get the next version number for training logs"""
    training_history_dir = f"{config.base_path}/logs/training_history"
    os.makedirs(training_history_dir, exist_ok=True)
    
    # Find existing version files
    version_files = []
    for file in os.listdir(training_history_dir):
        if file.startswith("v") and file.endswith(".log"):
            try:
                version_num = int(file[1:-4])  # Extract number from v{num}.log
                version_files.append(version_num)
            except ValueError:
                continue
    
    # Return next version number
    if version_files:
        return max(version_files) + 1
    else:
        return 1

def get_next_model_version():
    """Get the next model version number"""
    models_dir = f"{config.output_dir}"
    os.makedirs(models_dir, exist_ok=True)
    
    # Find existing model version directories
    model_versions = []
    for dir_name in os.listdir(models_dir):
        if dir_name.startswith("final-best-model-v"):
            try:
                version_num = int(dir_name.split("-v")[1])
                model_versions.append(version_num)
            except (ValueError, IndexError):
                continue
    
    # Return next model version
    if model_versions:
        return max(model_versions) + 1
    else:
        return 1

def setup_logging():
    """Setup logging with version tracking"""
    try:
        os.makedirs(config.log_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create log directory: {e}")
        config.log_dir = "."  # Fallback to current directory
    
    # Get next version number
    version_number = get_next_version_number()
    
    # Configure logging with versioned filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = random.randint(1000, 9999)
    
    # Create both versioned and timestamped log files
    training_history_dir = f"{config.base_path}/logs/training_history"
    versioned_log_filename = f"{training_history_dir}/v{version_number}.log"
    timestamped_log_filename = f"{config.log_dir}/training_{timestamp}_{unique_id}.log"
    
    # Create file handlers for both logs
    versioned_handler = logging.FileHandler(versioned_log_filename)
    timestamped_handler = logging.FileHandler(timestamped_log_filename)
    console_handler = logging.StreamHandler()
    
    # Configure logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    versioned_handler.setFormatter(formatter)
    timestamped_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(versioned_handler)
    logger.addHandler(timestamped_handler)
    logger.addHandler(console_handler)
    
    # Log version information
    logger.info(f"=== CengBot Training Session v{version_number} ===")
    logger.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Versioned log: {versioned_log_filename}")
    logger.info(f"Timestamped log: {timestamped_log_filename}")
    logger.info(f"Configuration: {config}")
    
    return logger

logger = setup_logging()

class SimpleProgressCallback(TrainerCallback):
    """Simple progress logging"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Extract key metrics
            train_loss = logs.get('loss', None)
            eval_loss = logs.get('eval_loss', None)
            
            if train_loss is not None:
                logger.info(f"Step {state.global_step}/{state.max_steps} - Train Loss: {train_loss:.4f}")
            if eval_loss is not None:
                logger.info(f"Step {state.global_step}/{state.max_steps} - Eval Loss: {eval_loss:.4f}")

# Custom Callbacks
class DualLanguageEvalCallback(TrainerCallback):
    """Custom callback for separate TR/EN validation loss tracking"""
    
    def __init__(self, eval_dataset_tr, eval_dataset_en, tokenizer):
        self.eval_dataset_tr = eval_dataset_tr
        self.eval_dataset_en = eval_dataset_en
        self.tokenizer = tokenizer
        self.metrics_history = []
    
    def on_evaluate(self, args, state, control, model, **kwargs):
        """Calculate separate losses for TR and EN"""
        model.eval()
        
        # Turkish evaluation
        tr_loss = self._calculate_loss(model, self.eval_dataset_tr)
        en_loss = self._calculate_loss(model, self.eval_dataset_en)
        
        # Handle None values
        if tr_loss is None or en_loss is None:
            logger.warning("Evaluation returned None loss, skipping metrics logging")
            return control
        
        # Log metrics
        metrics = {
            "eval/loss_turkish": tr_loss,
            "eval/loss_english": en_loss,
            "eval/loss_avg": (tr_loss + en_loss) / 2,
            "step": state.global_step
        }
        
        self.metrics_history.append(metrics)
        
        # Simple console logging
        logger.info(f"Step {state.global_step}: TR Loss: {tr_loss:.4f}, EN Loss: {en_loss:.4f}, Avg Loss: {(tr_loss + en_loss) / 2:.4f}")
        
        return control
    
    def _calculate_loss(self, model, dataset):
        """Calculate average loss for a dataset"""
        from torch.utils.data import DataLoader
        from transformers import DataCollatorForLanguageModeling
        
        total_loss = 0
        total_tokens = 0
        
        # Create proper data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Use the same batch size as eval
        batch_size = config.per_device_eval_batch_size
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=False  # Deterministic evaluation
        )
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(
                    input_ids=batch["input_ids"].to(model.device),
                    attention_mask=batch["attention_mask"].to(model.device),
                    labels=batch["labels"].to(model.device)
                )
                
                # Calculate token count (non-padding tokens)
                num_tokens = (batch["labels"] != -100).sum().item()
                
                # Weighted loss
                total_loss += outputs.loss.item() * num_tokens
                total_tokens += num_tokens
        
        return total_loss / total_tokens if total_tokens > 0 else None

# Data Processing
class DataProcessor:
    """Handle all data loading and processing"""
    
    def __init__(self, tokenizer, logger=None):
        self.tokenizer = tokenizer
        self.logger = logger or logging.getLogger(__name__)
        
    def load_dataset(self) -> List[Dict]:
        """Load JSONL dataset with automatic database export"""
        logger.info(f"Loading dataset from {config.dataset_path}")
        
        # Check if dataset file exists and is recent
        dataset_exists = os.path.exists(config.dataset_path)
        if dataset_exists:
            file_age = time.time() - os.path.getmtime(config.dataset_path)
            file_age_hours = file_age / 3600
            logger.info(f"Dataset file age: {file_age_hours:.1f} hours")
        
        # Auto-export from database if file doesn't exist or is older than 1 hour
        if not dataset_exists or file_age_hours > 1:
            logger.info("Dataset file missing or outdated, exporting from database...")
            try:
                from database_to_training import DatabaseToTrainingConverter
                converter = DatabaseToTrainingConverter(config.base_path)
                export_result = converter.full_export(create_backup=True)
                
                if export_result["success"]:
                    logger.info(f"✅ Database export successful: {export_result['message']}")
                    stats = export_result['statistics']
                    logger.info(f"📊 Exported {stats['total_entries']} entries: {stats['by_language']}")
                else:
                    logger.error(f"❌ Database export failed: {export_result['message']}")
                    if dataset_exists:
                        logger.info("Using existing dataset file...")
                    else:
                        raise FileNotFoundError("No dataset file available and database export failed")
            except Exception as e:
                logger.error(f"Database export error: {e}")
                if not dataset_exists:
                    raise FileNotFoundError(f"Dataset file not found: {config.dataset_path}")
                logger.info("Using existing dataset file...")
        
        # Load the dataset
        data = []
        with open(config.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        logger.info(f"Loaded {len(data)} examples")
        
        # Apply duplicate detection
        data = self.deduplicate_dataset(data)
        
        return data
    
    def deduplicate_dataset(self, data: List[Dict]) -> List[Dict]:
        """Remove duplicates using high-precision semantic similarity"""
        if not DUPLICATE_DETECTION_AVAILABLE:
            logger.warning("Duplicate detection not available - sentence-transformers not installed")
            return data
        
        if not data:
            return data
            
        logger.info("🔍 Starting duplicate detection with high-precision thresholds...")
        
        # Configuration
        ANSWER_SIMILARITY_THRESHOLD = 0.94  # 94% similarity for answers
        QUESTION_SIMILARITY_THRESHOLD = 0.98  # 98% similarity for questions
        
        try:
            # Initialize sentence transformer model
            logger.info("Loading sentence transformer model...")
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # Group by language for better comparison
            turkish_data = [d for d in data if d.get('language', '').lower() == 'turkish']
            english_data = [d for d in data if d.get('language', '').lower() == 'english']
            
            logger.info(f"Processing {len(turkish_data)} Turkish and {len(english_data)} English examples")
            
            # Process each language separately
            filtered_turkish = self._remove_duplicates_by_language(turkish_data, model, ANSWER_SIMILARITY_THRESHOLD, QUESTION_SIMILARITY_THRESHOLD, "Turkish")
            filtered_english = self._remove_duplicates_by_language(english_data, model, ANSWER_SIMILARITY_THRESHOLD, QUESTION_SIMILARITY_THRESHOLD, "English")
            
            # Combine results
            final_data = filtered_turkish + filtered_english
            
            # Log results
            original_count = len(data)
            final_count = len(final_data)
            removed_count = original_count - final_count
            
            logger.info(f"✅ Duplicate detection complete:")
            logger.info(f"   Original: {original_count} examples")
            logger.info(f"   Final: {final_count} examples")
            logger.info(f"   Removed: {removed_count} duplicates ({removed_count/original_count*100:.1f}%)")
            
            return final_data
            
        except Exception as e:
            logger.error(f"Error in duplicate detection: {e}")
            logger.warning("Falling back to original dataset without deduplication")
            return data
    
    def _remove_duplicates_by_language(self, data: List[Dict], model, answer_threshold: float, question_threshold: float, language: str) -> List[Dict]:
        """Remove duplicates within a single language"""
        if not data:
            return data
            
        logger.info(f"Processing {language} examples for duplicates...")
        
        # Extract texts
        questions = [item['question'] for item in data]
        answers = [item['answer'] for item in data]
        
        # Compute embeddings
        logger.info(f"Computing embeddings for {len(questions)} {language} examples...")
        question_embeddings = model.encode(questions, batch_size=32, show_progress_bar=False)
        answer_embeddings = model.encode(answers, batch_size=32, show_progress_bar=False)
        
        # Find duplicates
        duplicates_to_remove = set()
        
        # Check answer similarity (primary check)
        answer_similarities = cosine_similarity(answer_embeddings)
        for i in range(len(data)):
            if i in duplicates_to_remove:
                continue
                
            for j in range(i + 1, len(data)):
                if j in duplicates_to_remove:
                    continue
                    
                # Check answer similarity
                answer_sim = answer_similarities[i, j]
                if answer_sim >= answer_threshold:
                    # Also check question similarity to avoid false positives
                    question_sim = cosine_similarity([question_embeddings[i]], [question_embeddings[j]])[0, 0]
                    
                    if question_sim >= question_threshold:
                        # Both answer and question are very similar - mark as duplicate
                        duplicates_to_remove.add(j)  # Keep the first occurrence
                        logger.debug(f"Duplicate found: Q{i+1} and Q{j+1} (A_sim: {answer_sim:.3f}, Q_sim: {question_sim:.3f})")
        
        # Filter out duplicates
        filtered_data = [item for idx, item in enumerate(data) if idx not in duplicates_to_remove]
        
        removed_count = len(data) - len(filtered_data)
        logger.info(f"   {language}: Removed {removed_count} duplicates ({removed_count/len(data)*100:.1f}%)")
        
        return filtered_data
    
    def create_balanced_split(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """Create balanced train/val split with separate TR/EN validation sets"""
        # Separate by language
        turkish_data = [d for d in data if d.get('language', '').lower() == 'turkish']
        english_data = [d for d in data if d.get('language', '').lower() == 'english']
        
        logger.info(f"Turkish examples: {len(turkish_data)}, English examples: {len(english_data)}")
        
        # Shuffle for random selection
        np.random.seed(42)
        np.random.shuffle(turkish_data)
        np.random.shuffle(english_data)
        
        # Calculate splits - proportional to each language's size (7% each)
        tr_val_size = int(len(turkish_data) * config.val_split_ratio)
        en_val_size = int(len(english_data) * config.val_split_ratio)
        
        # Random split using shuffled data
        val_tr = turkish_data[:tr_val_size]
        val_en = english_data[:en_val_size]
        train_tr = turkish_data[tr_val_size:]
        train_en = english_data[en_val_size:]
        
        # Combine training data
        train_data = train_tr + train_en
        np.random.shuffle(train_data)
        
        logger.info(f"Train: {len(train_data)}, Val TR: {len(val_tr)}, Val EN: {len(val_en)}")
        
        return train_data, val_tr, val_en, val_tr + val_en
    
    def format_example(self, example: Dict) -> str:
        """Format example with appropriate template"""
        template = config.turkish_template if example.get('language', '').lower() == 'turkish' else config.english_template
        return template.format(
            question=example['question'],
            answer=example['answer']
        )
    
    def tokenize_and_mask(self, examples):
        """Tokenize with response masking"""
        formatted_texts = [self.format_example(ex) for ex in examples['raw']]
        
        # Tokenize with padding to max_length
        model_inputs = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
            return_tensors=None
        )
        
        # Apply response masking using token-level approach
        labels = []
        for i, text in enumerate(formatted_texts):
            # Find where answer starts in the text
            answer_start_text = "Assistant: "
            
            # Tokenize the prompt part separately to find where answer starts
            prompt_end_idx = text.find(answer_start_text) + len(answer_start_text)
            prompt_text = text[:prompt_end_idx]
            
            # Tokenize prompt to get its length in tokens
            prompt_tokens = self.tokenizer(
                prompt_text,
                truncation=True,
                padding=False,
                max_length=config.max_length,
                return_tensors=None
            )
            prompt_length = len(prompt_tokens["input_ids"])
            
            # Create labels with masking
            input_ids = model_inputs["input_ids"][i]
            label = [-100] * len(input_ids)
            
            # Only train on the answer part if we have enough tokens
            if prompt_length < len(input_ids) - 1:  # -1 to ensure at least one token is trained
                label[prompt_length:] = input_ids[prompt_length:]
            else:
                # If prompt is too long, at least train on the last token
                self.logger.warning(f"Prompt too long for example {i}, training on last token only")
                label[-1] = input_ids[-1]
                
            labels.append(label)
        
        model_inputs["labels"] = labels
        return model_inputs

# Model Setup
def setup_model_and_tokenizer():
    """Setup model with LoRA"""
    logger.info("Setting up model and tokenizer...")
    
    # Get HuggingFace token from environment
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        raise ValueError("HUGGING_FACE_TOKEN not found in environment variables")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        token=hf_token,
        trust_remote_code=True,
        use_fast=True  # Llama 3.2 uses fast tokenizer
    )
    
    # Configure tokenizer - Llama uses <|end_of_text|> for eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Use left padding
    
    # Model - 3B model 
    import shutil
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache_dir):
        llama_dirs = [d for d in os.listdir(cache_dir) if "Llama" in d and os.path.isdir(os.path.join(cache_dir, d))]
        if llama_dirs:
            logger.info(f"Found cached Llama models: {llama_dirs}")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,  # BF16 optimal for Ada Lovelace
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
        use_cache=False,
        force_download=False  # Use cache but check
    )
    
    # Debug: Print model information
    logger.info(f"Loaded model: {config.model_name}")
    logger.info(f"Model config: {model.config.model_type}")
    logger.info(f"Hidden size: {model.config.hidden_size}")
    logger.info(f"Num layers: {model.config.num_hidden_layers}")
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"Total parameters: {total_params:.1f}B")
    logger.info(f"Model dtype: {model.dtype}")
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing for SOTA training
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    # LoRA configuration - SOTA for small datasets
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

# Training Setup
def create_trainer(model, tokenizer, train_dataset, eval_dataset, eval_dataset_tr, eval_dataset_en):
    """Create trainer with all configurations"""
    
    # Calculate steps with proper rounding
    steps_per_epoch = math.ceil(len(train_dataset) / (config.per_device_train_batch_size * config.gradient_accumulation_steps))
    total_steps = steps_per_epoch * config.num_epochs
    # Ensure we get exactly evals_per_epoch evaluations
    eval_steps = max(1, steps_per_epoch // config.evals_per_epoch)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        tf32=True,
        dataloader_num_workers=14,  # Optimal for 16 core CPU
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=4,
        torch_compile=False,  # Not stable yet for RTX 5090
        remove_unused_columns=False,
        report_to=[],  # No reporting
        run_name=f"cengbot-{datetime.now().strftime('%Y%m%d-%H%M')}",
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,  # Much larger bucket (in RAM)
        dataloader_drop_last=False,
        fp16_full_eval=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },
        optim="adamw_torch_fused",  # Fused optimizer for RTX 5090
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        seed=42,
    )
    
    # Data collator with caching
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Pre-cache datasets in RAM
    logger.info("Pre-caching datasets in RAM...")
    train_dataset = train_dataset.with_format("torch")
    eval_dataset = eval_dataset.with_format("torch")
    eval_dataset_tr = eval_dataset_tr.with_format("torch")
    eval_dataset_en = eval_dataset_en.with_format("torch")
    logger.info("Datasets cached in RAM")
    
    # Callbacks - Clean setup
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience),
        DualLanguageEvalCallback(eval_dataset_tr, eval_dataset_en, tokenizer),
        SimpleProgressCallback()
    ]
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks
    )
    
    return trainer

# Model Saving Functions
def save_model_method1(trainer, save_path):
    """Method 1: Using trainer.save_model()"""
    try:
        trainer.save_model(save_path)
        logger.info(f"✓ Method 1 successful: Saved to {save_path}")
        return True
    except Exception as e:
        logger.error(f"✗ Method 1 failed: {e}")
        return False

def save_model_method2(model, tokenizer, save_path):
    """Method 2: Using model.save_pretrained()"""
    try:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"✓ Method 2 successful: Saved to {save_path}")
        return True
    except Exception as e:
        logger.error(f"✗ Method 2 failed: {e}")
        return False

def save_model_method3(model, save_path):
    """Method 3: Using torch.save() with state_dict"""
    try:
        save_dict = {
            'model_state_dict': model.state_dict(),
            'timestamp': datetime.now().isoformat()
        }
        # Add LoRA config if available
        if hasattr(model, 'peft_config'):
            save_dict['lora_config'] = model.peft_config
            
        torch.save(save_dict, f"{save_path}/pytorch_model.bin")
        logger.info(f"✓ Method 3 successful: Saved to {save_path}/pytorch_model.bin")
        return True
    except Exception as e:
        logger.error(f"✗ Method 3 failed: {e}")
        return False

def save_model_all_methods(trainer, model, tokenizer, base_path):
    """Try all three saving methods"""
    os.makedirs(base_path, exist_ok=True)
    
    success_count = 0
    success_count += save_model_method1(trainer, f"{base_path}/method1")
    success_count += save_model_method2(model, tokenizer, f"{base_path}/method2")
    success_count += save_model_method3(model, f"{base_path}/method3")
    
    if success_count == 0:
        raise Exception("All save methods failed!")
    
    logger.info(f"Successfully saved model using {success_count}/3 methods")

# Main Training Function
def main():
    """Main training pipeline"""
    logger.info("Starting CengBot training pipeline...")
    
    # GPU Detection and Warning
    try:
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info(f"GPU detected: {gpu_name} (Count: {gpu_count})")
        else:
            logger.warning("GPU not detected - training will use CPU (much slower)")
    except Exception as e:
        logger.warning(f"GPU detection failed: {e}")
        logger.warning("This may be due to driver/library version mismatch")
        logger.warning("Training will attempt to proceed with available hardware")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Load and process data
    processor = DataProcessor(tokenizer, logger)
    raw_data = processor.load_dataset()
    
    # Create balanced splits
    train_data, val_tr, val_en, val_combined = processor.create_balanced_split(raw_data)
    
    # Create datasets with caching
    def prepare_dataset(data):
        dataset = Dataset.from_dict({"raw": data})
        # Map with larger batch size for better RAM utilization
        dataset = dataset.map(
            processor.tokenize_and_mask,
            batched=True,
            batch_size=2000,  # Large batch for 128GB RAM
            num_proc=14,  # 16 core CPU - leave 2 cores for system
            load_from_cache_file=False,
            desc="Tokenizing dataset",
            remove_columns=["raw"],
            writer_batch_size=5000  # Faster disk writing
        )
        # Keep everything in memory
        dataset.set_format(type="torch")
        return dataset
    
    train_dataset = prepare_dataset(train_data)
    eval_dataset = prepare_dataset(val_combined)
    eval_dataset_tr = prepare_dataset(val_tr)
    eval_dataset_en = prepare_dataset(val_en)
    
    # Create trainer
    trainer = create_trainer(
        model, tokenizer, train_dataset, eval_dataset, 
        eval_dataset_tr, eval_dataset_en
    )
    
    try:
        # Train
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Get model version and save final model
        model_version = get_next_model_version()
        logger.info(f"Saving final model as version {model_version}...")
        final_save_path = f"{config.output_dir}/final-best-model-v{model_version}"
        save_model_all_methods(trainer, model, tokenizer, final_save_path)
        
        # Save tokenizer with special tokens
        tokenizer.save_pretrained(final_save_path)
        
        # Get version number for training info
        version_number = get_next_version_number() - 1  # Current version (already incremented)
        
        # Save training info with version
        training_info = {
            "version": f"v{version_number}",
            "model_version": f"v{model_version}",
            "final_loss": train_result.metrics["train_loss"],
            "total_steps": train_result.global_step,
            "dataset_size": len(train_dataset),
            "model_name": config.model_name,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "validation_split": config.val_split_ratio,
            "timestamp": datetime.now().isoformat(),
            "training_duration": None,  # Will be calculated if needed
            "log_file": f"v{version_number}.log",
            "model_save_path": final_save_path
        }
        
        with open(f"{final_save_path}/training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        # Also save version-specific training info in training history
        training_history_dir = f"{config.base_path}/logs/training_history"
        with open(f"{training_history_dir}/v{version_number}_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        # Save metrics history with error handling
        if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history') and trainer.state.log_history:
            try:
                with open(f"{config.log_dir}/metrics_history.csv", 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=trainer.state.log_history[0].keys())
                    writer.writeheader()
                    writer.writerows(trainer.state.log_history)
            except Exception as e:
                logger.warning(f"Failed to save metrics history: {e}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        error_response = handle_training_error(e, "main_training")
        logger.error(f"Training failed: {e}")
        logger.error(f"Error context: {error_response['error']['context']}")
        raise

if __name__ == "__main__":
    main()