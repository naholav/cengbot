"""
Environment Configuration Loader for CengBot
============================================

This module handles loading environment variables from .env files
and provides a centralized configuration management system.

Author: naholav
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union
from dotenv import load_dotenv

# Setup logging
logger = logging.getLogger(__name__)


class EnvironmentConfig:
    """
    Environment configuration manager for CengBot.
    
    This class loads configuration from environment variables and .env files,
    providing default values and type conversion for different settings.
    """
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize environment configuration.
        
        Args:
            env_file: Optional path to .env file. If None, looks for .env in project root.
        """
        self.project_root = Path(__file__).parent.parent
        self.env_file = env_file or self.project_root / ".env"
        
        # Load environment variables from file if it exists
        if self.env_file.exists():
            load_dotenv(self.env_file)
            logger.info(f"Loaded environment from {self.env_file}")
        else:
            logger.warning(f"Environment file not found: {self.env_file}")
    
    def get_str(self, key: str, default: str = "") -> str:
        """Get string value from environment."""
        return os.getenv(key, default)
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer value from environment."""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            logger.warning(f"Invalid integer value for {key}, using default: {default}")
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float value from environment."""
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            logger.warning(f"Invalid float value for {key}, using default: {default}")
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean value from environment."""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on', 'enabled')
    
    def get_list(self, key: str, default: list = None, separator: str = ",") -> list:
        """Get list value from environment (comma-separated by default)."""
        if default is None:
            default = []
        value = os.getenv(key, "")
        if not value:
            return default
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    def get_path(self, key: str, default: str = "") -> Path:
        """Get Path object from environment."""
        path_str = self.get_str(key, default)
        if path_str.startswith('/'):
            return Path(path_str)  # Absolute path
        return self.project_root / path_str  # Relative to project root
    
    # =============================================================================
    # TELEGRAM BOT CONFIGURATION
    # =============================================================================
    
    @property
    def telegram_bot_token(self) -> str:
        """Telegram bot token from environment."""
        token = self.get_str("TELEGRAM_BOT_TOKEN")
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required but not set")
        return token
    
    @property
    def telegram_topic_id(self) -> Optional[int]:
        """Optional Telegram topic ID for group chats."""
        topic_id = self.get_str("TELEGRAM_TOPIC_ID")
        return int(topic_id) if topic_id else None
    
    # =============================================================================
    # DATABASE CONFIGURATION
    # =============================================================================
    
    @property
    def database_url(self) -> str:
        """Database connection URL."""
        return self.get_str("DATABASE_URL", "sqlite:///university_bot.db")
    
    @property
    def db_pool_size(self) -> int:
        """Database connection pool size."""
        return self.get_int("DB_POOL_SIZE", 10)
    
    @property
    def db_max_overflow(self) -> int:
        """Database connection pool max overflow."""
        return self.get_int("DB_MAX_OVERFLOW", 20)
    
    # =============================================================================
    # RABBITMQ CONFIGURATION
    # =============================================================================
    
    @property
    def rabbitmq_url(self) -> str:
        """RabbitMQ connection URL."""
        return self.get_str("RABBITMQ_URL", "amqp://localhost:5672")
    
    @property
    def rabbitmq_host(self) -> str:
        """RabbitMQ host."""
        return self.get_str("RABBITMQ_HOST", "localhost")
    
    @property
    def rabbitmq_port(self) -> int:
        """RabbitMQ port."""
        return self.get_int("RABBITMQ_PORT", 5672)
    
    @property
    def questions_queue(self) -> str:
        """RabbitMQ questions queue name."""
        return self.get_str("QUESTIONS_QUEUE", "questions")
    
    @property
    def answers_queue(self) -> str:
        """RabbitMQ answers queue name."""
        return self.get_str("ANSWERS_QUEUE", "answers")
    
    # =============================================================================
    # AI MODEL CONFIGURATION
    # =============================================================================
    
    @property
    def base_model_name(self) -> str:
        """Base model name from Hugging Face."""
        return self.get_str("BASE_MODEL_NAME", "meta-llama/Llama-3.2-3B")
    
    @property
    def lora_model_path(self) -> Path:
        """LoRA adapter model path."""
        return self.get_path("LORA_MODEL_PATH", "models/active-model/method1")
    
    @property
    def model_cache_dir(self) -> Path:
        """Model cache directory."""
        return self.get_path("MODEL_CACHE_DIR", "model_cache")
    
    @property
    def model_temperature(self) -> float:
        """Model generation temperature."""
        return self.get_float("MODEL_TEMPERATURE", 0.7)
    
    @property
    def model_max_new_tokens(self) -> int:
        """Maximum new tokens to generate."""
        return self.get_int("MODEL_MAX_NEW_TOKENS", 200)
    
    @property
    def model_top_p(self) -> float:
        """Model top-p sampling parameter."""
        return self.get_float("MODEL_TOP_P", 0.95)
    
    @property
    def model_top_k(self) -> int:
        """Model top-k sampling parameter."""
        return self.get_int("MODEL_TOP_K", 50)
    
    @property
    def model_repetition_penalty(self) -> float:
        """Model repetition penalty."""
        return self.get_float("MODEL_REPETITION_PENALTY", 1.1)
    
    # =============================================================================
    # API SERVER CONFIGURATION
    # =============================================================================
    
    @property
    def api_host(self) -> str:
        """API server host."""
        return self.get_str("API_HOST", "0.0.0.0")
    
    @property
    def api_port(self) -> int:
        """API server port."""
        return self.get_int("API_PORT", 8001)
    
    @property
    def api_reload(self) -> bool:
        """Enable API auto-reload for development."""
        return self.get_bool("API_RELOAD", False)
    
    @property
    def cors_origins(self) -> list:
        """CORS allowed origins."""
        return self.get_list("CORS_ORIGINS", ["http://localhost:3000"])
    
    # =============================================================================
    # LOGGING CONFIGURATION
    # =============================================================================
    
    @property
    def log_level(self) -> str:
        """Logging level."""
        return self.get_str("LOG_LEVEL", "INFO").upper()
    
    @property
    def log_dir(self) -> Path:
        """Log directory path."""
        return self.get_path("LOG_DIR", "logs")
    
    @property
    def worker_log_file(self) -> Path:
        """Worker log file path."""
        return self.get_path("WORKER_LOG_FILE", "logs/worker.log")
    
    @property
    def bot_log_file(self) -> Path:
        """Bot log file path."""
        return self.get_path("BOT_LOG_FILE", "logs/bot.log")
    
    @property
    def api_log_file(self) -> Path:
        """API log file path."""
        return self.get_path("API_LOG_FILE", "logs/admin.log")
    
    # =============================================================================
    # PERFORMANCE CONFIGURATION
    # =============================================================================
    
    @property
    def use_cuda(self) -> bool:
        """Enable CUDA acceleration."""
        return self.get_bool("USE_CUDA", True)
    
    @property
    def model_precision(self) -> str:
        """Model precision: float16, float32, bfloat16."""
        return self.get_str("MODEL_PRECISION", "bfloat16")
    
    @property
    def use_4bit_quantization(self) -> bool:
        """Enable 4-bit quantization."""
        return self.get_bool("USE_4BIT_QUANTIZATION", False)
    
    @property
    def max_concurrent_requests(self) -> int:
        """Maximum concurrent model inference requests."""
        return self.get_int("MAX_CONCURRENT_REQUESTS", 3)
    
    # =============================================================================
    # DEVELOPMENT CONFIGURATION
    # =============================================================================
    
    @property
    def debug(self) -> bool:
        """Enable debug mode."""
        return self.get_bool("DEBUG", False)
    
    @property
    def skip_model_loading(self) -> bool:
        """Skip model loading for testing."""
        return self.get_bool("SKIP_MODEL_LOADING", False)


# Global configuration instance
config = EnvironmentConfig()


def load_config(env_file: Optional[str] = None) -> EnvironmentConfig:
    """
    Load configuration from environment file.
    
    Args:
        env_file: Optional path to .env file
        
    Returns:
        EnvironmentConfig instance
    """
    return EnvironmentConfig(env_file)


def get_config() -> EnvironmentConfig:
    """
    Get the global configuration instance.
    
    Returns:
        Global EnvironmentConfig instance
    """
    return config