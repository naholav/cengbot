#!/usr/bin/env python3
"""
Centralized error handling for CengBot system
"""

import logging
import traceback
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

class ErrorLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorHandler:
    """Centralized error handling and logging"""
    
    def __init__(self, logger_name: str = "CengBot"):
        self.logger = logging.getLogger(logger_name)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        if not self.logger.handlers:
            # Get the project root directory dynamically
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            log_file = os.path.join(project_root, 'logs', 'error.log')
            
            # Create logs directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # Create file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.ERROR)
            
            # Create console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.WARNING)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.DEBUG)
    
    def handle_error(
        self,
        error: Exception,
        context: str,
        level: ErrorLevel = ErrorLevel.MEDIUM,
        user_message: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle errors with appropriate logging and response generation
        
        Args:
            error: The exception that occurred
            context: Context where error occurred (e.g., "telegram_bot", "model_loading")
            level: Error severity level
            user_message: Optional user message
            additional_data: Optional additional context data
            
        Returns:
            Dict containing error response information
        """
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "level": level.value,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "additional_data": additional_data or {}
        }
        
        # Log based on severity
        if level == ErrorLevel.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR in {context}: {error}")
            self.logger.critical(f"Traceback: {traceback.format_exc()}")
        elif level == ErrorLevel.HIGH:
            self.logger.error(f"HIGH ERROR in {context}: {error}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        elif level == ErrorLevel.MEDIUM:
            self.logger.warning(f"MEDIUM ERROR in {context}: {error}")
            self.logger.warning(f"Traceback: {traceback.format_exc()}")
        else:
            self.logger.info(f"LOW ERROR in {context}: {error}")
        
        # Return response
        return {
            "success": False,
            "error": {
                "type": error_data["error_type"],
                "message": user_message or str(error),
                "context": context,
                "level": level.value,
                "timestamp": error_data["timestamp"]
            }
        }
    
    def handle_validation_error(
        self,
        field: str,
        value: Any,
        expected: str,
        context: str = "validation"
    ) -> Dict[str, Any]:
        """Handle validation errors"""
        error_msg = f"Validation failed for field '{field}': expected {expected}, got {type(value).__name__}"
        
        return {
            "success": False,
            "error": {
                "type": "ValidationError",
                "message": error_msg,
                "field": field,
                "value": str(value),
                "expected": expected,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def handle_database_error(
        self,
        error: Exception,
        operation: str,
        table: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle database-specific errors"""
        context = f"database_{operation}"
        if table:
            context += f"_{table}"
            
        return self.handle_error(
            error,
            context,
            level=ErrorLevel.HIGH,
            user_message="Database operation failed. Please try again.",
            additional_data={"operation": operation, "table": table}
        )
    
    def handle_api_error(
        self,
        error: Exception,
        endpoint: str,
        method: str = "GET"
    ) -> Dict[str, Any]:
        """Handle API-specific errors"""
        context = f"api_{method.lower()}_{endpoint.replace('/', '_')}"
        
        return self.handle_error(
            error,
            context,
            level=ErrorLevel.MEDIUM,
            user_message="API request failed. Please try again.",
            additional_data={"endpoint": endpoint, "method": method}
        )
    
    def handle_model_error(
        self,
        error: Exception,
        operation: str = "inference"
    ) -> Dict[str, Any]:
        """Handle model-specific errors"""
        context = f"model_{operation}"
        
        return self.handle_error(
            error,
            context,
            level=ErrorLevel.HIGH,
            user_message="Model operation failed. Please try again.",
            additional_data={"operation": operation}
        )
    
    def handle_telegram_error(
        self,
        error: Exception,
        user_id: Optional[int] = None,
        message_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Handle Telegram bot errors"""
        context = "telegram_bot"
        
        return self.handle_error(
            error,
            context,
            level=ErrorLevel.MEDIUM,
            user_message="Bot operation failed. Please try again.",
            additional_data={"user_id": user_id, "message_id": message_id}
        )
    
    def handle_training_error(
        self,
        error: Exception,
        step: str,
        epoch: Optional[int] = None
    ) -> Dict[str, Any]:
        """Handle training-specific errors"""
        context = f"training_{step}"
        
        return self.handle_error(
            error,
            context,
            level=ErrorLevel.CRITICAL,
            user_message="Training operation failed.",
            additional_data={"step": step, "epoch": epoch}
        )

# Global error handler instance
error_handler = ErrorHandler()

# Convenience functions
def handle_error(error: Exception, context: str, level: ErrorLevel = ErrorLevel.MEDIUM, **kwargs):
    """Convenience function for error handling"""
    return error_handler.handle_error(error, context, level, **kwargs)

def handle_validation_error(field: str, value: Any, expected: str, context: str = "validation"):
    """Convenience function for validation errors"""
    return error_handler.handle_validation_error(field, value, expected, context)

def handle_database_error(error: Exception, operation: str, table: Optional[str] = None):
    """Convenience function for database errors"""
    return error_handler.handle_database_error(error, operation, table)

def handle_api_error(error: Exception, endpoint: str, method: str = "GET"):
    """Convenience function for API errors"""
    return error_handler.handle_api_error(error, endpoint, method)

def handle_model_error(error: Exception, operation: str = "inference"):
    """Convenience function for model errors"""
    return error_handler.handle_model_error(error, operation)

def handle_telegram_error(error: Exception, user_id: Optional[int] = None, message_id: Optional[int] = None):
    """Convenience function for Telegram errors"""
    return error_handler.handle_telegram_error(error, user_id, message_id)

def handle_training_error(error: Exception, step: str, epoch: Optional[int] = None):
    """Convenience function for training errors"""
    return error_handler.handle_training_error(error, step, epoch)