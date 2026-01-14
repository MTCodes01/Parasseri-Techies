"""
Logging Module
Provides structured logging for the emotion music player.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

class EmotionLogger:
    """Custom logger for the emotion music player application."""
    
    def __init__(
        self,
        name: str = "EmotionMusicPlayer",
        log_file: Optional[str] = None,
        level: str = "INFO",
        log_to_console: bool = True,
        log_to_file: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 3
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_file: Path to log file
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_console: Whether to log to console
            log_to_file: Whether to log to file
            max_bytes: Maximum size of log file before rotation
            backup_count: Number of backup files to keep
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if log_to_file and log_file:
            try:
                # Create logs directory if it doesn't exist
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.warning(f"Could not create file handler: {e}")
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)
    
    def emotion_detected(self, emotion: str, confidence: float) -> None:
        """Log emotion detection event."""
        self.info(f"Emotion detected: {emotion} (confidence: {confidence:.2%})")
    
    def emotion_changed(self, old_emotion: str, new_emotion: str, confidence: float) -> None:
        """Log emotion change event."""
        self.info(f"Emotion changed: {old_emotion} -> {new_emotion} (confidence: {confidence:.2%})")
    
    def track_playing(self, emotion: str, track_path: str) -> None:
        """Log music playback event."""
        track_name = os.path.basename(track_path)
        self.info(f"Playing {emotion} track: {track_name}")
    
    def track_error(self, track_path: str, error: str) -> None:
        """Log track playback error."""
        self.error(f"Error playing track {track_path}: {error}")


# Global logger instance
_logger_instance: Optional[EmotionLogger] = None

def get_logger() -> EmotionLogger:
    """
    Get the global logger instance.
    
    Returns:
        EmotionLogger instance
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = EmotionLogger()
    return _logger_instance

def init_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    log_to_console: bool = True,
    log_to_file: bool = True
) -> EmotionLogger:
    """
    Initialize the global logger with custom settings.
    
    Args:
        log_file: Path to log file
        level: Logging level
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        
    Returns:
        Initialized EmotionLogger instance
    """
    global _logger_instance
    _logger_instance = EmotionLogger(
        log_file=log_file,
        level=level,
        log_to_console=log_to_console,
        log_to_file=log_to_file
    )
    return _logger_instance
