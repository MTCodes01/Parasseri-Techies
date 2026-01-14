"""
Configuration Manager Module
Handles loading, validating, and accessing configuration settings.
"""

import json
import os
from typing import Any, Dict

class ConfigManager:
    """Manages application configuration from config.json"""
    
    DEFAULT_CONFIG = {
        "emotion_detection": {
            "confidence_threshold": 0.6,
            "minimum_emotion_duration": 5.0,
            "smoothing_window_size": 15,
            "buffer_duration": 10
        },
        "camera": {
            "device_index": 0,
            "frame_width": 640,
            "frame_height": 480,
            "scale_factor": 1.3,
            "min_neighbors": 5
        },
        "audio": {
            "supported_formats": [".mp3", ".wav", ".ogg", ".flac"],
            "fade_duration_ms": 1000,
            "default_volume": 0.7
        },
        "ui": {
            "window_width": 512,
            "window_height": 850,
            "show_confidence": True,
            "theme": "dark"
        },
        "logging": {
            "enabled": True,
            "level": "INFO",
            "log_to_file": True,
            "log_file": "emotion_music_player.log"
        }
    }
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Returns:
            Configuration dictionary
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults (user config overrides defaults)
                config = self._merge_configs(self.DEFAULT_CONFIG.copy(), user_config)
                print(f"Configuration loaded from {self.config_path}")
                return config
            except json.JSONDecodeError as e:
                print(f"Error parsing config file: {e}")
                print("Using default configuration")
                return self.DEFAULT_CONFIG.copy()
            except Exception as e:
                print(f"Error loading config file: {e}")
                print("Using default configuration")
                return self.DEFAULT_CONFIG.copy()
        else:
            print(f"Config file not found at {self.config_path}")
            print("Using default configuration")
            # Create default config file
            self._save_config(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG.copy()
    
    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """
        Recursively merge user config with default config.
        
        Args:
            default: Default configuration
            user: User configuration
            
        Returns:
            Merged configuration
        """
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                default[key] = self._merge_configs(default[key], value)
            else:
                default[key] = value
        return default
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary to save
        """
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            *keys: Keys to traverse (e.g., 'emotion_detection', 'confidence_threshold')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, *keys: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            *keys: Keys to traverse (last key is the one to set)
            value: Value to set
        """
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
    
    def save(self) -> None:
        """Save current configuration to file."""
        self._save_config(self.config)
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()


# Global configuration instance
_config_instance = None

def get_config() -> ConfigManager:
    """
    Get the global configuration instance.
    
    Returns:
        ConfigManager instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance
