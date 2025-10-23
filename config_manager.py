"""
Configuration management for the LLM StreamChat Assistant.

This module handles loading, validating, and updating configuration settings
from YAML files, providing a centralized interface for all configuration needs.
"""

import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigurationManager:
    """Manages application configuration from YAML files."""
    
    def __init__(self, config_path: str = "config.yml"):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
        # Load configuration on initialization
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            The loaded configuration dictionary
            
        Raises:
            ConfigurationError: If config file cannot be loaded or is invalid
        """
        try:
            if not self.config_path.exists():
                raise ConfigurationError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file) or {}
            
            self.validate_config(self.config)
            self.logger.info(f"Configuration loaded successfully from {self.config_path}")
            return self.config
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading config file: {e}")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure and values.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        required_sections = ['platforms', 'llm', 'personality', 'behavior', 'dashboard']
        
        # Check required top-level sections
        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"Missing required configuration section: {section}")
        
        # Validate platforms section
        self._validate_platforms(config['platforms'])
        
        # Validate LLM section
        self._validate_llm(config['llm'])
        
        # Validate personality section
        self._validate_personality(config['personality'])
        
        # Validate behavior section
        self._validate_behavior(config['behavior'])
        
        # Validate dashboard section
        self._validate_dashboard(config['dashboard'])
        
        return True
    
    def _validate_platforms(self, platforms: Dict[str, Any]) -> None:
        """Validate platforms configuration."""
        if 'twitch' not in platforms and 'youtube' not in platforms:
            raise ConfigurationError("At least one platform (twitch or youtube) must be configured")
        
        # Validate Twitch config if present
        if 'twitch' in platforms:
            twitch = platforms['twitch']
            required_twitch = ['enabled', 'token', 'nick', 'channel']
            for field in required_twitch:
                if field not in twitch:
                    raise ConfigurationError(f"Missing required Twitch field: {field}")
        
        # Validate YouTube config if present
        if 'youtube' in platforms:
            youtube = platforms['youtube']
            required_youtube = ['enabled', 'video_id']
            for field in required_youtube:
                if field not in youtube:
                    raise ConfigurationError(f"Missing required YouTube field: {field}")
    
    def _validate_llm(self, llm: Dict[str, Any]) -> None:
        """Validate LLM configuration."""
        required_llm = ['backend', 'model', 'base_url', 'timeout']
        for field in required_llm:
            if field not in llm:
                raise ConfigurationError(f"Missing required LLM field: {field}")
        
        valid_backends = ['ollama', 'lmstudio', 'lemonade']
        if llm['backend'] not in valid_backends:
            raise ConfigurationError(f"Invalid LLM backend: {llm['backend']}. Must be one of: {valid_backends}")
        
        if not isinstance(llm['timeout'], (int, float)) or llm['timeout'] <= 0:
            raise ConfigurationError("LLM timeout must be a positive number")
    
    def _validate_personality(self, personality: Dict[str, Any]) -> None:
        """Validate personality configuration."""
        required_personality = ['preset', 'response_types', 'response_length']
        for field in required_personality:
            if field not in personality:
                raise ConfigurationError(f"Missing required personality field: {field}")
        
        valid_presets = ['friendly', 'professional', 'humorous', 'supportive']
        if personality['preset'] not in valid_presets:
            raise ConfigurationError(f"Invalid personality preset: {personality['preset']}. Must be one of: {valid_presets}")
        
        valid_response_types = ['conversational', 'informational', 'humorous', 'supportive']
        if not isinstance(personality['response_types'], list):
            raise ConfigurationError("response_types must be a list")
        
        for response_type in personality['response_types']:
            if response_type not in valid_response_types:
                raise ConfigurationError(f"Invalid response type: {response_type}. Must be one of: {valid_response_types}")
        
        valid_lengths = ['short', 'medium', 'long']
        if personality['response_length'] not in valid_lengths:
            raise ConfigurationError(f"Invalid response length: {personality['response_length']}. Must be one of: {valid_lengths}")
    
    def _validate_behavior(self, behavior: Dict[str, Any]) -> None:
        """Validate behavior configuration."""
        required_behavior = ['response_frequency', 'min_response_interval', 'max_queue_size']
        for field in required_behavior:
            if field not in behavior:
                raise ConfigurationError(f"Missing required behavior field: {field}")
        
        if not isinstance(behavior['response_frequency'], (int, float)) or not (0.0 <= behavior['response_frequency'] <= 1.0):
            raise ConfigurationError("response_frequency must be a number between 0.0 and 1.0")
        
        if not isinstance(behavior['min_response_interval'], int) or behavior['min_response_interval'] < 0:
            raise ConfigurationError("min_response_interval must be a non-negative integer")
        
        if not isinstance(behavior['max_queue_size'], int) or behavior['max_queue_size'] <= 0:
            raise ConfigurationError("max_queue_size must be a positive integer")
    
    def _validate_dashboard(self, dashboard: Dict[str, Any]) -> None:
        """Validate dashboard configuration."""
        required_dashboard = ['host', 'port', 'debug']
        for field in required_dashboard:
            if field not in dashboard:
                raise ConfigurationError(f"Missing required dashboard field: {field}")
        
        if not isinstance(dashboard['port'], int) or not (1 <= dashboard['port'] <= 65535):
            raise ConfigurationError("Dashboard port must be an integer between 1 and 65535")
        
        if not isinstance(dashboard['debug'], bool):
            raise ConfigurationError("Dashboard debug must be a boolean")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            
        Raises:
            ConfigurationError: If updates are invalid
        """
        # Create a copy of current config for validation
        updated_config = self.config.copy()
        
        # Apply updates recursively
        self._deep_update(updated_config, updates)
        
        # Validate the updated configuration
        self.validate_config(updated_config)
        
        # If validation passes, apply the updates
        self.config = updated_config
        self.logger.info("Configuration updated successfully")
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Recursively update nested dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save_config(self) -> None:
        """Save current configuration to file.
        
        Raises:
            ConfigurationError: If config cannot be saved
        """
        try:
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
            self.logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            raise ConfigurationError(f"Error saving config file: {e}")
    
    def get_platform_config(self) -> Dict[str, Any]:
        """Get platform configuration."""
        return self.config.get('platforms', {})
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self.config.get('llm', {})
    
    def get_personality_config(self) -> Dict[str, Any]:
        """Get personality configuration."""
        return self.config.get('personality', {})
    
    def get_behavior_config(self) -> Dict[str, Any]:
        """Get behavior configuration."""
        return self.config.get('behavior', {})
    
    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration."""
        return self.config.get('dashboard', {})
    
    def is_platform_enabled(self, platform: str) -> bool:
        """Check if a platform is enabled.
        
        Args:
            platform: Platform name ('twitch' or 'youtube')
            
        Returns:
            True if platform is enabled, False otherwise
        """
        platforms = self.get_platform_config()
        return platforms.get(platform, {}).get('enabled', False)
    
    def get_enabled_platforms(self) -> List[str]:
        """Get list of enabled platforms.
        
        Returns:
            List of enabled platform names
        """
        platforms = self.get_platform_config()
        enabled = []
        
        for platform_name, platform_config in platforms.items():
            if platform_config.get('enabled', False):
                enabled.append(platform_name)
        
        return enabled
    
    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary.
        
        Returns:
            The complete configuration dictionary
        """
        return self.config.copy()