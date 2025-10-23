#!/usr/bin/env python3
"""
StreamChat Assistant Unified Startup Script

Main entry point for starting the complete StreamChat Assistant system.
Supports multiple run modes and comprehensive configuration options.
"""

import asyncio
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from controller import StreamChatController


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    else:
        handlers.append(logging.FileHandler('streamchat.log'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def validate_run_mode(mode: str, config_path: Path) -> bool:
    """Validate that the run mode is compatible with configuration."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        platforms = config.get('platforms', {})
        twitch_enabled = platforms.get('twitch', {}).get('enabled', False)
        youtube_enabled = platforms.get('youtube', {}).get('enabled', False)
        
        if mode == 'twitch-only' and not twitch_enabled:
            print(f"Error: Twitch is not enabled in {config_path}")
            return False
        elif mode == 'youtube-only' and not youtube_enabled:
            print(f"Error: YouTube is not enabled in {config_path}")
            return False
        elif mode == 'full' and not (twitch_enabled or youtube_enabled):
            print(f"Error: At least one platform must be enabled in {config_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error validating configuration: {e}")
        return False


async def run_full_system(config_path: str, logger):
    """Run the complete StreamChat Assistant system."""
    logger.info("Starting complete StreamChat Assistant system...")
    
    controller = StreamChatController(config_path)
    
    try:
        await controller.initialize()
        await controller.start()
        
        # Wait for shutdown signal
        await controller.shutdown_event.wait()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    finally:
        await controller.stop()


async def run_platform_only(platform: str, config_path: str, logger):
    """Run system with only specified platform enabled."""
    logger.info(f"Starting StreamChat Assistant in {platform}-only mode...")
    
    if platform == 'twitch':
        from twitch_bot import TwitchBotWrapper
        bot = TwitchBotWrapper(config_path)
    elif platform == 'youtube':
        from youtube_bot import YouTubeBotWrapper
        bot = YouTubeBotWrapper(config_path)
    else:
        raise ValueError(f"Unsupported platform: {platform}")
    
    await bot.run()


async def main():
    """Main entry point with comprehensive argument parsing."""
    parser = argparse.ArgumentParser(
        description='StreamChat Assistant - AI-powered chat bot for Twitch and YouTube',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run Modes:
  full         Start complete system with all enabled platforms (default)
  twitch-only  Start with Twitch integration only
  youtube-only Start with YouTube integration only
  dashboard    Start web dashboard only (requires system to be running)

Examples:
  python start_streamchat.py                    # Full system with default config
  python start_streamchat.py --mode twitch-only # Twitch only
  python start_streamchat.py --config my.yml    # Custom config file
  python start_streamchat.py --log-level DEBUG  # Debug logging
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['full', 'twitch-only', 'youtube-only', 'dashboard'],
        default='full',
        help='Run mode (default: full)'
    )
    parser.add_argument(
        '--config', 
        default='config.yml',
        help='Path to configuration file (default: config.yml)'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    parser.add_argument(
        '--log-file',
        help='Custom log file path (default: streamchat.log)'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Validate configuration and exit'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='StreamChat Assistant v1.0.0'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Please create a config.yml file or specify a different path with --config")
        logger.info("You can copy config.example.yml to config.yml as a starting point")
        sys.exit(1)
    
    # Validate configuration
    if not validate_run_mode(args.mode, config_path):
        sys.exit(1)
    
    if args.validate_only:
        logger.info("Configuration validation passed")
        return
    
    logger.info(f"Starting StreamChat Assistant in '{args.mode}' mode")
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Log level: {args.log_level}")
    
    try:
        if args.mode == 'full':
            await run_full_system(str(config_path), logger)
        elif args.mode == 'twitch-only':
            await run_platform_only('twitch', str(config_path), logger)
        elif args.mode == 'youtube-only':
            await run_platform_only('youtube', str(config_path), logger)
        elif args.mode == 'dashboard':
            logger.info("Dashboard-only mode not yet implemented")
            logger.info("Use full mode to start the complete system including dashboard")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())