#!/usr/bin/env python3
"""
Twitch Bot Startup Script

Entry point for starting the StreamChat Assistant with Twitch-only configuration.
This script ensures Twitch is enabled and starts the full system.
"""

import asyncio
import sys
import argparse
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from twitch_bot import TwitchBotWrapper


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('twitch_bot.log')
        ]
    )


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Start Twitch StreamChat Assistant')
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
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Please create a config.yml file or specify a different path with --config")
        sys.exit(1)
    
    logger.info(f"Starting Twitch bot with config: {config_path}")
    
    try:
        # Create and run Twitch bot
        bot = TwitchBotWrapper(str(config_path))
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())