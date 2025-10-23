"""
YouTube Bot Integration for StreamChat Assistant

This module provides a simplified entry point for YouTube-only operation.
It integrates with the new ChatMonitor architecture while maintaining
backward compatibility with existing startup scripts.
"""

import asyncio
import logging
import sys
from controller import StreamChatController

logger = logging.getLogger(__name__)


class YouTubeBotWrapper:
    """
    Wrapper class for YouTube-only bot operation.
    
    Provides backward compatibility while using the new ChatMonitor architecture.
    """
    
    def __init__(self, config_path: str = "config.yml"):
        self.controller = StreamChatController(config_path)
        
    async def start(self):
        """Start YouTube bot using the new architecture."""
        try:
            # Initialize the full system
            await self.controller.initialize()
            
            # Verify YouTube is enabled
            enabled_platforms = self.controller.config_manager.get_enabled_platforms()
            if 'youtube' not in enabled_platforms:
                logger.error("YouTube is not enabled in configuration")
                return False
            
            # Start the system
            await self.controller.start()
            
            logger.info("YouTube bot started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start YouTube bot: {e}")
            return False
    
    async def stop(self):
        """Stop the YouTube bot."""
        await self.controller.stop()
    
    async def run(self):
        """Run the YouTube bot until interrupted."""
        if await self.start():
            try:
                # Wait for shutdown signal
                await self.controller.shutdown_event.wait()
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
            finally:
                await self.stop()


async def main():
    """Main entry point for YouTube bot."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('youtube_bot.log')
        ]
    )
    
    logger.info("Starting YouTube bot...")
    
    bot = YouTubeBotWrapper()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())