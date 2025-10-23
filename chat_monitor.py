"""
Chat monitoring system for the LLM StreamChat Assistant.

This module provides the ChatMonitor class that manages connections to multiple
chat platforms (Twitch and YouTube) and routes messages to the processing queue.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, Callable, Any
from enum import Enum

from models import ChatMessage, QueuedMessage
from config_manager import ConfigurationManager
from message_queue import MessageQueue, MessagePriority


class ConnectionState(Enum):
    """Represents the connection state of a platform."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class ChatMonitor:
    """
    Manages connections to chat platforms and routes messages to processing queue.
    
    Handles Twitch and YouTube chat monitoring with automatic reconnection,
    configurable platform enable/disable, and message routing.
    """
    
    def __init__(self, config_manager: ConfigurationManager, message_queue: Optional[MessageQueue] = None, message_callback: Optional[Callable] = None):
        """
        Initialize ChatMonitor with configuration and message processing.
        
        Args:
            config_manager: Configuration manager instance
            message_queue: Optional message queue for async processing
            message_callback: Optional callback function for message processing (legacy)
        """
        self.config_manager = config_manager
        self.message_queue = message_queue
        self.message_callback = message_callback
        self.logger = logging.getLogger(__name__)
        
        # Connection state tracking
        self.connection_states: Dict[str, ConnectionState] = {
            'twitch': ConnectionState.DISCONNECTED,
            'youtube': ConnectionState.DISCONNECTED
        }
        
        # Platform clients
        self.twitch_bot = None
        self.youtube_monitor = None
        
        # Control flags
        self.is_running = False
        self.reconnect_tasks: Dict[str, Optional[asyncio.Task]] = {
            'twitch': None,
            'youtube': None
        }
        
        # Reconnection settings
        self.max_reconnect_attempts = 5
        self.base_reconnect_delay = 2.0  # seconds
        self.max_reconnect_delay = 60.0  # seconds
        
    async def start_monitoring(self) -> None:
        """Start monitoring all enabled chat platforms."""
        if self.is_running:
            self.logger.warning("Chat monitoring is already running")
            return
            
        self.is_running = True
        self.logger.info("Starting chat monitoring system")
        
        config = self.config_manager.get_platform_config()
        
        # Start enabled platforms
        tasks = []
        
        if config.get('twitch', {}).get('enabled', False):
            self.logger.info("Starting Twitch chat monitoring")
            tasks.append(self.connect_twitch())
            
        if config.get('youtube', {}).get('enabled', False):
            self.logger.info("Starting YouTube chat monitoring")
            tasks.append(self.connect_youtube())
            
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            self.logger.warning("No chat platforms are enabled in configuration")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring all chat platforms and cleanup connections."""
        if not self.is_running:
            return
            
        self.logger.info("Stopping chat monitoring system")
        self.is_running = False
        
        # Cancel reconnection tasks
        for platform, task in self.reconnect_tasks.items():
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Disconnect platforms
        await self._disconnect_twitch()
        await self._disconnect_youtube()
        
        self.logger.info("Chat monitoring stopped")
    
    async def connect_twitch(self) -> None:
        """Establish connection to Twitch chat."""
        if self.connection_states['twitch'] in [ConnectionState.CONNECTING, ConnectionState.CONNECTED]:
            return
            
        self.connection_states['twitch'] = ConnectionState.CONNECTING
        
        try:
            config = self.config_manager.get_platform_config()
            twitch_config = config.get('twitch', {})
            
            if not twitch_config.get('enabled', False):
                self.logger.info("Twitch is disabled in configuration")
                self.connection_states['twitch'] = ConnectionState.DISCONNECTED
                return
            
            try:
                # Import twitchio here to avoid import errors if not needed
                import twitchio
            except ImportError:
                raise ConnectionError("twitchio library is required for Twitch integration")
            
            # Create Twitch bot instance
            self.twitch_bot = TwitchBot(
                token=twitch_config.get('token'),
                nick=twitch_config.get('nick'),
                initial_channels=[twitch_config.get('channel')],
                message_handler=self._handle_twitch_message
            )
            
            # Start the bot
            await self.twitch_bot.start()
            self.connection_states['twitch'] = ConnectionState.CONNECTED
            self.logger.info(f"Connected to Twitch chat: {twitch_config.get('channel')}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Twitch: {e}")
            self.connection_states['twitch'] = ConnectionState.ERROR
            
            if self.is_running:
                await self._schedule_reconnect('twitch')
    
    async def connect_youtube(self) -> None:
        """Establish connection to YouTube chat."""
        if self.connection_states['youtube'] in [ConnectionState.CONNECTING, ConnectionState.CONNECTED]:
            return
            
        self.connection_states['youtube'] = ConnectionState.CONNECTING
        
        try:
            config = self.config_manager.get_platform_config()
            youtube_config = config.get('youtube', {})
            
            if not youtube_config.get('enabled', False):
                self.logger.info("YouTube is disabled in configuration")
                self.connection_states['youtube'] = ConnectionState.DISCONNECTED
                return
            
            try:
                # Import pytchat here to avoid import errors if not needed
                import pytchat
            except ImportError:
                raise ConnectionError("pytchat library is required for YouTube integration")
            
            # Create YouTube chat monitor
            video_id = youtube_config.get('video_id')
            if not video_id:
                raise ValueError("YouTube video_id is required but not configured")
            
            self.youtube_monitor = YouTubeMonitor(
                video_id=video_id,
                message_handler=self._handle_youtube_message
            )
            
            # Start monitoring
            await self.youtube_monitor.start()
            self.connection_states['youtube'] = ConnectionState.CONNECTED
            self.logger.info(f"Connected to YouTube chat: {video_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to YouTube: {e}")
            self.connection_states['youtube'] = ConnectionState.ERROR
            
            if self.is_running:
                await self._schedule_reconnect('youtube')
    
    async def _disconnect_twitch(self) -> None:
        """Disconnect from Twitch chat."""
        if self.twitch_bot:
            try:
                await self.twitch_bot.close()
            except Exception as e:
                self.logger.error(f"Error disconnecting from Twitch: {e}")
            finally:
                self.twitch_bot = None
                self.connection_states['twitch'] = ConnectionState.DISCONNECTED
    
    async def _disconnect_youtube(self) -> None:
        """Disconnect from YouTube chat."""
        if self.youtube_monitor:
            try:
                await self.youtube_monitor.stop()
            except Exception as e:
                self.logger.error(f"Error disconnecting from YouTube: {e}")
            finally:
                self.youtube_monitor = None
                self.connection_states['youtube'] = ConnectionState.DISCONNECTED
    
    async def _schedule_reconnect(self, platform: str) -> None:
        """Schedule reconnection for a platform with exponential backoff."""
        if not self.is_running:
            return
            
        # Cancel existing reconnect task
        if self.reconnect_tasks[platform]:
            self.reconnect_tasks[platform].cancel()
        
        # Create new reconnect task
        self.reconnect_tasks[platform] = asyncio.create_task(
            self._reconnect_with_backoff(platform)
        )
    
    async def _reconnect_with_backoff(self, platform: str) -> None:
        """Reconnect to platform with exponential backoff."""
        attempt = 0
        delay = self.base_reconnect_delay
        
        while self.is_running and attempt < self.max_reconnect_attempts:
            attempt += 1
            self.connection_states[platform] = ConnectionState.RECONNECTING
            
            self.logger.info(f"Reconnecting to {platform} (attempt {attempt}/{self.max_reconnect_attempts}) in {delay}s")
            
            try:
                await asyncio.sleep(delay)
                
                if not self.is_running:
                    break
                
                # Attempt reconnection
                if platform == 'twitch':
                    await self.connect_twitch()
                elif platform == 'youtube':
                    await self.connect_youtube()
                
                # Check if connection was successful
                if self.connection_states[platform] == ConnectionState.CONNECTED:
                    self.logger.info(f"Successfully reconnected to {platform}")
                    return
                    
            except Exception as e:
                self.logger.error(f"Reconnection attempt {attempt} failed for {platform}: {e}")
            
            # Exponential backoff with jitter
            delay = min(delay * 2, self.max_reconnect_delay)
        
        # Max attempts reached
        self.logger.error(f"Failed to reconnect to {platform} after {self.max_reconnect_attempts} attempts")
        self.connection_states[platform] = ConnectionState.ERROR
    
    async def _handle_twitch_message(self, message_data: Dict[str, Any]) -> None:
        """Handle incoming Twitch message."""
        try:
            chat_message = ChatMessage(
                platform='twitch',
                username=message_data.get('username', 'unknown'),
                content=message_data.get('content', ''),
                timestamp=datetime.now(),
                channel=message_data.get('channel', ''),
                user_id=message_data.get('user_id'),
                message_id=message_data.get('message_id')
            )
            
            await self._route_message(chat_message)
            
        except Exception as e:
            self.logger.error(f"Error handling Twitch message: {e}")
    
    async def _handle_youtube_message(self, message_data: Dict[str, Any]) -> None:
        """Handle incoming YouTube message."""
        try:
            chat_message = ChatMessage(
                platform='youtube',
                username=message_data.get('username', 'unknown'),
                content=message_data.get('content', ''),
                timestamp=datetime.now(),
                channel=message_data.get('channel', ''),
                user_id=message_data.get('user_id'),
                message_id=message_data.get('message_id')
            )
            
            await self._route_message(chat_message)
            
        except Exception as e:
            self.logger.error(f"Error handling YouTube message: {e}")
    
    async def _route_message(self, message: ChatMessage) -> None:
        """Route message to processing queue or callback."""
        try:
            # Determine message priority based on content or user
            priority = self._determine_message_priority(message)
            
            # Route to message queue if available
            if self.message_queue:
                success = await self.message_queue.enqueue_message(message, priority)
                if success:
                    self.logger.debug(f"Queued message from {message.username} on {message.platform}")
                else:
                    self.logger.warning(f"Failed to queue message from {message.username} - queue full")
            
            # Also call legacy callback if provided
            if self.message_callback:
                await self.message_callback(message)
                
        except Exception as e:
            self.logger.error(f"Error routing message: {e}")
    
    def _determine_message_priority(self, message: ChatMessage) -> MessagePriority:
        """
        Determine message priority based on content and context.
        
        Args:
            message: Chat message to prioritize
            
        Returns:
            Message priority level
        """
        content = message.content.lower()
        
        # High priority for questions or direct mentions
        if any(keyword in content for keyword in ['?', 'help', 'question', '@']):
            return MessagePriority.HIGH
        
        # Low priority for common spam patterns
        if any(pattern in content for pattern in ['!', 'lol', 'lmao', 'kappa', 'poggers']):
            return MessagePriority.LOW
        
        # Normal priority for everything else
        return MessagePriority.NORMAL
    
    def get_connection_status(self) -> Dict[str, str]:
        """Get current connection status for all platforms."""
        return {
            platform: state.value 
            for platform, state in self.connection_states.items()
        }
    
    def is_platform_connected(self, platform: str) -> bool:
        """Check if a specific platform is connected."""
        return self.connection_states.get(platform) == ConnectionState.CONNECTED


# Platform-specific implementations
class TwitchBot:
    """
    Twitch chat bot implementation using twitchio.
    
    Handles connection to Twitch IRC, message processing, and automatic reconnection
    with exponential backoff on connection failures.
    """
    
    def __init__(self, token: str, nick: str, initial_channels: list, message_handler: Callable):
        """
        Initialize Twitch bot.
        
        Args:
            token: Twitch OAuth token (with oauth: prefix)
            nick: Bot nickname
            initial_channels: List of channels to join
            message_handler: Async callback for handling messages
        """
        self.token = token
        self.nick = nick
        self.initial_channels = initial_channels
        self.message_handler = message_handler
        self.logger = logging.getLogger(f"{__name__}.TwitchBot")
        
        self.bot_instance = None
        self.is_running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
    async def start(self):
        """Start the Twitch bot and connect to IRC."""
        if self.is_running:
            return
            
        try:
            import twitchio
            from twitchio.ext import commands
            
            # Create bot instance
            self.bot_instance = TwitchBotClient(
                token=self.token,
                nick=self.nick,
                initial_channels=self.initial_channels,
                message_handler=self.message_handler,
                parent_bot=self
            )
            
            self.is_running = True
            self.logger.info(f"Starting Twitch bot for channels: {self.initial_channels}")
            
            # Start the bot (this will block until disconnected)
            await self.bot_instance.start()
            
        except Exception as e:
            self.logger.error(f"Failed to start Twitch bot: {e}")
            self.is_running = False
            raise
    
    async def close(self):
        """Close the Twitch bot connection."""
        if self.bot_instance:
            try:
                await self.bot_instance.close()
            except Exception as e:
                self.logger.error(f"Error closing Twitch bot: {e}")
            finally:
                self.bot_instance = None
                self.is_running = False
    
    async def send_message(self, channel: str, content: str):
        """Send a message to a Twitch channel."""
        if self.bot_instance and self.is_running:
            try:
                channel_obj = self.bot_instance.get_channel(channel)
                if channel_obj:
                    await channel_obj.send(content)
                else:
                    self.logger.warning(f"Channel {channel} not found or not joined")
            except Exception as e:
                self.logger.error(f"Failed to send message to {channel}: {e}")


class TwitchBotClient:
    """
    Internal twitchio bot client implementation.
    
    This class handles the actual twitchio Bot implementation and event handling.
    """
    
    def __init__(self, token: str, nick: str, initial_channels: list, message_handler: Callable, parent_bot):
        """Initialize the twitchio bot client."""
        import twitchio
        from twitchio.ext import commands
        
        self.message_handler = message_handler
        self.parent_bot = parent_bot
        self.logger = logging.getLogger(f"{__name__}.TwitchBotClient")
        
        # Create the actual twitchio bot
        class InternalBot(commands.Bot):
            def __init__(self, token, nick, initial_channels, outer_self):
                super().__init__(
                    token=token,
                    nick=nick,
                    initial_channels=initial_channels
                )
                self.outer_self = outer_self
            
            async def event_ready(self):
                """Called when the bot is ready and connected."""
                self.outer_self.logger.info(f'Twitch bot ready | {self.nick}')
                self.outer_self.parent_bot.reconnect_attempts = 0  # Reset on successful connection
            
            async def event_message(self, message):
                """Handle incoming chat messages."""
                # Ignore messages from the bot itself
                if message.echo:
                    return
                
                try:
                    # Convert twitchio message to our format
                    message_data = {
                        'username': message.author.name if message.author else 'unknown',
                        'content': message.content,
                        'channel': message.channel.name if message.channel else '',
                        'user_id': str(message.author.id) if message.author else None,
                        'message_id': str(message.id) if hasattr(message, 'id') else None
                    }
                    
                    # Route to message handler
                    await self.outer_self.message_handler(message_data)
                    
                except Exception as e:
                    self.outer_self.logger.error(f"Error processing Twitch message: {e}")
            
            async def event_channel_joined(self, channel):
                """Called when the bot joins a channel."""
                self.outer_self.logger.info(f'Joined Twitch channel: {channel.name}')
            
            async def event_error(self, error, data=None):
                """Handle bot errors."""
                self.outer_self.logger.error(f'Twitch bot error: {error}')
            
            async def event_websocket_closed(self, websocket):
                """Handle websocket disconnection."""
                self.outer_self.logger.warning('Twitch websocket connection closed')
                
                # Trigger reconnection logic in parent
                if self.outer_self.parent_bot.is_running:
                    await self.outer_self._handle_disconnection()
        
        self.bot = InternalBot(token, nick, initial_channels, self)
    
    async def start(self):
        """Start the twitchio bot."""
        try:
            await self.bot.start()
        except Exception as e:
            self.logger.error(f"Twitch bot start failed: {e}")
            await self._handle_disconnection()
    
    async def close(self):
        """Close the twitchio bot."""
        if self.bot:
            await self.bot.close()
    
    def get_channel(self, channel_name: str):
        """Get a channel object by name."""
        return self.bot.get_channel(channel_name)
    
    async def _handle_disconnection(self):
        """Handle disconnection and attempt reconnection."""
        if not self.parent_bot.is_running:
            return
            
        self.parent_bot.reconnect_attempts += 1
        
        if self.parent_bot.reconnect_attempts <= self.parent_bot.max_reconnect_attempts:
            delay = min(2 ** self.parent_bot.reconnect_attempts, 60)  # Exponential backoff, max 60s
            
            self.logger.info(
                f"Attempting Twitch reconnection {self.parent_bot.reconnect_attempts}/"
                f"{self.parent_bot.max_reconnect_attempts} in {delay}s"
            )
            
            await asyncio.sleep(delay)
            
            if self.parent_bot.is_running:
                try:
                    # Close current connection
                    await self.close()
                    
                    # Restart the bot
                    await self.parent_bot.start()
                    
                except Exception as e:
                    self.logger.error(f"Twitch reconnection failed: {e}")
                    await self._handle_disconnection()  # Try again
        else:
            self.logger.error(f"Twitch reconnection failed after {self.parent_bot.max_reconnect_attempts} attempts")
            self.parent_bot.is_running = False


class YouTubeMonitor:
    """
    YouTube chat monitor implementation using pytchat.
    
    Handles connection to YouTube live chat, message processing, and automatic
    reconnection with exponential backoff on connection failures.
    """
    
    def __init__(self, video_id: str, message_handler: Callable):
        """
        Initialize YouTube chat monitor.
        
        Args:
            video_id: YouTube video ID for live stream
            message_handler: Async callback for handling messages
        """
        self.video_id = video_id
        self.message_handler = message_handler
        self.logger = logging.getLogger(f"{__name__}.YouTubeMonitor")
        
        self.chat = None
        self.is_running = False
        self.monitor_task = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
    async def start(self):
        """Start YouTube chat monitoring."""
        if self.is_running:
            return
            
        try:
            import pytchat
            
            self.logger.info(f"Starting YouTube chat monitor for video: {self.video_id}")
            
            # Create pytchat instance
            self.chat = pytchat.create(video_id=self.video_id)
            
            if not self.chat.is_alive():
                raise ConnectionError("Failed to connect to YouTube chat - stream may not be live")
            
            self.is_running = True
            self.reconnect_attempts = 0
            
            # Start monitoring task
            self.monitor_task = asyncio.create_task(self._monitor_chat())
            
            self.logger.info("YouTube chat monitor started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start YouTube chat monitor: {e}")
            self.is_running = False
            raise
    
    async def stop(self):
        """Stop YouTube chat monitoring."""
        self.is_running = False
        
        # Cancel monitoring task
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        # Close chat connection
        if self.chat:
            try:
                self.chat.terminate()
            except Exception as e:
                self.logger.error(f"Error terminating YouTube chat: {e}")
            finally:
                self.chat = None
        
        self.logger.info("YouTube chat monitor stopped")
    
    async def _monitor_chat(self):
        """Main monitoring loop for YouTube chat messages."""
        try:
            while self.is_running and self.chat and self.chat.is_alive():
                try:
                    # Get new messages (non-blocking)
                    messages = self.chat.get().sync()
                    
                    for message in messages:
                        if not self.is_running:
                            break
                            
                        await self._process_message(message)
                    
                    # Small delay to prevent excessive polling
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    self.logger.error(f"Error in YouTube chat monitoring loop: {e}")
                    
                    # If chat is no longer alive, trigger reconnection
                    if self.chat and not self.chat.is_alive():
                        await self._handle_disconnection()
                        break
                    
                    # Brief pause before continuing
                    await asyncio.sleep(1)
            
            # If we exit the loop and should still be running, handle disconnection
            if self.is_running:
                await self._handle_disconnection()
                
        except Exception as e:
            self.logger.error(f"YouTube chat monitoring failed: {e}")
            if self.is_running:
                await self._handle_disconnection()
    
    async def _process_message(self, message):
        """Process a single YouTube chat message."""
        try:
            # Convert pytchat message to our format
            message_data = {
                'username': message.author.name if hasattr(message, 'author') and message.author else 'unknown',
                'content': message.message if hasattr(message, 'message') else '',
                'channel': self.video_id,  # Use video_id as channel identifier
                'user_id': message.author.channelId if hasattr(message, 'author') and message.author else None,
                'message_id': message.id if hasattr(message, 'id') else None
            }
            
            # Route to message handler
            await self.message_handler(message_data)
            
        except Exception as e:
            self.logger.error(f"Error processing YouTube message: {e}")
    
    async def _handle_disconnection(self):
        """Handle disconnection and attempt reconnection."""
        if not self.is_running:
            return
            
        self.reconnect_attempts += 1
        
        if self.reconnect_attempts <= self.max_reconnect_attempts:
            delay = min(2 ** self.reconnect_attempts, 60)  # Exponential backoff, max 60s
            
            self.logger.info(
                f"Attempting YouTube reconnection {self.reconnect_attempts}/"
                f"{self.max_reconnect_attempts} in {delay}s"
            )
            
            await asyncio.sleep(delay)
            
            if self.is_running:
                try:
                    # Close current connection
                    if self.chat:
                        self.chat.terminate()
                        self.chat = None
                    
                    # Restart monitoring
                    await self.start()
                    
                except Exception as e:
                    self.logger.error(f"YouTube reconnection failed: {e}")
                    await self._handle_disconnection()  # Try again
        else:
            self.logger.error(f"YouTube reconnection failed after {self.max_reconnect_attempts} attempts")
            self.is_running = False
    
    def is_connected(self) -> bool:
        """Check if YouTube chat is currently connected."""
        return self.is_running and self.chat and self.chat.is_alive()
    
    async def send_message(self, content: str):
        """
        Send a message to YouTube chat.
        
        Note: pytchat is read-only, so this is a placeholder for potential future
        implementation using YouTube Data API.
        """
        self.logger.warning("YouTube message sending not implemented - pytchat is read-only")