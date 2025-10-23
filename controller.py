"""
Main controller for the LLM StreamChat Assistant.

This module orchestrates all components of the system including configuration management,
chat monitoring, response generation, and the web dashboard.
"""

import asyncio
import logging
import signal
import sys
from typing import Optional, Dict, Any
from datetime import datetime

from config_manager import ConfigurationManager, ConfigurationError
from chat_monitor import ChatMonitor
from response_generator import ResponseGenerator
from message_queue import MessageQueue
from web_dashboard import WebDashboard
from llm_client import LLMClientFactory, LLMClientError


class StreamChatController:
    """
    Main controller that orchestrates all components of the StreamChat Assistant.
    
    Manages the lifecycle of all system components, handles graceful startup and shutdown,
    and provides centralized error handling and logging.
    """
    
    def __init__(self, config_path: str = "config.yml"):
        """
        Initialize the StreamChat Controller.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.config_manager: Optional[ConfigurationManager] = None
        self.message_queue: Optional[MessageQueue] = None
        self.chat_monitor: Optional[ChatMonitor] = None
        self.response_generator: Optional[ResponseGenerator] = None
        self.web_dashboard: Optional[WebDashboard] = None
        self.llm_client = None
        
        # System state
        self.is_running = False
        self.startup_time: Optional[datetime] = None
        self.shutdown_event = asyncio.Event()
        
        # Component tasks
        self.dashboard_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> None:
        """Initialize all system components."""
        self.logger.info("Initializing StreamChat Assistant...")
        
        try:
            # Initialize configuration manager
            self.logger.info("Loading configuration...")
            self.config_manager = ConfigurationManager(self.config_path)
            
            # Validate startup configuration
            await self._validate_startup_configuration()
            
            # Initialize message queue
            self.logger.info("Initializing message queue...")
            behavior_config = self.config_manager.get_behavior_config()
            max_queue_size = behavior_config.get('max_queue_size', 100)
            self.message_queue = MessageQueue(max_size=max_queue_size)
            
            # Initialize LLM client
            self.logger.info("Initializing LLM client...")
            await self._initialize_llm_client()
            
            # Initialize chat monitor
            self.logger.info("Initializing chat monitor...")
            self.chat_monitor = ChatMonitor(
                config_manager=self.config_manager,
                message_queue=self.message_queue
            )
            
            # Initialize response generator
            self.logger.info("Initializing response generator...")
            self.response_generator = ResponseGenerator(
                config_manager=self.config_manager,
                message_queue=self.message_queue
            )
            
            # Set up response delivery callbacks
            await self._setup_response_callbacks()
            
            # Initialize web dashboard
            self.logger.info("Initializing web dashboard...")
            self.web_dashboard = WebDashboard(
                config_manager=self.config_manager,
                chat_monitor=self.chat_monitor,
                response_generator=self.response_generator,
                llm_client=self.llm_client
            )
            
            # Connect dashboard to other components for real-time updates
            self._connect_dashboard_callbacks()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def _validate_startup_configuration(self) -> None:
        """Validate configuration for startup requirements."""
        config = self.config_manager.get_config()
        
        # Check that at least one platform is enabled
        enabled_platforms = self.config_manager.get_enabled_platforms()
        if not enabled_platforms:
            raise ConfigurationError("At least one chat platform must be enabled")
        
        # Validate LLM configuration
        llm_config = config.get('llm', {})
        required_llm_fields = ['backend', 'model', 'base_url']
        for field in required_llm_fields:
            if not llm_config.get(field):
                raise ConfigurationError(f"LLM configuration missing required field: {field}")
        
        # Validate dashboard configuration
        dashboard_config = config.get('dashboard', {})
        if not isinstance(dashboard_config.get('port'), int):
            raise ConfigurationError("Dashboard port must be an integer")
        
        self.logger.info(f"Configuration validated - enabled platforms: {enabled_platforms}")
    
    async def _initialize_llm_client(self) -> None:
        """Initialize the LLM client based on configuration."""
        try:
            llm_config = self.config_manager.get_llm_config()
            backend = llm_config.get('backend', 'ollama')
            
            client_kwargs = {
                'base_url': llm_config.get('base_url', 'http://localhost:11434'),
                'timeout': llm_config.get('timeout', 30.0)
            }
            
            # Add backend-specific configuration
            if backend in ['lmstudio', 'lemonade']:
                client_kwargs['api_key'] = llm_config.get('api_key', 'default-key')
            
            self.llm_client = LLMClientFactory.create_client(backend, **client_kwargs)
            
            # Test connection
            if await self.llm_client.test_connection():
                self.logger.info(f"LLM client ({backend}) connected successfully")
            else:
                self.logger.warning(f"LLM client ({backend}) connection test failed")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            raise
    
    async def _setup_response_callbacks(self) -> None:
        """Set up response delivery callbacks for chat platforms."""
        if not self.response_generator or not self.chat_monitor:
            return
        
        # Register Twitch response callback
        async def twitch_response_callback(original_message, response):
            """Deliver response to Twitch chat."""
            try:
                if self.chat_monitor.twitch_bot and self.chat_monitor.is_platform_connected('twitch'):
                    await self.chat_monitor.twitch_bot.send_message(original_message.channel, response)
                    self.logger.debug(f"Sent Twitch response: {response[:50]}...")
                    
                    # Notify dashboard
                    if self.web_dashboard:
                        self.web_dashboard.add_response(original_message, response, 'twitch')
                else:
                    self.logger.warning("Twitch bot not connected - cannot send response")
            except Exception as e:
                self.logger.error(f"Failed to send Twitch response: {e}")
        
        # Register YouTube response callback
        async def youtube_response_callback(original_message, response):
            """Deliver response to YouTube chat."""
            try:
                # Note: YouTube chat is read-only with pytchat
                # This is a placeholder for potential future implementation
                self.logger.info(f"YouTube response (read-only): {response}")
                
                # Notify dashboard
                if self.web_dashboard:
                    self.web_dashboard.add_response(original_message, response, 'youtube')
            except Exception as e:
                self.logger.error(f"Failed to send YouTube response: {e}")
        
        # Register callbacks
        self.response_generator.register_response_callback('twitch', twitch_response_callback)
        self.response_generator.register_response_callback('youtube', youtube_response_callback)
        
        self.logger.info("Response delivery callbacks registered")
    
    def _connect_dashboard_callbacks(self) -> None:
        """Connect dashboard to other components for real-time updates."""
        if not self.web_dashboard:
            return
        
        # Set up message callback for dashboard updates
        async def dashboard_message_callback(message):
            """Forward chat messages to dashboard."""
            self.web_dashboard.add_chat_message(message)
        
        # Add dashboard callback to chat monitor
        if self.chat_monitor:
            self.chat_monitor.message_callback = dashboard_message_callback
        
        self.logger.info("Dashboard callbacks connected")
    
    async def start(self) -> None:
        """Start all system components."""
        if self.is_running:
            self.logger.warning("System is already running")
            return
        
        try:
            self.startup_time = datetime.now()
            self.logger.info("Starting StreamChat Assistant...")
            
            # Start message queue
            if self.message_queue:
                await self.message_queue.start_processing()
                self.logger.info("Message queue started")
            
            # Start response generator
            if self.response_generator:
                await self.response_generator.start()
                self.logger.info("Response generator started")
            
            # Start chat monitoring for enabled platforms
            if self.chat_monitor:
                await self.chat_monitor.start_monitoring()
                self.logger.info("Chat monitoring started")
            
            # Start web dashboard in background task
            if self.web_dashboard:
                self.dashboard_task = asyncio.create_task(self._run_dashboard())
                self.logger.info("Web dashboard starting...")
            
            self.is_running = True
            
            # Log startup summary
            enabled_platforms = self.config_manager.get_enabled_platforms()
            llm_config = self.config_manager.get_llm_config()
            dashboard_config = self.config_manager.get_dashboard_config()
            
            self.logger.info(
                f"StreamChat Assistant started successfully!\n"
                f"  - Platforms: {', '.join(enabled_platforms)}\n"
                f"  - LLM Backend: {llm_config.get('backend')} ({llm_config.get('model')})\n"
                f"  - Dashboard: http://{dashboard_config.get('host', 'localhost')}:{dashboard_config.get('port', 5000)}"
            )
            
            # Add startup log to dashboard
            if self.web_dashboard:
                self.web_dashboard.add_system_log(
                    'info', 
                    f'StreamChat Assistant started with platforms: {", ".join(enabled_platforms)}'
                )
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            await self.stop()
            raise
    
    async def _run_dashboard(self) -> None:
        """Run the web dashboard in a separate task."""
        try:
            # Run dashboard in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.web_dashboard.start_server)
        except Exception as e:
            self.logger.error(f"Dashboard error: {e}")
            if self.web_dashboard:
                self.web_dashboard.add_system_log('error', f'Dashboard error: {e}')
    
    async def stop(self) -> None:
        """Stop all system components gracefully."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping StreamChat Assistant...")
        self.is_running = False
        
        try:
            # Stop web dashboard
            if self.dashboard_task and not self.dashboard_task.done():
                self.dashboard_task.cancel()
                try:
                    await self.dashboard_task
                except asyncio.CancelledError:
                    pass
            
            if self.web_dashboard:
                self.web_dashboard.stop_server()
                self.logger.info("Web dashboard stopped")
            
            # Stop chat monitoring
            if self.chat_monitor:
                await self.chat_monitor.stop_monitoring()
                self.logger.info("Chat monitoring stopped")
            
            # Stop response generator
            if self.response_generator:
                await self.response_generator.stop()
                self.logger.info("Response generator stopped")
            
            # Stop message queue
            if self.message_queue:
                await self.message_queue.stop_processing()
                self.logger.info("Message queue stopped")
            
            # Close LLM client
            if self.llm_client:
                await self.llm_client.close()
                self.logger.info("LLM client closed")
            
            # Calculate uptime
            if self.startup_time:
                uptime = datetime.now() - self.startup_time
                self.logger.info(f"StreamChat Assistant stopped (uptime: {uptime})")
            else:
                self.logger.info("StreamChat Assistant stopped")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def restart(self) -> None:
        """Restart the entire system."""
        self.logger.info("Restarting StreamChat Assistant...")
        await self.stop()
        await asyncio.sleep(2)  # Brief pause
        await self.start()
    
    async def reload_configuration(self) -> None:
        """Reload configuration and restart affected components."""
        self.logger.info("Reloading configuration...")
        
        try:
            # Reload configuration
            old_config = self.config_manager.get_config()
            self.config_manager.load_config()
            new_config = self.config_manager.get_config()
            
            # Check what changed and restart affected components
            changes = self._detect_config_changes(old_config, new_config)
            
            if changes.get('llm_changed'):
                self.logger.info("LLM configuration changed - reinitializing LLM client")
                await self._initialize_llm_client()
                if self.response_generator:
                    self.response_generator.llm_client = self.llm_client
            
            if changes.get('platforms_changed'):
                self.logger.info("Platform configuration changed - restarting chat monitor")
                if self.chat_monitor:
                    await self.chat_monitor.stop_monitoring()
                    await self.chat_monitor.start_monitoring()
            
            if changes.get('dashboard_changed'):
                self.logger.info("Dashboard configuration changed - restart required")
                # Dashboard restart requires full system restart
                await self.restart()
                return
            
            self.logger.info("Configuration reloaded successfully")
            
            if self.web_dashboard:
                self.web_dashboard.add_system_log('info', 'Configuration reloaded')
            
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}")
            if self.web_dashboard:
                self.web_dashboard.add_system_log('error', f'Configuration reload failed: {e}')
    
    def _detect_config_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, bool]:
        """Detect what configuration sections have changed."""
        changes = {
            'llm_changed': old_config.get('llm') != new_config.get('llm'),
            'platforms_changed': old_config.get('platforms') != new_config.get('platforms'),
            'personality_changed': old_config.get('personality') != new_config.get('personality'),
            'behavior_changed': old_config.get('behavior') != new_config.get('behavior'),
            'dashboard_changed': old_config.get('dashboard') != new_config.get('dashboard')
        }
        return changes
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status information."""
        status = {
            'running': self.is_running,
            'startup_time': self.startup_time.isoformat() if self.startup_time else None,
            'uptime_seconds': (datetime.now() - self.startup_time).total_seconds() if self.startup_time else 0,
            'components': {}
        }
        
        # Component status
        if self.config_manager:
            status['components']['config_manager'] = 'initialized'
            status['enabled_platforms'] = self.config_manager.get_enabled_platforms()
        
        if self.message_queue:
            status['components']['message_queue'] = 'running' if self.message_queue.is_processing else 'stopped'
            status['queue_size'] = len(self.message_queue._queue) if hasattr(self.message_queue, '_queue') else 0
        
        if self.chat_monitor:
            status['components']['chat_monitor'] = 'running' if self.chat_monitor.is_running else 'stopped'
            status['platform_connections'] = self.chat_monitor.get_connection_status()
        
        if self.response_generator:
            status['components']['response_generator'] = 'running' if self.response_generator.is_running else 'stopped'
            status['response_stats'] = self.response_generator.get_stats()
        
        if self.web_dashboard:
            status['components']['web_dashboard'] = 'running'
            dashboard_config = self.config_manager.get_dashboard_config()
            status['dashboard_url'] = f"http://{dashboard_config.get('host', 'localhost')}:{dashboard_config.get('port', 5000)}"
        
        return status
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of all components."""
        health = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {},
            'issues': []
        }
        
        try:
            # Check configuration
            if self.config_manager:
                try:
                    self.config_manager.validate_config(self.config_manager.get_config())
                    health['components']['configuration'] = 'healthy'
                except Exception as e:
                    health['components']['configuration'] = 'error'
                    health['issues'].append(f'Configuration error: {e}')
            
            # Check LLM client
            if self.llm_client:
                try:
                    llm_health = await self.response_generator.test_llm_connection()
                    health['components']['llm_client'] = 'healthy' if llm_health.get('connected') else 'error'
                    if not llm_health.get('connected'):
                        health['issues'].append(f"LLM connection error: {llm_health.get('error', 'Unknown')}")
                except Exception as e:
                    health['components']['llm_client'] = 'error'
                    health['issues'].append(f'LLM health check failed: {e}')
            
            # Check chat platforms
            if self.chat_monitor:
                connection_status = self.chat_monitor.get_connection_status()
                enabled_platforms = self.config_manager.get_enabled_platforms()
                
                for platform in enabled_platforms:
                    if connection_status.get(platform) == 'connected':
                        health['components'][f'{platform}_chat'] = 'healthy'
                    else:
                        health['components'][f'{platform}_chat'] = 'error'
                        health['issues'].append(f'{platform.title()} chat not connected')
            
            # Check message queue
            if self.message_queue:
                if self.message_queue.is_processing:
                    health['components']['message_queue'] = 'healthy'
                else:
                    health['components']['message_queue'] = 'error'
                    health['issues'].append('Message queue not running')
            
            # Determine overall status
            if health['issues']:
                health['overall_status'] = 'degraded' if len(health['issues']) < 3 else 'unhealthy'
            
        except Exception as e:
            health['overall_status'] = 'error'
            health['issues'].append(f'Health check failed: {e}')
        
        return health


async def main():
    """Main entry point for the StreamChat Assistant."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('streamchat.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Create controller
    controller = StreamChatController()
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(controller.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize and start the system
        await controller.initialize()
        await controller.start()
        
        # Wait for shutdown signal
        await controller.shutdown_event.wait()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await controller.stop()


if __name__ == "__main__":
    asyncio.run(main())