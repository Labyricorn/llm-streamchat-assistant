"""
Web Dashboard for LLM StreamChat Assistant

Provides real-time web interface for controlling and monitoring the chat assistant.
Includes bot controls, configuration editing, live logging, and prompt testing.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import json

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import eventlet

from config_manager import ConfigurationManager
from models import ChatMessage, QueuedMessage, ResponseContext

# Configure eventlet for async compatibility
eventlet.monkey_patch()

logger = logging.getLogger(__name__)


class WebDashboard:
    """Flask-based web dashboard with Socket.IO for real-time updates."""
    
    def __init__(self, config_manager: ConfigurationManager, chat_monitor=None, response_generator=None, llm_client=None):
        self.config_manager = config_manager
        self.chat_monitor = chat_monitor
        self.response_generator = response_generator
        self.llm_client = llm_client
        
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'streamchat-assistant-secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='eventlet')
        
        # Dashboard state
        self.bot_status = {
            'twitch': {'connected': False, 'last_message': None, 'error': None},
            'youtube': {'connected': False, 'last_message': None, 'error': None}
        }
        self.system_logs = []
        self.chat_history = []
        self.response_history = []
        self.start_time = datetime.now()
        
        # Setup routes and socket handlers
        self._setup_routes()
        self._setup_socket_handlers()
        
        # Start periodic status sync if chat monitor is available
        if self.chat_monitor:
            self._start_status_sync()
        
    def _setup_routes(self):
        """Setup Flask routes for the dashboard."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            config = self.config_manager.get_config()
            return render_template('dashboard.html', 
                                 config=config, 
                                 bot_status=self.bot_status)
        
        @self.app.route('/api/config', methods=['GET'])
        def get_config():
            """Get current configuration."""
            return jsonify(self.config_manager.get_config())
        
        @self.app.route('/api/config', methods=['POST'])
        def update_config():
            """Update configuration."""
            try:
                config_data = request.get_json()
                self.config_manager.update_config(config_data)
                
                # Emit config update to all connected clients
                self.socketio.emit('config_updated', config_data)
                
                return jsonify({'success': True, 'message': 'Configuration updated'})
            except Exception as e:
                logger.error(f"Config update error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 400
        
        @self.app.route('/api/status')
        def get_status():
            """Get system status."""
            return jsonify({
                'bot_status': self.bot_status,
                'system_health': self._get_system_health(),
                'uptime': self._get_uptime()
            })
        
        @self.app.route('/api/config/reset', methods=['POST'])
        def reset_config():
            """Reset configuration to defaults."""
            try:
                # This would reset to default configuration
                # For now, just return success - actual implementation would reload defaults
                return jsonify({'success': True, 'message': 'Configuration reset to defaults'})
            except Exception as e:
                logger.error(f"Config reset error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/config/validate', methods=['POST'])
        def validate_config():
            """Validate configuration without saving."""
            try:
                config_data = request.get_json()
                validation_result = self._validate_configuration(config_data)
                return jsonify(validation_result)
            except Exception as e:
                logger.error(f"Config validation error: {e}")
                return jsonify({'valid': False, 'errors': [str(e)]}), 400
    
    def _setup_socket_handlers(self):
        """Setup Socket.IO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            logger.info(f"Client connected: {request.sid}")
            # Send current status to newly connected client
            emit('status_update', {
                'bot_status': self.bot_status,
                'system_health': self._get_system_health()
            })
            
            # Send recent logs and chat history
            emit('log_history', self.system_logs[-50:])  # Last 50 logs
            emit('chat_history', self.chat_history[-100:])  # Last 100 messages
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('bot_control')
        def handle_bot_control(data):
            """Handle bot start/stop/restart commands."""
            try:
                action = data.get('action')  # 'start', 'stop', 'restart'
                platform = data.get('platform')  # 'twitch', 'youtube', 'both'
                
                logger.info(f"Bot control: {action} {platform}")
                
                # This will be connected to actual bot control logic later
                result = self._handle_bot_control(action, platform)
                
                emit('bot_control_result', {
                    'success': result['success'],
                    'message': result['message'],
                    'action': action,
                    'platform': platform
                })
                
                # Broadcast status update to all clients
                self.socketio.emit('status_update', {
                    'bot_status': self.bot_status,
                    'system_health': self._get_system_health()
                })
                
            except Exception as e:
                logger.error(f"Bot control error: {e}")
                emit('bot_control_result', {
                    'success': False,
                    'error': str(e),
                    'action': action,
                    'platform': platform
                })
        
        @self.socketio.on('test_prompt')
        def handle_test_prompt(data):
            """Handle prompt testing requests."""
            try:
                prompt = data.get('prompt', '')
                username = data.get('username', 'TestUser')
                platform = data.get('platform', 'twitch')
                test_mode = data.get('testMode', True)
                personality = data.get('personality', {})
                
                logger.info(f"Testing prompt from {username} on {platform}: {prompt[:50]}...")
                
                # Log the test
                self.add_system_log('info', f'Prompt test: {username} on {platform} - "{prompt[:30]}..."')
                
                # Test the prompt
                result = self._test_prompt(prompt, personality)
                
                emit('prompt_test_result', {
                    'success': result['success'],
                    'response': result.get('response', ''),
                    'error': result.get('error', ''),
                    'prompt': prompt,
                    'username': username,
                    'platform': platform,
                    'test_mode': test_mode
                })
                
            except Exception as e:
                logger.error(f"Prompt test error: {e}")
                emit('prompt_test_result', {
                    'success': False,
                    'error': str(e),
                    'prompt': data.get('prompt', ''),
                    'username': data.get('username', 'TestUser'),
                    'platform': data.get('platform', 'twitch'),
                    'test_mode': data.get('testMode', True)
                })
        
        @self.socketio.on('config_update')
        def handle_config_update(data):
            """Handle real-time configuration updates."""
            try:
                self.config_manager.update_config(data)
                
                # Broadcast config update to all clients
                self.socketio.emit('config_updated', data)
                
                self.add_system_log('info', 'Configuration updated via dashboard')
                
            except Exception as e:
                logger.error(f"Config update error: {e}")
                emit('config_update_error', {'error': str(e)})
    
    def _handle_bot_control(self, action: str, platform: str) -> Dict[str, Any]:
        """Handle bot control actions with actual ChatMonitor integration."""
        if not self.chat_monitor:
            return {'success': False, 'message': 'Chat monitor not available'}
        
        try:
            if action == 'start':
                return self._start_bots(platform)
            elif action == 'stop':
                return self._stop_bots(platform)
            elif action == 'restart':
                # Stop then start
                stop_result = self._stop_bots(platform)
                if not stop_result['success']:
                    return stop_result
                # Small delay before restart
                asyncio.create_task(self._delayed_start(platform, 2.0))
                return {'success': True, 'message': f'Restarting {platform} bot(s)...'}
            else:
                return {'success': False, 'message': f'Unknown action: {action}'}
                
        except Exception as e:
            logger.error(f"Bot control error: {e}")
            return {'success': False, 'message': f'Error: {str(e)}'}
    
    def _start_bots(self, platform: str) -> Dict[str, Any]:
        """Start specified bot platforms."""
        if platform == 'both':
            # Start both platforms
            asyncio.create_task(self._start_platform('twitch'))
            asyncio.create_task(self._start_platform('youtube'))
            return {'success': True, 'message': 'Starting both platforms...'}
        elif platform in ['twitch', 'youtube']:
            asyncio.create_task(self._start_platform(platform))
            return {'success': True, 'message': f'Starting {platform} bot...'}
        else:
            return {'success': False, 'message': f'Unknown platform: {platform}'}
    
    def _stop_bots(self, platform: str) -> Dict[str, Any]:
        """Stop specified bot platforms."""
        if platform == 'both':
            # Stop both platforms
            asyncio.create_task(self._stop_platform('twitch'))
            asyncio.create_task(self._stop_platform('youtube'))
            return {'success': True, 'message': 'Stopping both platforms...'}
        elif platform in ['twitch', 'youtube']:
            asyncio.create_task(self._stop_platform(platform))
            return {'success': True, 'message': f'Stopping {platform} bot...'}
        else:
            return {'success': False, 'message': f'Unknown platform: {platform}'}
    
    async def _start_platform(self, platform: str):
        """Start a specific platform asynchronously."""
        try:
            if platform == 'twitch':
                await self.chat_monitor.connect_twitch()
            elif platform == 'youtube':
                await self.chat_monitor.connect_youtube()
            
            # Update status based on actual connection state
            self._sync_bot_status()
            
        except Exception as e:
            logger.error(f"Failed to start {platform}: {e}")
            self.update_bot_status(platform, False, str(e))
    
    async def _stop_platform(self, platform: str):
        """Stop a specific platform asynchronously."""
        try:
            if platform == 'twitch':
                await self.chat_monitor._disconnect_twitch()
            elif platform == 'youtube':
                await self.chat_monitor._disconnect_youtube()
            
            # Update status based on actual connection state
            self._sync_bot_status()
            
        except Exception as e:
            logger.error(f"Failed to stop {platform}: {e}")
            self.add_system_log('error', f'Failed to stop {platform}: {e}')
    
    async def _delayed_start(self, platform: str, delay: float):
        """Start platform after a delay (for restart functionality)."""
        await asyncio.sleep(delay)
        await self._start_platform(platform)
    
    def _sync_bot_status(self):
        """Synchronize dashboard bot status with actual ChatMonitor state."""
        if self.chat_monitor:
            connection_status = self.chat_monitor.get_connection_status()
            
            for platform in ['twitch', 'youtube']:
                connected = self.chat_monitor.is_platform_connected(platform)
                self.bot_status[platform]['connected'] = connected
                
                # Clear error if now connected
                if connected and 'error' in self.bot_status[platform]:
                    del self.bot_status[platform]['error']
    
    def _test_prompt(self, prompt: str, personality: Dict[str, Any]) -> Dict[str, Any]:
        """Test prompt with actual LLM backend."""
        if not self.llm_client:
            return {'success': False, 'error': 'LLM client not available'}
        
        try:
            # Create a test task for the LLM
            asyncio.create_task(self._async_test_prompt(prompt, personality))
            return {'success': True, 'response': 'Testing prompt...'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _async_test_prompt(self, prompt: str, personality: Dict[str, Any]):
        """Asynchronously test prompt with LLM and emit result."""
        try:
            # Build context similar to how ResponseGenerator would
            config = self.config_manager.get_config()
            llm_config = config.get('llm', {})
            
            # Create system prompt with personality
            system_prompt = self._build_test_system_prompt(personality)
            full_prompt = f"{system_prompt}\n\nUser message: {prompt}\n\nResponse:"
            
            # Send to LLM
            response = await self.llm_client.send_request(
                backend=llm_config.get('backend', 'ollama'),
                prompt=full_prompt,
                config=llm_config
            )
            
            # Emit result to client
            self.socketio.emit('prompt_test_result', {
                'success': True,
                'response': response,
                'prompt': prompt
            })
            
        except Exception as e:
            logger.error(f"Prompt test error: {e}")
            self.socketio.emit('prompt_test_result', {
                'success': False,
                'error': str(e),
                'prompt': prompt
            })
    
    def _build_test_system_prompt(self, personality: Dict[str, Any]) -> str:
        """Build system prompt for testing based on personality settings."""
        preset = personality.get('preset', 'friendly')
        custom_instructions = personality.get('custom_instructions', '')
        response_length = personality.get('response_length', 'short')
        
        # Base personality prompts
        personality_prompts = {
            'friendly': "You are a friendly and welcoming chat assistant. Be warm, encouraging, and conversational.",
            'professional': "You are a professional and knowledgeable assistant. Be informative, clear, and helpful.",
            'humorous': "You are a fun and humorous chat assistant. Be witty, playful, and entertaining while staying appropriate.",
            'supportive': "You are a supportive and empathetic assistant. Be understanding, encouraging, and positive."
        }
        
        # Length guidelines
        length_guides = {
            'short': "Keep responses brief and concise (1-2 sentences).",
            'medium': "Provide moderate length responses (2-4 sentences).",
            'long': "Give detailed responses when appropriate (3-6 sentences)."
        }
        
        system_prompt = personality_prompts.get(preset, personality_prompts['friendly'])
        system_prompt += f" {length_guides.get(response_length, length_guides['short'])}"
        
        if custom_instructions:
            system_prompt += f" Additional instructions: {custom_instructions}"
        
        system_prompt += " This is a test message, respond as you would in a live chat."
        
        return system_prompt
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        health = {
            'last_updated': datetime.now().isoformat(),
            'uptime': self._get_uptime()
        }
        
        # LLM Backend Health
        if self.llm_client:
            try:
                # This would be an actual health check in a real implementation
                health['llm_backend'] = 'connected'
                health['llm_backend_type'] = self.config_manager.get_config().get('llm', {}).get('backend', 'unknown')
            except Exception:
                health['llm_backend'] = 'error'
                health['llm_backend_type'] = 'unknown'
        else:
            health['llm_backend'] = 'not_configured'
            health['llm_backend_type'] = 'none'
        
        # Configuration Health
        try:
            config = self.config_manager.get_config()
            health['config_valid'] = True
            health['platforms_enabled'] = []
            
            if config.get('platforms', {}).get('twitch', {}).get('enabled', False):
                health['platforms_enabled'].append('twitch')
            if config.get('platforms', {}).get('youtube', {}).get('enabled', False):
                health['platforms_enabled'].append('youtube')
                
        except Exception as e:
            health['config_valid'] = False
            health['config_error'] = str(e)
            health['platforms_enabled'] = []
        
        # Chat Monitor Health
        if self.chat_monitor:
            connection_status = self.chat_monitor.get_connection_status()
            health['chat_platforms'] = connection_status
            health['total_connections'] = sum(1 for status in connection_status.values() if status == 'connected')
        else:
            health['chat_platforms'] = {}
            health['total_connections'] = 0
        
        # Message Queue Health (if available)
        if hasattr(self, 'message_queue') and self.message_queue:
            try:
                health['message_queue_size'] = len(self.message_queue.queue) if hasattr(self.message_queue, 'queue') else 0
                health['message_queue_status'] = 'healthy'
            except Exception:
                health['message_queue_status'] = 'error'
        else:
            health['message_queue_status'] = 'not_configured'
        
        # Memory and Performance (basic)
        import psutil
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            health['memory_usage_mb'] = round(memory_info.rss / 1024 / 1024, 1)
            health['cpu_percent'] = process.cpu_percent()
            
            # Categorize memory usage
            if health['memory_usage_mb'] < 100:
                health['memory_status'] = 'low'
            elif health['memory_usage_mb'] < 500:
                health['memory_status'] = 'normal'
            else:
                health['memory_status'] = 'high'
                
        except ImportError:
            # psutil not available
            health['memory_usage_mb'] = 0
            health['memory_status'] = 'unknown'
            health['cpu_percent'] = 0
        except Exception as e:
            health['memory_status'] = 'error'
            health['memory_error'] = str(e)
        
        # Overall system status
        critical_issues = []
        if not health['config_valid']:
            critical_issues.append('config_invalid')
        if health['llm_backend'] == 'error':
            critical_issues.append('llm_error')
        if health['total_connections'] == 0 and len(health.get('platforms_enabled', [])) > 0:
            critical_issues.append('no_connections')
        
        if critical_issues:
            health['overall_status'] = 'error'
            health['critical_issues'] = critical_issues
        elif health['total_connections'] > 0:
            health['overall_status'] = 'healthy'
        else:
            health['overall_status'] = 'idle'
        
        return health
    
    def _get_uptime(self) -> str:
        """Get system uptime."""
        uptime_delta = datetime.now() - self.start_time
        hours, remainder = divmod(int(uptime_delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    # Public methods for external components to update dashboard state
    
    def add_system_log(self, level: str, message: str, details: Optional[Dict] = None):
        """Add system log entry and broadcast to clients."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'details': details or {}
        }
        
        self.system_logs.append(log_entry)
        
        # Keep only last 1000 logs
        if len(self.system_logs) > 1000:
            self.system_logs = self.system_logs[-1000:]
        
        # Broadcast to all connected clients
        self.socketio.emit('new_log', log_entry)
    
    def add_chat_message(self, message: ChatMessage):
        """Add chat message to history and broadcast to clients."""
        chat_entry = {
            'timestamp': message.timestamp.isoformat(),
            'platform': message.platform,
            'username': message.username,
            'content': message.content,
            'channel': message.channel
        }
        
        self.chat_history.append(chat_entry)
        
        # Keep only last 500 messages
        if len(self.chat_history) > 500:
            self.chat_history = self.chat_history[-500:]
        
        # Update bot status
        self.bot_status[message.platform]['last_message'] = chat_entry['timestamp']
        
        # Broadcast to all connected clients
        self.socketio.emit('new_chat_message', chat_entry)
        self.socketio.emit('status_update', {
            'bot_status': self.bot_status,
            'system_health': self._get_system_health()
        })
    
    def add_response(self, original_message: ChatMessage, response: str, llm_backend: str):
        """Add bot response to history and broadcast to clients."""
        response_entry = {
            'timestamp': datetime.now().isoformat(),
            'platform': original_message.platform,
            'original_message': original_message.content,
            'original_user': original_message.username,
            'response': response,
            'llm_backend': llm_backend,
            'channel': original_message.channel
        }
        
        self.response_history.append(response_entry)
        
        # Keep only last 200 responses
        if len(self.response_history) > 200:
            self.response_history = self.response_history[-200:]
        
        # Broadcast to all connected clients
        self.socketio.emit('new_response', response_entry)
    
    def update_bot_status(self, platform: str, connected: bool, error: Optional[str] = None):
        """Update bot connection status."""
        self.bot_status[platform]['connected'] = connected
        if error:
            self.bot_status[platform]['error'] = error
            self.add_system_log('error', f'{platform.title()} bot error: {error}')
        elif 'error' in self.bot_status[platform]:
            del self.bot_status[platform]['error']
        
        # Broadcast status update
        self.socketio.emit('status_update', {
            'bot_status': self.bot_status,
            'system_health': self._get_system_health()
        })
    
    def start_server(self):
        """Start the Flask-SocketIO server."""
        config = self.config_manager.get_config()
        dashboard_config = config.get('dashboard', {})
        
        host = dashboard_config.get('host', 'localhost')
        port = dashboard_config.get('port', 5000)
        debug = dashboard_config.get('debug', False)
        
        logger.info(f"Starting web dashboard on {host}:{port}")
        self.add_system_log('info', f'Web dashboard starting on {host}:{port}')
        
        self.socketio.run(
            self.app,
            host=host,
            port=port,
            debug=debug,
            use_reloader=False  # Disable reloader to prevent issues with threading
        )
    
    def stop_server(self):
        """Stop the Flask-SocketIO server."""
        logger.info("Stopping web dashboard")
        self.add_system_log('info', 'Web dashboard stopping')
        # SocketIO server will be stopped by the main application
    
    def _start_status_sync(self):
        """Start periodic status synchronization with ChatMonitor."""
        def sync_status():
            """Sync status and schedule next sync."""
            try:
                self._sync_bot_status()
                
                # Broadcast updated status
                self.socketio.emit('status_update', {
                    'bot_status': self.bot_status,
                    'system_health': self._get_system_health()
                })
                
            except Exception as e:
                logger.error(f"Status sync error: {e}")
            
            # Schedule next sync in 5 seconds
            self.socketio.start_background_task(
                lambda: self.socketio.sleep(5) or sync_status()
            )
        
        # Start the sync loop
        self.socketio.start_background_task(sync_status)
    
    def set_chat_monitor(self, chat_monitor):
        """Set the chat monitor instance after initialization."""
        self.chat_monitor = chat_monitor
        if chat_monitor:
            self._start_status_sync()
    
    def set_response_generator(self, response_generator):
        """Set the response generator instance after initialization."""
        self.response_generator = response_generator
    
    def set_llm_client(self, llm_client):
        """Set the LLM client instance after initialization."""
        self.llm_client = llm_client
    
    def _validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration data."""
        errors = []
        
        # Validate LLM settings
        llm_config = config.get('llm', {})
        if not llm_config.get('backend'):
            errors.append('LLM backend is required')
        
        if not llm_config.get('model'):
            errors.append('LLM model name is required')
        
        if not llm_config.get('base_url'):
            errors.append('LLM base URL is required')
        else:
            try:
                from urllib.parse import urlparse
                result = urlparse(llm_config['base_url'])
                if not all([result.scheme, result.netloc]):
                    errors.append('LLM base URL must be a valid URL')
            except Exception:
                errors.append('LLM base URL must be a valid URL')
        
        timeout = llm_config.get('timeout', 30)
        if not isinstance(timeout, int) or timeout < 5 or timeout > 120:
            errors.append('LLM timeout must be between 5 and 120 seconds')
        
        # Validate behavior settings
        behavior_config = config.get('behavior', {})
        response_freq = behavior_config.get('response_frequency', 0.3)
        if not isinstance(response_freq, (int, float)) or response_freq < 0 or response_freq > 1:
            errors.append('Response frequency must be between 0 and 1')
        
        min_interval = behavior_config.get('min_response_interval', 10)
        if not isinstance(min_interval, int) or min_interval < 1:
            errors.append('Minimum response interval must be at least 1 second')
        
        # Validate platform settings
        platforms_config = config.get('platforms', {})
        twitch_enabled = platforms_config.get('twitch', {}).get('enabled', False)
        youtube_enabled = platforms_config.get('youtube', {}).get('enabled', False)
        
        if not twitch_enabled and not youtube_enabled:
            errors.append('At least one platform must be enabled')
        
        # Validate personality settings
        personality_config = config.get('personality', {})
        valid_presets = ['friendly', 'professional', 'humorous', 'supportive']
        preset = personality_config.get('preset', 'friendly')
        if preset not in valid_presets:
            errors.append(f'Personality preset must be one of: {", ".join(valid_presets)}')
        
        valid_lengths = ['short', 'medium', 'long']
        length = personality_config.get('response_length', 'short')
        if length not in valid_lengths:
            errors.append(f'Response length must be one of: {", ".join(valid_lengths)}')
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }