"""
Integration tests for the Web Dashboard functionality.

Tests Socket.IO communication, real-time updates, configuration updates,
and bot controls to ensure the dashboard works correctly.
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import socketio

from web_dashboard import WebDashboard
from config_manager import ConfigurationManager
from models import ChatMessage


class TestWebDashboard:
    """Test suite for WebDashboard functionality."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock configuration manager."""
        config_manager = Mock()
        
        # Mock the config attribute directly
        config_manager.config = {
            'platforms': {
                'twitch': {'enabled': True, 'token': 'test_token', 'nick': 'testbot', 'channel': 'testchannel'},
                'youtube': {'enabled': False, 'video_id': 'test_video_id'}
            },
            'llm': {
                'backend': 'ollama',
                'model': 'llama2',
                'base_url': 'http://localhost:11434',
                'timeout': 30
            },
            'personality': {
                'preset': 'friendly',
                'custom_instructions': 'Be helpful',
                'response_length': 'short'
            },
            'behavior': {
                'response_frequency': 0.3,
                'min_response_interval': 10,
                'max_queue_size': 50
            },
            'dashboard': {
                'host': 'localhost',
                'port': 5000,
                'debug': False
            }
        }
        
        # Mock the individual config methods
        config_manager.get_platform_config.return_value = config_manager.config['platforms']
        config_manager.get_llm_config.return_value = config_manager.config['llm']
        config_manager.get_personality_config.return_value = config_manager.config['personality']
        config_manager.get_behavior_config.return_value = config_manager.config['behavior']
        config_manager.get_dashboard_config.return_value = config_manager.config['dashboard']
        
        # Add a get_config method for compatibility
        config_manager.get_config.return_value = config_manager.config
        
        return config_manager
    
    @pytest.fixture
    def mock_chat_monitor(self):
        """Create a mock chat monitor."""
        chat_monitor = Mock()
        chat_monitor.get_connection_status.return_value = {
            'twitch': 'connected',
            'youtube': 'disconnected'
        }
        chat_monitor.is_platform_connected.side_effect = lambda platform: platform == 'twitch'
        chat_monitor.connect_twitch = AsyncMock()
        chat_monitor.connect_youtube = AsyncMock()
        chat_monitor._disconnect_twitch = AsyncMock()
        chat_monitor._disconnect_youtube = AsyncMock()
        return chat_monitor
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        llm_client = Mock()
        llm_client.send_request = AsyncMock(return_value="Test response from LLM")
        return llm_client
    
    @pytest.fixture
    def dashboard(self, mock_config_manager, mock_chat_monitor, mock_llm_client):
        """Create a WebDashboard instance with mocked dependencies."""
        dashboard = WebDashboard(
            config_manager=mock_config_manager,
            chat_monitor=mock_chat_monitor,
            llm_client=mock_llm_client
        )
        return dashboard
    
    def test_dashboard_initialization(self, dashboard, mock_config_manager):
        """Test that dashboard initializes correctly."""
        assert dashboard.config_manager == mock_config_manager
        assert dashboard.chat_monitor is not None
        assert dashboard.llm_client is not None
        assert dashboard.app is not None
        assert dashboard.socketio is not None
        
        # Check initial bot status
        assert 'twitch' in dashboard.bot_status
        assert 'youtube' in dashboard.bot_status
        assert dashboard.bot_status['twitch']['connected'] is False
        assert dashboard.bot_status['youtube']['connected'] is False
    
    def test_flask_routes_exist(self, dashboard):
        """Test that all required Flask routes are registered."""
        with dashboard.app.test_client() as client:
            # Test main dashboard route
            response = client.get('/')
            assert response.status_code == 200
            
            # Test API routes
            response = client.get('/api/config')
            assert response.status_code == 200
            
            response = client.get('/api/status')
            assert response.status_code == 200
    
    def test_get_config_api(self, dashboard, mock_config_manager):
        """Test the GET /api/config endpoint."""
        with dashboard.app.test_client() as client:
            response = client.get('/api/config')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'platforms' in data
            assert 'llm' in data
            assert 'personality' in data
            assert 'behavior' in data
    
    def test_update_config_api(self, dashboard, mock_config_manager):
        """Test the POST /api/config endpoint."""
        test_config = {
            'personality': {'preset': 'humorous'},
            'behavior': {'response_frequency': 0.5}
        }
        
        with dashboard.app.test_client() as client:
            response = client.post('/api/config', 
                                 data=json.dumps(test_config),
                                 content_type='application/json')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert data['success'] is True
            mock_config_manager.update_config.assert_called_once_with(test_config)
    
    def test_get_status_api(self, dashboard):
        """Test the GET /api/status endpoint."""
        with dashboard.app.test_client() as client:
            response = client.get('/api/status')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'bot_status' in data
            assert 'system_health' in data
            assert 'uptime' in data
    
    def test_system_health_monitoring(self, dashboard):
        """Test system health monitoring functionality."""
        health = dashboard._get_system_health()
        
        assert 'last_updated' in health
        assert 'uptime' in health
        assert 'llm_backend' in health
        assert 'config_valid' in health
        assert 'total_connections' in health
        assert 'overall_status' in health
    
    def test_bot_control_start(self, dashboard, mock_chat_monitor):
        """Test bot control start functionality."""
        result = dashboard._handle_bot_control('start', 'twitch')
        
        assert result['success'] is True
        assert 'Starting twitch bot' in result['message']
    
    def test_bot_control_stop(self, dashboard, mock_chat_monitor):
        """Test bot control stop functionality."""
        result = dashboard._handle_bot_control('stop', 'twitch')
        
        assert result['success'] is True
        assert 'Stopping twitch bot' in result['message']
    
    def test_bot_control_restart(self, dashboard, mock_chat_monitor):
        """Test bot control restart functionality."""
        result = dashboard._handle_bot_control('restart', 'twitch')
        
        assert result['success'] is True
        assert 'Restarting twitch bot' in result['message']
    
    def test_bot_control_both_platforms(self, dashboard, mock_chat_monitor):
        """Test bot control for both platforms."""
        result = dashboard._handle_bot_control('start', 'both')
        
        assert result['success'] is True
        assert 'both platforms' in result['message']
    
    def test_bot_control_invalid_action(self, dashboard):
        """Test bot control with invalid action."""
        result = dashboard._handle_bot_control('invalid', 'twitch')
        
        assert result['success'] is False
        assert 'Unknown action' in result['message']
    
    def test_add_system_log(self, dashboard):
        """Test system log functionality."""
        initial_count = len(dashboard.system_logs)
        
        dashboard.add_system_log('info', 'Test log message', {'detail': 'test'})
        
        assert len(dashboard.system_logs) == initial_count + 1
        log_entry = dashboard.system_logs[-1]
        assert log_entry['level'] == 'info'
        assert log_entry['message'] == 'Test log message'
        assert log_entry['details']['detail'] == 'test'
        assert 'timestamp' in log_entry
    
    def test_add_chat_message(self, dashboard):
        """Test chat message handling."""
        message = ChatMessage(
            platform='twitch',
            username='testuser',
            content='Hello world!',
            timestamp=datetime.now(),
            channel='testchannel'
        )
        
        initial_count = len(dashboard.chat_history)
        dashboard.add_chat_message(message)
        
        assert len(dashboard.chat_history) == initial_count + 1
        chat_entry = dashboard.chat_history[-1]
        assert chat_entry['platform'] == 'twitch'
        assert chat_entry['username'] == 'testuser'
        assert chat_entry['content'] == 'Hello world!'
        assert chat_entry['channel'] == 'testchannel'
    
    def test_add_response(self, dashboard):
        """Test bot response handling."""
        original_message = ChatMessage(
            platform='twitch',
            username='testuser',
            content='Hello!',
            timestamp=datetime.now(),
            channel='testchannel'
        )
        
        initial_count = len(dashboard.response_history)
        dashboard.add_response(original_message, 'Hello there!', 'ollama')
        
        assert len(dashboard.response_history) == initial_count + 1
        response_entry = dashboard.response_history[-1]
        assert response_entry['platform'] == 'twitch'
        assert response_entry['original_user'] == 'testuser'
        assert response_entry['response'] == 'Hello there!'
        assert response_entry['llm_backend'] == 'ollama'
    
    def test_update_bot_status(self, dashboard):
        """Test bot status updates."""
        dashboard.update_bot_status('twitch', True)
        assert dashboard.bot_status['twitch']['connected'] is True
        
        dashboard.update_bot_status('twitch', False, 'Connection error')
        assert dashboard.bot_status['twitch']['connected'] is False
        assert dashboard.bot_status['twitch']['error'] == 'Connection error'
        
        # Test error clearing
        dashboard.update_bot_status('twitch', True)
        assert 'error' not in dashboard.bot_status['twitch']
    
    def test_configuration_validation(self, dashboard):
        """Test configuration validation."""
        # Valid configuration
        valid_config = {
            'llm': {
                'backend': 'ollama',
                'model': 'llama2',
                'base_url': 'http://localhost:11434',
                'timeout': 30
            },
            'behavior': {
                'response_frequency': 0.5,
                'min_response_interval': 10
            },
            'platforms': {
                'twitch': {'enabled': True},
                'youtube': {'enabled': False}
            },
            'personality': {
                'preset': 'friendly',
                'response_length': 'short'
            }
        }
        
        result = dashboard._validate_configuration(valid_config)
        assert result['valid'] is True
        assert len(result['errors']) == 0
        
        # Invalid configuration
        invalid_config = {
            'llm': {
                'backend': '',  # Empty backend
                'model': '',    # Empty model
                'base_url': 'invalid-url',  # Invalid URL
                'timeout': 200  # Too high timeout
            },
            'behavior': {
                'response_frequency': 1.5,  # Out of range
                'min_response_interval': 0   # Too low
            },
            'platforms': {
                'twitch': {'enabled': False},
                'youtube': {'enabled': False}  # No platforms enabled
            },
            'personality': {
                'preset': 'invalid',  # Invalid preset
                'response_length': 'invalid'  # Invalid length
            }
        }
        
        result = dashboard._validate_configuration(invalid_config)
        assert result['valid'] is False
        assert len(result['errors']) > 0
    
    @pytest.mark.asyncio
    async def test_async_prompt_testing(self, dashboard, mock_llm_client):
        """Test asynchronous prompt testing functionality."""
        prompt = "Hello, how are you?"
        personality = {'preset': 'friendly'}
        
        # Mock the emit method
        dashboard.socketio.emit = Mock()
        
        await dashboard._async_test_prompt(prompt, personality)
        
        # Verify LLM client was called
        mock_llm_client.send_request.assert_called_once()
        
        # Verify result was emitted
        dashboard.socketio.emit.assert_called_once()
        call_args = dashboard.socketio.emit.call_args
        assert call_args[0][0] == 'prompt_test_result'
        assert call_args[0][1]['success'] is True
        assert call_args[0][1]['response'] == "Test response from LLM"
    
    def test_build_test_system_prompt(self, dashboard):
        """Test system prompt building for testing."""
        personality = {
            'preset': 'humorous',
            'custom_instructions': 'Be extra funny',
            'response_length': 'medium'
        }
        
        prompt = dashboard._build_test_system_prompt(personality)
        
        assert 'humorous' in prompt.lower() or 'fun' in prompt.lower()
        assert 'Be extra funny' in prompt
        assert 'medium' in prompt.lower() or '2-4 sentences' in prompt
        assert 'test message' in prompt.lower()
    
    def test_sync_bot_status(self, dashboard, mock_chat_monitor):
        """Test bot status synchronization."""
        # Set up mock return values
        mock_chat_monitor.get_connection_status.return_value = {
            'twitch': 'connected',
            'youtube': 'disconnected'
        }
        mock_chat_monitor.is_platform_connected.side_effect = lambda p: p == 'twitch'
        
        dashboard._sync_bot_status()
        
        assert dashboard.bot_status['twitch']['connected'] is True
        assert dashboard.bot_status['youtube']['connected'] is False
    
    def test_log_history_management(self, dashboard):
        """Test that log history is properly managed."""
        # Add more logs than the limit
        for i in range(1200):  # More than the 1000 limit
            dashboard.add_system_log('info', f'Test log {i}')
        
        # Should keep only the last 1000
        assert len(dashboard.system_logs) == 1000
        
        # Should have the most recent logs
        assert dashboard.system_logs[-1]['message'] == 'Test log 1199'
    
    def test_chat_history_management(self, dashboard):
        """Test that chat history is properly managed."""
        # Add more messages than the limit
        for i in range(600):  # More than the 500 limit
            message = ChatMessage(
                platform='twitch',
                username=f'user{i}',
                content=f'Message {i}',
                timestamp=datetime.now(),
                channel='testchannel'
            )
            dashboard.add_chat_message(message)
        
        # Should keep only the last 500
        assert len(dashboard.chat_history) == 500
        
        # Should have the most recent messages
        assert dashboard.chat_history[-1]['content'] == 'Message 599'
    
    def test_response_history_management(self, dashboard):
        """Test that response history is properly managed."""
        # Add more responses than the limit
        for i in range(250):  # More than the 200 limit
            message = ChatMessage(
                platform='twitch',
                username=f'user{i}',
                content=f'Question {i}',
                timestamp=datetime.now(),
                channel='testchannel'
            )
            dashboard.add_response(message, f'Response {i}', 'ollama')
        
        # Should keep only the last 200
        assert len(dashboard.response_history) == 200
        
        # Should have the most recent responses
        assert dashboard.response_history[-1]['response'] == 'Response 249'


class TestWebDashboardSocketIO:
    """Test suite for Socket.IO functionality."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock configuration manager."""
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.get_config.return_value = {
            'platforms': {'twitch': {'enabled': True}, 'youtube': {'enabled': False}},
            'llm': {'backend': 'ollama', 'model': 'llama2'},
            'personality': {'preset': 'friendly'},
            'behavior': {'response_frequency': 0.3}
        }
        return config_manager
    
    @pytest.fixture
    def dashboard(self, mock_config_manager):
        """Create a WebDashboard instance for Socket.IO testing."""
        return WebDashboard(config_manager=mock_config_manager)
    
    def test_socketio_client_connection(self, dashboard):
        """Test Socket.IO client connection handling."""
        client = socketio.test_client(dashboard.app, socketio=dashboard.socketio)
        
        # Test connection
        assert client.is_connected()
        
        # Should receive initial status update
        received = client.get_received()
        status_updates = [msg for msg in received if msg['name'] == 'status_update']
        assert len(status_updates) > 0
        
        client.disconnect()
    
    def test_socketio_bot_control_events(self, dashboard):
        """Test Socket.IO bot control events."""
        client = socketio.test_client(dashboard.app, socketio=dashboard.socketio)
        
        # Test bot control event
        client.emit('bot_control', {
            'action': 'start',
            'platform': 'twitch'
        })
        
        # Should receive bot control result
        received = client.get_received()
        control_results = [msg for msg in received if msg['name'] == 'bot_control_result']
        assert len(control_results) > 0
        
        result = control_results[0]['args'][0]
        assert result['success'] is True
        assert result['action'] == 'start'
        assert result['platform'] == 'twitch'
        
        client.disconnect()
    
    def test_socketio_config_update_events(self, dashboard, mock_config_manager):
        """Test Socket.IO configuration update events."""
        client = socketio.test_client(dashboard.app, socketio=dashboard.socketio)
        
        # Test config update event
        test_config = {'personality': {'preset': 'humorous'}}
        client.emit('config_update', test_config)
        
        # Should call config manager
        mock_config_manager.update_config.assert_called_with(test_config)
        
        # Should receive config updated broadcast
        received = client.get_received()
        config_updates = [msg for msg in received if msg['name'] == 'config_updated']
        assert len(config_updates) > 0
        
        client.disconnect()
    
    def test_socketio_prompt_test_events(self, dashboard):
        """Test Socket.IO prompt testing events."""
        client = socketio.test_client(dashboard.app, socketio=dashboard.socketio)
        
        # Test prompt test event
        client.emit('test_prompt', {
            'prompt': 'Hello world',
            'username': 'testuser',
            'platform': 'twitch',
            'testMode': True,
            'personality': {'preset': 'friendly'}
        })
        
        # Should receive prompt test result
        received = client.get_received()
        test_results = [msg for msg in received if msg['name'] == 'prompt_test_result']
        assert len(test_results) > 0
        
        result = test_results[0]['args'][0]
        assert 'success' in result
        assert result['prompt'] == 'Hello world'
        assert result['username'] == 'testuser'
        assert result['platform'] == 'twitch'
        
        client.disconnect()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])