"""
Unit tests for the ResponseGenerator class.

Tests cover message processing, context building, personality application,
response formatting, and delivery system functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from response_generator import ResponseGenerator, ResponseStats
from models import ChatMessage, QueuedMessage, ResponseContext
from config_manager import ConfigurationManager
from message_queue import MessageQueue
from llm_client import LLMResponse


class TestResponseGenerator:
    """Test cases for ResponseGenerator class."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock configuration manager."""
        config_manager = Mock(spec=ConfigurationManager)
        
        # Default configuration
        config_manager.get_llm_config.return_value = {
            'backend': 'ollama',
            'model': 'llama2',
            'base_url': 'http://localhost:11434',
            'timeout': 30.0
        }
        
        config_manager.get_personality_config.return_value = {
            'preset': 'friendly',
            'response_types': ['conversational'],
            'response_length': 'short',
            'custom_instructions': '',
            'temperature': 0.7
        }
        
        config_manager.get_behavior_config.return_value = {
            'response_frequency': 0.5,
            'min_response_interval': 30,
            'max_queue_size': 100
        }
        
        config_manager.get_enabled_platforms.return_value = ['twitch', 'youtube']
        
        return config_manager
    
    @pytest.fixture
    def mock_message_queue(self):
        """Create a mock message queue."""
        return Mock(spec=MessageQueue)
    
    @pytest.fixture
    def response_generator(self, mock_config_manager, mock_message_queue):
        """Create a ResponseGenerator instance with mocked dependencies."""
        with patch('response_generator.LLMClientFactory.create_client'):
            generator = ResponseGenerator(mock_config_manager, mock_message_queue)
            generator.llm_client = AsyncMock()
            return generator
    
    @pytest.fixture
    def sample_chat_message(self):
        """Create a sample chat message for testing."""
        return ChatMessage(
            platform='twitch',
            username='test_user',
            content='Hello, how are you?',
            timestamp=datetime.now(),
            channel='test_channel'
        )
    
    @pytest.fixture
    def sample_queued_message(self, sample_chat_message):
        """Create a sample queued message for testing."""
        return QueuedMessage.from_chat_message(sample_chat_message)


class TestResponseGeneratorInitialization(TestResponseGenerator):
    """Test ResponseGenerator initialization."""
    
    def test_initialization(self, mock_config_manager, mock_message_queue):
        """Test proper initialization of ResponseGenerator."""
        with patch('response_generator.LLMClientFactory.create_client') as mock_factory:
            mock_client = AsyncMock()
            mock_factory.return_value = mock_client
            
            generator = ResponseGenerator(mock_config_manager, mock_message_queue)
            
            assert generator.config_manager == mock_config_manager
            assert generator.message_queue == mock_message_queue
            assert generator.is_running is False
            assert isinstance(generator.stats, ResponseStats)
            assert generator.response_callbacks == {}
    
    def test_llm_client_initialization_failure(self, mock_config_manager, mock_message_queue):
        """Test handling of LLM client initialization failure."""
        with patch('response_generator.LLMClientFactory.create_client', side_effect=Exception("Connection failed")):
            generator = ResponseGenerator(mock_config_manager, mock_message_queue)
            assert generator.llm_client is None


class TestMessageProcessing(TestResponseGenerator):
    """Test message processing functionality."""
    
    @pytest.mark.asyncio
    async def test_process_message_success(self, response_generator, sample_queued_message):
        """Test successful message processing."""
        # Mock LLM response
        mock_llm_response = LLMResponse(
            content="Hello! I'm doing great, thanks for asking!",
            model="llama2",
            backend="ollama",
            success=True
        )
        response_generator.llm_client.send_request.return_value = mock_llm_response
        
        # Mock response callback
        mock_callback = AsyncMock()
        response_generator.register_response_callback('twitch', mock_callback)
        
        # Mock random to ensure response passes frequency check
        with patch('random.random', return_value=0.1):  # Pass frequency check
            # Process message
            await response_generator.process_message(sample_queued_message)
        
        # Verify stats updated
        assert response_generator.stats.messages_processed == 1
        assert response_generator.stats.responses_generated == 1
        assert response_generator.stats.responses_sent == 1
    
    @pytest.mark.asyncio
    async def test_process_message_llm_error(self, response_generator, sample_queued_message):
        """Test message processing with LLM error."""
        # Mock LLM error response
        mock_llm_response = LLMResponse(
            content="",
            model="llama2",
            backend="ollama",
            success=False,
            error="Connection timeout"
        )
        response_generator.llm_client.send_request.return_value = mock_llm_response
        
        # Mock random to ensure response passes frequency check
        with patch('random.random', return_value=0.1):  # Pass frequency check
            # Process message
            await response_generator.process_message(sample_queued_message)
        
        # Verify stats updated
        assert response_generator.stats.messages_processed == 1
        assert response_generator.stats.llm_errors == 1
        assert response_generator.stats.responses_sent == 0
    
    @pytest.mark.asyncio
    async def test_should_respond_frequency_check(self, response_generator, sample_chat_message):
        """Test response frequency checking."""
        # Set low frequency to test rejection
        response_generator.config_manager.get_behavior_config.return_value = {
            'response_frequency': 0.0,  # Never respond
            'min_response_interval': 30
        }
        
        with patch('random.random', return_value=0.5):
            should_respond = await response_generator._should_respond(sample_chat_message)
            assert should_respond is False
    
    @pytest.mark.asyncio
    async def test_should_respond_rate_limiting(self, response_generator, sample_chat_message):
        """Test rate limiting functionality."""
        # Set recent response time
        channel_key = f"{sample_chat_message.platform}:{sample_chat_message.channel}"
        response_generator.last_response_time[channel_key] = datetime.now()
        
        with patch('random.random', return_value=0.1):  # Pass frequency check
            should_respond = await response_generator._should_respond(sample_chat_message)
            assert should_respond is False
            assert response_generator.stats.rate_limited == 1


class TestContextBuilding(TestResponseGenerator):
    """Test response context building."""
    
    def test_build_response_context(self, response_generator, sample_chat_message):
        """Test building response context."""
        # Add some recent messages
        channel_key = f"{sample_chat_message.platform}:{sample_chat_message.channel}"
        response_generator.recent_messages[channel_key] = [
            ChatMessage('twitch', 'user1', 'Previous message', datetime.now(), 'test_channel'),
            sample_chat_message
        ]
        
        context = response_generator._build_response_context(sample_chat_message)
        
        assert isinstance(context, ResponseContext)
        assert context.message == sample_chat_message
        assert len(context.recent_messages) == 2
        assert 'message_count' in context.user_history
    
    def test_build_prompt(self, response_generator, sample_chat_message):
        """Test prompt building with personality integration."""
        context = ResponseContext(
            message=sample_chat_message,
            personality=response_generator.config_manager.get_personality_config()
        )
        
        prompt = response_generator._build_prompt(context)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert 'friendly' in prompt.lower()
        assert sample_chat_message.content in prompt
        assert sample_chat_message.username in prompt


class TestPersonalityApplication(TestResponseGenerator):
    """Test personality configuration application."""
    
    def test_get_personality_instructions_friendly(self, response_generator):
        """Test personality instructions for friendly preset."""
        personality_config = {
            'preset': 'friendly',
            'response_types': ['conversational'],
            'response_length': 'short'
        }
        
        instructions = response_generator._get_personality_instructions(personality_config)
        
        assert 'warm' in instructions.lower() or 'welcoming' in instructions.lower()
        assert 'concise' in instructions.lower()
    
    def test_get_personality_instructions_professional(self, response_generator):
        """Test personality instructions for professional preset."""
        personality_config = {
            'preset': 'professional',
            'response_types': ['informational'],
            'response_length': 'medium'
        }
        
        instructions = response_generator._get_personality_instructions(personality_config)
        
        assert 'professional' in instructions.lower()
        assert 'information' in instructions.lower()
    
    def test_apply_personality_formatting_friendly(self, response_generator, sample_chat_message):
        """Test personality formatting for friendly preset."""
        response_generator.config_manager.get_personality_config.return_value = {
            'preset': 'friendly',
            'response_types': ['conversational'],
            'response_length': 'short'
        }
        
        with patch('random.random', return_value=0.1):  # Trigger friendly prefix
            formatted = response_generator._apply_personality_formatting(
                "that sounds great", sample_chat_message
            )
            
            # Should add friendly prefix
            assert formatted.startswith(('Hey!', 'Oh,', 'Nice!'))
    
    def test_apply_personality_formatting_professional(self, response_generator, sample_chat_message):
        """Test personality formatting for professional preset."""
        response_generator.config_manager.get_personality_config.return_value = {
            'preset': 'professional',
            'response_types': ['informational'],
            'response_length': 'medium'
        }
        
        formatted = response_generator._apply_personality_formatting(
            "that is correct", sample_chat_message
        )
        
        # Should capitalize and add period
        assert formatted.startswith('T')
        assert formatted.endswith('.')
    
    def test_personality_presets(self, response_generator):
        """Test available personality presets."""
        presets = response_generator.get_personality_presets()
        
        assert 'friendly' in presets
        assert 'professional' in presets
        assert 'humorous' in presets
        assert 'supportive' in presets
        
        for preset_name, preset_config in presets.items():
            assert 'preset' in preset_config
            assert 'response_types' in preset_config
            assert 'response_length' in preset_config


class TestResponseFormatting(TestResponseGenerator):
    """Test response formatting and sanitization."""
    
    def test_format_response_basic(self, response_generator, sample_chat_message):
        """Test basic response formatting."""
        raw_response = "Response: Hello there! How can I help you today?"
        
        # Mock random to prevent friendly prefix addition
        with patch('random.random', return_value=0.5):  # Don't trigger friendly prefix
            formatted = response_generator._format_response(raw_response, sample_chat_message)
        
        assert formatted == "Hello there! How can I help you today?"
        assert not formatted.startswith("Response:")
    
    def test_format_response_length_limit(self, response_generator, sample_chat_message):
        """Test response length limiting."""
        # Create a very long response
        long_response = "This is a very long response. " * 20
        
        formatted = response_generator._format_response(long_response, sample_chat_message)
        
        assert len(formatted) <= 200  # Default short length limit
        assert formatted.endswith("...")
    
    def test_format_for_twitch(self, response_generator, sample_chat_message):
        """Test Twitch-specific formatting."""
        response = "Hello there!"
        
        with patch('random.random', return_value=0.1):  # Trigger @mention
            formatted = response_generator._format_for_twitch(response, sample_chat_message)
            
            assert formatted.startswith(f"@{sample_chat_message.username}")
            assert len(formatted) <= 500
    
    def test_format_for_youtube(self, response_generator, sample_chat_message):
        """Test YouTube-specific formatting."""
        response = "@someone Hello there!"
        
        formatted = response_generator._format_for_youtube(response, sample_chat_message)
        
        # Should remove @mention for YouTube
        assert not formatted.startswith("@")
        assert len(formatted) <= 200
    
    def test_validate_twitch_response(self, response_generator):
        """Test Twitch response validation."""
        valid_response = "Hello there!"
        invalid_response = "Check out this link: http://example.com"
        
        assert response_generator._validate_twitch_response(valid_response) is True
        assert response_generator._validate_twitch_response(invalid_response) is False
    
    def test_validate_youtube_response(self, response_generator):
        """Test YouTube response validation."""
        valid_response = "Hello there!"
        invalid_response = "@someone hello"
        
        assert response_generator._validate_youtube_response(valid_response) is True
        assert response_generator._validate_youtube_response(invalid_response) is False
    
    def test_sanitize_response(self, response_generator):
        """Test response sanitization."""
        dirty_response = "HELLO!!! Check this out??? FREE MONEY!!!"
        
        sanitized = response_generator.sanitize_response(dirty_response)
        
        assert "!!!" not in sanitized
        assert "???" not in sanitized
        assert "FREE MONEY" not in sanitized


class TestResponseDelivery(TestResponseGenerator):
    """Test response delivery system."""
    
    @pytest.mark.asyncio
    async def test_register_response_callback(self, response_generator):
        """Test registering response callbacks."""
        mock_callback = AsyncMock()
        
        response_generator.register_response_callback('twitch', mock_callback)
        
        assert 'twitch' in response_generator.response_callbacks
        assert response_generator.response_callbacks['twitch'] == mock_callback
    
    @pytest.mark.asyncio
    async def test_deliver_response_success(self, response_generator, sample_chat_message):
        """Test successful response delivery."""
        mock_callback = AsyncMock()
        response_generator.register_response_callback('twitch', mock_callback)
        
        await response_generator._deliver_response(sample_chat_message, "Hello!")
        
        mock_callback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_deliver_response_no_callback(self, response_generator, sample_chat_message):
        """Test response delivery with no registered callback."""
        # Should not raise exception
        await response_generator._deliver_response(sample_chat_message, "Hello!")
    
    @pytest.mark.asyncio
    async def test_test_response_delivery(self, response_generator):
        """Test response delivery testing functionality."""
        mock_callback = AsyncMock()
        response_generator.register_response_callback('twitch', mock_callback)
        
        result = await response_generator.test_response_delivery('twitch')
        
        assert result['success'] is True
        assert result['platform'] == 'twitch'
        assert 'formatted_response' in result
    
    def test_get_response_routing_info(self, response_generator):
        """Test getting response routing information."""
        response_generator.register_response_callback('twitch', AsyncMock())
        
        info = response_generator.get_response_routing_info()
        
        assert 'registered_platforms' in info
        assert 'enabled_platforms' in info
        assert 'delivery_stats' in info
        assert 'twitch' in info['registered_platforms']


class TestStatistics(TestResponseGenerator):
    """Test statistics tracking."""
    
    def test_stats_initialization(self, response_generator):
        """Test statistics initialization."""
        stats = response_generator.stats
        
        assert stats.messages_processed == 0
        assert stats.responses_generated == 0
        assert stats.responses_sent == 0
        assert stats.llm_errors == 0
        assert stats.average_response_time == 0.0
    
    def test_get_stats(self, response_generator):
        """Test getting statistics."""
        # Simulate some activity
        response_generator.stats.messages_processed = 10
        response_generator.stats.responses_sent = 5
        
        stats_dict = response_generator.get_stats()
        
        assert stats_dict['messages_processed'] == 10
        assert stats_dict['responses_sent'] == 5
        assert isinstance(stats_dict, dict)
    
    def test_reset_stats(self, response_generator):
        """Test resetting statistics."""
        # Set some stats
        response_generator.stats.messages_processed = 10
        response_generator.stats.responses_sent = 5
        
        response_generator.reset_stats()
        
        assert response_generator.stats.messages_processed == 0
        assert response_generator.stats.responses_sent == 0


class TestLLMIntegration(TestResponseGenerator):
    """Test LLM integration functionality."""
    
    @pytest.mark.asyncio
    async def test_test_llm_connection_success(self, response_generator):
        """Test successful LLM connection test."""
        response_generator.llm_client.health_check.return_value = {
            'status': 'healthy',
            'backend': 'ollama',
            'base_url': 'http://localhost:11434',
            'available_models': ['llama2']
        }
        
        result = await response_generator.test_llm_connection()
        
        assert result['connected'] is True
        assert result['backend'] == 'ollama'
        assert 'available_models' in result
    
    @pytest.mark.asyncio
    async def test_test_llm_connection_failure(self, response_generator):
        """Test LLM connection test failure."""
        response_generator.llm_client = None
        
        result = await response_generator.test_llm_connection()
        
        assert result['connected'] is False
        assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_generate_test_response(self, response_generator):
        """Test generating test responses."""
        mock_llm_response = LLMResponse(
            content="This is a test response",
            model="llama2",
            backend="ollama",
            success=True
        )
        response_generator.llm_client.send_request.return_value = mock_llm_response
        
        result = await response_generator.generate_test_response("Hello test")
        
        assert result['success'] is True
        assert 'response' in result
        assert 'personality' in result
        assert result['model'] == 'llama2'


if __name__ == '__main__':
    pytest.main([__file__])