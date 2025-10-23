#!/usr/bin/env python3
"""
Integration tests for chat platform connections and message routing.

Tests Twitch and YouTube connection establishment, message routing,
and queue functionality. Covers requirements 1.1, 1.2, 1.6, 1.7.
"""

import asyncio
import pytest
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import List, Dict, Any

from chat_monitor import ChatMonitor, ConnectionState, TwitchBot, YouTubeMonitor
from message_queue import MessageQueue, MessagePriority
from config_manager import ConfigurationManager
from models import ChatMessage, QueuedMessage


class TestChatMonitorInitialization:
    """Test ChatMonitor initialization and configuration."""
    
    def test_chat_monitor_initialization(self):
        """Test ChatMonitor initialization with different configurations."""
        # Mock configuration manager
        config_manager = MagicMock(spec=ConfigurationManager)
        
        # Test basic initialization
        monitor = ChatMonitor(config_manager)
        assert monitor.config_manager == config_manager
        assert monitor.message_callback is None
        assert monitor.message_queue is None
        assert monitor.is_running == False
        assert monitor.connection_states['twitch'] == ConnectionState.DISCONNECTED
        assert monitor.connection_states['youtube'] == ConnectionState.DISCONNECTED
        
        # Test initialization with message queue
        message_queue = MagicMock(spec=MessageQueue)
        monitor = ChatMonitor(config_manager, message_queue=message_queue)
        assert monitor.message_queue == message_queue
        
        # Test initialization with callback
        callback = AsyncMock()
        monitor = ChatMonitor(config_manager, message_callback=callback)
        assert monitor.message_callback == callback
    
    def test_connection_state_tracking(self):
        """Test connection state tracking functionality."""
        config_manager = MagicMock(spec=ConfigurationManager)
        monitor = ChatMonitor(config_manager)
        
        # Test initial states
        status = monitor.get_connection_status()
        assert status['twitch'] == 'disconnected'
        assert status['youtube'] == 'disconnected'
        
        # Test state changes
        monitor.connection_states['twitch'] = ConnectionState.CONNECTED
        assert monitor.is_platform_connected('twitch') == True
        assert monitor.is_platform_connected('youtube') == False
        
        status = monitor.get_connection_status()
        assert status['twitch'] == 'connected'
        assert status['youtube'] == 'disconnected'


class TestTwitchIntegration:
    """Test Twitch chat integration functionality."""
    
    @pytest.mark.asyncio
    async def test_twitch_connection_success(self):
        """Test successful Twitch connection establishment."""
        # Mock configuration
        config_manager = MagicMock(spec=ConfigurationManager)
        config_manager.get_platform_config.return_value = {
            'twitch': {
                'enabled': True,
                'token': 'oauth:test_token',
                'nick': 'test_bot',
                'channel': 'test_channel'
            },
            'youtube': {'enabled': False}
        }
        
        monitor = ChatMonitor(config_manager)
        
        # Mock TwitchBot
        mock_bot = AsyncMock(spec=TwitchBot)
        mock_bot.start = AsyncMock()
        
        with patch('chat_monitor.TwitchBot', return_value=mock_bot):
            await monitor.connect_twitch()
            
            # Verify bot was created and started
            mock_bot.start.assert_called_once()
            assert monitor.twitch_bot == mock_bot
            assert monitor.connection_states['twitch'] == ConnectionState.CONNECTED
    
    @pytest.mark.asyncio
    async def test_twitch_connection_disabled(self):
        """Test Twitch connection when disabled in config."""
        config_manager = MagicMock(spec=ConfigurationManager)
        config_manager.get_platform_config.return_value = {
            'twitch': {'enabled': False},
            'youtube': {'enabled': False}
        }
        
        monitor = ChatMonitor(config_manager)
        
        await monitor.connect_twitch()
        
        # Should remain disconnected
        assert monitor.connection_states['twitch'] == ConnectionState.DISCONNECTED
        assert monitor.twitch_bot is None
    
    @pytest.mark.asyncio
    async def test_twitch_connection_failure(self):
        """Test Twitch connection failure and error handling."""
        config_manager = MagicMock(spec=ConfigurationManager)
        config_manager.get_platform_config.return_value = {
            'twitch': {
                'enabled': True,
                'token': 'oauth:test_token',
                'nick': 'test_bot',
                'channel': 'test_channel'
            }
        }
        
        monitor = ChatMonitor(config_manager)
        monitor.is_running = True  # Enable reconnection logic
        
        # Mock TwitchBot that fails to start
        mock_bot = AsyncMock(spec=TwitchBot)
        mock_bot.start.side_effect = Exception("Connection failed")
        
        with patch('chat_monitor.TwitchBot', return_value=mock_bot):
            with patch.object(monitor, '_schedule_reconnect', new_callable=AsyncMock) as mock_reconnect:
                await monitor.connect_twitch()
                
                # Should be in error state and schedule reconnection
                assert monitor.connection_states['twitch'] == ConnectionState.ERROR
                mock_reconnect.assert_called_once_with('twitch')
    
    @pytest.mark.asyncio
    async def test_twitch_message_handling(self):
        """Test Twitch message processing and routing."""
        config_manager = MagicMock(spec=ConfigurationManager)
        message_queue = AsyncMock(spec=MessageQueue)
        message_queue.enqueue_message = AsyncMock(return_value=True)
        
        monitor = ChatMonitor(config_manager, message_queue=message_queue)
        
        # Test message data
        message_data = {
            'username': 'test_user',
            'content': 'Hello, world!',
            'channel': 'test_channel',
            'user_id': '12345',
            'message_id': 'msg_123'
        }
        
        await monitor._handle_twitch_message(message_data)
        
        # Verify message was queued
        message_queue.enqueue_message.assert_called_once()
        call_args = message_queue.enqueue_message.call_args
        
        chat_message = call_args[0][0]
        priority = call_args[0][1]
        
        assert isinstance(chat_message, ChatMessage)
        assert chat_message.platform == 'twitch'
        assert chat_message.username == 'test_user'
        assert chat_message.content == 'Hello, world!'
        assert chat_message.channel == 'test_channel'
        assert chat_message.user_id == '12345'
        assert isinstance(priority, MessagePriority)


class TestYouTubeIntegration:
    """Test YouTube chat integration functionality."""
    
    @pytest.mark.asyncio
    async def test_youtube_connection_success(self):
        """Test successful YouTube connection establishment."""
        config_manager = MagicMock(spec=ConfigurationManager)
        config_manager.get_platform_config.return_value = {
            'twitch': {'enabled': False},
            'youtube': {
                'enabled': True,
                'video_id': 'test_video_id'
            }
        }
        
        monitor = ChatMonitor(config_manager)
        
        # Mock YouTubeMonitor
        mock_monitor = AsyncMock(spec=YouTubeMonitor)
        mock_monitor.start = AsyncMock()
        
        with patch('chat_monitor.YouTubeMonitor', return_value=mock_monitor):
            await monitor.connect_youtube()
            
            # Verify monitor was created and started
            mock_monitor.start.assert_called_once()
            assert monitor.youtube_monitor == mock_monitor
            assert monitor.connection_states['youtube'] == ConnectionState.CONNECTED
    
    @pytest.mark.asyncio
    async def test_youtube_connection_disabled(self):
        """Test YouTube connection when disabled in config."""
        config_manager = MagicMock(spec=ConfigurationManager)
        config_manager.get_platform_config.return_value = {
            'twitch': {'enabled': False},
            'youtube': {'enabled': False}
        }
        
        monitor = ChatMonitor(config_manager)
        
        await monitor.connect_youtube()
        
        # Should remain disconnected
        assert monitor.connection_states['youtube'] == ConnectionState.DISCONNECTED
        assert monitor.youtube_monitor is None
    
    @pytest.mark.asyncio
    async def test_youtube_connection_missing_video_id(self):
        """Test YouTube connection failure when video_id is missing."""
        config_manager = MagicMock(spec=ConfigurationManager)
        config_manager.get_platform_config.return_value = {
            'youtube': {
                'enabled': True,
                'video_id': ''  # Empty video ID
            }
        }
        
        monitor = ChatMonitor(config_manager)
        monitor.is_running = True
        
        with patch.object(monitor, '_schedule_reconnect', new_callable=AsyncMock) as mock_reconnect:
            await monitor.connect_youtube()
            
            # Should be in error state
            assert monitor.connection_states['youtube'] == ConnectionState.ERROR
            mock_reconnect.assert_called_once_with('youtube')
    
    @pytest.mark.asyncio
    async def test_youtube_message_handling(self):
        """Test YouTube message processing and routing."""
        config_manager = MagicMock(spec=ConfigurationManager)
        message_queue = AsyncMock(spec=MessageQueue)
        message_queue.enqueue_message = AsyncMock(return_value=True)
        
        monitor = ChatMonitor(config_manager, message_queue=message_queue)
        
        # Test message data
        message_data = {
            'username': 'youtube_user',
            'content': 'Great stream!',
            'channel': 'test_video_id',
            'user_id': 'UC123456789',
            'message_id': 'yt_msg_123'
        }
        
        await monitor._handle_youtube_message(message_data)
        
        # Verify message was queued
        message_queue.enqueue_message.assert_called_once()
        call_args = message_queue.enqueue_message.call_args
        
        chat_message = call_args[0][0]
        priority = call_args[0][1]
        
        assert isinstance(chat_message, ChatMessage)
        assert chat_message.platform == 'youtube'
        assert chat_message.username == 'youtube_user'
        assert chat_message.content == 'Great stream!'
        assert chat_message.channel == 'test_video_id'
        assert chat_message.user_id == 'UC123456789'
        assert isinstance(priority, MessagePriority)


class TestMessageRouting:
    """Test message routing and priority determination."""
    
    def test_message_priority_determination(self):
        """Test message priority assignment based on content."""
        config_manager = MagicMock(spec=ConfigurationManager)
        monitor = ChatMonitor(config_manager)
        
        # Test high priority messages (questions)
        high_priority_messages = [
            ChatMessage('twitch', 'user1', 'How do I do this?', datetime.now(), 'channel'),
            ChatMessage('twitch', 'user2', 'Can you help me?', datetime.now(), 'channel'),
            ChatMessage('twitch', 'user3', '@streamer question about game', datetime.now(), 'channel'),
        ]
        
        for message in high_priority_messages:
            priority = monitor._determine_message_priority(message)
            assert priority == MessagePriority.HIGH
        
        # Test low priority messages (common expressions)
        low_priority_messages = [
            ChatMessage('twitch', 'user1', 'lol', datetime.now(), 'channel'),
            ChatMessage('twitch', 'user2', 'LMAO that was funny!', datetime.now(), 'channel'),
            ChatMessage('twitch', 'user3', 'Kappa', datetime.now(), 'channel'),
            ChatMessage('twitch', 'user4', 'Poggers!!!', datetime.now(), 'channel'),
        ]
        
        for message in low_priority_messages:
            priority = monitor._determine_message_priority(message)
            assert priority == MessagePriority.LOW
        
        # Test normal priority messages
        normal_priority_messages = [
            ChatMessage('twitch', 'user1', 'This is a normal message', datetime.now(), 'channel'),
            ChatMessage('twitch', 'user2', 'Great gameplay today', datetime.now(), 'channel'),
        ]
        
        for message in normal_priority_messages:
            priority = monitor._determine_message_priority(message)
            assert priority == MessagePriority.NORMAL
    
    @pytest.mark.asyncio
    async def test_message_routing_with_queue(self):
        """Test message routing to message queue."""
        config_manager = MagicMock(spec=ConfigurationManager)
        message_queue = AsyncMock(spec=MessageQueue)
        message_queue.enqueue_message = AsyncMock(return_value=True)
        
        monitor = ChatMonitor(config_manager, message_queue=message_queue)
        
        message = ChatMessage(
            platform='twitch',
            username='test_user',
            content='Test message',
            timestamp=datetime.now(),
            channel='test_channel'
        )
        
        await monitor._route_message(message)
        
        # Verify message was queued with correct priority
        message_queue.enqueue_message.assert_called_once()
        call_args = message_queue.enqueue_message.call_args
        
        queued_message = call_args[0][0]
        priority = call_args[0][1]
        
        assert queued_message == message
        assert priority == MessagePriority.NORMAL
    
    @pytest.mark.asyncio
    async def test_message_routing_with_callback(self):
        """Test message routing to callback function."""
        config_manager = MagicMock(spec=ConfigurationManager)
        callback = AsyncMock()
        
        monitor = ChatMonitor(config_manager, message_callback=callback)
        
        message = ChatMessage(
            platform='twitch',
            username='test_user',
            content='Test message',
            timestamp=datetime.now(),
            channel='test_channel'
        )
        
        await monitor._route_message(message)
        
        # Verify callback was called
        callback.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_message_routing_queue_full(self):
        """Test message routing when queue is full."""
        config_manager = MagicMock(spec=ConfigurationManager)
        message_queue = AsyncMock(spec=MessageQueue)
        message_queue.enqueue_message = AsyncMock(return_value=False)  # Queue full
        
        monitor = ChatMonitor(config_manager, message_queue=message_queue)
        
        message = ChatMessage(
            platform='twitch',
            username='test_user',
            content='Test message',
            timestamp=datetime.now(),
            channel='test_channel'
        )
        
        # Should not raise exception even when queue is full
        await monitor._route_message(message)
        
        message_queue.enqueue_message.assert_called_once()


class TestMultiPlatformMonitoring:
    """Test multi-platform monitoring functionality."""
    
    @pytest.mark.asyncio
    async def test_start_monitoring_both_platforms(self):
        """Test starting monitoring for both Twitch and YouTube."""
        config_manager = MagicMock(spec=ConfigurationManager)
        config_manager.get_platform_config.return_value = {
            'twitch': {
                'enabled': True,
                'token': 'oauth:test_token',
                'nick': 'test_bot',
                'channel': 'test_channel'
            },
            'youtube': {
                'enabled': True,
                'video_id': 'test_video_id'
            }
        }
        
        monitor = ChatMonitor(config_manager)
        
        with patch.object(monitor, 'connect_twitch', new_callable=AsyncMock) as mock_twitch:
            with patch.object(monitor, 'connect_youtube', new_callable=AsyncMock) as mock_youtube:
                await monitor.start_monitoring()
                
                # Both platforms should be started
                mock_twitch.assert_called_once()
                mock_youtube.assert_called_once()
                assert monitor.is_running == True
    
    @pytest.mark.asyncio
    async def test_start_monitoring_no_platforms_enabled(self):
        """Test starting monitoring when no platforms are enabled."""
        config_manager = MagicMock(spec=ConfigurationManager)
        config_manager.get_platform_config.return_value = {
            'twitch': {'enabled': False},
            'youtube': {'enabled': False}
        }
        
        monitor = ChatMonitor(config_manager)
        
        with patch.object(monitor, 'connect_twitch', new_callable=AsyncMock) as mock_twitch:
            with patch.object(monitor, 'connect_youtube', new_callable=AsyncMock) as mock_youtube:
                await monitor.start_monitoring()
                
                # No platforms should be started
                mock_twitch.assert_not_called()
                mock_youtube.assert_not_called()
                assert monitor.is_running == True  # Still running, just no connections
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self):
        """Test stopping monitoring and cleanup."""
        config_manager = MagicMock(spec=ConfigurationManager)
        monitor = ChatMonitor(config_manager)
        monitor.is_running = True
        
        # Mock platform connections
        monitor.twitch_bot = AsyncMock()
        monitor.youtube_monitor = AsyncMock()
        
        # Mock reconnect tasks
        mock_task = AsyncMock()
        mock_task.done.return_value = False
        monitor.reconnect_tasks['twitch'] = mock_task
        
        with patch.object(monitor, '_disconnect_twitch', new_callable=AsyncMock) as mock_disconnect_twitch:
            with patch.object(monitor, '_disconnect_youtube', new_callable=AsyncMock) as mock_disconnect_youtube:
                await monitor.stop_monitoring()
                
                # Should cancel tasks and disconnect platforms
                mock_task.cancel.assert_called_once()
                mock_disconnect_twitch.assert_called_once()
                mock_disconnect_youtube.assert_called_once()
                assert monitor.is_running == False


class TestReconnectionLogic:
    """Test automatic reconnection functionality."""
    
    @pytest.mark.asyncio
    async def test_reconnection_scheduling(self):
        """Test reconnection scheduling after connection failure."""
        config_manager = MagicMock(spec=ConfigurationManager)
        monitor = ChatMonitor(config_manager)
        monitor.is_running = True
        
        with patch.object(monitor, '_reconnect_with_backoff', new_callable=AsyncMock) as mock_reconnect:
            # Mock asyncio.create_task
            mock_task = AsyncMock()
            with patch('asyncio.create_task', return_value=mock_task) as mock_create_task:
                await monitor._schedule_reconnect('twitch')
                
                # Should create reconnection task
                mock_create_task.assert_called_once()
                assert monitor.reconnect_tasks['twitch'] == mock_task
    
    @pytest.mark.asyncio
    async def test_reconnection_backoff(self):
        """Test exponential backoff reconnection logic."""
        config_manager = MagicMock(spec=ConfigurationManager)
        monitor = ChatMonitor(config_manager)
        monitor.is_running = True
        monitor.max_reconnect_attempts = 2  # Limit for testing
        
        # Mock failed connections
        with patch.object(monitor, 'connect_twitch', new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")
            
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                await monitor._reconnect_with_backoff('twitch')
                
                # Should attempt reconnection with increasing delays
                assert mock_connect.call_count == 2
                assert mock_sleep.call_count == 2
                
                # Should be in error state after max attempts
                assert monitor.connection_states['twitch'] == ConnectionState.ERROR


class TestMessageQueueIntegration:
    """Test integration with message queue system."""
    
    @pytest.mark.asyncio
    async def test_message_queue_integration(self):
        """Test full integration with message queue."""
        # Create real message queue for integration test
        message_queue = MessageQueue(max_size=10)
        processed_messages = []
        
        async def process_callback(message):
            processed_messages.append(message)
        
        message_queue.set_processing_callback(process_callback)
        await message_queue.start_processing()
        
        # Create monitor with queue
        config_manager = MagicMock(spec=ConfigurationManager)
        monitor = ChatMonitor(config_manager, message_queue=message_queue)
        
        # Send test messages
        test_messages = [
            ChatMessage('twitch', 'user1', 'Hello!', datetime.now(), 'channel1'),
            ChatMessage('youtube', 'user2', 'Great stream!', datetime.now(), 'channel2'),
            ChatMessage('twitch', 'user3', 'How do I do this?', datetime.now(), 'channel1'),  # High priority
        ]
        
        for message in test_messages:
            await monitor._route_message(message)
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Stop queue processing
        await message_queue.stop_processing()
        
        # Verify messages were processed
        assert len(processed_messages) == 3
        
        # Verify high priority message was processed first
        assert processed_messages[0].content == 'How do I do this?'
    
    def test_queue_statistics_tracking(self):
        """Test that queue statistics are properly tracked."""
        message_queue = MessageQueue(max_size=5)
        config_manager = MagicMock(spec=ConfigurationManager)
        monitor = ChatMonitor(config_manager, message_queue=message_queue)
        
        # Check initial stats
        stats = message_queue.get_queue_stats()
        assert stats['messages_queued'] == 0
        assert stats['queue_size'] == 0
        assert stats['is_processing'] == False


# Test runner functions
async def run_integration_tests():
    """Run integration tests without pytest."""
    print("Running Chat Monitor Integration Tests...\n")
    
    test_classes = [
        TestChatMonitorInitialization,
        TestTwitchIntegration,
        TestYouTubeIntegration,
        TestMessageRouting,
        TestMultiPlatformMonitoring,
        TestReconnectionLogic,
        TestMessageQueueIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"Running {test_class.__name__}...")
        
        instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            method = getattr(instance, method_name)
            
            try:
                if asyncio.iscoroutinefunction(method):
                    await method()
                else:
                    method()
                
                print(f"  ✓ {method_name}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ❌ {method_name}: {e}")
    
    print(f"\nTest Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("🎉 All integration tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    # Run tests using pytest if available, otherwise run basic tests
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, running basic integration tests...")
        sys.exit(asyncio.run(run_integration_tests()))