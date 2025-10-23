#!/usr/bin/env python3
"""
Unit tests for message queue system.

Tests message queuing, priority handling, rate limiting, and async processing.
Covers core message queue functionality for the chat monitoring system.
"""

import asyncio
import pytest
import sys
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timedelta

from message_queue import MessageQueue, MessagePriority, PriorityQueueItem
from models import ChatMessage, QueuedMessage


class TestMessageQueueBasics:
    """Test basic message queue functionality."""
    
    def test_message_queue_initialization(self):
        """Test MessageQueue initialization."""
        # Default initialization
        queue = MessageQueue()
        assert queue.max_size == 100
        assert queue.processing_callback is None
        assert queue.is_processing == False
        assert queue.get_queue_size() == 0
        assert queue.is_empty() == True
        
        # Custom initialization
        callback = AsyncMock()
        queue = MessageQueue(max_size=50, processing_callback=callback)
        assert queue.max_size == 50
        assert queue.processing_callback == callback
    
    @pytest.mark.asyncio
    async def test_message_enqueue_basic(self):
        """Test basic message enqueuing."""
        queue = MessageQueue(max_size=10)
        
        message = ChatMessage(
            platform='twitch',
            username='test_user',
            content='Test message',
            timestamp=datetime.now(),
            channel='test_channel'
        )
        
        # Enqueue message
        result = await queue.enqueue_message(message, MessagePriority.NORMAL)
        assert result == True
        assert queue.get_queue_size() == 1
        assert queue.is_empty() == False
        
        # Check stats
        stats = queue.get_queue_stats()
        assert stats['messages_queued'] == 1
        assert stats['queue_size'] == 1
    
    @pytest.mark.asyncio
    async def test_message_priority_ordering(self):
        """Test that messages are processed in priority order."""
        queue = MessageQueue(max_size=10)
        processed_messages = []
        
        async def process_callback(message):
            processed_messages.append(message)
        
        queue.set_processing_callback(process_callback)
        
        # Create messages with different priorities
        messages = [
            (ChatMessage('twitch', 'user1', 'Normal message', datetime.now(), 'channel'), MessagePriority.NORMAL),
            (ChatMessage('twitch', 'user2', 'High priority message', datetime.now(), 'channel'), MessagePriority.HIGH),
            (ChatMessage('twitch', 'user3', 'Low priority message', datetime.now(), 'channel'), MessagePriority.LOW),
            (ChatMessage('twitch', 'user4', 'Another high priority', datetime.now(), 'channel'), MessagePriority.HIGH),
        ]
        
        # Enqueue all messages
        for message, priority in messages:
            await queue.enqueue_message(message, priority)
        
        # Start processing
        await queue.start_processing()
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Stop processing
        await queue.stop_processing()
        
        # Verify processing order (high priority first)
        assert len(processed_messages) == 4
        assert processed_messages[0].content == 'High priority message'
        assert processed_messages[1].content == 'Another high priority'
        # Normal and low priority can be in any order among themselves
    
    @pytest.mark.asyncio
    async def test_queue_size_limit(self):
        """Test queue size limits and message dropping."""
        queue = MessageQueue(max_size=2)
        
        # Fill queue to capacity
        message1 = ChatMessage('twitch', 'user1', 'Message 1', datetime.now(), 'channel')
        message2 = ChatMessage('twitch', 'user2', 'Message 2', datetime.now(), 'channel')
        
        result1 = await queue.enqueue_message(message1, MessagePriority.NORMAL)
        result2 = await queue.enqueue_message(message2, MessagePriority.NORMAL)
        
        assert result1 == True
        assert result2 == True
        assert queue.get_queue_size() == 2
        
        # Try to add one more (should be dropped)
        message3 = ChatMessage('twitch', 'user3', 'Message 3', datetime.now(), 'channel')
        result3 = await queue.enqueue_message(message3, MessagePriority.NORMAL)
        
        assert result3 == False
        assert queue.get_queue_size() == 2
        
        # Check that dropped message is counted in stats
        stats = queue.get_queue_stats()
        assert stats['messages_dropped'] >= 1
    
    @pytest.mark.asyncio
    async def test_low_priority_message_dropping(self):
        """Test that low priority messages are dropped when queue is full."""
        queue = MessageQueue(max_size=2)
        
        # Fill queue with low priority messages
        low_msg1 = ChatMessage('twitch', 'user1', 'Low 1', datetime.now(), 'channel')
        low_msg2 = ChatMessage('twitch', 'user2', 'Low 2', datetime.now(), 'channel')
        
        await queue.enqueue_message(low_msg1, MessagePriority.LOW)
        await queue.enqueue_message(low_msg2, MessagePriority.LOW)
        
        # Try to add high priority message (should succeed by dropping low priority)
        high_msg = ChatMessage('twitch', 'user3', 'High priority', datetime.now(), 'channel')
        result = await queue.enqueue_message(high_msg, MessagePriority.HIGH)
        
        assert result == True
        assert queue.get_queue_size() == 2


class TestMessageProcessing:
    """Test message processing functionality."""
    
    @pytest.mark.asyncio
    async def test_processing_start_stop(self):
        """Test starting and stopping message processing."""
        queue = MessageQueue()
        
        # Initially not processing
        assert queue.is_processing == False
        
        # Start processing
        await queue.start_processing()
        assert queue.is_processing == True
        
        # Stop processing
        await queue.stop_processing()
        assert queue.is_processing == False
    
    @pytest.mark.asyncio
    async def test_message_processing_callback(self):
        """Test message processing with callback."""
        queue = MessageQueue(max_size=10)
        processed_messages = []
        
        async def process_callback(message):
            processed_messages.append(message)
        
        queue.set_processing_callback(process_callback)
        
        # Add test message
        message = ChatMessage('twitch', 'test_user', 'Test message', datetime.now(), 'channel')
        await queue.enqueue_message(message, MessagePriority.NORMAL)
        
        # Start processing
        await queue.start_processing()
        
        # Wait for processing
        await asyncio.sleep(0.2)
        
        # Stop processing
        await queue.stop_processing()
        
        # Verify message was processed
        assert len(processed_messages) == 1
        assert processed_messages[0].username == 'test_user'
        assert processed_messages[0].content == 'Test message'
    
    @pytest.mark.asyncio
    async def test_processing_error_handling(self):
        """Test error handling during message processing."""
        queue = MessageQueue(max_size=10)
        
        async def failing_callback(message):
            raise Exception("Processing error")
        
        queue.set_processing_callback(failing_callback)
        
        # Add test message
        message = ChatMessage('twitch', 'test_user', 'Test message', datetime.now(), 'channel')
        await queue.enqueue_message(message, MessagePriority.NORMAL)
        
        # Start processing
        await queue.start_processing()
        
        # Wait for processing attempt
        await asyncio.sleep(0.2)
        
        # Stop processing
        await queue.stop_processing()
        
        # Check that error was counted
        stats = queue.get_queue_stats()
        assert stats['processing_errors'] >= 1


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    @pytest.mark.asyncio
    async def test_rate_limiting_basic(self):
        """Test basic rate limiting functionality."""
        queue = MessageQueue(max_size=10)
        queue.set_rate_limit(1.0)  # 1 second minimum interval
        
        processed_messages = []
        
        async def process_callback(message):
            processed_messages.append(message)
        
        queue.set_processing_callback(process_callback)
        
        # Add multiple messages from same user
        user_messages = [
            ChatMessage('twitch', 'same_user', 'Message 1', datetime.now(), 'channel'),
            ChatMessage('twitch', 'same_user', 'Message 2', datetime.now(), 'channel'),
            ChatMessage('twitch', 'same_user', 'Message 3', datetime.now(), 'channel'),
        ]
        
        for message in user_messages:
            await queue.enqueue_message(message, MessagePriority.NORMAL)
        
        # Start processing
        await queue.start_processing()
        
        # Wait briefly (less than rate limit)
        await asyncio.sleep(0.3)
        
        # Stop processing
        await queue.stop_processing()
        
        # Should only process one message due to rate limiting
        assert len(processed_messages) == 1
        assert processed_messages[0].content == 'Message 1'
    
    @pytest.mark.asyncio
    async def test_rate_limiting_different_users(self):
        """Test that rate limiting is per-user."""
        queue = MessageQueue(max_size=10)
        queue.set_rate_limit(1.0)  # 1 second minimum interval
        
        processed_messages = []
        
        async def process_callback(message):
            processed_messages.append(message)
        
        queue.set_processing_callback(process_callback)
        
        # Add messages from different users
        messages = [
            ChatMessage('twitch', 'user1', 'Message from user1', datetime.now(), 'channel'),
            ChatMessage('twitch', 'user2', 'Message from user2', datetime.now(), 'channel'),
            ChatMessage('twitch', 'user3', 'Message from user3', datetime.now(), 'channel'),
        ]
        
        for message in messages:
            await queue.enqueue_message(message, MessagePriority.NORMAL)
        
        # Start processing
        await queue.start_processing()
        
        # Wait briefly
        await asyncio.sleep(0.3)
        
        # Stop processing
        await queue.stop_processing()
        
        # Should process all messages since they're from different users
        assert len(processed_messages) == 3
    
    @pytest.mark.asyncio
    async def test_message_requeuing(self):
        """Test that rate-limited messages are requeued."""
        queue = MessageQueue(max_size=10)
        queue.set_rate_limit(0.5)  # 0.5 second minimum interval
        
        processed_messages = []
        
        async def process_callback(message):
            processed_messages.append(message)
        
        queue.set_processing_callback(process_callback)
        
        # Add messages from same user
        message1 = ChatMessage('twitch', 'same_user', 'Message 1', datetime.now(), 'channel')
        message2 = ChatMessage('twitch', 'same_user', 'Message 2', datetime.now(), 'channel')
        
        await queue.enqueue_message(message1, MessagePriority.NORMAL)
        await queue.enqueue_message(message2, MessagePriority.NORMAL)
        
        # Start processing
        await queue.start_processing()
        
        # Wait for rate limit to pass
        await asyncio.sleep(1.0)
        
        # Stop processing
        await queue.stop_processing()
        
        # Should eventually process both messages
        assert len(processed_messages) >= 1  # At least first message


class TestPlatformHandlers:
    """Test platform-specific message handlers."""
    
    @pytest.mark.asyncio
    async def test_platform_handler_registration(self):
        """Test registering platform-specific handlers."""
        queue = MessageQueue()
        
        twitch_messages = []
        youtube_messages = []
        
        async def twitch_handler(message):
            twitch_messages.append(message)
        
        async def youtube_handler(message):
            youtube_messages.append(message)
        
        # Register handlers
        queue.register_platform_handler('twitch', twitch_handler)
        queue.register_platform_handler('youtube', youtube_handler)
        
        # Check handlers are registered
        stats = queue.get_queue_stats()
        assert 'twitch' in stats['platform_handlers']
        assert 'youtube' in stats['platform_handlers']
        
        # Add messages from different platforms
        twitch_msg = ChatMessage('twitch', 'user1', 'Twitch message', datetime.now(), 'channel')
        youtube_msg = ChatMessage('youtube', 'user2', 'YouTube message', datetime.now(), 'channel')
        
        await queue.enqueue_message(twitch_msg, MessagePriority.NORMAL)
        await queue.enqueue_message(youtube_msg, MessagePriority.NORMAL)
        
        # Start processing
        await queue.start_processing()
        
        # Wait for processing
        await asyncio.sleep(0.2)
        
        # Stop processing
        await queue.stop_processing()
        
        # Verify platform-specific handlers were called
        assert len(twitch_messages) == 1
        assert len(youtube_messages) == 1
        assert twitch_messages[0].platform == 'twitch'
        assert youtube_messages[0].platform == 'youtube'
    
    def test_platform_handler_unregistration(self):
        """Test unregistering platform handlers."""
        queue = MessageQueue()
        
        async def dummy_handler(message):
            pass
        
        # Register and then unregister
        queue.register_platform_handler('twitch', dummy_handler)
        assert 'twitch' in queue.get_queue_stats()['platform_handlers']
        
        queue.unregister_platform_handler('twitch')
        assert 'twitch' not in queue.get_queue_stats()['platform_handlers']


class TestQueueManagement:
    """Test queue management operations."""
    
    @pytest.mark.asyncio
    async def test_clear_queue(self):
        """Test clearing all messages from queue."""
        queue = MessageQueue(max_size=10)
        
        # Add several messages
        for i in range(5):
            message = ChatMessage('twitch', f'user{i}', f'Message {i}', datetime.now(), 'channel')
            await queue.enqueue_message(message, MessagePriority.NORMAL)
        
        assert queue.get_queue_size() == 5
        
        # Clear queue
        cleared_count = await queue.clear_queue()
        
        assert cleared_count == 5
        assert queue.get_queue_size() == 0
        assert queue.is_empty() == True
    
    @pytest.mark.asyncio
    async def test_get_pending_messages(self):
        """Test getting pending messages from queue."""
        queue = MessageQueue(max_size=10)
        
        # Add messages from different platforms
        twitch_msg = ChatMessage('twitch', 'user1', 'Twitch message', datetime.now(), 'channel')
        youtube_msg = ChatMessage('youtube', 'user2', 'YouTube message', datetime.now(), 'channel')
        
        await queue.enqueue_message(twitch_msg, MessagePriority.NORMAL)
        await queue.enqueue_message(youtube_msg, MessagePriority.NORMAL)
        
        # Get all pending messages
        all_messages = await queue.get_pending_messages()
        assert len(all_messages) == 2
        
        # Get platform-specific messages
        twitch_messages = await queue.get_pending_messages(platform='twitch')
        youtube_messages = await queue.get_pending_messages(platform='youtube')
        
        assert len(twitch_messages) == 1
        assert len(youtube_messages) == 1
        assert twitch_messages[0].platform == 'twitch'
        assert youtube_messages[0].platform == 'youtube'
    
    def test_queue_statistics(self):
        """Test queue statistics tracking."""
        queue = MessageQueue(max_size=10)
        
        # Initial stats
        stats = queue.get_queue_stats()
        assert stats['messages_queued'] == 0
        assert stats['messages_processed'] == 0
        assert stats['messages_dropped'] == 0
        assert stats['queue_size'] == 0
        assert stats['processing_errors'] == 0
        assert stats['is_processing'] == False
        assert stats['platform_handlers'] == []


class TestPriorityQueueItem:
    """Test PriorityQueueItem functionality."""
    
    def test_priority_queue_item_comparison(self):
        """Test priority queue item comparison logic."""
        now = datetime.now()
        later = now + timedelta(seconds=1)
        
        # Create test messages
        msg1 = QueuedMessage('1', 'twitch', 'user1', 'content1', now, 'channel', priority=1)
        msg2 = QueuedMessage('2', 'twitch', 'user2', 'content2', later, 'channel', priority=2)
        msg3 = QueuedMessage('3', 'twitch', 'user3', 'content3', now, 'channel', priority=1)
        
        # Create priority queue items
        item1 = PriorityQueueItem(1, now, msg1)
        item2 = PriorityQueueItem(2, later, msg2)
        item3 = PriorityQueueItem(1, later, msg3)
        
        # Test priority comparison (lower number = higher priority)
        assert item1 < item2  # Priority 1 < Priority 2
        
        # Test timestamp comparison for same priority
        assert item1 < item3  # Same priority, earlier timestamp
        
        # Test sorting
        items = [item2, item3, item1]
        items.sort()
        
        # Should be sorted by priority first, then timestamp
        assert items[0] == item1  # Priority 1, earliest time
        assert items[1] == item3  # Priority 1, later time
        assert items[2] == item2  # Priority 2


# Test runner functions
async def run_message_queue_tests():
    """Run message queue tests without pytest."""
    print("Running Message Queue Tests...\n")
    
    test_classes = [
        TestMessageQueueBasics,
        TestMessageProcessing,
        TestRateLimiting,
        TestPlatformHandlers,
        TestQueueManagement,
        TestPriorityQueueItem
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
                import traceback
                traceback.print_exc()
    
    print(f"\nTest Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("🎉 All message queue tests passed!")
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
        print("pytest not available, running basic message queue tests...")
        sys.exit(asyncio.run(run_message_queue_tests()))