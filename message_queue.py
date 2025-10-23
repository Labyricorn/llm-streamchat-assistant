"""
Message queue system for the LLM StreamChat Assistant.

This module provides an asyncio-based message queuing system with priority handling,
queue size management, and message routing based on platform origin.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import IntEnum
import heapq

from models import ChatMessage, QueuedMessage


class MessagePriority(IntEnum):
    """Message priority levels (lower numbers = higher priority)."""
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class PriorityQueueItem:
    """Wrapper for priority queue items."""
    priority: int
    timestamp: datetime
    message: QueuedMessage
    
    def __lt__(self, other):
        """Compare items for priority queue ordering."""
        # First by priority (lower = higher priority)
        if self.priority != other.priority:
            return self.priority < other.priority
        # Then by timestamp (older = higher priority)
        return self.timestamp < other.timestamp


class MessageQueue:
    """
    Async message queue system with priority handling and size management.
    
    Provides message queuing, routing, and processing capabilities for chat messages
    from multiple platforms with configurable priority and size limits.
    """
    
    def __init__(self, max_size: int = 100, processing_callback: Optional[Callable] = None):
        """
        Initialize message queue.
        
        Args:
            max_size: Maximum number of messages in queue
            processing_callback: Async callback for processing messages
        """
        self.max_size = max_size
        self.processing_callback = processing_callback
        self.logger = logging.getLogger(__name__)
        
        # Priority queue for messages
        self._queue: List[PriorityQueueItem] = []
        self._queue_lock = asyncio.Lock()
        
        # Processing control
        self.is_processing = False
        self.processor_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'messages_queued': 0,
            'messages_processed': 0,
            'messages_dropped': 0,
            'queue_size': 0,
            'processing_errors': 0
        }
        
        # Platform routing
        self.platform_handlers: Dict[str, Callable] = {}
        
        # Rate limiting
        self.last_process_time: Dict[str, datetime] = {}
        self.min_interval_seconds = 1.0  # Minimum interval between processing messages from same user
    
    async def start_processing(self) -> None:
        """Start the message processing loop."""
        if self.is_processing:
            self.logger.warning("Message queue processing is already running")
            return
        
        self.is_processing = True
        self.processor_task = asyncio.create_task(self._process_messages())
        self.logger.info("Message queue processing started")
    
    async def stop_processing(self) -> None:
        """Stop the message processing loop."""
        if not self.is_processing:
            return
        
        self.is_processing = False
        
        if self.processor_task and not self.processor_task.done():
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Message queue processing stopped")
    
    async def enqueue_message(self, message: ChatMessage, priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """
        Add a message to the processing queue.
        
        Args:
            message: Chat message to queue
            priority: Message priority level
            
        Returns:
            True if message was queued, False if dropped due to size limits
        """
        async with self._queue_lock:
            # Check queue size limits
            if len(self._queue) >= self.max_size:
                # Drop oldest low-priority message if possible
                if not self._drop_low_priority_message():
                    self.stats['messages_dropped'] += 1
                    self.logger.warning(f"Queue full, dropping message from {message.username}")
                    return False
            
            # Create queued message
            queued_msg = QueuedMessage.from_chat_message(message, priority=priority.value)
            
            # Create priority queue item
            queue_item = PriorityQueueItem(
                priority=priority.value,
                timestamp=message.timestamp,
                message=queued_msg
            )
            
            # Add to priority queue
            heapq.heappush(self._queue, queue_item)
            
            self.stats['messages_queued'] += 1
            self.stats['queue_size'] = len(self._queue)
            
            self.logger.debug(f"Queued message from {message.username} on {message.platform} (priority: {priority.name})")
            return True
    
    def _drop_low_priority_message(self) -> bool:
        """
        Drop the oldest low-priority message from the queue.
        
        Returns:
            True if a message was dropped, False if no low-priority messages found
        """
        # Find and remove the oldest low-priority message
        for i, item in enumerate(self._queue):
            if item.priority == MessagePriority.LOW:
                del self._queue[i]
                heapq.heapify(self._queue)  # Restore heap property
                self.stats['messages_dropped'] += 1
                return True
        return False
    
    async def _process_messages(self) -> None:
        """Main message processing loop."""
        while self.is_processing:
            try:
                # Get next message from queue
                message_item = await self._get_next_message()
                
                if message_item is None:
                    # No messages available, wait briefly
                    await asyncio.sleep(0.1)
                    continue
                
                # Check rate limiting
                if not self._should_process_message(message_item.message):
                    # Re-queue with lower priority if rate limited
                    await self._requeue_message(message_item.message)
                    continue
                
                # Process the message
                await self._process_single_message(message_item.message)
                
            except Exception as e:
                self.logger.error(f"Error in message processing loop: {e}")
                self.stats['processing_errors'] += 1
                await asyncio.sleep(1)  # Brief pause on error
    
    async def _get_next_message(self) -> Optional[PriorityQueueItem]:
        """Get the next message from the priority queue."""
        async with self._queue_lock:
            if self._queue:
                item = heapq.heappop(self._queue)
                self.stats['queue_size'] = len(self._queue)
                return item
            return None
    
    def _should_process_message(self, message: QueuedMessage) -> bool:
        """
        Check if a message should be processed based on rate limiting.
        
        Args:
            message: Message to check
            
        Returns:
            True if message should be processed now, False if rate limited
        """
        user_key = f"{message.platform}:{message.username}"
        now = datetime.now()
        
        last_time = self.last_process_time.get(user_key)
        if last_time is None:
            return True
        
        time_since_last = (now - last_time).total_seconds()
        return time_since_last >= self.min_interval_seconds
    
    async def _requeue_message(self, message: QueuedMessage) -> None:
        """Re-queue a rate-limited message with lower priority."""
        # Increase retry count
        message.retry_count += 1
        
        # If too many retries, drop the message
        if message.retry_count > 3:
            self.logger.debug(f"Dropping message from {message.username} after {message.retry_count} retries")
            self.stats['messages_dropped'] += 1
            return
        
        # Re-queue with lower priority
        async with self._queue_lock:
            if len(self._queue) < self.max_size:
                queue_item = PriorityQueueItem(
                    priority=MessagePriority.LOW,
                    timestamp=datetime.now() + timedelta(seconds=self.min_interval_seconds),
                    message=message
                )
                heapq.heappush(self._queue, queue_item)
                self.stats['queue_size'] = len(self._queue)
    
    async def _process_single_message(self, message: QueuedMessage) -> None:
        """
        Process a single message.
        
        Args:
            message: Message to process
        """
        try:
            # Update rate limiting timestamp
            user_key = f"{message.platform}:{message.username}"
            self.last_process_time[user_key] = datetime.now()
            
            # Route to platform-specific handler if available
            platform_handler = self.platform_handlers.get(message.platform)
            if platform_handler:
                await platform_handler(message)
            
            # Call general processing callback
            if self.processing_callback:
                await self.processing_callback(message)
            
            # Mark as processed
            message.processed = True
            self.stats['messages_processed'] += 1
            
            self.logger.debug(f"Processed message from {message.username} on {message.platform}")
            
        except Exception as e:
            self.logger.error(f"Error processing message from {message.username}: {e}")
            self.stats['processing_errors'] += 1
    
    def register_platform_handler(self, platform: str, handler: Callable) -> None:
        """
        Register a platform-specific message handler.
        
        Args:
            platform: Platform name (e.g., 'twitch', 'youtube')
            handler: Async handler function for messages from this platform
        """
        self.platform_handlers[platform] = handler
        self.logger.info(f"Registered handler for platform: {platform}")
    
    def unregister_platform_handler(self, platform: str) -> None:
        """
        Unregister a platform-specific message handler.
        
        Args:
            platform: Platform name to unregister
        """
        if platform in self.platform_handlers:
            del self.platform_handlers[platform]
            self.logger.info(f"Unregistered handler for platform: {platform}")
    
    def set_processing_callback(self, callback: Callable) -> None:
        """
        Set the general message processing callback.
        
        Args:
            callback: Async callback function for processing messages
        """
        self.processing_callback = callback
        self.logger.info("Message processing callback updated")
    
    def set_rate_limit(self, min_interval_seconds: float) -> None:
        """
        Set the minimum interval between processing messages from the same user.
        
        Args:
            min_interval_seconds: Minimum interval in seconds
        """
        self.min_interval_seconds = min_interval_seconds
        self.logger.info(f"Rate limit updated to {min_interval_seconds}s")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue statistics."""
        return {
            **self.stats,
            'queue_size': len(self._queue),
            'is_processing': self.is_processing,
            'platform_handlers': list(self.platform_handlers.keys())
        }
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return len(self._queue)
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0
    
    async def clear_queue(self) -> int:
        """
        Clear all messages from the queue.
        
        Returns:
            Number of messages that were cleared
        """
        async with self._queue_lock:
            cleared_count = len(self._queue)
            self._queue.clear()
            self.stats['queue_size'] = 0
            self.stats['messages_dropped'] += cleared_count
            
        self.logger.info(f"Cleared {cleared_count} messages from queue")
        return cleared_count
    
    async def get_pending_messages(self, platform: Optional[str] = None) -> List[QueuedMessage]:
        """
        Get list of pending messages in the queue.
        
        Args:
            platform: Optional platform filter
            
        Returns:
            List of pending messages
        """
        async with self._queue_lock:
            messages = [item.message for item in self._queue]
            
            if platform:
                messages = [msg for msg in messages if msg.platform == platform]
            
            return messages