"""
Response generation engine for the LLM StreamChat Assistant.

This module handles processing queued chat messages, generating context-aware
prompts for LLM backends, and managing response frequency and rate limiting.
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field

from models import ChatMessage, QueuedMessage, ResponseContext
from config_manager import ConfigurationManager
from llm_client import LLMClientFactory, LLMRequest, LLMResponse, LLMClientError
from message_queue import MessageQueue


@dataclass
class ResponseStats:
    """Statistics for response generation."""
    
    messages_processed: int = 0
    responses_generated: int = 0
    responses_sent: int = 0
    responses_skipped: int = 0
    llm_errors: int = 0
    rate_limited: int = 0
    total_response_time: float = 0.0
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        if self.responses_generated == 0:
            return 0.0
        return self.total_response_time / self.responses_generated
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'messages_processed': self.messages_processed,
            'responses_generated': self.responses_generated,
            'responses_sent': self.responses_sent,
            'responses_skipped': self.responses_skipped,
            'llm_errors': self.llm_errors,
            'rate_limited': self.rate_limited,
            'average_response_time': self.average_response_time
        }


class ResponseGenerator:
    """
    Response generation engine that processes chat messages and generates LLM responses.
    
    Handles message processing from queue, applies personality and behavior configuration,
    manages response frequency and rate limiting, and coordinates with LLM backends.
    """
    
    def __init__(self, config_manager: ConfigurationManager, message_queue: MessageQueue):
        """
        Initialize the response generator.
        
        Args:
            config_manager: Configuration manager instance
            message_queue: Message queue for processing chat messages
        """
        self.config_manager = config_manager
        self.message_queue = message_queue
        self.logger = logging.getLogger(__name__)
        
        # LLM client
        self.llm_client = None
        self._initialize_llm_client()
        
        # Processing state
        self.is_running = False
        self.processor_task: Optional[asyncio.Task] = None
        
        # Rate limiting and frequency control
        self.last_response_time: Dict[str, datetime] = {}  # channel -> last response time
        self.user_response_history: Dict[str, List[datetime]] = {}  # user -> response times
        self.recent_messages: Dict[str, List[ChatMessage]] = {}  # channel -> recent messages
        
        # Response delivery callbacks
        self.response_callbacks: Dict[str, Callable] = {}  # platform -> callback
        
        # Statistics
        self.stats = ResponseStats()
        
        # Message context window
        self.max_context_messages = 10
        
    def _initialize_llm_client(self) -> None:
        """Initialize the LLM client based on configuration."""
        try:
            llm_config = self.config_manager.get_llm_config()
            backend = llm_config.get('backend', 'ollama')
            
            client_kwargs = {
                'base_url': llm_config.get('base_url', 'http://localhost:11434'),
                'timeout': llm_config.get('timeout', 30.0)
            }
            
            # Add backend-specific configuration
            if backend == 'lmstudio' or backend == 'lemonade':
                client_kwargs['api_key'] = llm_config.get('api_key', 'default-key')
            
            self.llm_client = LLMClientFactory.create_client(backend, **client_kwargs)
            self.logger.info(f"Initialized {backend} LLM client")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_client = None
    
    async def start(self) -> None:
        """Start the response generation engine."""
        if self.is_running:
            self.logger.warning("Response generator is already running")
            return
        
        if self.llm_client is None:
            self._initialize_llm_client()
            if self.llm_client is None:
                raise RuntimeError("Cannot start response generator without valid LLM client")
        
        # Test LLM connection
        try:
            if not await self.llm_client.test_connection():
                self.logger.warning("LLM connection test failed, but starting anyway")
        except Exception as e:
            self.logger.warning(f"LLM connection test error: {e}")
        
        self.is_running = True
        
        # Set message queue processing callback
        self.message_queue.set_processing_callback(self.process_message)
        
        self.logger.info("Response generator started")
    
    async def stop(self) -> None:
        """Stop the response generation engine."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.processor_task and not self.processor_task.done():
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        # Close LLM client
        if self.llm_client:
            await self.llm_client.close()
        
        self.logger.info("Response generator stopped")
    
    async def process_message(self, queued_message: QueuedMessage) -> None:
        """
        Process a queued chat message and potentially generate a response.
        
        Args:
            queued_message: The message to process
        """
        start_time = datetime.now()
        self.stats.messages_processed += 1
        
        try:
            # Convert to ChatMessage
            chat_message = ChatMessage(
                platform=queued_message.platform,
                username=queued_message.username,
                content=queued_message.content,
                timestamp=queued_message.timestamp,
                channel=queued_message.channel,
                message_id=queued_message.id
            )
            
            # Update recent messages for context
            self._update_recent_messages(chat_message)
            
            # Check if we should respond to this message
            if not await self._should_respond(chat_message):
                self.stats.responses_skipped += 1
                return
            
            # Generate response
            response = await self._generate_response(chat_message)
            
            if response and response.success:
                # Format and deliver response
                formatted_response = self._format_response(response.content, chat_message)
                
                if formatted_response:
                    await self._deliver_response(chat_message, formatted_response)
                    self.stats.responses_sent += 1
                    
                    # Update rate limiting
                    self._update_response_timing(chat_message)
                
                self.stats.responses_generated += 1
            else:
                self.stats.llm_errors += 1
                if response and response.error:
                    self.logger.error(f"LLM error for message from {chat_message.username}: {response.error}")
            
            # Update timing stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats.total_response_time += processing_time
            
        except Exception as e:
            self.logger.error(f"Error processing message from {queued_message.username}: {e}")
            self.stats.llm_errors += 1
    
    def _update_recent_messages(self, message: ChatMessage) -> None:
        """Update the recent messages context for a channel."""
        channel_key = f"{message.platform}:{message.channel}"
        
        if channel_key not in self.recent_messages:
            self.recent_messages[channel_key] = []
        
        # Add message to recent messages
        self.recent_messages[channel_key].append(message)
        
        # Keep only the most recent messages
        if len(self.recent_messages[channel_key]) > self.max_context_messages:
            self.recent_messages[channel_key] = self.recent_messages[channel_key][-self.max_context_messages:]
    
    async def _should_respond(self, message: ChatMessage) -> bool:
        """
        Determine if we should respond to a message based on frequency, rate limiting, and response types.
        
        Args:
            message: The chat message to evaluate
            
        Returns:
            True if we should respond, False otherwise
        """
        behavior_config = self.config_manager.get_behavior_config()
        personality_config = self.config_manager.get_personality_config()
        
        # Check if message content matches enabled response types
        context = ResponseContext(
            message=message,
            personality=personality_config
        )
        
        if not context.should_respond_with_type(message.content):
            return False
        
        # Check response frequency (random chance)
        response_frequency = behavior_config.get('response_frequency', 0.3)
        if random.random() > response_frequency:
            return False
        
        # Check minimum response interval for the channel
        min_interval = behavior_config.get('min_response_interval', 30)
        channel_key = f"{message.platform}:{message.channel}"
        
        if channel_key in self.last_response_time:
            time_since_last = (datetime.now() - self.last_response_time[channel_key]).total_seconds()
            if time_since_last < min_interval:
                self.stats.rate_limited += 1
                return False
        
        # Check user-specific rate limiting (don't respond to same user too frequently)
        user_key = f"{message.platform}:{message.username}"
        user_min_interval = min_interval // 2  # Half the channel interval for individual users
        
        if user_key in self.user_response_history:
            recent_responses = [
                t for t in self.user_response_history[user_key]
                if (datetime.now() - t).total_seconds() < user_min_interval * 3
            ]
            
            if recent_responses:
                last_user_response = max(recent_responses)
                time_since_user = (datetime.now() - last_user_response).total_seconds()
                if time_since_user < user_min_interval:
                    self.stats.rate_limited += 1
                    return False
        
        return True
    
    async def _generate_response(self, message: ChatMessage) -> Optional[LLMResponse]:
        """
        Generate an LLM response for a chat message.
        
        Args:
            message: The chat message to respond to
            
        Returns:
            LLMResponse or None if generation fails
        """
        if not self.llm_client:
            self.logger.error("No LLM client available for response generation")
            return None
        
        try:
            # Build response context
            context = self._build_response_context(message)
            
            # Generate prompt
            prompt = self._build_prompt(context)
            
            # Get LLM configuration
            llm_config = self.config_manager.get_llm_config()
            personality_config = self.config_manager.get_personality_config()
            
            # Create LLM request
            request = LLMRequest(
                prompt=prompt,
                model=llm_config.get('model', 'llama2'),
                config={},
                timeout=llm_config.get('timeout', 30.0),
                temperature=personality_config.get('temperature', 0.7),
                max_tokens=self._get_max_tokens_for_length(personality_config.get('response_length', 'short'))
            )
            
            # Send request to LLM
            response = await self.llm_client.send_request(request)
            return response
            
        except LLMClientError as e:
            self.logger.error(f"LLM client error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error generating response: {e}")
            return None
    
    def _build_response_context(self, message: ChatMessage) -> ResponseContext:
        """
        Build response context for LLM prompt generation.
        
        Args:
            message: The current chat message
            
        Returns:
            ResponseContext with relevant information
        """
        channel_key = f"{message.platform}:{message.channel}"
        recent_messages = self.recent_messages.get(channel_key, [])
        
        # Get user history (simplified for now)
        user_key = f"{message.platform}:{message.username}"
        user_history = {
            'message_count': len([m for m in recent_messages if m.username == message.username]),
            'recent_activity': user_key in self.user_response_history
        }
        
        return ResponseContext(
            message=message,
            personality=self.config_manager.get_personality_config(),
            recent_messages=recent_messages,
            user_history=user_history,
            response_config=self.config_manager.get_behavior_config()
        )
    
    def _build_prompt(self, context: ResponseContext) -> str:
        """
        Build the LLM prompt from response context with personality integration.
        
        Args:
            context: Response context information
            
        Returns:
            Formatted prompt string
        """
        # Get personality-based system prompt
        system_prompt = context.get_personality_prompt()
        
        # Get context string
        context_string = context.get_context_string()
        
        # Add personality-specific instructions
        personality_instructions = self._get_personality_instructions(context.personality)
        
        # Build full prompt
        prompt_parts = [
            system_prompt,
            "",
            "You are responding to messages in a live stream chat. Keep responses natural and engaging.",
            "Respond directly to the current message while being aware of the recent chat context.",
            "",
            personality_instructions,
            "",
            context_string,
            "",
            "Generate a response that fits the personality and length requirements. Do not include any prefixes like 'Response:' or the username."
        ]
        
        return "\n".join(prompt_parts)
    
    def _get_personality_instructions(self, personality_config: Dict[str, Any]) -> str:
        """
        Generate personality-specific instructions for the LLM.
        
        Args:
            personality_config: Personality configuration
            
        Returns:
            Formatted personality instructions
        """
        preset = personality_config.get('preset', 'friendly')
        response_types = personality_config.get('response_types', ['conversational'])
        response_length = personality_config.get('response_length', 'short')
        
        instructions = []
        
        # Preset-specific instructions
        preset_instructions = {
            'friendly': "Be warm, welcoming, and use casual language. Show genuine interest in the conversation.",
            'professional': "Maintain a professional tone while being helpful. Focus on providing accurate information.",
            'humorous': "Use appropriate humor, wordplay, and light-hearted responses. Keep it fun but respectful.",
            'supportive': "Be encouraging, positive, and uplifting. Celebrate achievements and offer comfort when needed."
        }
        
        if preset in preset_instructions:
            instructions.append(preset_instructions[preset])
        
        # Response type specific instructions
        if 'informational' in response_types:
            instructions.append("When appropriate, provide helpful information or explanations.")
        
        if 'humorous' in response_types and preset != 'humorous':
            instructions.append("Add light humor when the context allows.")
        
        if 'supportive' in response_types and preset != 'supportive':
            instructions.append("Be encouraging and positive in your responses.")
        
        # Length-specific instructions
        length_instructions = {
            'short': "Keep responses concise and to the point.",
            'medium': "Provide moderate detail in your responses.",
            'long': "You can elaborate when the topic warrants detailed explanation."
        }
        
        if response_length in length_instructions:
            instructions.append(length_instructions[response_length])
        
        return " ".join(instructions)
    
    def _get_max_tokens_for_length(self, response_length: str) -> int:
        """
        Get max tokens based on response length setting.
        
        Args:
            response_length: Length setting ('short', 'medium', 'long')
            
        Returns:
            Maximum token count
        """
        length_mapping = {
            'short': 50,
            'medium': 100,
            'long': 200
        }
        return length_mapping.get(response_length, 50)
    
    def _format_response(self, response_content: str, original_message: ChatMessage) -> Optional[str]:
        """
        Format and sanitize LLM response for chat delivery with personality-based formatting.
        
        Args:
            response_content: Raw LLM response
            original_message: Original chat message being responded to
            
        Returns:
            Formatted response string or None if invalid
        """
        if not response_content or not response_content.strip():
            return None
        
        # Clean up the response
        formatted = response_content.strip()
        
        # Remove common LLM artifacts
        artifacts_to_remove = [
            "Response:",
            "Assistant:",
            "Bot:",
            f"{original_message.username}:",
            "Chat response:",
            "Reply:",
            "Answer:",
            "Output:"
        ]
        
        for artifact in artifacts_to_remove:
            if formatted.startswith(artifact):
                formatted = formatted[len(artifact):].strip()
        
        # Apply personality-based formatting
        formatted = self._apply_personality_formatting(formatted, original_message)
        
        if not formatted:
            return None
        
        # Ensure response isn't too long for chat platforms
        personality_config = self.config_manager.get_personality_config()
        max_length = self._get_max_length_for_response_length(personality_config.get('response_length', 'short'))
        
        if len(formatted) > max_length:
            formatted = formatted[:max_length-3] + "..."
        
        # Basic sanitization - remove potentially problematic characters
        formatted = formatted.replace('\n', ' ').replace('\r', ' ')
        
        # Remove excessive whitespace
        formatted = ' '.join(formatted.split())
        
        return formatted if formatted else None
    
    def _apply_personality_formatting(self, response: str, original_message: ChatMessage) -> str:
        """
        Apply personality-specific formatting to the response.
        
        Args:
            response: The response text to format
            original_message: Original message for context
            
        Returns:
            Formatted response with personality touches
        """
        personality_config = self.config_manager.get_personality_config()
        preset = personality_config.get('preset', 'friendly')
        
        # Apply preset-specific formatting
        if preset == 'friendly':
            # Add occasional friendly touches
            if random.random() < 0.3:  # 30% chance
                friendly_prefixes = ["Hey! ", "Oh, ", "Nice! "]
                if not any(response.startswith(prefix.strip()) for prefix in friendly_prefixes):
                    response = random.choice(friendly_prefixes) + response.lower()[0] + response[1:]
        
        elif preset == 'humorous':
            # Ensure response has some personality (already handled by LLM prompt)
            pass
        
        elif preset == 'supportive':
            # Add supportive elements if not already present
            supportive_indicators = ['great', 'awesome', 'good', 'nice', 'well done', 'keep it up']
            if not any(indicator in response.lower() for indicator in supportive_indicators):
                if random.random() < 0.2:  # 20% chance to add supportive touch
                    supportive_additions = [" Keep it up!", " You're doing great!", " Nice!"]
                    response += random.choice(supportive_additions)
        
        elif preset == 'professional':
            # Ensure professional tone (capitalize first letter, proper punctuation)
            if response and response[0].islower():
                response = response[0].upper() + response[1:]
            
            if response and not response.endswith(('.', '!', '?')):
                response += '.'
        
        return response
    
    def _get_max_length_for_response_length(self, response_length: str) -> int:
        """
        Get maximum character length based on response length setting.
        
        Args:
            response_length: Length setting ('short', 'medium', 'long')
            
        Returns:
            Maximum character count
        """
        length_mapping = {
            'short': 200,
            'medium': 350,
            'long': 500
        }
        return length_mapping.get(response_length, 200)
    
    async def _deliver_response(self, original_message: ChatMessage, response: str) -> None:
        """
        Deliver response to the appropriate chat platform with platform-specific formatting.
        
        Args:
            original_message: Original message being responded to
            response: Formatted response to send
        """
        platform = original_message.platform
        
        # Apply platform-specific formatting
        platform_formatted_response = self._format_for_platform(response, platform, original_message)
        
        if not platform_formatted_response:
            self.logger.warning(f"Response formatting failed for platform {platform}")
            return
        
        # Validate response for platform requirements
        if not self._validate_response_for_platform(platform_formatted_response, platform):
            self.logger.warning(f"Response validation failed for platform {platform}")
            return
        
        if platform in self.response_callbacks:
            try:
                await self.response_callbacks[platform](original_message, platform_formatted_response)
                self.logger.debug(f"Delivered response to {platform}: {platform_formatted_response[:50]}...")
            except Exception as e:
                self.logger.error(f"Error delivering response to {platform}: {e}")
        else:
            self.logger.warning(f"No response callback registered for platform: {platform}")
    
    def _format_for_platform(self, response: str, platform: str, original_message: ChatMessage) -> Optional[str]:
        """
        Apply platform-specific formatting to the response.
        
        Args:
            response: The response text to format
            platform: Target platform ('twitch', 'youtube')
            original_message: Original message for context
            
        Returns:
            Platform-formatted response or None if formatting fails
        """
        if not response or not response.strip():
            return None
        
        formatted = response.strip()
        
        if platform == 'twitch':
            # Twitch-specific formatting
            formatted = self._format_for_twitch(formatted, original_message)
        elif platform == 'youtube':
            # YouTube-specific formatting
            formatted = self._format_for_youtube(formatted, original_message)
        else:
            self.logger.warning(f"Unknown platform for formatting: {platform}")
        
        return formatted
    
    def _format_for_twitch(self, response: str, original_message: ChatMessage) -> str:
        """
        Format response for Twitch chat requirements.
        
        Args:
            response: Response text
            original_message: Original message
            
        Returns:
            Twitch-formatted response
        """
        # Twitch has a 500 character limit
        max_length = 500
        
        # Sometimes mention the user (but not always to avoid spam)
        if random.random() < 0.3 and not response.startswith('@'):  # 30% chance
            response = f"@{original_message.username} {response}"
        
        # Ensure length limit
        if len(response) > max_length:
            response = response[:max_length-3] + "..."
        
        # Remove problematic characters for Twitch
        response = response.replace('\n', ' ').replace('\r', ' ')
        
        return response
    
    def _format_for_youtube(self, response: str, original_message: ChatMessage) -> str:
        """
        Format response for YouTube chat requirements.
        
        Args:
            response: Response text
            original_message: Original message
            
        Returns:
            YouTube-formatted response
        """
        # YouTube has a 200 character limit for live chat
        max_length = 200
        
        # YouTube doesn't support @mentions in the same way, so avoid them
        if response.startswith('@'):
            # Remove @mention and adjust
            parts = response.split(' ', 1)
            if len(parts) > 1:
                response = parts[1]
            else:
                response = response[len(parts[0]):].strip()
        
        # Ensure length limit (more restrictive for YouTube)
        if len(response) > max_length:
            response = response[:max_length-3] + "..."
        
        # Remove problematic characters
        response = response.replace('\n', ' ').replace('\r', ' ')
        
        return response
    
    def _validate_response_for_platform(self, response: str, platform: str) -> bool:
        """
        Validate response meets platform requirements.
        
        Args:
            response: Response to validate
            platform: Target platform
            
        Returns:
            True if response is valid for platform, False otherwise
        """
        if not response or not response.strip():
            return False
        
        # Common validation
        if len(response.strip()) == 0:
            return False
        
        # Platform-specific validation
        if platform == 'twitch':
            return self._validate_twitch_response(response)
        elif platform == 'youtube':
            return self._validate_youtube_response(response)
        
        return True
    
    def _validate_twitch_response(self, response: str) -> bool:
        """
        Validate response for Twitch chat.
        
        Args:
            response: Response to validate
            
        Returns:
            True if valid for Twitch
        """
        # Check length limit
        if len(response) > 500:
            return False
        
        # Check for prohibited content (basic checks)
        prohibited_patterns = [
            'http://',  # Avoid links unless specifically allowed
            'https://',
            'www.',
        ]
        
        response_lower = response.lower()
        for pattern in prohibited_patterns:
            if pattern in response_lower:
                return False
        
        return True
    
    def _validate_youtube_response(self, response: str) -> bool:
        """
        Validate response for YouTube chat.
        
        Args:
            response: Response to validate
            
        Returns:
            True if valid for YouTube
        """
        # Check length limit (more restrictive)
        if len(response) > 200:
            return False
        
        # YouTube is more restrictive about content
        prohibited_patterns = [
            'http://',
            'https://',
            'www.',
            '@',  # Avoid @ mentions
        ]
        
        response_lower = response.lower()
        for pattern in prohibited_patterns:
            if pattern in response_lower:
                return False
        
        return True
    
    def _update_response_timing(self, message: ChatMessage) -> None:
        """Update response timing for rate limiting."""
        now = datetime.now()
        
        # Update channel timing
        channel_key = f"{message.platform}:{message.channel}"
        self.last_response_time[channel_key] = now
        
        # Update user timing
        user_key = f"{message.platform}:{message.username}"
        if user_key not in self.user_response_history:
            self.user_response_history[user_key] = []
        
        self.user_response_history[user_key].append(now)
        
        # Clean up old entries (keep only last 24 hours)
        cutoff_time = now - timedelta(hours=24)
        self.user_response_history[user_key] = [
            t for t in self.user_response_history[user_key] if t > cutoff_time
        ]
    
    def register_response_callback(self, platform: str, callback: Callable) -> None:
        """
        Register a callback for delivering responses to a platform.
        
        Args:
            platform: Platform name ('twitch', 'youtube')
            callback: Async callback function(original_message, response)
        """
        self.response_callbacks[platform] = callback
        self.logger.info(f"Registered response callback for platform: {platform}")
    
    def unregister_response_callback(self, platform: str) -> None:
        """
        Unregister a response callback for a platform.
        
        Args:
            platform: Platform name to unregister
        """
        if platform in self.response_callbacks:
            del self.response_callbacks[platform]
            self.logger.info(f"Unregistered response callback for platform: {platform}")
    
    def sanitize_response(self, response: str) -> str:
        """
        Sanitize response content to remove potentially harmful or inappropriate content.
        
        Args:
            response: Response text to sanitize
            
        Returns:
            Sanitized response text
        """
        if not response:
            return ""
        
        sanitized = response.strip()
        
        # Remove or replace potentially problematic content
        replacements = {
            # Remove excessive punctuation
            '!!!': '!',
            '???': '?',
            '...': '.',
            
            # Remove potential spam patterns
            'CLICK HERE': '',
            'FREE MONEY': '',
            'SUBSCRIBE NOW': '',
        }
        
        for old, new in replacements.items():
            sanitized = sanitized.replace(old, new)
        
        # Remove excessive capitalization (more than 3 consecutive caps)
        import re
        sanitized = re.sub(r'[A-Z]{4,}', lambda m: m.group(0)[:3], sanitized)
        
        # Remove excessive whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized
    
    def get_response_routing_info(self) -> Dict[str, Any]:
        """
        Get information about response routing and delivery.
        
        Returns:
            Dictionary with routing information
        """
        return {
            'registered_platforms': list(self.response_callbacks.keys()),
            'enabled_platforms': self.config_manager.get_enabled_platforms(),
            'delivery_stats': {
                'responses_sent': self.stats.responses_sent,
                'delivery_errors': self.stats.llm_errors,  # Includes delivery errors
                'rate_limited': self.stats.rate_limited
            }
        }
    
    async def test_response_delivery(self, platform: str, test_message: str = "Test message") -> Dict[str, Any]:
        """
        Test response delivery for a specific platform.
        
        Args:
            platform: Platform to test
            test_message: Test message content
            
        Returns:
            Dictionary with test results
        """
        if platform not in self.response_callbacks:
            return {
                'success': False,
                'error': f'No callback registered for platform: {platform}',
                'platform': platform
            }
        
        try:
            # Create test message
            test_chat_message = ChatMessage(
                platform=platform,
                username="test_user",
                content=test_message,
                timestamp=datetime.now(),
                channel="test_channel"
            )
            
            # Generate test response
            test_response = f"Test response for {platform}"
            
            # Format for platform
            formatted_response = self._format_for_platform(test_response, platform, test_chat_message)
            
            if not formatted_response:
                return {
                    'success': False,
                    'error': 'Response formatting failed',
                    'platform': platform
                }
            
            # Validate response
            if not self._validate_response_for_platform(formatted_response, platform):
                return {
                    'success': False,
                    'error': 'Response validation failed',
                    'platform': platform,
                    'formatted_response': formatted_response
                }
            
            return {
                'success': True,
                'platform': platform,
                'original_response': test_response,
                'formatted_response': formatted_response,
                'validation_passed': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'platform': platform
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get response generation statistics."""
        return self.stats.to_dict()
    
    def reset_stats(self) -> None:
        """Reset response generation statistics."""
        self.stats = ResponseStats()
        self.logger.info("Response generation statistics reset")
    
    async def test_llm_connection(self) -> Dict[str, Any]:
        """
        Test the LLM connection and return status information.
        
        Returns:
            Dictionary with connection status and details
        """
        if not self.llm_client:
            return {
                'connected': False,
                'error': 'No LLM client initialized'
            }
        
        try:
            health_info = await self.llm_client.health_check()
            return {
                'connected': health_info.get('status') == 'healthy',
                'backend': health_info.get('backend'),
                'base_url': health_info.get('base_url'),
                'available_models': health_info.get('available_models', []),
                'error': health_info.get('error')
            }
        except Exception as e:
            return {
                'connected': False,
                'error': str(e)
            }
    
    def update_personality_config(self, personality_updates: Dict[str, Any]) -> None:
        """
        Update personality configuration and reinitialize if needed.
        
        Args:
            personality_updates: Dictionary of personality configuration updates
        """
        try:
            # Update configuration through config manager
            self.config_manager.update_config({'personality': personality_updates})
            self.logger.info("Personality configuration updated")
        except Exception as e:
            self.logger.error(f"Failed to update personality configuration: {e}")
    
    def get_personality_presets(self) -> Dict[str, Dict[str, Any]]:
        """
        Get available personality presets with their configurations.
        
        Returns:
            Dictionary of preset names and their configurations
        """
        return {
            'friendly': {
                'preset': 'friendly',
                'response_types': ['conversational', 'supportive'],
                'response_length': 'short',
                'custom_instructions': 'Be warm and welcoming to everyone in chat.'
            },
            'professional': {
                'preset': 'professional',
                'response_types': ['informational', 'conversational'],
                'response_length': 'medium',
                'custom_instructions': 'Provide helpful information while maintaining professionalism.'
            },
            'humorous': {
                'preset': 'humorous',
                'response_types': ['humorous', 'conversational'],
                'response_length': 'short',
                'custom_instructions': 'Keep things light and fun with appropriate humor.'
            },
            'supportive': {
                'preset': 'supportive',
                'response_types': ['supportive', 'conversational'],
                'response_length': 'medium',
                'custom_instructions': 'Be encouraging and positive in all interactions.'
            }
        }
    
    def apply_personality_preset(self, preset_name: str) -> bool:
        """
        Apply a personality preset configuration.
        
        Args:
            preset_name: Name of the preset to apply
            
        Returns:
            True if preset was applied successfully, False otherwise
        """
        presets = self.get_personality_presets()
        
        if preset_name not in presets:
            self.logger.error(f"Unknown personality preset: {preset_name}")
            return False
        
        try:
            preset_config = presets[preset_name]
            self.update_personality_config(preset_config)
            return True
        except Exception as e:
            self.logger.error(f"Failed to apply personality preset {preset_name}: {e}")
            return False
    
    async def generate_test_response(self, test_message: str, platform: str = "test", username: str = "test_user") -> Dict[str, Any]:
        """
        Generate a test response for prompt testing with current personality settings.
        
        Args:
            test_message: Test message content
            platform: Platform name for context
            username: Username for context
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Create test message
            test_chat_message = ChatMessage(
                platform=platform,
                username=username,
                content=test_message,
                timestamp=datetime.now(),
                channel="test_channel"
            )
            
            # Generate response
            response = await self._generate_response(test_chat_message)
            
            if response and response.success:
                formatted = self._format_response(response.content, test_chat_message)
                
                # Include personality information in response
                personality_config = self.config_manager.get_personality_config()
                
                return {
                    'success': True,
                    'response': formatted,
                    'raw_response': response.content,
                    'model': response.model,
                    'backend': response.backend,
                    'personality': {
                        'preset': personality_config.get('preset'),
                        'response_types': personality_config.get('response_types'),
                        'response_length': personality_config.get('response_length')
                    },
                    'metadata': response.metadata
                }
            else:
                return {
                    'success': False,
                    'error': response.error if response else 'Unknown error',
                    'response': None
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response': None
            }