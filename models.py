"""
Core data models for the LLM StreamChat Assistant.

This module defines the primary data structures used throughout the application
for representing chat messages, queued processing items, and response contexts.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid


@dataclass
class ChatMessage:
    """Represents a chat message from any supported platform."""
    
    platform: str  # "twitch" or "youtube"
    username: str
    content: str
    timestamp: datetime
    channel: str
    message_id: Optional[str] = None
    user_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate message ID if not provided."""
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())


@dataclass
class QueuedMessage:
    """Represents a message in the processing queue."""
    
    id: str
    platform: str
    username: str
    content: str
    timestamp: datetime
    channel: str
    priority: int = 1
    processed: bool = False
    retry_count: int = 0
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    @classmethod
    def from_chat_message(cls, chat_message: ChatMessage, priority: int = 1) -> 'QueuedMessage':
        """Create a QueuedMessage from a ChatMessage."""
        return cls(
            id=chat_message.message_id or str(uuid.uuid4()),
            platform=chat_message.platform,
            username=chat_message.username,
            content=chat_message.content,
            timestamp=chat_message.timestamp,
            channel=chat_message.channel,
            priority=priority
        )


@dataclass
class ResponseContext:
    """Context information for generating LLM responses."""
    
    message: ChatMessage
    personality: Dict[str, Any]
    recent_messages: List[ChatMessage] = field(default_factory=list)
    user_history: Dict[str, Any] = field(default_factory=dict)
    response_config: Dict[str, Any] = field(default_factory=dict)
    
    def get_context_string(self) -> str:
        """Generate a context string for LLM prompt."""
        context_parts = []
        
        # Add recent messages for context
        if self.recent_messages:
            context_parts.append("Recent chat context:")
            for msg in self.recent_messages[-5:]:  # Last 5 messages
                context_parts.append(f"{msg.username}: {msg.content}")
        
        # Add current message
        context_parts.append(f"\nCurrent message from {self.message.username}: {self.message.content}")
        
        return "\n".join(context_parts)
    
    def get_personality_prompt(self) -> str:
        """Generate personality-based system prompt."""
        preset = self.personality.get('preset', 'friendly')
        custom_instructions = self.personality.get('custom_instructions', '')
        response_length = self.personality.get('response_length', 'short')
        response_types = self.personality.get('response_types', ['conversational'])
        
        base_prompts = {
            'friendly': "You are a friendly and welcoming streaming assistant. Be warm and engaging.",
            'professional': "You are a professional streaming assistant. Be helpful and informative.",
            'humorous': "You are a humorous streaming assistant. Be witty and entertaining while staying appropriate.",
            'supportive': "You are a supportive streaming assistant. Be encouraging and positive."
        }
        
        prompt = base_prompts.get(preset, base_prompts['friendly'])
        
        # Add response type guidance
        if response_types:
            type_guidance = self._get_response_type_guidance(response_types)
            if type_guidance:
                prompt += f" {type_guidance}"
        
        if custom_instructions:
            prompt += f" {custom_instructions}"
        
        length_instructions = {
            'short': " Keep responses brief (1-2 sentences).",
            'medium': " Keep responses moderate length (2-3 sentences).",
            'long': " You can provide detailed responses when appropriate."
        }
        
        prompt += length_instructions.get(response_length, length_instructions['short'])
        
        return prompt
    
    def _get_response_type_guidance(self, response_types: List[str]) -> str:
        """Generate guidance based on enabled response types."""
        type_descriptions = {
            'conversational': "Engage in natural conversation",
            'informational': "Provide helpful information when relevant",
            'humorous': "Add appropriate humor and wit",
            'supportive': "Be encouraging and positive"
        }
        
        enabled_types = [type_descriptions.get(rt) for rt in response_types if rt in type_descriptions]
        
        if len(enabled_types) == 1:
            return enabled_types[0] + "."
        elif len(enabled_types) > 1:
            return "Focus on: " + ", ".join(enabled_types[:-1]) + f", and {enabled_types[-1]}."
        
        return ""
    
    def should_respond_with_type(self, message_content: str) -> bool:
        """Determine if message warrants a response based on enabled response types."""
        response_types = self.personality.get('response_types', ['conversational'])
        content_lower = message_content.lower()
        
        # Simple heuristics for response type matching
        type_indicators = {
            'conversational': ['hello', 'hi', 'hey', 'how', 'what', 'chat', 'talk'],
            'informational': ['?', 'help', 'how to', 'what is', 'explain', 'tell me'],
            'humorous': ['lol', 'funny', 'joke', 'haha', 'laugh', '😂', '🤣'],
            'supportive': ['thanks', 'thank you', 'good', 'great', 'awesome', 'love', '❤️', '💖']
        }
        
        # Check if message content matches any enabled response types
        for response_type in response_types:
            if response_type in type_indicators:
                indicators = type_indicators[response_type]
                if any(indicator in content_lower for indicator in indicators):
                    return True
        
        # Default to conversational if no specific type matches
        return 'conversational' in response_types