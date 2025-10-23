"""
LLM Client abstraction layer for StreamChat Assistant.

This module provides a unified interface for communicating with different
local LLM backends including Ollama, LM Studio, and Lemonade AI Server.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import json
import requests


logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    """Represents a request to an LLM backend."""
    
    prompt: str
    model: str
    config: Dict[str, Any]
    timeout: float = 30.0
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API requests."""
        data = {
            'prompt': self.prompt,
            'model': self.model,
        }
        
        if self.max_tokens is not None:
            data['max_tokens'] = self.max_tokens
        if self.temperature is not None:
            data['temperature'] = self.temperature
            
        # Add any additional config parameters
        data.update(self.config)
        return data


@dataclass
class LLMResponse:
    """Represents a response from an LLM backend."""
    
    content: str
    model: str
    backend: str
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass


class LLMConnectionError(LLMClientError):
    """Raised when connection to LLM backend fails."""
    pass


class LLMTimeoutError(LLMClientError):
    """Raised when LLM request times out."""
    pass


class LLMValidationError(LLMClientError):
    """Raised when request validation fails."""
    pass


class LLMClient(ABC):
    """
    Abstract base class for LLM backend clients.
    
    Provides a unified interface for communicating with different LLM backends
    while handling common concerns like error handling, timeouts, and connection management.
    """
    
    def __init__(self, base_url: str, timeout: float = 30.0, **kwargs):
        """
        Initialize the LLM client.
        
        Args:
            base_url: Base URL for the LLM backend API
            timeout: Default timeout for requests in seconds
            **kwargs: Additional backend-specific configuration
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.config = kwargs
        self._session = requests.Session()
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
            
    async def close(self):
        """Close the HTTP session."""
        if self._session:
            self._session.close()
            
    @abstractmethod
    async def send_request(self, request: LLMRequest) -> LLMResponse:
        """
        Send a request to the LLM backend.
        
        Args:
            request: The LLM request to send
            
        Returns:
            LLMResponse containing the generated text and metadata
            
        Raises:
            LLMConnectionError: If connection to backend fails
            LLMTimeoutError: If request times out
            LLMValidationError: If request validation fails
        """
        pass
        
    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Test connection to the LLM backend.
        
        Returns:
            True if connection is successful, False otherwise
        """
        pass
        
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the LLM backend.
        
        Returns:
            Dictionary containing health status information
        """
        pass
        
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the backend.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request data for POST requests
            timeout: Request timeout override
            
        Returns:
            Response data as dictionary
            
        Raises:
            LLMConnectionError: If connection fails
            LLMTimeoutError: If request times out
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_timeout = timeout or self.timeout
        
        try:
            # Use asyncio.to_thread to run the synchronous requests call in a thread
            if method.upper() == 'GET':
                response = await asyncio.to_thread(
                    self._session.get, 
                    url, 
                    timeout=request_timeout
                )
            elif method.upper() == 'POST':
                headers = {'Content-Type': 'application/json'}
                response = await asyncio.to_thread(
                    self._session.post, 
                    url, 
                    json=data, 
                    headers=headers, 
                    timeout=request_timeout
                )
            else:
                raise LLMValidationError(f"Unsupported HTTP method: {method}")
                
            return self._handle_response(response)
                    
        except requests.exceptions.Timeout:
            logger.error(f"Request to {url} timed out after {request_timeout}s")
            raise LLMTimeoutError(f"Request timed out after {request_timeout}s")
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection error to {url}: {e}")
            raise LLMConnectionError(f"Failed to connect to LLM backend: {e}")
            
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Handle HTTP response from backend.
        
        Args:
            response: requests response object
            
        Returns:
            Response data as dictionary
            
        Raises:
            LLMConnectionError: If response indicates an error
        """
        try:
            if response.status_code == 200:
                return response.json()
            else:
                error_text = response.text
                logger.error(f"HTTP {response.status_code} error: {error_text}")
                raise LLMConnectionError(f"HTTP {response.status_code}: {error_text}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise LLMConnectionError(f"Invalid JSON response: {e}")
            
    def _validate_request(self, request: LLMRequest) -> None:
        """
        Validate an LLM request.
        
        Args:
            request: The request to validate
            
        Raises:
            LLMValidationError: If validation fails
        """
        if not request.prompt or not request.prompt.strip():
            raise LLMValidationError("Prompt cannot be empty")
            
        if not request.model or not request.model.strip():
            raise LLMValidationError("Model name cannot be empty")
            
        if request.timeout <= 0:
            raise LLMValidationError("Timeout must be positive")
            
        if request.max_tokens is not None and request.max_tokens <= 0:
            raise LLMValidationError("max_tokens must be positive")
            
        if request.temperature is not None and not (0.0 <= request.temperature <= 2.0):
            raise LLMValidationError("temperature must be between 0.0 and 2.0")


class OllamaClient(LLMClient):
    """
    Client for Ollama REST API integration.
    
    Handles communication with Ollama's generate endpoint using HTTP requests.
    Supports streaming and non-streaming responses.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 30.0, **kwargs):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        super().__init__(base_url, timeout, **kwargs)
        self.stream = kwargs.get('stream', False)
        
    async def send_request(self, request: LLMRequest) -> LLMResponse:
        """
        Send a request to Ollama's generate endpoint.
        
        Args:
            request: The LLM request to send
            
        Returns:
            LLMResponse containing the generated text
            
        Raises:
            LLMConnectionError: If connection to Ollama fails
            LLMTimeoutError: If request times out
            LLMValidationError: If request validation fails
        """
        self._validate_request(request)
        
        # Prepare Ollama-specific request format
        ollama_data = {
            'model': request.model,
            'prompt': request.prompt,
            'stream': self.stream
        }
        
        # Add optional parameters if provided
        if request.max_tokens is not None:
            ollama_data['options'] = ollama_data.get('options', {})
            ollama_data['options']['num_predict'] = request.max_tokens
            
        if request.temperature is not None:
            ollama_data['options'] = ollama_data.get('options', {})
            ollama_data['options']['temperature'] = request.temperature
            
        # Add any additional config parameters to options
        if request.config:
            ollama_data['options'] = ollama_data.get('options', {})
            ollama_data['options'].update(request.config)
        
        try:
            logger.debug(f"Sending request to Ollama: {ollama_data}")
            response_data = await self._make_request('POST', '/api/generate', ollama_data, request.timeout)
            
            # Handle Ollama response format
            if 'response' in response_data:
                content = response_data['response']
            elif 'message' in response_data and 'content' in response_data['message']:
                content = response_data['message']['content']
            else:
                logger.error(f"Unexpected Ollama response format: {response_data}")
                raise LLMConnectionError("Unexpected response format from Ollama")
                
            return LLMResponse(
                content=content,
                model=request.model,
                backend='ollama',
                success=True,
                metadata={
                    'done': response_data.get('done', True),
                    'total_duration': response_data.get('total_duration'),
                    'load_duration': response_data.get('load_duration'),
                    'prompt_eval_count': response_data.get('prompt_eval_count'),
                    'eval_count': response_data.get('eval_count'),
                    'eval_duration': response_data.get('eval_duration')
                }
            )
            
        except (LLMConnectionError, LLMTimeoutError):
            # Re-raise these as they're already properly formatted
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Ollama request: {e}")
            return LLMResponse(
                content="",
                model=request.model,
                backend='ollama',
                success=False,
                error=str(e)
            )
            
    async def test_connection(self) -> bool:
        """
        Test connection to Ollama server.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            await self._make_request('GET', '/api/tags', timeout=5.0)
            return True
        except Exception as e:
            logger.warning(f"Ollama connection test failed: {e}")
            return False
            
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on Ollama server.
        
        Returns:
            Dictionary containing health status and available models
        """
        try:
            # Get list of available models
            models_response = await self._make_request('GET', '/api/tags', timeout=10.0)
            models = [model['name'] for model in models_response.get('models', [])]
            
            return {
                'status': 'healthy',
                'backend': 'ollama',
                'base_url': self.base_url,
                'available_models': models,
                'model_count': len(models)
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'backend': 'ollama',
                'base_url': self.base_url,
                'error': str(e),
                'available_models': [],
                'model_count': 0
            }
            
    async def list_models(self) -> list:
        """
        Get list of available models from Ollama.
        
        Returns:
            List of available model names
            
        Raises:
            LLMConnectionError: If unable to retrieve models
        """
        try:
            response = await self._make_request('GET', '/api/tags')
            return [model['name'] for model in response.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            raise LLMConnectionError(f"Failed to retrieve model list: {e}")


class LMStudioClient(LLMClient):
    """
    Client for LM Studio OpenAI-compatible API integration.
    
    Handles communication with LM Studio server using OpenAI-compatible
    chat completion endpoints.
    """
    
    def __init__(self, base_url: str = "http://localhost:1234", timeout: float = 30.0, **kwargs):
        """
        Initialize LM Studio client.
        
        Args:
            base_url: LM Studio server URL (default: http://localhost:1234)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        super().__init__(base_url, timeout, **kwargs)
        self.api_key = kwargs.get('api_key', 'lm-studio')  # LM Studio uses placeholder key
        
    async def send_request(self, request: LLMRequest) -> LLMResponse:
        """
        Send a request to LM Studio's OpenAI-compatible chat completion endpoint.
        
        Args:
            request: The LLM request to send
            
        Returns:
            LLMResponse containing the generated text
            
        Raises:
            LLMConnectionError: If connection to LM Studio fails
            LLMTimeoutError: If request times out
            LLMValidationError: If request validation fails
        """
        self._validate_request(request)
        
        # Convert prompt to OpenAI chat format
        messages = [
            {"role": "user", "content": request.prompt}
        ]
        
        # Prepare OpenAI-compatible request format
        lmstudio_data = {
            'model': request.model,
            'messages': messages,
            'stream': False  # Non-streaming for now
        }
        
        # Add optional parameters if provided
        if request.max_tokens is not None:
            lmstudio_data['max_tokens'] = request.max_tokens
            
        if request.temperature is not None:
            lmstudio_data['temperature'] = request.temperature
            
        # Add any additional config parameters
        if request.config:
            lmstudio_data.update(request.config)
        
        try:
            logger.debug(f"Sending request to LM Studio: {lmstudio_data}")
            
            # Use the base class _make_request method but override headers
            original_session_post = self._session.post
            
            def post_with_auth(*args, **kwargs):
                # Add Authorization header for OpenAI compatibility
                headers = kwargs.get('headers', {})
                headers['Authorization'] = f'Bearer {self.api_key}'
                kwargs['headers'] = headers
                return original_session_post(*args, **kwargs)
            
            # Temporarily replace the post method
            self._session.post = post_with_auth
            
            try:
                response_data = await self._make_request('POST', '/v1/chat/completions', lmstudio_data, request.timeout)
            finally:
                # Restore original post method
                self._session.post = original_session_post
            
            # Handle OpenAI-compatible response format
            if 'choices' in response_data and len(response_data['choices']) > 0:
                choice = response_data['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    content = choice['message']['content']
                else:
                    logger.error(f"Unexpected LM Studio response format: {response_data}")
                    raise LLMConnectionError("Unexpected response format from LM Studio")
            else:
                logger.error(f"No choices in LM Studio response: {response_data}")
                raise LLMConnectionError("No response choices from LM Studio")
                
            return LLMResponse(
                content=content,
                model=request.model,
                backend='lmstudio',
                success=True,
                metadata={
                    'id': response_data.get('id'),
                    'object': response_data.get('object'),
                    'created': response_data.get('created'),
                    'usage': response_data.get('usage', {}),
                    'finish_reason': choice.get('finish_reason')
                }
            )
            
        except (LLMConnectionError, LLMTimeoutError):
            # Re-raise these as they're already properly formatted
            raise
        except requests.exceptions.Timeout:
            logger.error(f"Request to LM Studio timed out after {request.timeout}s")
            raise LLMTimeoutError(f"Request timed out after {request.timeout}s")
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection error to LM Studio: {e}")
            raise LLMConnectionError(f"Failed to connect to LM Studio: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in LM Studio request: {e}")
            return LLMResponse(
                content="",
                model=request.model,
                backend='lmstudio',
                success=False,
                error=str(e)
            )
            
    async def test_connection(self) -> bool:
        """
        Test connection to LM Studio server.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Use the base class _make_request method but override headers
            original_session_get = self._session.get
            
            def get_with_auth(*args, **kwargs):
                # Add Authorization header for OpenAI compatibility
                headers = kwargs.get('headers', {})
                headers['Authorization'] = f'Bearer {self.api_key}'
                kwargs['headers'] = headers
                return original_session_get(*args, **kwargs)
            
            # Temporarily replace the get method
            self._session.get = get_with_auth
            
            try:
                await self._make_request('GET', '/v1/models', timeout=5.0)
                return True
            finally:
                # Restore original get method
                self._session.get = original_session_get
        except Exception as e:
            logger.warning(f"LM Studio connection test failed: {e}")
            return False
            
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on LM Studio server.
        
        Returns:
            Dictionary containing health status and available models
        """
        try:
            # Use the base class _make_request method but override headers
            original_session_get = self._session.get
            
            def get_with_auth(*args, **kwargs):
                # Add Authorization header for OpenAI compatibility
                headers = kwargs.get('headers', {})
                headers['Authorization'] = f'Bearer {self.api_key}'
                kwargs['headers'] = headers
                return original_session_get(*args, **kwargs)
            
            # Temporarily replace the get method
            self._session.get = get_with_auth
            
            try:
                models_response = await self._make_request('GET', '/v1/models', timeout=10.0)
                models = [model['id'] for model in models_response.get('data', [])]
                
                return {
                    'status': 'healthy',
                    'backend': 'lmstudio',
                    'base_url': self.base_url,
                    'available_models': models,
                    'model_count': len(models)
                }
            finally:
                # Restore original get method
                self._session.get = original_session_get
        except Exception as e:
            return {
                'status': 'unhealthy',
                'backend': 'lmstudio',
                'base_url': self.base_url,
                'error': str(e),
                'available_models': [],
                'model_count': 0
            }
            
    async def list_models(self) -> list:
        """
        Get list of available models from LM Studio.
        
        Returns:
            List of available model names
            
        Raises:
            LLMConnectionError: If unable to retrieve models
        """
        try:
            # Use the base class _make_request method but override headers
            original_session_get = self._session.get
            
            def get_with_auth(*args, **kwargs):
                # Add Authorization header for OpenAI compatibility
                headers = kwargs.get('headers', {})
                headers['Authorization'] = f'Bearer {self.api_key}'
                kwargs['headers'] = headers
                return original_session_get(*args, **kwargs)
            
            # Temporarily replace the get method
            self._session.get = get_with_auth
            
            try:
                models_response = await self._make_request('GET', '/v1/models', timeout=10.0)
                return [model['id'] for model in models_response.get('data', [])]
            finally:
                # Restore original get method
                self._session.get = original_session_get
        except Exception as e:
            logger.error(f"Failed to list LM Studio models: {e}")
            raise LLMConnectionError(f"Failed to retrieve model list: {e}")


class LemonadeAIClient(LLMClient):
    """
    Client for Lemonade AI Server OpenAI-compatible API integration.
    
    Handles communication with Lemonade AI Server using OpenAI-compatible
    chat completion endpoints with Lemonade-specific authentication.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0, **kwargs):
        """
        Initialize Lemonade AI client.
        
        Args:
            base_url: Lemonade AI server URL (default: http://localhost:8000)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options including api_key
        """
        super().__init__(base_url, timeout, **kwargs)
        self.api_key = kwargs.get('api_key', 'lemonade-default-key')
        
    async def send_request(self, request: LLMRequest) -> LLMResponse:
        """
        Send a request to Lemonade AI Server's OpenAI-compatible chat completion endpoint.
        
        Args:
            request: The LLM request to send
            
        Returns:
            LLMResponse containing the generated text
            
        Raises:
            LLMConnectionError: If connection to Lemonade AI Server fails
            LLMTimeoutError: If request times out
            LLMValidationError: If request validation fails
        """
        self._validate_request(request)
        
        # Convert prompt to OpenAI chat format
        messages = [
            {"role": "user", "content": request.prompt}
        ]
        
        # Prepare OpenAI-compatible request format for Lemonade AI
        lemonade_data = {
            'model': request.model,
            'messages': messages,
            'stream': False  # Non-streaming for now
        }
        
        # Add optional parameters if provided
        if request.max_tokens is not None:
            lemonade_data['max_tokens'] = request.max_tokens
            
        if request.temperature is not None:
            lemonade_data['temperature'] = request.temperature
            
        # Add any additional config parameters
        if request.config:
            lemonade_data.update(request.config)
        
        try:
            logger.debug(f"Sending request to Lemonade AI Server: {lemonade_data}")
            
            # Use the base class _make_request method but override headers
            original_session_post = self._session.post
            
            def post_with_auth(*args, **kwargs):
                # Add Authorization header for OpenAI compatibility
                headers = kwargs.get('headers', {})
                headers['Authorization'] = f'Bearer {self.api_key}'
                headers['Content-Type'] = 'application/json'
                kwargs['headers'] = headers
                return original_session_post(*args, **kwargs)
            
            # Temporarily replace the post method
            self._session.post = post_with_auth
            
            try:
                response_data = await self._make_request('POST', '/v1/chat/completions', lemonade_data, request.timeout)
            finally:
                # Restore original post method
                self._session.post = original_session_post
            
            # Handle OpenAI-compatible response format
            if 'choices' in response_data and len(response_data['choices']) > 0:
                choice = response_data['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    content = choice['message']['content']
                else:
                    logger.error(f"Unexpected Lemonade AI response format: {response_data}")
                    raise LLMConnectionError("Unexpected response format from Lemonade AI Server")
            else:
                logger.error(f"No choices in Lemonade AI response: {response_data}")
                raise LLMConnectionError("No response choices from Lemonade AI Server")
                
            return LLMResponse(
                content=content,
                model=request.model,
                backend='lemonade',
                success=True,
                metadata={
                    'id': response_data.get('id'),
                    'object': response_data.get('object'),
                    'created': response_data.get('created'),
                    'usage': response_data.get('usage', {}),
                    'finish_reason': choice.get('finish_reason')
                }
            )
            
        except (LLMConnectionError, LLMTimeoutError):
            # Re-raise these as they're already properly formatted
            raise
        except requests.exceptions.Timeout:
            logger.error(f"Request to Lemonade AI Server timed out after {request.timeout}s")
            raise LLMTimeoutError(f"Request timed out after {request.timeout}s")
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection error to Lemonade AI Server: {e}")
            raise LLMConnectionError(f"Failed to connect to Lemonade AI Server: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in Lemonade AI request: {e}")
            return LLMResponse(
                content="",
                model=request.model,
                backend='lemonade',
                success=False,
                error=str(e)
            )
            
    async def test_connection(self) -> bool:
        """
        Test connection to Lemonade AI Server.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Use the base class _make_request method but override headers
            original_session_get = self._session.get
            
            def get_with_auth(*args, **kwargs):
                # Add Authorization header for OpenAI compatibility
                headers = kwargs.get('headers', {})
                headers['Authorization'] = f'Bearer {self.api_key}'
                kwargs['headers'] = headers
                return original_session_get(*args, **kwargs)
            
            # Temporarily replace the get method
            self._session.get = get_with_auth
            
            try:
                await self._make_request('GET', '/v1/models', timeout=5.0)
                return True
            finally:
                # Restore original get method
                self._session.get = original_session_get
        except Exception as e:
            logger.warning(f"Lemonade AI Server connection test failed: {e}")
            return False
            
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on Lemonade AI Server.
        
        Returns:
            Dictionary containing health status and available models
        """
        try:
            # Use the base class _make_request method but override headers
            original_session_get = self._session.get
            
            def get_with_auth(*args, **kwargs):
                # Add Authorization header for OpenAI compatibility
                headers = kwargs.get('headers', {})
                headers['Authorization'] = f'Bearer {self.api_key}'
                kwargs['headers'] = headers
                return original_session_get(*args, **kwargs)
            
            # Temporarily replace the get method
            self._session.get = get_with_auth
            
            try:
                models_response = await self._make_request('GET', '/v1/models', timeout=10.0)
                models = [model['id'] for model in models_response.get('data', [])]
                
                return {
                    'status': 'healthy',
                    'backend': 'lemonade',
                    'base_url': self.base_url,
                    'available_models': models,
                    'model_count': len(models)
                }
            finally:
                # Restore original get method
                self._session.get = original_session_get
        except Exception as e:
            return {
                'status': 'unhealthy',
                'backend': 'lemonade',
                'base_url': self.base_url,
                'error': str(e),
                'available_models': [],
                'model_count': 0
            }
            
    async def list_models(self) -> list:
        """
        Get list of available models from Lemonade AI Server.
        
        Returns:
            List of available model names
            
        Raises:
            LLMConnectionError: If unable to retrieve models
        """
        try:
            # Use the base class _make_request method but override headers
            original_session_get = self._session.get
            
            def get_with_auth(*args, **kwargs):
                # Add Authorization header for OpenAI compatibility
                headers = kwargs.get('headers', {})
                headers['Authorization'] = f'Bearer {self.api_key}'
                kwargs['headers'] = headers
                return original_session_get(*args, **kwargs)
            
            # Temporarily replace the get method
            self._session.get = get_with_auth
            
            try:
                models_response = await self._make_request('GET', '/v1/models', timeout=10.0)
                return [model['id'] for model in models_response.get('data', [])]
            finally:
                # Restore original get method
                self._session.get = original_session_get
        except Exception as e:
            logger.error(f"Failed to list Lemonade AI models: {e}")
            raise LLMConnectionError(f"Failed to retrieve model list: {e}")


class LLMClientFactory:
    """Factory for creating LLM client instances."""
    
    _clients = {
        'ollama': OllamaClient,
        'lmstudio': LMStudioClient,
        'lemonade': LemonadeAIClient
    }
    
    @classmethod
    def register_client(cls, backend_name: str, client_class: type):
        """Register a client class for a backend."""
        cls._clients[backend_name.lower()] = client_class
        
    @classmethod
    def create_client(cls, backend: str, **kwargs) -> LLMClient:
        """
        Create an LLM client instance.
        
        Args:
            backend: Backend name (ollama, lmstudio, lemonade)
            **kwargs: Client configuration parameters
            
        Returns:
            LLMClient instance
            
        Raises:
            ValueError: If backend is not supported
        """
        backend_lower = backend.lower()
        if backend_lower not in cls._clients:
            available = ', '.join(cls._clients.keys())
            raise ValueError(f"Unsupported backend '{backend}'. Available: {available}")
            
        client_class = cls._clients[backend_lower]
        return client_class(**kwargs)