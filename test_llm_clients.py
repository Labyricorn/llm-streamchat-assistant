#!/usr/bin/env python3
"""
Unit tests for LLM client implementations.

Tests all LLM client classes with mocked backend responses and error scenarios.
Covers requirements 3.1, 3.2, 3.3, 3.4 from the specification.
"""

import asyncio
import pytest
import sys
from unittest.mock import AsyncMock, patch, MagicMock
import requests
import json

from llm_client import (
    LLMClient, OllamaClient, LMStudioClient, LemonadeAIClient, LLMClientFactory,
    LLMRequest, LLMResponse, LLMClientError, LLMConnectionError, 
    LLMTimeoutError, LLMValidationError
)


class TestLLMRequest:
    """Test LLMRequest data class."""
    
    def test_llm_request_creation(self):
        """Test basic LLMRequest creation."""
        request = LLMRequest(
            prompt="Test prompt",
            model="test-model",
            config={"param": "value"}
        )
        
        assert request.prompt == "Test prompt"
        assert request.model == "test-model"
        assert request.config == {"param": "value"}
        assert request.timeout == 30.0
        assert request.max_tokens is None
        assert request.temperature is None
    
    def test_llm_request_to_dict(self):
        """Test LLMRequest to_dict conversion."""
        request = LLMRequest(
            prompt="Test prompt",
            model="test-model",
            config={"custom": "param"},
            max_tokens=100,
            temperature=0.7
        )
        
        result = request.to_dict()
        expected = {
            'prompt': 'Test prompt',
            'model': 'test-model',
            'max_tokens': 100,
            'temperature': 0.7,
            'custom': 'param'
        }
        
        assert result == expected


class TestLLMResponse:
    """Test LLMResponse data class."""
    
    def test_llm_response_creation(self):
        """Test basic LLMResponse creation."""
        response = LLMResponse(
            content="Test response",
            model="test-model",
            backend="test-backend"
        )
        
        assert response.content == "Test response"
        assert response.model == "test-model"
        assert response.backend == "test-backend"
        assert response.success == True
        assert response.error is None
        assert response.metadata == {}
    
    def test_llm_response_with_error(self):
        """Test LLMResponse with error."""
        response = LLMResponse(
            content="",
            model="test-model",
            backend="test-backend",
            success=False,
            error="Test error",
            metadata={"key": "value"}
        )
        
        assert response.success == False
        assert response.error == "Test error"
        assert response.metadata == {"key": "value"}


class TestOllamaClient:
    """Test OllamaClient implementation."""
    
    def test_initialization(self):
        """Test OllamaClient initialization."""
        # Default initialization
        client = OllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.timeout == 30.0
        assert client.stream == False
        
        # Custom initialization
        client = OllamaClient(
            base_url="http://custom:8080",
            timeout=60.0,
            stream=True
        )
        assert client.base_url == "http://custom:8080"
        assert client.timeout == 60.0
        assert client.stream == True
    
    @pytest.mark.asyncio
    async def test_send_request_success(self):
        """Test successful Ollama request."""
        client = OllamaClient()
        
        mock_response = {
            'response': 'Generated response text',
            'done': True,
            'total_duration': 1000000,
            'eval_count': 15,
            'prompt_eval_count': 5
        }
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            request = LLMRequest(
                prompt="Hello, world!",
                model="llama2",
                config={},
                max_tokens=100,
                temperature=0.7
            )
            
            response = await client.send_request(request)
            
            # Verify request formatting
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == 'POST'
            assert call_args[0][1] == '/api/generate'
            
            request_data = call_args[0][2]
            assert request_data['model'] == 'llama2'
            assert request_data['prompt'] == 'Hello, world!'
            assert request_data['stream'] == False
            assert request_data['options']['num_predict'] == 100
            assert request_data['options']['temperature'] == 0.7
            
            # Verify response
            assert isinstance(response, LLMResponse)
            assert response.content == 'Generated response text'
            assert response.model == 'llama2'
            assert response.backend == 'ollama'
            assert response.success == True
            assert response.metadata['done'] == True
            assert response.metadata['eval_count'] == 15
    
    @pytest.mark.asyncio
    async def test_send_request_connection_error(self):
        """Test Ollama request with connection error."""
        client = OllamaClient()
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = LLMConnectionError("Connection failed")
            
            request = LLMRequest(
                prompt="Test prompt",
                model="llama2",
                config={}
            )
            
            with pytest.raises(LLMConnectionError):
                await client.send_request(request)
    
    @pytest.mark.asyncio
    async def test_send_request_timeout_error(self):
        """Test Ollama request with timeout error."""
        client = OllamaClient()
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = LLMTimeoutError("Request timed out")
            
            request = LLMRequest(
                prompt="Test prompt",
                model="llama2",
                config={}
            )
            
            with pytest.raises(LLMTimeoutError):
                await client.send_request(request)
    
    @pytest.mark.asyncio
    async def test_send_request_validation_error(self):
        """Test Ollama request with validation error."""
        client = OllamaClient()
        
        # Empty prompt should raise validation error
        request = LLMRequest(
            prompt="",
            model="llama2",
            config={}
        )
        
        with pytest.raises(LLMValidationError):
            await client.send_request(request)
        
        # Empty model should raise validation error
        request = LLMRequest(
            prompt="Test prompt",
            model="",
            config={}
        )
        
        with pytest.raises(LLMValidationError):
            await client.send_request(request)
    
    @pytest.mark.asyncio
    async def test_send_request_unexpected_response_format(self):
        """Test Ollama request with unexpected response format."""
        client = OllamaClient()
        
        # Response without 'response' or 'message' field
        mock_response = {
            'done': True,
            'total_duration': 1000000
        }
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            request = LLMRequest(
                prompt="Test prompt",
                model="llama2",
                config={}
            )
            
            with pytest.raises(LLMConnectionError, match="Unexpected response format"):
                await client.send_request(request)
    
    @pytest.mark.asyncio
    async def test_test_connection_success(self):
        """Test successful Ollama connection test."""
        client = OllamaClient()
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {'models': []}
            
            result = await client.test_connection()
            assert result == True
            mock_request.assert_called_once_with('GET', '/api/tags', timeout=5.0)
    
    @pytest.mark.asyncio
    async def test_test_connection_failure(self):
        """Test failed Ollama connection test."""
        client = OllamaClient()
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("Connection failed")
            
            result = await client.test_connection()
            assert result == False
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful Ollama health check."""
        client = OllamaClient()
        
        mock_models_response = {
            'models': [
                {'name': 'llama2'},
                {'name': 'codellama'},
                {'name': 'mistral'}
            ]
        }
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_models_response
            
            health = await client.health_check()
            
            assert health['status'] == 'healthy'
            assert health['backend'] == 'ollama'
            assert health['model_count'] == 3
            assert 'llama2' in health['available_models']
            assert 'codellama' in health['available_models']
            assert 'mistral' in health['available_models']
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed Ollama health check."""
        client = OllamaClient()
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("Connection failed")
            
            health = await client.health_check()
            
            assert health['status'] == 'unhealthy'
            assert health['backend'] == 'ollama'
            assert health['model_count'] == 0
            assert 'Connection failed' in health['error']


class TestLMStudioClient:
    """Test LMStudioClient implementation."""
    
    def test_initialization(self):
        """Test LMStudioClient initialization."""
        # Default initialization
        client = LMStudioClient()
        assert client.base_url == "http://localhost:1234"
        assert client.timeout == 30.0
        assert client.api_key == "lm-studio"
        
        # Custom initialization
        client = LMStudioClient(
            base_url="http://custom:5678",
            timeout=60.0,
            api_key="custom-key"
        )
        assert client.base_url == "http://custom:5678"
        assert client.timeout == 60.0
        assert client.api_key == "custom-key"
    
    @pytest.mark.asyncio
    async def test_send_request_success(self):
        """Test successful LM Studio request."""
        client = LMStudioClient()
        
        mock_response = {
            'id': 'chatcmpl-123',
            'object': 'chat.completion',
            'created': 1677652288,
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'Generated response from LM Studio'
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 15,
                'total_tokens': 25
            }
        }
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            request = LLMRequest(
                prompt="Hello, world!",
                model="llama-2-7b-chat",
                config={},
                max_tokens=100,
                temperature=0.7
            )
            
            response = await client.send_request(request)
            
            # Verify request formatting
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == 'POST'
            assert call_args[0][1] == '/v1/chat/completions'
            
            request_data = call_args[0][2]
            assert request_data['model'] == 'llama-2-7b-chat'
            assert request_data['messages'][0]['role'] == 'user'
            assert request_data['messages'][0]['content'] == 'Hello, world!'
            assert request_data['stream'] == False
            assert request_data['max_tokens'] == 100
            assert request_data['temperature'] == 0.7
            
            # Verify response
            assert isinstance(response, LLMResponse)
            assert response.content == 'Generated response from LM Studio'
            assert response.model == 'llama-2-7b-chat'
            assert response.backend == 'lmstudio'
            assert response.success == True
            assert response.metadata['id'] == 'chatcmpl-123'
            assert response.metadata['usage']['total_tokens'] == 25
    
    @pytest.mark.asyncio
    async def test_send_request_no_choices(self):
        """Test LM Studio request with no choices in response."""
        client = LMStudioClient()
        
        mock_response = {
            'id': 'chatcmpl-123',
            'object': 'chat.completion',
            'choices': []
        }
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            request = LLMRequest(
                prompt="Test prompt",
                model="llama-2-7b-chat",
                config={}
            )
            
            with pytest.raises(LLMConnectionError, match="No response choices"):
                await client.send_request(request)
    
    @pytest.mark.asyncio
    async def test_send_request_timeout_error(self):
        """Test LM Studio request with timeout error."""
        client = LMStudioClient()
        
        with patch.object(client, '_session') as mock_session:
            mock_session.post.side_effect = requests.exceptions.Timeout("Request timed out")
            
            request = LLMRequest(
                prompt="Test prompt",
                model="llama-2-7b-chat",
                config={}
            )
            
            with pytest.raises(LLMTimeoutError):
                await client.send_request(request)
    
    @pytest.mark.asyncio
    async def test_send_request_connection_error(self):
        """Test LM Studio request with connection error."""
        client = LMStudioClient()
        
        with patch.object(client, '_session') as mock_session:
            mock_session.post.side_effect = requests.exceptions.ConnectionError("Connection failed")
            
            request = LLMRequest(
                prompt="Test prompt",
                model="llama-2-7b-chat",
                config={}
            )
            
            with pytest.raises(LLMConnectionError):
                await client.send_request(request)
    
    @pytest.mark.asyncio
    async def test_test_connection_success(self):
        """Test successful LM Studio connection test."""
        client = LMStudioClient()
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {'object': 'list', 'data': []}
            
            result = await client.test_connection()
            assert result == True
            mock_request.assert_called_once_with('GET', '/v1/models', timeout=5.0)
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful LM Studio health check."""
        client = LMStudioClient()
        
        mock_models_response = {
            'object': 'list',
            'data': [
                {'id': 'llama-2-7b-chat', 'object': 'model'},
                {'id': 'codellama-7b-instruct', 'object': 'model'}
            ]
        }
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_models_response
            
            health = await client.health_check()
            
            assert health['status'] == 'healthy'
            assert health['backend'] == 'lmstudio'
            assert health['model_count'] == 2
            assert 'llama-2-7b-chat' in health['available_models']
            assert 'codellama-7b-instruct' in health['available_models']


class TestLemonadeAIClient:
    """Test LemonadeAIClient implementation."""
    
    def test_initialization(self):
        """Test LemonadeAIClient initialization."""
        # Default initialization
        client = LemonadeAIClient()
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 30.0
        assert client.api_key == "lemonade-default-key"
        
        # Custom initialization
        client = LemonadeAIClient(
            base_url="http://custom:9000",
            timeout=60.0,
            api_key="custom-lemonade-key"
        )
        assert client.base_url == "http://custom:9000"
        assert client.timeout == 60.0
        assert client.api_key == "custom-lemonade-key"
    
    @pytest.mark.asyncio
    async def test_send_request_success(self):
        """Test successful Lemonade AI request."""
        client = LemonadeAIClient()
        
        mock_response = {
            'id': 'lemonade-123',
            'object': 'chat.completion',
            'created': 1677652288,
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'Generated response from Lemonade AI'
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': 8,
                'completion_tokens': 12,
                'total_tokens': 20
            }
        }
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            request = LLMRequest(
                prompt="Hello, world!",
                model="lemonade-model",
                config={},
                max_tokens=100,
                temperature=0.7
            )
            
            response = await client.send_request(request)
            
            # Verify request formatting
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == 'POST'
            assert call_args[0][1] == '/v1/chat/completions'
            
            request_data = call_args[0][2]
            assert request_data['model'] == 'lemonade-model'
            assert request_data['messages'][0]['role'] == 'user'
            assert request_data['messages'][0]['content'] == 'Hello, world!'
            assert request_data['stream'] == False
            assert request_data['max_tokens'] == 100
            assert request_data['temperature'] == 0.7
            
            # Verify response
            assert isinstance(response, LLMResponse)
            assert response.content == 'Generated response from Lemonade AI'
            assert response.model == 'lemonade-model'
            assert response.backend == 'lemonade'
            assert response.success == True
            assert response.metadata['id'] == 'lemonade-123'
            assert response.metadata['usage']['total_tokens'] == 20
    
    @pytest.mark.asyncio
    async def test_send_request_unexpected_format(self):
        """Test Lemonade AI request with unexpected response format."""
        client = LemonadeAIClient()
        
        mock_response = {
            'id': 'lemonade-123',
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant'
                    # Missing 'content' field
                }
            }]
        }
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            request = LLMRequest(
                prompt="Test prompt",
                model="lemonade-model",
                config={}
            )
            
            with pytest.raises(LLMConnectionError, match="Unexpected response format"):
                await client.send_request(request)
    
    @pytest.mark.asyncio
    async def test_test_connection_success(self):
        """Test successful Lemonade AI connection test."""
        client = LemonadeAIClient()
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {'object': 'list', 'data': []}
            
            result = await client.test_connection()
            assert result == True
            mock_request.assert_called_once_with('GET', '/v1/models', timeout=5.0)
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful Lemonade AI health check."""
        client = LemonadeAIClient()
        
        mock_models_response = {
            'object': 'list',
            'data': [
                {'id': 'lemonade-7b', 'object': 'model'},
                {'id': 'lemonade-13b', 'object': 'model'}
            ]
        }
        
        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_models_response
            
            health = await client.health_check()
            
            assert health['status'] == 'healthy'
            assert health['backend'] == 'lemonade'
            assert health['model_count'] == 2
            assert 'lemonade-7b' in health['available_models']
            assert 'lemonade-13b' in health['available_models']


class TestLLMClientFactory:
    """Test LLMClientFactory functionality."""
    
    def test_create_ollama_client(self):
        """Test creating Ollama client via factory."""
        client = LLMClientFactory.create_client('ollama', base_url='http://test:1234')
        assert isinstance(client, OllamaClient)
        assert client.base_url == 'http://test:1234'
    
    def test_create_lmstudio_client(self):
        """Test creating LM Studio client via factory."""
        client = LLMClientFactory.create_client('lmstudio', base_url='http://test:5678')
        assert isinstance(client, LMStudioClient)
        assert client.base_url == 'http://test:5678'
    
    def test_create_lemonade_client(self):
        """Test creating Lemonade AI client via factory."""
        client = LLMClientFactory.create_client('lemonade', base_url='http://test:9000')
        assert isinstance(client, LemonadeAIClient)
        assert client.base_url == 'http://test:9000'
    
    def test_case_insensitive_backend_names(self):
        """Test that backend names are case insensitive."""
        client1 = LLMClientFactory.create_client('OLLAMA')
        assert isinstance(client1, OllamaClient)
        
        client2 = LLMClientFactory.create_client('LMStudio')
        assert isinstance(client2, LMStudioClient)
        
        client3 = LLMClientFactory.create_client('LEMONADE')
        assert isinstance(client3, LemonadeAIClient)
    
    def test_unsupported_backend(self):
        """Test error handling for unsupported backend."""
        with pytest.raises(ValueError, match="Unsupported backend 'unknown'"):
            LLMClientFactory.create_client('unknown')
    
    def test_register_custom_client(self):
        """Test registering a custom client class."""
        class CustomClient(LLMClient):
            def __init__(self, base_url="http://localhost:8080", **kwargs):
                super().__init__(base_url, **kwargs)
            
            async def send_request(self, request):
                pass
            async def test_connection(self):
                pass
            async def health_check(self):
                pass
        
        LLMClientFactory.register_client('custom', CustomClient)
        client = LLMClientFactory.create_client('custom')
        assert isinstance(client, CustomClient)


class TestLLMClientValidation:
    """Test LLM client request validation."""
    
    def test_validate_empty_prompt(self):
        """Test validation of empty prompt."""
        client = OllamaClient()
        
        request = LLMRequest(
            prompt="",
            model="test-model",
            config={}
        )
        
        with pytest.raises(LLMValidationError, match="Prompt cannot be empty"):
            client._validate_request(request)
    
    def test_validate_empty_model(self):
        """Test validation of empty model."""
        client = OllamaClient()
        
        request = LLMRequest(
            prompt="Test prompt",
            model="",
            config={}
        )
        
        with pytest.raises(LLMValidationError, match="Model name cannot be empty"):
            client._validate_request(request)
    
    def test_validate_negative_timeout(self):
        """Test validation of negative timeout."""
        client = OllamaClient()
        
        request = LLMRequest(
            prompt="Test prompt",
            model="test-model",
            config={},
            timeout=-1.0
        )
        
        with pytest.raises(LLMValidationError, match="Timeout must be positive"):
            client._validate_request(request)
    
    def test_validate_negative_max_tokens(self):
        """Test validation of negative max_tokens."""
        client = OllamaClient()
        
        request = LLMRequest(
            prompt="Test prompt",
            model="test-model",
            config={},
            max_tokens=-10
        )
        
        with pytest.raises(LLMValidationError, match="max_tokens must be positive"):
            client._validate_request(request)
    
    def test_validate_invalid_temperature(self):
        """Test validation of invalid temperature."""
        client = OllamaClient()
        
        # Temperature too low
        request = LLMRequest(
            prompt="Test prompt",
            model="test-model",
            config={},
            temperature=-0.1
        )
        
        with pytest.raises(LLMValidationError, match="temperature must be between 0.0 and 2.0"):
            client._validate_request(request)
        
        # Temperature too high
        request = LLMRequest(
            prompt="Test prompt",
            model="test-model",
            config={},
            temperature=2.1
        )
        
        with pytest.raises(LLMValidationError, match="temperature must be between 0.0 and 2.0"):
            client._validate_request(request)
    
    def test_validate_valid_request(self):
        """Test validation of valid request."""
        client = OllamaClient()
        
        request = LLMRequest(
            prompt="Test prompt",
            model="test-model",
            config={},
            timeout=30.0,
            max_tokens=100,
            temperature=0.7
        )
        
        # Should not raise any exception
        client._validate_request(request)


if __name__ == "__main__":
    # Run tests using pytest if available, otherwise run basic tests
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, running basic tests...")
        asyncio.run(test_basic_functionality())


@pytest.mark.asyncio
async def test_basic_functionality():
    """Basic test runner when pytest is not available."""
    print("Running basic LLM client tests...")
    
    # Test data classes
    request = LLMRequest("test", "model", {})
    assert request.prompt == "test"
    
    response = LLMResponse("content", "model", "backend")
    assert response.content == "content"
    
    # Test factory
    client = LLMClientFactory.create_client('ollama')
    assert isinstance(client, OllamaClient)
    
    print("✓ Basic tests passed")