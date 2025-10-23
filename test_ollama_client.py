#!/usr/bin/env python3
"""
Simple test script for OllamaClient implementation.

This script tests the basic functionality of the OllamaClient without requiring
an actual Ollama server to be running.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, patch
from llm_client import OllamaClient, LMStudioClient, LLMRequest, LLMResponse, LLMClientFactory


async def test_ollama_client_initialization():
    """Test OllamaClient initialization."""
    print("Testing OllamaClient initialization...")
    
    # Test default initialization
    client = OllamaClient()
    assert client.base_url == "http://localhost:11434"
    assert client.timeout == 30.0
    assert client.stream == False
    
    # Test custom initialization
    client = OllamaClient(
        base_url="http://custom:8080", 
        timeout=60.0, 
        stream=True
    )
    assert client.base_url == "http://custom:8080"
    assert client.timeout == 60.0
    assert client.stream == True
    
    print("✓ OllamaClient initialization tests passed")


async def test_ollama_request_formatting():
    """Test Ollama-specific request formatting."""
    print("Testing Ollama request formatting...")
    
    client = OllamaClient()
    
    # Mock the _make_request method to capture the formatted request
    mock_response = {
        'response': 'Test response',
        'done': True,
        'total_duration': 1000000,
        'eval_count': 10
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
        
        # Verify the request was formatted correctly for Ollama
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        
        assert call_args[0][0] == 'POST'  # method
        assert call_args[0][1] == '/api/generate'  # endpoint
        
        request_data = call_args[0][2]  # data
        assert request_data['model'] == 'llama2'
        assert request_data['prompt'] == 'Hello, world!'
        assert request_data['stream'] == False
        assert request_data['options']['num_predict'] == 100
        assert request_data['options']['temperature'] == 0.7
        
        # Verify response formatting
        assert isinstance(response, LLMResponse)
        assert response.content == 'Test response'
        assert response.model == 'llama2'
        assert response.backend == 'ollama'
        assert response.success == True
        assert response.metadata['done'] == True
        assert response.metadata['eval_count'] == 10
    
    print("✓ Ollama request formatting tests passed")


async def test_ollama_health_check():
    """Test Ollama health check functionality."""
    print("Testing Ollama health check...")
    
    client = OllamaClient()
    
    # Mock successful health check
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
    
    # Mock failed health check
    with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
        mock_request.side_effect = Exception("Connection failed")
        
        health = await client.health_check()
        
        assert health['status'] == 'unhealthy'
        assert health['backend'] == 'ollama'
        assert health['model_count'] == 0
        assert 'Connection failed' in health['error']
    
    print("✓ Ollama health check tests passed")


async def test_ollama_connection_test():
    """Test Ollama connection testing."""
    print("Testing Ollama connection test...")
    
    client = OllamaClient()
    
    # Mock successful connection
    with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
        mock_request.return_value = {'models': []}
        
        result = await client.test_connection()
        assert result == True
        mock_request.assert_called_once_with('GET', '/api/tags', timeout=5.0)
    
    # Mock failed connection
    with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
        mock_request.side_effect = Exception("Connection failed")
        
        result = await client.test_connection()
        assert result == False
    
    print("✓ Ollama connection test passed")


async def test_lmstudio_client_initialization():
    """Test LMStudioClient initialization."""
    print("Testing LMStudioClient initialization...")
    
    # Test default initialization
    client = LMStudioClient()
    assert client.base_url == "http://localhost:1234"
    assert client.timeout == 30.0
    assert client.api_key == "lm-studio"
    
    # Test custom initialization
    client = LMStudioClient(
        base_url="http://custom:5678", 
        timeout=60.0, 
        api_key="custom-key"
    )
    assert client.base_url == "http://custom:5678"
    assert client.timeout == 60.0
    assert client.api_key == "custom-key"
    
    print("✓ LMStudioClient initialization tests passed")


async def test_lmstudio_request_formatting():
    """Test LM Studio OpenAI-compatible request formatting."""
    print("Testing LM Studio request formatting...")
    
    client = LMStudioClient()
    
    # Mock response data in OpenAI format
    mock_response_data = {
        'id': 'chatcmpl-123',
        'object': 'chat.completion',
        'created': 1677652288,
        'choices': [{
            'index': 0,
            'message': {
                'role': 'assistant',
                'content': 'Test response from LM Studio'
            },
            'finish_reason': 'stop'
        }],
        'usage': {
            'prompt_tokens': 10,
            'completion_tokens': 5,
            'total_tokens': 15
        }
    }
    
    # Mock the _make_request method to capture the formatted request
    with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response_data
        
        request = LLMRequest(
            prompt="Hello, world!",
            model="llama-2-7b-chat",
            config={},
            max_tokens=100,
            temperature=0.7
        )
        
        response = await client.send_request(request)
        
        # Verify the request was formatted correctly for OpenAI API
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        
        assert call_args[0][0] == 'POST'  # method
        assert call_args[0][1] == '/v1/chat/completions'  # endpoint
        
        request_data = call_args[0][2]  # data
        assert request_data['model'] == 'llama-2-7b-chat'
        assert request_data['messages'][0]['role'] == 'user'
        assert request_data['messages'][0]['content'] == 'Hello, world!'
        assert request_data['stream'] == False
        assert request_data['max_tokens'] == 100
        assert request_data['temperature'] == 0.7
        
        # Verify response formatting
        assert isinstance(response, LLMResponse)
        assert response.content == 'Test response from LM Studio'
        assert response.model == 'llama-2-7b-chat'
        assert response.backend == 'lmstudio'
        assert response.success == True
        assert response.metadata['id'] == 'chatcmpl-123'
        assert response.metadata['usage']['total_tokens'] == 15
    
    print("✓ LM Studio request formatting tests passed")


async def test_lmstudio_health_check():
    """Test LM Studio health check functionality."""
    print("Testing LM Studio health check...")
    
    client = LMStudioClient()
    
    # Mock successful health check
    mock_models_response = {
        'object': 'list',
        'data': [
            {'id': 'llama-2-7b-chat', 'object': 'model'},
            {'id': 'codellama-7b-instruct', 'object': 'model'},
            {'id': 'mistral-7b-instruct', 'object': 'model'}
        ]
    }
    
    with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_models_response
        
        health = await client.health_check()
        
        assert health['status'] == 'healthy'
        assert health['backend'] == 'lmstudio'
        assert health['model_count'] == 3
        assert 'llama-2-7b-chat' in health['available_models']
        assert 'codellama-7b-instruct' in health['available_models']
        assert 'mistral-7b-instruct' in health['available_models']
    
    # Mock failed health check
    with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
        mock_request.side_effect = Exception("Connection failed")
        
        health = await client.health_check()
        
        assert health['status'] == 'unhealthy'
        assert health['backend'] == 'lmstudio'
        assert health['model_count'] == 0
        assert 'Connection failed' in health['error']
    
    print("✓ LM Studio health check tests passed")


async def test_lmstudio_connection_test():
    """Test LM Studio connection testing."""
    print("Testing LM Studio connection test...")
    
    client = LMStudioClient()
    
    # Mock successful connection
    with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
        mock_request.return_value = {'object': 'list', 'data': []}
        
        result = await client.test_connection()
        assert result == True
        mock_request.assert_called_once_with('GET', '/v1/models', timeout=5.0)
    
    # Mock failed connection
    with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
        mock_request.side_effect = Exception("Connection failed")
        
        result = await client.test_connection()
        assert result == False
    
    print("✓ LM Studio connection test passed")


async def test_factory_registration():
    """Test that both clients are properly registered in the factory."""
    print("Testing factory registration...")
    
    # Test that ollama client can be created via factory
    client = LLMClientFactory.create_client('ollama', base_url='http://test:1234')
    assert isinstance(client, OllamaClient)
    assert client.base_url == 'http://test:1234'
    
    # Test that lmstudio client can be created via factory
    client = LLMClientFactory.create_client('lmstudio', base_url='http://test:5678')
    assert isinstance(client, LMStudioClient)
    assert client.base_url == 'http://test:5678'
    
    # Test case insensitive
    client = LLMClientFactory.create_client('OLLAMA')
    assert isinstance(client, OllamaClient)
    
    client = LLMClientFactory.create_client('LMSTUDIO')
    assert isinstance(client, LMStudioClient)
    
    print("✓ Factory registration tests passed")


async def main():
    """Run all tests."""
    print("Running LLM Client tests...\n")
    
    try:
        # Ollama tests
        await test_ollama_client_initialization()
        await test_ollama_request_formatting()
        await test_ollama_health_check()
        await test_ollama_connection_test()
        
        # LM Studio tests
        await test_lmstudio_client_initialization()
        await test_lmstudio_request_formatting()
        await test_lmstudio_health_check()
        await test_lmstudio_connection_test()
        
        # Factory tests
        await test_factory_registration()
        
        print("\n🎉 All LLM Client tests passed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))