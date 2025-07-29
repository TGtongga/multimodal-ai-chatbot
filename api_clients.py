"""
API clients for different AI providers
Handles communication with OpenAI, Gemini, and Anthropic APIs
"""

import os
import asyncio
from typing import AsyncGenerator, List, Dict, Any, Optional
import aiohttp
import json
import base64
import logging
from abc import ABC, abstractmethod

# Configure logging
logger = logging.getLogger(__name__)

class BaseClient(ABC):
    """Abstract base class for AI API clients"""
    
    def __init__(self, api_key: str, api_url: str):
        self.api_key = api_key
        self.api_url = api_url
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    @abstractmethod
    async def stream_chat(
        self, 
        messages: List[Dict[str, Any]], 
        model: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion"""
        pass
    
    def _format_messages_for_api(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for API - to be overridden by subclasses"""
        return messages

class OpenAIClient(BaseClient):
    """OpenAI API client with streaming support"""
    
    def __init__(self):
        super().__init__(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_url=os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
        )
    
    def _format_messages_for_api(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for OpenAI API"""
        formatted_messages = []
        
        for msg in messages:
            formatted_msg = {"role": msg["role"]}
            
            if msg.get("images"):
                # Handle multimodal content (images)
                content = [{"type": "text", "text": msg["content"]}]
                for image_data in msg["images"]:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    })
                formatted_msg["content"] = content
            else:
                # Text-only content (including documents embedded in text)
                formatted_msg["content"] = msg["content"]
            
            formatted_messages.append(formatted_msg)
        
        return formatted_messages
    
    async def stream_chat(
        self, 
        messages: List[Dict[str, Any]], 
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion from OpenAI"""
        
        session = await self._get_session()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Use the dynamic parameters passed from the UI
        data = {
            "model": model,
            "messages": self._format_messages_for_api(messages),
            "stream": True,
            "temperature": float(temperature),  # Ensure proper type conversion
            "max_tokens": int(max_tokens)       # Ensure proper type conversion
        }
        
        try:
            async with session.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=data
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    yield f"OpenAI API Error: {response.status} - {error_text}"
                    return
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if line.startswith('data: '):
                        line = line[6:]  # Remove 'data: ' prefix
                        
                        if line == '[DONE]':
                            break
                        
                        try:
                            chunk = json.loads(line)
                            if 'choices' in chunk and chunk['choices']:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            continue
                        
        except Exception as e:
            yield f"OpenAI streaming error: {str(e)}"


class GeminiClient(BaseClient):
    """Google Gemini API client with streaming support"""
    
    def __init__(self):
        super().__init__(
            api_key=os.getenv("GEMINI_API_KEY"),
            api_url=os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta")
        )
    
    def _format_messages_for_api(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for Gemini API"""
        formatted_messages = []
        
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            
            if msg.get("images"):
                # Handle multimodal content (images)
                parts = [{"text": msg["content"]}]
                for image_data in msg["images"]:
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_data
                        }
                    })
                formatted_messages.append({"role": role, "parts": parts})
            else:
                # Text-only content (including documents embedded in text)
                formatted_messages.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })
        
        return formatted_messages
    
    async def stream_chat(
        self, 
        messages: List[Dict[str, Any]], 
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion from Gemini"""
        
        session = await self._get_session()
        
        url = f"{self.api_url}/models/{model}:streamGenerateContent"
        params = {"key": self.api_key}
        
                    # Use the dynamic parameters passed from the UI
        data = {
            "contents": self._format_messages_for_api(messages),
            "generationConfig": {
                "temperature": float(temperature),      # Ensure proper type conversion
                "maxOutputTokens": int(max_tokens),     # Ensure proper type conversion
            }
        }
        
        try:
            async with session.post(
                url,
                params=params,
                json=data
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    yield f"Gemini API Error: {response.status} - {error_text}"
                    return
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if line and not line.startswith('[') and not line.startswith(']'):
                        try:
                            if line.endswith(','):
                                line = line[:-1]
                            
                            chunk = json.loads(line)
                            if 'candidates' in chunk and chunk['candidates']:
                                candidate = chunk['candidates'][0]
                                if 'content' in candidate and 'parts' in candidate['content']:
                                    for part in candidate['content']['parts']:
                                        if 'text' in part:
                                            yield part['text']
                        except json.JSONDecodeError:
                            continue
                        
        except Exception as e:
            yield f"Gemini streaming error: {str(e)}"


class AnthropicClient(BaseClient):
    """Anthropic Claude API client with streaming support"""
    
    def __init__(self):
        super().__init__(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            api_url=os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com")
        )
    
    def _format_messages_for_api(self, messages: List[Dict[str, Any]]) -> tuple:
        """Format messages for Anthropic API"""
        system_message = ""
        formatted_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
                continue
            
            # Validate message content
            content = msg.get("content", "")
            if not content:
                continue  # Skip empty messages
            
            # Ensure content is string for text-only messages
            if isinstance(content, list):
                # This shouldn't happen, but if it does, convert to string
                content = str(content)
                logger.warning(f"Converted list content to string: {content[:100]}...")
            
            if msg.get("images"):
                # Handle multimodal content (images + text)
                # Anthropic requires content to be an array of content blocks
                content_blocks = []
                
                # Add text content first
                if content and content.strip():
                    content_blocks.append({"type": "text", "text": content})
                
                # Add images
                for image_data in msg["images"]:
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data
                        }
                    })
                
                formatted_messages.append({"role": msg["role"], "content": content_blocks})
            else:
                # Text-only content
                # For text-only messages, content should be a string, not an array
                formatted_messages.append({
                    "role": msg["role"],
                    "content": content
                })
        
        return system_message, formatted_messages
    
    async def stream_chat(
        self, 
        messages: List[Dict[str, Any]], 
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion from Anthropic"""
        
        session = await self._get_session()
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        system_message, formatted_messages = self._format_messages_for_api(messages)
        
        # Use the dynamic parameters passed from the UI
        data = {
            "model": model,
            "messages": formatted_messages,
            "stream": True,
            "temperature": float(temperature),  # Ensure proper type conversion
            "max_tokens": int(max_tokens)       # Ensure proper type conversion
        }
        
        if system_message:
            data["system"] = system_message
        
        try:
            async with session.post(
                f"{self.api_url}/v1/messages",
                headers=headers,
                json=data
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    yield f"Anthropic API Error: {response.status} - {error_text}"
                    return
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if line.startswith('data: '):
                        line = line[6:]  # Remove 'data: ' prefix
                        
                        if line == '[DONE]':
                            break
                        
                        try:
                            chunk = json.loads(line)
                            if chunk.get('type') == 'content_block_delta':
                                if 'delta' in chunk and 'text' in chunk['delta']:
                                    yield chunk['delta']['text']
                        except json.JSONDecodeError:
                            continue
                        
        except Exception as e:
            yield f"Anthropic streaming error: {str(e)}"