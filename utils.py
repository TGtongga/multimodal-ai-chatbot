"""
Utility functions for the multimodal chatbot
"""

import base64
import io
import logging
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import mimetypes
import os
from datetime import datetime

from config import UI_CONFIG, MODEL_CAPABILITIES, ERROR_MESSAGES, SECURITY_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def encode_image(image_path_or_file) -> str:
    """
    Encode image to base64 string
    
    Args:
        image_path_or_file: File path or file-like object
        
    Returns:
        Base64 encoded string of the image
    """
    try:
        if isinstance(image_path_or_file, str):
            # File path
            with open(image_path_or_file, "rb") as image_file:
                image_data = image_file.read()
        else:
            # File-like object (from Gradio file upload)
            if hasattr(image_path_or_file, 'name'):
                # Gradio file object
                with open(image_path_or_file.name, "rb") as image_file:
                    image_data = image_file.read()
            else:
                # Direct file content
                image_data = image_path_or_file
        
        # Resize image if too large
        image_data = resize_image_if_needed(image_data)
        
        return base64.b64encode(image_data).decode('utf-8')
    
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        raise ValueError(f"Failed to encode image: {str(e)}")

def resize_image_if_needed(image_data: bytes, max_size: int = 1024) -> bytes:
    """
    Resize image if it's too large while maintaining aspect ratio
    
    Args:
        image_data: Raw image bytes
        max_size: Maximum dimension size
        
    Returns:
        Resized image bytes
    """
    try:
        # Open image
        image = Image.open(io.BytesIO(image_data))
        
        # Check if resize is needed
        if max(image.size) <= max_size:
            return image_data
        
        # Calculate new size maintaining aspect ratio
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        
        # Resize image
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert back to bytes
        output = io.BytesIO()
        # Preserve original format if possible, otherwise use JPEG
        format_to_use = image.format if image.format in ['JPEG', 'PNG', 'WEBP'] else 'JPEG'
        resized_image.save(output, format=format_to_use, quality=85, optimize=True)
        
        return output.getvalue()
    
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        return image_data

def validate_file(file_obj) -> bool:
    """
    Validate uploaded file (supports both images and other file types)
    
    Args:
        file_obj: Uploaded file object
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if file_obj is None:
            return False
        
        # Get file path
        if hasattr(file_obj, 'name'):
            file_path = file_obj.name
        else:
            file_path = str(file_obj)
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > UI_CONFIG['max_file_size']:
            logger.warning(f"File too large: {file_size} bytes")
            return False
        
        # Check file format
        mime_type, _ = mimetypes.guess_type(file_path)
        
        # Handle case where mimetypes doesn't detect properly
        if mime_type is None:
            # Check by file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in UI_CONFIG['supported_file_formats']:
                return True
        
        if mime_type not in SECURITY_CONFIG['allowed_file_types']:
            logger.warning(f"Unsupported file format: {mime_type}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"File validation failed: {str(e)}")
        return False

def is_image_file(file_obj) -> bool:
    """
    Check if the file is an image
    
    Args:
        file_obj: File object to check
        
    Returns:
        True if it's an image file, False otherwise
    """
    try:
        if hasattr(file_obj, 'name'):
            file_path = file_obj.name
        else:
            file_path = str(file_obj)
        
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type and mime_type.startswith('image/')
    except:
        return False

def read_text_file(file_path: str) -> str:
    """
    Read content from a text file
    
    Args:
        file_path: Path to the text file
        
    Returns:
        File content as string
    """
    try:
        # Try different encodings
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    # Limit content size for safety
                    max_length = SECURITY_CONFIG['max_message_length'] * 5  # Allow longer for files
                    if len(content) > max_length:
                        content = content[:max_length] + "\n... (file truncated due to length)"
                    return content
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, try binary read
        with open(file_path, 'rb') as f:
            content = f.read()
            # Try to decode as UTF-8 with error handling
            return content.decode('utf-8', errors='replace')[:SECURITY_CONFIG['max_message_length'] * 5]
    
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {str(e)}")
        return f"Error reading file: {str(e)}"

def process_file(file_obj) -> dict:
    """
    Process uploaded file and return appropriate content
    
    Args:
        file_obj: Uploaded file object
        
    Returns:
        Dictionary with file type and processed content
    """
    try:
        if hasattr(file_obj, 'name'):
            file_path = file_obj.name
        else:
            file_path = str(file_obj)
        
        file_name = os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if is_image_file(file_obj):
            # Process as image
            return {
                'type': 'image',
                'name': file_name,
                'data': encode_image(file_obj),
                'mime_type': mime_type
            }
        else:
            # Process as text/document
            content = read_text_file(file_path)
            return {
                'type': 'document',
                'name': file_name,
                'content': content,
                'mime_type': mime_type
            }
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return {
            'type': 'error',
            'name': 'unknown',
            'content': f"Error processing file: {str(e)}",
            'mime_type': None
        }

def validate_file_path(file_path: str) -> bool:
    """
    Validate file path
    
    Args:
        file_path: Path to file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if not file_path or not isinstance(file_path, str):
            return False
        
        if not os.path.exists(file_path):
            logger.warning(f"File path does not exist: {file_path}")
            return False
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > UI_CONFIG['max_file_size']:
            logger.warning(f"File too large: {file_size} bytes")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"File path validation failed: {str(e)}")
        return False

def process_file_from_path(file_path: str) -> dict:
    """
    Process file from path and return appropriate content
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file type and processed content
    """
    try:
        file_name = os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if is_image_file_path(file_path):
            # Process as image
            return {
                'type': 'image',
                'name': file_name,
                'data': encode_image_from_path(file_path),
                'mime_type': mime_type
            }
        else:
            # Process as text/document
            content = read_text_file(file_path)
            return {
                'type': 'document',
                'name': file_name,
                'content': content,
                'mime_type': mime_type
            }
    
    except Exception as e:
        logger.error(f"Error processing file from path: {str(e)}")
        return {
            'type': 'error',
            'name': 'unknown',
            'content': f"Error processing file: {str(e)}",
            'mime_type': None
        }

def is_image_file_path(file_path: str) -> bool:
    """
    Check if the file path points to an image
    
    Args:
        file_path: Path to check
        
    Returns:
        True if it's an image file, False otherwise
    """
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type and mime_type.startswith('image/')
    except:
        return False

def encode_image_from_path(file_path: str) -> str:
    """
    Encode image from file path to base64 string
    
    Args:
        file_path: Path to image file
        
    Returns:
        Base64 encoded string of the image
    """
    try:
        with open(file_path, "rb") as image_file:
            image_data = image_file.read()
        
        # Resize image if needed
        image_data = resize_image_if_needed(image_data)
        
        return base64.b64encode(image_data).decode('utf-8')
    
    except Exception as e:
        logger.error(f"Error encoding image from path: {str(e)}")
        raise ValueError(f"Failed to encode image: {str(e)}")
    """
    Process uploaded file and return appropriate content
    
    Args:
        file_obj: Uploaded file object
        
    Returns:
        Dictionary with file type and processed content
    """
    try:
        if hasattr(file_obj, 'name'):
            file_path = file_obj.name
        else:
            file_path = str(file_obj)
        
        file_name = os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if is_image_file(file_obj):
            # Process as image
            return {
                'type': 'image',
                'name': file_name,
                'data': encode_image(file_obj),
                'mime_type': mime_type
            }
        else:
            # Process as text/document
            content = read_text_file(file_path)
            return {
                'type': 'document',
                'name': file_name,
                'content': content,
                'mime_type': mime_type
            }
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return {
            'type': 'error',
            'name': 'unknown',
            'content': f"Error processing file: {str(e)}",
            'mime_type': None
        }

def format_message(content: str, role: str = "user") -> Dict[str, Any]:
    """
    Format message for consistent structure
    
    Args:
        content: Message content
        role: Message role (user, assistant, system)
        
    Returns:
        Formatted message dictionary
    """
    return {
        "role": role,
        "content": sanitize_content(content),
        "timestamp": datetime.now().isoformat()
    }

def sanitize_content(content: str) -> str:
    """
    Sanitize message content for security
    
    Args:
        content: Raw content string
        
    Returns:
        Sanitized content string
    """
    if not isinstance(content, str):
        content = str(content)
    
    # Limit message length
    if len(content) > SECURITY_CONFIG['max_message_length']:
        content = content[:SECURITY_CONFIG['max_message_length']] + "... (truncated)"
    
    # Basic HTML sanitization if enabled
    if SECURITY_CONFIG.get('sanitize_html', True):
        # Remove potentially dangerous HTML tags
        dangerous_tags = ['<script', '<iframe', '<object', '<embed', '<form']
        for tag in dangerous_tags:
            content = content.replace(tag, f"&lt;{tag[1:]}")
    
    return content.strip()

def get_model_provider(model_name: str) -> Optional[str]:
    """
    Get the provider for a given model name
    
    Args:
        model_name: Name of the model
        
    Returns:
        Provider name or None if not found
    """
    from config import AVAILABLE_MODELS
    
    for provider, models in AVAILABLE_MODELS.items():
        if model_name in models:
            return provider
    return None

def is_vision_model(model_name: str) -> bool:
    """
    Check if a model supports vision capabilities
    
    Args:
        model_name: Name of the model
        
    Returns:
        True if model supports vision, False otherwise
    """
    return MODEL_CAPABILITIES.get(model_name, {}).get('vision', False)

def get_model_max_tokens(model_name: str) -> int:
    """
    Get maximum tokens for a model
    
    Args:
        model_name: Name of the model
        
    Returns:
        Maximum token limit
    """
    return MODEL_CAPABILITIES.get(model_name, {}).get('max_tokens', 2000)

def truncate_conversation_history(
    messages: List[Dict[str, Any]], 
    max_length: int = None
) -> List[Dict[str, Any]]:
    """
    Truncate conversation history to stay within limits
    
    Args:
        messages: List of message dictionaries
        max_length: Maximum number of messages to keep
        
    Returns:
        Truncated message list
    """
    if max_length is None:
        max_length = UI_CONFIG.get('max_conversation_length', 50)
    
    if len(messages) <= max_length:
        return messages
    
    # Keep system message if present and recent messages
    system_messages = [msg for msg in messages if msg.get('role') == 'system']
    other_messages = [msg for msg in messages if msg.get('role') != 'system']
    
    # Keep most recent messages
    recent_messages = other_messages[-(max_length - len(system_messages)):]
    
    return system_messages + recent_messages

def format_error_message(error_type: str, details: str = "") -> str:
    """
    Format error message for display
    
    Args:
        error_type: Type of error
        details: Additional error details
        
    Returns:
        Formatted error message
    """
    base_message = ERROR_MESSAGES.get(error_type, ERROR_MESSAGES['general_error'])
    
    if details:
        return f"{base_message}\n\nDetails: {details}"
    
    return base_message

def log_conversation(messages: List[Dict[str, Any]], model: str, response_time: float):
    """
    Log conversation for monitoring and debugging
    
    Args:
        messages: Conversation messages
        model: Model used
        response_time: Time taken for response
    """
    try:
        logger.info(f"Conversation logged - Model: {model}, "
                   f"Messages: {len(messages)}, "
                   f"Response time: {response_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to log conversation: {str(e)}")

def extract_text_from_messages(messages: List[Dict[str, Any]]) -> str:
    """
    Extract text content from messages for analysis
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Combined text content
    """
    text_parts = []
    for msg in messages:
        if isinstance(msg.get('content'), str):
            text_parts.append(msg['content'])
        elif isinstance(msg.get('content'), list):
            # Handle multimodal content
            for part in msg['content']:
                if isinstance(part, dict) and part.get('type') == 'text':
                    text_parts.append(part.get('text', ''))
    
    return ' '.join(text_parts)

def get_conversation_stats(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about the conversation
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Dictionary with conversation statistics
    """
    stats = {
        'total_messages': len(messages),
        'user_messages': len([m for m in messages if m.get('role') == 'user']),
        'assistant_messages': len([m for m in messages if m.get('role') == 'assistant']),
        'total_characters': len(extract_text_from_messages(messages)),
        'has_images': any(m.get('images') for m in messages),
        'image_count': sum(len(m.get('images', [])) for m in messages)
    }
    
    return stats

def create_system_message(custom_prompt: str = None) -> Dict[str, Any]:
    """
    Create system message with default or custom prompt
    
    Args:
        custom_prompt: Custom system prompt
        
    Returns:
        System message dictionary
    """
    from config import DEFAULT_SETTINGS
    
    prompt = custom_prompt or DEFAULT_SETTINGS['system_prompt']
    
    return {
        'role': 'system',
        'content': prompt,
        'timestamp': datetime.now().isoformat()
    }