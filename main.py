import gradio as gr
import os
from dotenv import load_dotenv
from typing import Generator, List, Tuple, Optional
import asyncio
from dataclasses import dataclass
import threading
import queue
import logging

from api_clients import OpenAIClient, GeminiClient, AnthropicClient
from utils import (format_message, encode_image, validate_file, is_image_file, process_file,
                   validate_file_path, process_file_from_path, is_image_file_path, encode_image_from_path)
from config import AVAILABLE_MODELS, THINKING_SUPPORTED_MODELS, MODEL_CAPABILITIES

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global event loop for async operations
_loop = None
_loop_thread = None

def detect_media_type_from_path(file_path: str) -> str:
    """Detect media type from file extension"""
    import mimetypes
    from pathlib import Path
    
    # Get the file extension
    ext = Path(file_path).suffix.lower()
    
    # Map common image extensions to media types
    media_type_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff'
    }
    
    # Try our map first
    if ext in media_type_map:
        return media_type_map[ext]
    
    # Fall back to mimetypes
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith('image/'):
        return mime_type
    
    # Final fallback
    return 'image/jpeg'

def get_or_create_event_loop():
    """Get or create a persistent event loop running in a separate thread"""
    global _loop, _loop_thread
    
    if _loop is None or _loop.is_closed():
        
        def run_loop():
            global _loop
            _loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_loop)
            _loop.run_forever()
        
        _loop_thread = threading.Thread(target=run_loop, daemon=True)
        _loop_thread.start()
        
        # Wait for loop to be created
        import time
        while _loop is None:
            time.sleep(0.01)
    
    return _loop

@dataclass
class ChatState:
    """Manages chat state and conversation history"""
    messages: List[dict]
    current_model: str
    
    def __init__(self):
        self.messages = []
        self.current_model = list(AVAILABLE_MODELS.keys())[0]
    
    def add_message(self, role: str, content: str, images: Optional[List] = None):
        """Add a message to the conversation history"""
        message = {"role": role, "content": content}
        if images:
            message["images"] = images
        self.messages.append(message)
    
    def clear_history(self):
        """Clear conversation history"""
        self.messages = []
    
    def get_history_for_api(self):
        """Get formatted history for API calls"""
        return self.messages.copy()

class MultimodalChatbot:
    """Main chatbot class handling all model interactions"""
    
    def __init__(self):
        self.clients = {
            'openai': OpenAIClient(),
            'gemini': GeminiClient(),
            'anthropic': AnthropicClient()
        }
        self.chat_state = ChatState()
    
    def get_client_for_model(self, model_name: str):
        """Get the appropriate client for a given model"""
        for provider, models in AVAILABLE_MODELS.items():
            if model_name in models:
                return self.clients[provider]
        raise ValueError(f"Model {model_name} not found")
    
    def stream_response(
        self,
        message,
        history: List[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        thinking_enabled: bool = False,
        thinking_budget: int = 10000,
        use_1m_context: bool = False,  # ADDED: Support for 1M token context window
        system_prompt_template: str = "Default Assistant",
        custom_system_prompt: str = ""
    ) -> Generator[str, None, None]:
        """Stream response from the selected model with dynamic parameters"""
        
        try:
            # Import templates here to avoid circular import
            from config import SYSTEM_PROMPT_TEMPLATES
            
            # Determine which system prompt to use
            if system_prompt_template == "Custom":
                final_system_prompt = custom_system_prompt
            else:
                final_system_prompt = SYSTEM_PROMPT_TEMPLATES.get(
                    system_prompt_template,
                    SYSTEM_PROMPT_TEMPLATES["Default Assistant"]
                )
            
            # Update chat state
            self.chat_state.current_model = model
            
            # DEBUG: Log the raw inputs from Gradio
            logger.info(f"=== DEBUG: Raw inputs from Gradio ===")
            logger.info(f"Message type: {type(message)}")
            logger.info(f"Message content: {message}")
            logger.info(f"History type: {type(history)}")
            logger.info(f"History length: {len(history) if history else 0}")
            logger.info(f"Model: {model}")
            logger.info(f"Temperature: {temperature}, Max tokens: {max_tokens}")
            logger.info(f"System prompt template: {system_prompt_template}")
            logger.info(f"Final system prompt: {final_system_prompt[:100]}..." if len(final_system_prompt) > 100 else f"Final system prompt: {final_system_prompt}")
            
            if history:
                logger.info(f"History sample: {history[:2] if len(history) >= 2 else history}")
            logger.info("=" * 50)
            
            # Handle multimodal message format
            text_content = ""
            processed_images = []
            
            if isinstance(message, dict):
                # New multimodal format from Gradio
                text_content = message.get("text", "")
                files = message.get("files", [])
                
                logger.info(f"Multimodal message - Text: {len(text_content)} chars, Files: {len(files)}")
                
                for i, file_info in enumerate(files):
                    logger.info(f"File {i}: {file_info} (type: {type(file_info)})")
                    
                    # Handle different file formats from Gradio
                    if isinstance(file_info, str):
                        # It's a file path string
                        if validate_file_path(file_info):
                            # MODIFIED: Detect media type from file path
                            detected_media_type = detect_media_type_from_path(file_info)
                            processed_file = process_file_from_path(file_info)
                            if processed_file['type'] == 'image':
                                # MODIFIED: Use detected media type with fallback
                                processed_images.append({
                                    'data': processed_file['data'],
                                    'media_type': processed_file.get('media_type', detected_media_type)
                                })
                            elif processed_file['type'] == 'document':  # FIX: Moved inside validate_file_path block
                                doc_text = f"\n\n--- Content from {processed_file['name']} ---\n{processed_file['content']}\n--- End of {processed_file['name']} ---\n\n"
                                text_content += doc_text
                    
                    elif isinstance(file_info, dict) and 'path' in file_info:
                        # It's a Gradio FileData object
                        file_path = file_info['path']
                        if validate_file_path(file_path):
                            # MODIFIED: Detect media type from file path
                            detected_media_type = detect_media_type_from_path(file_path)
                            processed_file = process_file_from_path(file_path)
                            if processed_file['type'] == 'image':
                                # MODIFIED: Use detected media type with fallback
                                processed_images.append({
                                    'data': processed_file['data'],
                                    'media_type': processed_file.get('media_type', detected_media_type)
                                })
                            elif processed_file['type'] == 'document':  # FIX: Moved inside validate_file_path block
                                doc_text = f"\n\n--- Content from {processed_file['name']} ---\n{processed_file['content']}\n--- End of {processed_file['name']} ---\n\n"
                                text_content += doc_text
                    else:
                        logger.warning(f"Unknown file format: {file_info}")
            
            elif isinstance(message, str):
                # Simple text message
                text_content = message
                logger.info(f"Text message: {len(text_content)} chars")
            else:
                logger.error(f"Unknown message format: {type(message)} - {message}")
                yield "Error: Unknown message format received"
                return
            
            # Convert history to messages format and clean up malformed messages
            messages = []
            
            # Add system message if provided
            if final_system_prompt and final_system_prompt.strip():
                messages.append({
                    "role": "system",
                    "content": final_system_prompt.strip()
                })
            
            # Process history and add current message (rest of the method remains the same)
            for i, msg in enumerate(history):
                logger.info(f"History entry {i}: {type(msg)} - {msg}")
                
                if isinstance(msg, dict) and msg.get("role") in ["user", "assistant"]:
                    # Properly formatted message from ChatInterface
                    content = msg["content"]
                    
                    # CRITICAL: Check for malformed content that causes API errors
                    if isinstance(content, list):
                        # This is the source of our API errors!
                        logger.error(f"FOUND MALFORMED CONTENT: {content}")
                        if len(content) == 1 and isinstance(content[0], str):
                            # This is a single-item list with a file path message
                            if content[0].startswith('/') or content[0].startswith('\\'):
                                # This is a file path - SKIP this malformed message
                                logger.warning(f"Skipping malformed file path message: {content[0]}")
                                continue
                            else:
                                # Convert single-item list to string
                                content = content[0]
                        else:
                            # Multiple items or complex structure - convert to string
                            content = str(content)
                        logger.warning(f"Converted complex content to string: {content[:100]}...")
                    
                    # Only add if content is valid string
                    if isinstance(content, str) and content.strip():
                        messages.append({
                            "role": msg["role"],
                            "content": content
                        })
                    else:
                        logger.warning(f"Skipping message with invalid content: {content}")
                
                elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                    # Old format: [user_message, assistant_message]
                    user_msg, assistant_msg = msg
                    if user_msg and isinstance(user_msg, str):
                        messages.append({"role": "user", "content": user_msg})
                    if assistant_msg and isinstance(assistant_msg, str):
                        messages.append({"role": "assistant", "content": assistant_msg})
                else:
                    logger.warning(f"Unknown history format: {type(msg)} - {msg}")
            
            # Add current message
            if text_content.strip() or processed_images:
                current_message = {"role": "user", "content": text_content}
                if processed_images:
                    current_message["images"] = processed_images
                messages.append(current_message)
            else:
                yield "Please provide some text or upload a file to continue the conversation."
                return
            
            # Get appropriate client
            client = self.get_client_for_model(model)
            
            # Use persistent event loop for streaming
            loop = get_or_create_event_loop()
            result_queue = queue.Queue()
            
            async def stream_to_queue():
                try:
                    async for chunk in client.stream_chat(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        thinking_enabled=thinking_enabled,
                        thinking_budget=thinking_budget,
                        use_1m_context=use_1m_context  # ADDED: Pass 1M context parameter
                    ):
                        if chunk:
                            result_queue.put(('chunk', chunk))  # MODIFIED: Removed double accumulation
                    result_queue.put(('done', None))
                except Exception as e:
                    logger.error(f"Streaming error for {model}: {str(e)}")
                    result_queue.put(('error', str(e)))
            
            # Schedule the async function on the persistent loop
            future = asyncio.run_coroutine_threadsafe(stream_to_queue(), loop)
            
            # Yield chunks as they arrive
            while True:
                try:
                    queue_item = result_queue.get(timeout=30)
                    
                    # Handle tuple unpacking safely
                    if isinstance(queue_item, tuple):
                        if len(queue_item) == 2:
                            msg_type, content = queue_item
                        elif len(queue_item) == 1:
                            msg_type = queue_item[0]
                            content = None
                        else:
                            logger.error(f"Unexpected queue item format: {queue_item}")
                            continue
                    else:
                        logger.error(f"Queue item is not a tuple: {queue_item}")
                        continue
                    
                    if msg_type == 'chunk':
                        yield content
                    elif msg_type == 'done':
                        break
                    elif msg_type == 'error':
                        yield f"Streaming error: {content}"
                        break
                except queue.Empty:
                    yield "Request timed out"
                    break
        
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            yield error_msg
    
    def clear_chat(self):
        """Clear chat history"""
        self.chat_state.clear_history()
        return []

# Initialize chatbot
chatbot = MultimodalChatbot()

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="Multimodal AI Chatbot",
        theme=gr.themes.Soft(),
        css="""
        /* Import elegant fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Source+Serif+Pro:wght@400;600;700&display=swap');
        
        /* Global elegant typography */
        .gradio-container {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            font-weight: 400;
            line-height: 1.6;
        }
        
        /* Target Gradio's internal chatbot message classes */
        .message-wrap .message {
            font-family: 'Source Serif Pro', Georgia, 'Times New Roman', serif !important;
            font-size: 14px !important;
            line-height: 1.6 !important;
        }
        
        /* Target chatbot container content */
        .chatbot .prose, .chatbot .prose p {
            font-family: 'Source Serif Pro', Georgia, 'Times New Roman', serif !important;
            font-size: 14px !important;
            line-height: 1.6 !important;
        }
        
        /* Code blocks should be smaller (10px) - UNIFIED SIZING */
        .chatbot pre *,
        .chatbot code,
        .chatbot code * {
            font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace !important;
            font-size: 10px !important;
            line-height: 1.4 !important;
        }
        
        /* Inline code */
        .chatbot code:not(pre code) {
            font-size: 10px !important;
            background: #f5f5f9 !important;
            color: #1e293b !important;
            padding: 2px 4px !important;
            border-radius: 3px !important;
        }
        
        /* Force uniform font size for all syntax highlighting elements */
        .chatbot pre span,
        .chatbot pre .token,
        .chatbot pre .keyword,
        .chatbot pre .string,
        .chatbot pre .comment,
        .chatbot pre .function,
        .chatbot pre .number,
        .chatbot pre .operator,
        .chatbot pre .punctuation,
        .chatbot code span,
        .chatbot code .token,
        .chatbot code .keyword,
        .chatbot code .string,
        .chatbot code .comment,
        .chatbot code .function,
        .chatbot code .number,
        .chatbot code .operator,
        .chatbot code .punctuation {
            font-size: 10px !important;
            font-family: inherit !important;
        }
        
        /* Dark mode fixes */
        .dark .gradio-container h1,
        .dark h1 {
            color: #f7fafc !important;
        }
        
        .dark textarea,
        .dark input[type="text"] {
            color: #f7fafc !important;
        }
        
        /* Override dark mode header colors */
        .dark .chatbot h1,
        .dark .chatbot h2,
        .dark .chatbot h3,
        .dark .chatbot h4,
        .dark .chatbot h5,
        .dark .chatbot h6 {
            color: #f7fafc !important;
        }
        
        .chatbot code:not(pre code) {
            background: #2d3748 !important;
            color: #e2e8f0 !important;
        }
        
        .dark .chatbot code:not(pre code) {
            background: #2d3748 !important;
            color: #e2e8f0 !important;
        }
        
        /* Target specific message elements */
        .chatbot .message.user .content,
        .chatbot .message.bot .content,
        .chatbot .message .content {
            font-family: 'Source Serif Pro', serif !important;
            font-size: 14px !important;
            line-height: 1.6 !important;
        }
        
        /* Headers in chatbot messages (override both light & dark mode) */
        .chatbot h1,
        .chatbot h2,
        .chatbot h3,
        .chatbot h4,
        .chatbot h5,
        .chatbot h6 {
            font-family: 'Source Serif Pro', serif !important;
            font-weight: 600 !important;
            color: #1a202c !important;
            margin: 12px 0 6px 0 !important;
        }
        
        /* Font sizes */
        .chatbot h1 { font-size: 18px !important; }
        .chatbot h2 { font-size: 17px !important; }
        .chatbot h3 { font-size: 16px !important;  }
        .chatbot h4, .chatbot h5, .chatbot h6 { font-size: 15px !important; }
        
        /* Lists */
        .chatbot ul, .chatbot ol {
            font-size: 14px !important;
            line-height: 1.6 !important;
            margin: 8px 0 !important;
            padding-left: 20px !important;
        }
        
        .chatbot li {
            font-size: 14px !important;
            margin: 4px 0 !important;
        }
        
        /* Tables */
        .chatbot table {
            font-size: 14px !important;
        }
        
        .chatbot th, .chatbot td {
            font-size: 14px !important;
            padding: 8px 12px !important;
        }
        
        /* Blockquotes */
        .chatbot blockquote {
            font-size: 14px !important;
            font-style: italic !important;
            border-left: 4px solid #e2e8f0 !important;
            padding-left: 16px !important;
            margin: 12px 0 !important;
        }
        
        /* Strong and emphasis */
        .chatbot strong, .chatbot b {
            font-weight: 600 !important;
            font-size: 14px !important;
        }
        
        .chatbot em, .chatbot i {
            font-style: italic !important;
            font-size: 14px !important;
        }
        
        /* Links */
        .chatbot a {
            font-size: 14px !important;
            color: #3b82f6 !important;
            text-decoration: underline !important;
        }
        
        /* Ensure paragraph and list text under markdown also uses Source Serif Pro */
        .chatbot .prose p,
        .chatbot .prose li {
            font-family: 'Source Serif Pro', serif !important;
            color: inherit !important;
        }
        
        /* Input textbox styling */
        .chatbot-input textarea, textarea {
            font-family: 'Inter', sans-serif !important;
            font-size: 15px !important;
            line-height: 1.5 !important;
            color: #2d3748 !important;
        }
        
        /* Model selector styling */
        .model-selector {
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 10px;
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
        }
        
        /* Chat container */
        .chat-container {
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Button styling */
        button {
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
            font-size: 14px !important;
        }
        
        /* Header and title styling */
        .gradio-container h1 {
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
            font-size: 32px !important;
            color: #1a202c !important;
        }
        
        /* Accordion and sidebar text */
        .accordion-content, .sidebar-content {
            font-family: 'Inter', sans-serif !important;
            font-size: 14px !important;
            line-height: 1.6 !important;
            color: #4a5568 !important;
        }
        
        /* Ensure all chatbot content uses correct size */
        .chatbot div, .chatbot span, .chatbot p {
            font-size: 14px !important;
        }
        
        /* Override any default Gradio styles */
        .svelte-1wnfpfb {
            font-size: 14px !important;
        }
        
        /* Ensure proper text rendering */
        * {
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            text-rendering: optimizeLegibility;
        }
        """
    ) as demo:
        
        gr.Markdown(
            """
            # 🤖 Multimodal AI Chatbot
            
            Chat with multiple AI models including OpenAI GPT, Google Gemini, and Anthropic Claude.
            Support for text and images in conversations.
            """,
            elem_classes=["header"]
        )
        
        # Import the new templates and defaults from config
        from config import SYSTEM_PROMPT_TEMPLATES, MODEL_DEFAULTS
        
        # Get all available models
        all_models = []
        for provider_models in AVAILABLE_MODELS.values():
            all_models.extend(provider_models)
        
        # Create components that will be used as additional_inputs (not pre-rendered)
        # These will appear below chatbot automatically
        model_dropdown = gr.Dropdown(
            choices=all_models,
            value=all_models[0] if all_models else None,
            label="🎯 Select Model",
            elem_classes=["model-selector"],
            interactive=True,
            render=False  # Will be rendered in additional_inputs below chatbot
        )
        
        temperature = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=MODEL_DEFAULTS.get(all_models[0], {}).get('temperature', 0.7) if all_models else 0.7,
            step=0.1,
            label="🌡️ Temperature",
            info="Controls randomness in responses",
            render=False  # Will be rendered in additional_inputs below chatbot
        )
        
        # MODIFIED: Set initial max based on first model's capabilities
        initial_model = all_models[0] if all_models else None
        initial_max_tokens = MODEL_CAPABILITIES.get(initial_model, {}).get('max_tokens', 64000) if initial_model else 64000
        
        max_tokens = gr.Slider(
            minimum=1,
            maximum=initial_max_tokens,  # MODIFIED: Dynamic maximum based on model capability
            value=initial_max_tokens,    # MODIFIED: Default to max capability
            step=50,
            label="📝 Max Tokens",
            info="Maximum response length",
            render=False  # Will be rendered in additional_inputs below chatbot
        )
        
        # Extended thinking controls (for Anthropic models)
        thinking_enabled = gr.Checkbox(
            label="🧠 Enable Extended Thinking (Anthropic only)",
            value=True,
            info="Enable Claude's extended reasoning capabilities",
            visible=any(model in all_models[0] for model in THINKING_SUPPORTED_MODELS) if all_models else False,
            render=False
        )
        
        thinking_budget = gr.Slider(
            minimum=1024,
            maximum=64000,
            value=10000,
            step=1024,
            label="💭 Thinking Budget (tokens)",
            info="Token budget for Claude's internal reasoning",
            visible=False,  # Keep as False since checkbox starts unchecked
            render=False
        )
        
        # ADDED: 1M token context window support (Claude Sonnet 4 and 4.5 only)
        use_1m_context = gr.Checkbox(
            label="📚 Enable 1M Token Context Window (Beta)",
            value=True,
            info="Enable 1M token context for Claude Sonnet 4/4.5 (*requires usage tier 4). Premium pricing applies.",
            visible=any("claude-sonnet-4" in model for model in all_models) if all_models else False,
            render=False
        )
        
        # System prompt template selector
        system_prompt_template = gr.Dropdown(
            choices=list(SYSTEM_PROMPT_TEMPLATES.keys()),
            value="Default Assistant",
            label="📋 System Prompt Template",
            info="Choose a pre-built template",
            interactive=True,
            render=False  # Will be rendered in additional_inputs below chatbot
        )
        
        # System prompt editor - always visible for editing the selected template
        system_prompt_editor = gr.Textbox(
            value=SYSTEM_PROMPT_TEMPLATES["Default Assistant"],
            label="✏️ System Prompt Editor",
            info="Edit the selected template or create your own custom prompt",
            lines=6,
            max_lines=16,
            interactive=True,
            render=False  # Will be rendered in additional_inputs below chatbot
        )
        
        # Main chat Interface - components will appear below it
        chatinterface = gr.ChatInterface(
            fn=chatbot.stream_response,
            type="messages",
            multimodal=True,
            title="",
            description="",
            examples=[
                ["Hello! How can you help me today?"],
                ["Explain quantum computing in simple terms"],
                ["What can you see in this image?"],
                ["Write a creative short story"],
                ["Help me debug this Python code"],
                ["Compare different machine learning algorithms"],
                ["Summarize the key points from this document"],
                ["Generate a marketing email for a new product"],
            ],
            cache_examples=False,
            submit_btn="Send 📤",
            stop_btn="Stop 🛑",
            chatbot=gr.Chatbot(
                show_label=True,
                container=True,
                resizable=True,
                editable='user',
                show_copy_button=True,
                show_copy_all_button=True,
                height=700
            ),  # 2x larger height
            textbox=gr.MultimodalTextbox(
                placeholder="Type your message here... (Supports text, images, and documents)",
                file_types=[
                    "image", ".text", ".pdf", ".doc", ".docx",
                    ".txt", ".md", ".py", ".js", ".html", ".css",
                    ".json", ".xml", ".csv", ".xlsx", ".xls"
                ],
                file_count="multiple"
            ),
            additional_inputs=[model_dropdown, temperature, max_tokens, thinking_enabled, thinking_budget,
                             use_1m_context, system_prompt_template, system_prompt_editor],  # ADDED: use_1m_context
            additional_inputs_accordion="Model Settings",  # FIX: Only pass title string
            save_history=True  # Enable chat history with sidebar
        )
        
        # Event handlers for dynamic behavior
        def update_model_defaults(selected_model):
            """Update temperature/max_tokens based on selected model"""
            # MODIFIED: Get defaults and capabilities
            defaults = MODEL_DEFAULTS.get(selected_model, {'temperature': 0.7, 'max_tokens': 2000})
            capabilities = MODEL_CAPABILITIES.get(selected_model, {'max_tokens': 64000})
            
            # MODIFIED: Get the actual max_tokens capability for this model
            max_capability = capabilities.get('max_tokens', 64000)
            
            return (
                gr.Slider(value=defaults['temperature']),  # Update temperature value
                gr.Slider(
                    value=max_capability,          # MODIFIED: Set value to max capability
                    maximum=max_capability,        # MODIFIED: Update maximum bound
                    minimum=1,                     # Keep minimum consistent
                    step=50                        # Keep step consistent
                )
            )
        
        def update_system_prompt(template_name):
            """
            Update system prompt editor content and visibility based on the selected template.
            """
            if template_name == "Custom":
                return gr.update(visible=True), gr.update(value="")
            else:
                prompt_text = SYSTEM_PROMPT_TEMPLATES.get(
                    system_prompt_template,
                    SYSTEM_PROMPT_TEMPLATES["Default Assistant"]
                )
                return gr.update(value=prompt_text, visible=True), gr.update(value=template_name)
        
        def get_system_prompt_for_chat(template_name, custom_prompt):
            """Get the actual system prompt text for the chat function"""
            if template_name == "Custom":
                return custom_prompt
            else:
                return SYSTEM_PROMPT_TEMPLATES.get(template_name, SYSTEM_PROMPT_TEMPLATES["Default Assistant"])
        
        def update_thinking_visibility(selected_model, thinking_enabled_value):
            """Show/hide thinking budget based on model and checkbox"""
            from config import THINKING_SUPPORTED_MODELS
            
            is_anthropic = any(model in selected_model for model in THINKING_SUPPORTED_MODELS)
            
            # MODIFIED: Return updates for both checkbox and slider
            return (
                gr.update(visible=is_anthropic, value=thinking_enabled_value if is_anthropic else False),
                gr.update(visible=is_anthropic and thinking_enabled_value)
            )
        
        # MODIFIED: Bind the event with proper outputs
        model_dropdown.change(
            fn=update_thinking_visibility,
            inputs=[model_dropdown, thinking_enabled],
            outputs=[thinking_enabled, thinking_budget]
        )
        
        # MODIFIED: Update slider visibility when checkbox changes
        thinking_enabled.change(
            fn=lambda enabled, model: gr.update(visible=enabled and any(m in model for m in THINKING_SUPPORTED_MODELS)),
            inputs=[thinking_enabled, model_dropdown],
            outputs=[thinking_budget]
        )
        
        # ADDED: Update 1M context visibility when model changes
        def update_1m_context_visibility(selected_model):
            """Show/hide 1M context checkbox for Claude Sonnet 4 and 4.5 only"""
            is_sonnet_4_or_45 = "claude-sonnet-4" in selected_model
            return gr.update(visible=is_sonnet_4_or_45, value=False if not is_sonnet_4_or_45 else None)
        
        model_dropdown.change(
            fn=update_1m_context_visibility,
            inputs=[model_dropdown],
            outputs=[use_1m_context]
        )
        
        system_prompt_template.change(
            fn=lambda template_name: gr.update(value="" if template_name == "Custom" else
                                              SYSTEM_PROMPT_TEMPLATES.get(template_name,
                                                                         SYSTEM_PROMPT_TEMPLATES["Default Assistant"]),
                                              visible=True),
            inputs=[system_prompt_template],
            outputs=[system_prompt_editor]
        )
        
        # MODIFIED: Update model defaults when model changes
        model_dropdown.change(
            fn=update_model_defaults,
            inputs=[model_dropdown],
            outputs=[temperature, max_tokens]  # Update both sliders
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    
    # Launch with appropriate settings
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True,
        quiet=False
    )