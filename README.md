# 🤖 Next-Generation Multimodal AI Chatbot

A state-of-the-art Gradio-based chatbot application featuring the latest 2025 AI models from OpenAI, Google Gemini, and Anthropic Claude with comprehensive multimodal capabilities and advanced prompt engineering.

## ✨ Key Features

- **🚀 Latest 2025 AI Models**: GPT-4.1, Gemini 2.5 Pro, Claude Opus 4.0, and more
- **🎯 Advanced Multimodal Support**: Text, image, and document processing
- **⚡ Real-time Streaming**: Live response streaming with async processing
- **🎛️ Dynamic Model Selection**: Seamless switching between 15+ cutting-edge models
- **🖼️ Intelligent Image Processing**: Auto-optimization, format validation, and encoding
- **💬 Smart Conversation Management**: Persistent chat history with context preservation
- **🔧 Professional Settings Panel**: Temperature, token limits, and custom system prompts
- **📝 Expert Prompt Templates**: Pre-built templates for coding, writing, research, and business
- **🎨 Modern Interface**: Responsive Gradio UI with professional styling
- **⚙️ Robust Architecture**: Async-first design with comprehensive error handling

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project files
# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Copy the `.env.example` file to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your API credentials:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_URL=https://api.openai.com/v1

# Google Gemini Configuration  
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_API_URL=https://generativelanguage.googleapis.com/v1beta

# Anthropic Claude Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_API_URL=https://api.anthropic.com
```

### 3. Run the Application

```bash
python main.py
```

The application will be available at `http://localhost:7860`

## 🏗️ Project Architecture

```
chatbot/
├── main.py              # 🚀 Main application with Gradio interface
├── main.ipynb          # 📓 Jupyter notebook for development/testing
├── api_clients.py      # 🔌 Async API clients for all providers
├── config.py           # ⚙️ Model configurations and prompt templates
├── utils.py            # 🛠️ Image processing and utility functions
├── requirements.txt    # 📦 Python dependencies
├── .env.example       # 🔐 Environment variables template
└── README.md          # 📖 This documentation
```

## 🎯 Latest 2025 AI Models

### 🧠 OpenAI Models (GPT-4.1 Series)
- **GPT-4.1**: Flagship model with 1M context window
- **GPT-4.1 Mini**: Fast and efficient with vision capabilities
- **GPT-4.1 Nano**: Fastest, cost-effective text-only model
- **GPT-4o/GPT-4o Mini**: Advanced multimodal capabilities
- **GPT-4 Turbo**: Previous generation with reliable performance

### 🔮 Google Gemini Models (2.5 & 2.0 Series)
- **Gemini 2.5 Pro**: State-of-the-art reasoning with 2M context
- **Gemini 2.5 Flash**: Best price-performance with thinking capabilities
- **Gemini 2.0 Flash**: Next-gen features with 1M context window
- **Gemini 2.0 Pro Experimental**: Strongest coding performance
- **Flash Lite variants**: Cost-efficient, high-throughput options

### 🎭 Anthropic Claude Models (4.0 Series)
- **Claude Opus 4.0**: Most capable, best coding model
- **Claude Sonnet 4.0**: High-performance, efficient reasoning
- **Claude 3.7 Sonnet**: Advanced with extended thinking capabilities
- **Claude 3.5 Sonnet**: Previous generation (still powerful)

## 🖼️ Advanced Image & File Support

**Supported Formats:**
- **Images**: JPEG, PNG, GIF, WebP, BMP
- **Documents**: Coming soon (PDF, DOCX, TXT)

**Smart Processing Features:**
- ✅ Automatic image optimization and resizing
- ✅ Intelligent base64 encoding for API compatibility
- ✅ File size validation (max 10MB per file)
- ✅ Security-focused format validation
- ✅ Batch file processing support
- ✅ Error handling with detailed feedback

## 📝 Professional Prompt Templates

Choose from expertly crafted system prompts:

### 🎨 **Creative Writer**
Advanced storytelling, poetry, and narrative development with genre expertise and literary analysis capabilities.

### 💻 **Technical Expert** 
Senior-level programming mentor covering full-stack development, DevOps, ML/AI, cybersecurity, and system architecture.

### 🔬 **Research Assistant**
Academic research specialist with literature review, methodology design, statistical analysis, and citation management expertise.

### 💼 **Business Consultant**
Strategic advisor for business strategy, market analysis, financial planning, operations optimization, and digital transformation.

### 🎓 **Educational Tutor**
Adaptive learning specialist using multiple teaching methods, Socratic questioning, and personalized instruction across all subjects.

## ⚙️ Advanced Configuration

### 🎛️ Model Settings
- **Temperature**: Response creativity control (0.0 - 2.0)
- **Max Tokens**: Response length limits (up to 2M tokens for Gemini 2.5 Pro)
- **System Prompts**: Custom AI behavior with expert templates
- **Dynamic Updates**: Real-time setting adjustments per model

### 🎨 Interface Customization
Customize the experience in `config.py`:
- Modern theme with professional styling
- Adjustable chat height (700px default)
- Example prompts for quick start
- Comprehensive error messaging
- Debug mode for development

### 🔒 Security & Performance
- **Input Validation**: Comprehensive sanitization and validation
- **Rate Limiting**: Built-in protection against API abuse  
- **Async Architecture**: Non-blocking operations with event loops
- **Error Recovery**: Graceful handling of API failures
- **Memory Management**: Efficient conversation history handling

## 🔧 Advanced Usage & Development

### 🌐 Custom API Endpoints
Configure custom endpoints for enterprise deployments:

```env
OPENAI_API_URL=https://your-enterprise-openai.com/v1
GEMINI_API_URL=https://your-custom-gemini.com/v1beta
ANTHROPIC_API_URL=https://your-claude-proxy.com
```

### 🔌 Adding New Models
1. Update `AVAILABLE_MODELS` in `config.py`
2. Define model capabilities in `MODEL_CAPABILITIES`
3. Implement API client in `api_clients.py` if needed
4. Test with the development notebook (`main.ipynb`)

### ⚡ Custom Streaming Implementation
The async streaming system uses generators for real-time responses:
- Modify `stream_chat` methods in `api_clients.py`
- Customize response formatting in `utils.py`
- Add custom error handling and recovery logic

### 🧪 Development & Testing
- Use `main.ipynb` for interactive development
- Built-in debug logging with configurable levels
- Comprehensive error tracking and reporting
- Performance monitoring and optimization tools

## 🛡️ Enterprise-Grade Security

- ✅ **Input Sanitization**: Comprehensive validation for all user inputs
- ✅ **File Security**: Type validation, size limits, and malware protection
- ✅ **API Key Protection**: Secure environment variable handling
- ✅ **Rate Limiting**: Built-in throttling and abuse prevention
- ✅ **Content Filtering**: Optional content moderation capabilities
- ✅ **Error Isolation**: Secure error handling without data leakage
- ✅ **Session Management**: Secure conversation state handling

## 🐛 Troubleshooting & Support

### Common Issues & Solutions

**❌ API Key Errors**
```
Error: API key not found or invalid
```
**✅ Solutions:**
- Verify `.env` file exists in project root
- Check API key format and permissions
- Ensure your subscription includes the requested models
- Test API keys with provider's official tools

**❌ Model Availability Issues**
```
Error: Model not supported or unavailable
```
**✅ Solutions:**
- Verify model name matches `config.py` exactly
- Check your API subscription tier and model access
- Some models require special access or waitlist approval
- Try alternative models from the same provider

**❌ Image Upload Problems**
```
Error: Image processing failed
```
**✅ Solutions:**
- Supported formats: JPEG, PNG, GIF, WebP, BMP only
- Maximum file size: 10MB per image
- Try converting to JPEG for best compatibility
- Check image isn't corrupted or password-protected

**❌ Streaming & Connection Issues**
```
Streaming error: Connection timeout or interrupted
```
**✅ Solutions:**
- Verify stable internet connection
- Check API endpoint accessibility
- Reduce `max_tokens` setting for faster responses
- Try switching to a different model or provider

### 🔧 Debug Mode
Enable comprehensive debugging:
```env
GRADIO_DEBUG=true
LOG_LEVEL=DEBUG
```

### 📊 Performance Optimization
- Use lighter models (GPT-4.1 Nano, Gemini Flash Lite) for speed
- Reduce context window for faster processing
- Enable async processing for better responsiveness
- Monitor API usage and rate limits

## � API Documentation & Resources

### 🔗 Official Documentation
- **OpenAI**: [Platform Docs](https://platform.openai.com/docs) | [GPT-4.1 Guide](https://platform.openai.com/docs/models/gpt-4-1)
- **Google Gemini**: [AI Studio](https://ai.google.dev/docs) | [Gemini 2.5 Pro](https://ai.google.dev/gemini-api/docs/models/gemini-pro)
- **Anthropic Claude**: [API Docs](https://docs.anthropic.com) | [Claude 4.0 Features](https://docs.anthropic.com/claude/docs/models-overview)

### 💡 Best Practices
- **Model Selection**: Choose based on use case (speed vs capability)
- **Prompt Engineering**: Use specific, detailed prompts for better results
- **Context Management**: Keep conversations focused for optimal performance
- **Rate Limiting**: Implement proper throttling for production use
- **Cost Optimization**: Monitor token usage and choose cost-effective models

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 🆘 Support

If you encounter issues or have questions:

1. Check the troubleshooting section above
2. Review the API documentation for your provider
3. Create an issue on the project repository
4. Check that your API keys have the necessary permissions

## 🔄 Roadmap & Future Updates

The application is designed for continuous evolution. Upcoming features include:

### 🚀 **Q2 2025**
- 🎤 **Voice Integration**: Speech-to-text and text-to-speech capabilities
- 📄 **Document Analysis**: PDF, DOCX, and TXT file processing
- 🌐 **Web Search**: Real-time web information retrieval
- 💾 **Export Features**: Conversation export in multiple formats

### 🚀 **Q3 2025**
- 🧠 **Custom Model Training**: Fine-tuning integration for specialized tasks
- 🤖 **Agent Workflows**: Multi-step reasoning and task automation
- 📊 **Analytics Dashboard**: Usage statistics and performance metrics
- 🌍 **Multi-language Support**: International language interface

### 🚀 **Q4 2025**
- 🎨 **Advanced UI**: Customizable themes and layouts
- 🔌 **Plugin System**: Third-party integrations and extensions
- ☁️ **Cloud Deployment**: Scalable cloud infrastructure options
- 🛡️ **Enterprise Features**: Advanced security and compliance tools

---

**💡 Note**: This application requires valid API keys from the respective AI providers. Usage costs vary by provider and model. Please refer to each provider's current pricing documentation for detailed cost information.