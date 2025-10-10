# 🤖 Next-Generation Multimodal AI Chatbot

A production-grade Gradio-based chatbot application featuring the latest 2025 AI models from OpenAI, Google Gemini, and Anthropic Claude. Built with advanced multimodal capabilities, extended reasoning support, and professional prompt engineering.

## ✨ Key Features

- **🚀 Latest 2025 AI Models**: GPT-4.1 series, Gemini 2.5/2.0, Claude 4.x with Opus 4.1
- **🧠 Extended Thinking Mode**: Claude's advanced reasoning with configurable token budgets
- **📚 1M Context Window**: Beta support for Claude Sonnet 4/4.5 (requires usage tier 4)
- **🎯 Advanced Multimodal Support**: Text, image, and document processing with intelligent optimization
- **⚡ Real-time Streaming**: Async-first architecture with persistent event loops
- **🎛️ Dynamic Model Configuration**: Auto-adjusting temperature and token limits per model
- **🖼️ Intelligent Image Processing**: Auto-optimization, format validation, and base64 encoding
- **💬 Smart Conversation Management**: Persistent history with malformed content detection
- **📝 Expert Prompt Templates**: 12+ professional templates including AI Engineer, Prompt Engineer, Alpha Researcher
- **🎨 Modern Typography**: Custom fonts optimized for readability (Source Serif Pro + Inter)
- **⚙️ Robust Architecture**: Comprehensive error handling and validation

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- API keys from at least one provider (OpenAI, Google, or Anthropic)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/TGtongga/multimodal-ai-chatbot.git
cd multimodal-ai-chatbot

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Copy the `.env.example` file to `.env` and configure your API credentials:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

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

**Note**: You only need API keys for the providers you plan to use.

### 3. Run the Application

```bash
python main.py
```

The application will launch at `http://localhost:7860`

---

## 🏗️ Project Architecture

```
chatbot/
├── main.py              # 🚀 Gradio interface with async streaming
├── main.ipynb          # 📓 Development/testing notebook
├── api_clients.py      # 🔌 Async API clients (OpenAI, Gemini, Anthropic)
├── config.py           # ⚙️ Model configurations, templates, defaults
├── utils.py            # 🛠️ Image processing and validation utilities
├── requirements.txt    # 📦 Python dependencies
├── .env.example       # 🔐 Environment variables template
└── README.md          # 📖 Documentation
```

---

## 🎯 Supported AI Models (2025)

### 🧠 OpenAI Models (GPT-4.1 Series)

| Model | Context | Max Output | Vision | Best For |
|-------|---------|------------|--------|----------|
| **GPT-4.1** | 1M tokens | 32K | ✅ | Flagship reasoning and analysis |
| **GPT-4.1 Mini** | 1M tokens | 32K | ✅ | Fast, efficient multimodal tasks |
| **GPT-4.1 Nano** | 1M tokens | 32K | ✅ | High-speed, cost-effective processing |
| **GPT-4o** | 128K tokens | 16K | ✅ | Advanced multimodal capabilities |
| **GPT-4o Mini** | 128K tokens | 16K | ✅ | Cost-efficient multimodal |

### 🔮 Google Gemini Models (2.5 & 2.0 Series)

| Model | Context | Max Output | Vision | Best For |
|-------|---------|------------|--------|----------|
| **Gemini 2.5 Pro** | 1M tokens | 64K | ✅ | State-of-the-art reasoning |
| **Gemini 2.5 Flash** | 1M tokens | 64K | ✅ | Best price-performance with thinking |
| **Gemini 2.5 Flash Lite** | 1M tokens | 64K | ✅ | High-throughput, cost-efficient |
| **Gemini 2.0 Flash** | 1M tokens | 8K | ✅ | Next-gen features, balanced performance |
| **Gemini 2.0 Flash Lite** | 1M tokens | 8K | ✅ | Low-latency, cost-optimized |

### 🎭 Anthropic Claude Models (4.x Series)

| Model | Context | Max Output | Vision | Extended Thinking | Best For |
|-------|---------|------------|--------|-------------------|----------|
| **Claude Sonnet 4.5** | 200K (1M β) | 64K | ✅ | ✅ | Smartest model, balanced performance |
| **Claude Sonnet 4.0** | 200K (1M β) | 64K | ✅ | ✅ | High-performance reasoning |
| **Claude Opus 4.1** | 200K | 32K | ✅ | ✅ | Most capable, best for coding |
| **Claude 3.7 Sonnet** | 200K | 64K | ❌ | ✅ | Extended thinking, text-only |

> **Note**: 1M context (β) requires usage tier 4 and has premium pricing. Extended thinking available for all Claude 4.x models.

---

## 🧠 Extended Thinking Mode (Anthropic)

Claude models support **Extended Thinking**, an advanced reasoning capability that shows the model's internal thought process.

### Configuration

- **Enable Thinking**: Toggle in Model Settings
- **Thinking Budget**: 1,024 - 64,000 tokens (default: 10,000)
- **Temperature**: Automatically set to 1.0 when thinking is enabled
- **Supported Models**: All Claude 4.x models

### Output Format

When enabled, responses display:
```
🧠 Thinking Process:
[Model's internal reasoning chain]

---

📤 Response:
[Final answer based on reasoning]
```

### Use Cases

- Complex problem-solving requiring multi-step reasoning
- Code debugging and architecture decisions
- Mathematical proofs and scientific analysis
- Strategic planning and decision-making

---

## 🖼️ Advanced Multimodal Support

### Supported File Types

**Images**: JPEG, PNG, GIF, WebP, BMP  
**Documents**: TXT, MD, PY, JS, HTML, CSS, JSON, XML, CSV  
**Office**: PDF, DOC, DOCX, XLS, XLSX

### Smart Processing Features

- ✅ **Auto-optimization**: Images resized to 1024px max dimension
- ✅ **Intelligent encoding**: Base64 with media type detection
- ✅ **Security validation**: File size limits (10MB), format verification
- ✅ **Batch processing**: Multiple files per message (max 10)
- ✅ **Document parsing**: Text extraction with encoding detection
- ✅ **Error recovery**: Graceful handling with detailed feedback

### Usage

1. Click the attachment icon in the chat input
2. Select one or more files (images or documents)
3. Add your text prompt
4. Send to process with the selected model

**Note**: Not all models support vision. Check the table above for vision capabilities.

---

## 📝 Professional Prompt Templates

Choose from 12 expertly crafted system prompts designed for specific use cases:

### 🎨 Creative Writer
Advanced storytelling, poetry, and narrative development with genre expertise and literary analysis.

### 💻 Technical Expert
Senior-level programming mentor covering full-stack development, DevOps, ML/AI, cybersecurity, and system architecture.

### 🔬 Research Assistant
Academic research specialist with literature review, methodology design, statistical analysis, and citation management.

### 💼 Business Consultant
Strategic advisor for business strategy, market analysis, financial planning, operations optimization, and digital transformation.

### 🎓 Educational Tutor
Adaptive learning specialist using multiple teaching methods, Socratic questioning, and personalized instruction across all subjects.

### 📊 Data Analyst
Expert data scientist with statistical analysis, machine learning, visualization, and business intelligence capabilities.

### 🤖 AI Engineer
Elite AI systems engineer specializing in production ML, agent architectures, LLM optimization, and scalable infrastructure.

### 🎯 Prompt Engineer
World-leading prompt optimization specialist with expertise in advanced techniques, evaluation frameworks, and production implementation.

### 📈 Alpha Researcher
Senior quantitative researcher specializing in systematic alpha generation, market microstructure, and portfolio construction.

### 🧮 Machine Learning Engineer
Senior ML engineer specializing in deep learning architectures, production systems, and MLOps best practices.

### 🧪 Scientific Advisor
Scientific expert committed to empirical rigor across physics, chemistry, biology, mathematics, and interdisciplinary research.

### ✏️ Custom
Write your own system prompt for specialized use cases.

**Access**: Select from the "System Prompt Template" dropdown in Model Settings.

---

## ⚙️ Advanced Configuration

### 🎛️ Model Settings

All settings are configurable per conversation and adjust dynamically based on the selected model:

| Setting | Range | Description |
|---------|-------|-------------|
| **Temperature** | 0.0 - 2.0 | Controls response creativity and randomness |
| **Max Tokens** | 1 - 64K+ | Maximum response length (model-dependent) |
| **Extended Thinking** | Toggle | Enable Claude's reasoning mode (Anthropic only) |
| **Thinking Budget** | 1K - 64K | Token allocation for internal reasoning |
| **1M Context** | Toggle | Enable 1M token context window (Claude Sonnet 4/4.5 β) |
| **System Prompt** | Template/Custom | Choose pre-built template or write custom prompt |

### 🔄 Dynamic Defaults

The application automatically adjusts settings when you switch models:

- **Temperature**: Optimized per model (e.g., 0.3 for Gemini 2.5 Pro, 1.0 for Claude Opus)
- **Max Tokens**: Set to each model's maximum output capability
- **Thinking Controls**: Shown/hidden based on model support

### 🎨 Interface Customization

Typography optimized for readability:
- **Content**: Source Serif Pro 14px (messages, markdown)
- **UI Elements**: Inter sans-serif (controls, headers)
- **Code Blocks**: SF Mono/Monaco 10px (optimized for density)
- **Chat Height**: 700px (resizable)
- **Dark Mode**: Full support with automatic theme detection

---

## 🔧 Development & Customization

### 🌐 Custom API Endpoints

For enterprise deployments or proxy configurations:

```env
OPENAI_API_URL=https://your-enterprise-openai.com/v1
GEMINI_API_URL=https://your-custom-gemini.com/v1beta
ANTHROPIC_API_URL=https://your-claude-proxy.com
```

### 🔌 Adding New Models

1. **Update `config.py`**:
   ```python
   AVAILABLE_MODELS = {
       'provider': ['new-model-name']
   }
   
   MODEL_CAPABILITIES = {
       'new-model-name': {
           'text': True,
           'vision': True,
           'max_tokens': 32000,
           'context_window': 128000
       }
   }
   
   MODEL_DEFAULTS = {
       'new-model-name': {'temperature': 0.7, 'max_tokens': 32000}
   }
   ```

2. **Update API client** in `api_clients.py` if using a new provider

3. **Test** with `main.ipynb` for interactive development

### ⚡ Custom Streaming Implementation

All API clients use async generators for real-time streaming:

```python
async def stream_chat(
    self,
    messages: List[Dict[str, Any]],
    model: str,
    **kwargs
) -> AsyncGenerator[str, None]:
    # Your implementation
    accumulated_content = ""
    async for chunk in response:
        accumulated_content += chunk
        yield accumulated_content
```

### 🧪 Development Tools

- **Interactive Development**: Use `main.ipynb` for testing
- **Debug Logging**: Set `GRADIO_DEBUG=true` and `LOG_LEVEL=DEBUG` in `.env`
- **Performance Monitoring**: Built-in logging for response times and errors
- **Error Tracking**: Comprehensive exception handling with detailed messages

---

## 🛡️ Security & Best Practices

### 🔒 Security Features

- ✅ **Input Sanitization**: Comprehensive validation for all user inputs
- ✅ **File Security**: Type validation, size limits (10MB), format verification
- ✅ **API Key Protection**: Secure environment variable handling
- ✅ **Content Filtering**: HTML sanitization, dangerous tag removal
- ✅ **Error Isolation**: Secure error handling without data leakage
- ✅ **Session Management**: Malformed content detection and cleanup

### ⚠️ Important Considerations

**1M Context Window (Beta)**
- Requires Anthropic usage tier 4
- Premium pricing applies (5x standard rate)
- Currently available for Claude Sonnet 4.0 and 4.5

**Extended Thinking Mode**
- Automatically sets temperature to 1.0 (required by API)
- Consumes tokens from thinking budget separate from response
- Best for complex reasoning tasks, not simple queries

**Rate Limits**
- Varies by provider and subscription tier
- Implement exponential backoff for production use
- Monitor API usage through provider dashboards

---

## 🐛 Troubleshooting

### Common Issues

#### ❌ API Key Errors
```
Error: API key not found or invalid
```
**Solutions:**
- Verify `.env` file exists in project root
- Check API key format (no quotes or spaces)
- Ensure subscription includes requested models
- Test keys with provider's official tools

#### ❌ Model Not Available
```
Error: Model not supported or unavailable
```
**Solutions:**
- Verify model name matches `config.py` exactly
- Check API subscription tier and model access
- Some models require waitlist approval or special access
- Try alternative models from the same provider

#### ❌ Image Processing Failed
```
Error: Image processing failed
```
**Solutions:**
- Supported formats: JPEG, PNG, GIF, WebP, BMP only
- Maximum file size: 10MB per image
- Try converting to JPEG for best compatibility
- Ensure image isn't corrupted or password-protected

#### ❌ Streaming Errors
```
Streaming error: Connection timeout or interrupted
```
**Solutions:**
- Verify stable internet connection
- Check API endpoint accessibility (try `curl` or Postman)
- Reduce `max_tokens` setting for faster responses
- Try switching to a different model or provider

#### ❌ Extended Thinking Issues
```
Error: Extended thinking not available
```
**Solutions:**
- Only supported for Claude 4.x models
- Ensure `anthropic-beta` header is set correctly
- Temperature automatically set to 1.0 (cannot be overridden)
- Check Anthropic API documentation for latest requirements

### 🔧 Debug Mode

Enable comprehensive debugging:

```env
GRADIO_DEBUG=true
LOG_LEVEL=DEBUG
```

Check `chatbot.log` for detailed error traces and API interactions.

---

## 📊 Performance Optimization

### Speed Optimization
- **Fastest Models**: GPT-4.1 Nano, Gemini 2.0/2.5 Flash Lite
- **Reduce Context**: Keep conversations focused, limit history length
- **Lower Max Tokens**: Set to actual needs, not maximum
- **Async Processing**: Built-in for non-blocking operations

### Cost Optimization
- **Cost-Effective Models**: GPT-4o Mini, Gemini Flash Lite, Claude Sonnet
- **Monitor Token Usage**: Track input/output tokens via API responses
- **Optimize Prompts**: Concise prompts reduce costs significantly
- **Disable Thinking**: For simple queries, save on thinking token budget

### Quality Optimization
- **Best Models**: Claude Opus 4.1, GPT-4.1, Gemini 2.5 Pro
- **Extended Thinking**: For complex reasoning and problem-solving
- **1M Context**: For long document analysis (Claude Sonnet 4/4.5)
- **Custom Prompts**: Specialized templates for specific tasks

---

## 📚 API Documentation

### Provider Resources

- **OpenAI**: [Platform Docs](https://platform.openai.com/docs) | [GPT-4.1 Guide](https://platform.openai.com/docs/models/gpt-4-1)
- **Google Gemini**: [AI Studio](https://ai.google.dev/docs) | [Gemini API](https://ai.google.dev/gemini-api/docs)
- **Anthropic Claude**: [API Docs](https://docs.anthropic.com) | [Claude Models](https://docs.anthropic.com/claude/docs/models-overview)

### Best Practices

**Model Selection**
- Choose based on use case: speed vs capability vs cost
- Test multiple models for your specific task
- Monitor performance metrics (latency, quality, cost)

**Prompt Engineering**
- Be specific and detailed in instructions
- Provide examples for complex tasks
- Use system prompts for consistent behavior
- Iterate based on results

**Context Management**
- Keep conversations focused and relevant
- Clear history when switching topics
- Use system prompts to set context
- Monitor token usage to avoid cutoffs

**Production Deployment**
- Implement proper rate limiting and retry logic
- Add monitoring and alerting
- Version control prompts and configurations
- Test thoroughly before deployment

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add some amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Development Guidelines

- Follow existing code style and conventions
- Add tests for new features
- Update documentation for API changes
- Test with multiple models and edge cases
- Include error handling and validation

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) file for details.

---

## 🆘 Support

If you encounter issues:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review [API Documentation](#-api-documentation)
3. Check existing [GitHub Issues](https://github.com/TGtongga/multimodal-ai-chatbot/issues)
4. Create a new issue with:
   - Error message and logs
   - Steps to reproduce
   - Environment details (Python version, OS)
   - API provider and model used

---

## 🔄 Changelog

### Latest Updates (2025)

**New Models**
- ✨ Added GPT-4.1 series (4.1, 4.1 Mini, 4.1 Nano)
- ✨ Added Gemini 2.5/2.0 series (Pro, Flash, Flash Lite)
- ✨ Added Claude 4.x series (Opus 4.1, Sonnet 4.5/4.0, 3.7 Sonnet)
- 🗑️ Removed deprecated older models

**Features**
- 🧠 Extended Thinking Mode with configurable token budgets
- 📚 1M Context Window support (Claude Sonnet 4/4.5 beta)
- 🎯 Dynamic model defaults (auto-adjusting settings)
- 📝 12+ professional prompt templates
- 🎨 Custom typography (Source Serif Pro + Inter)
- 🔍 Malformed content detection and cleanup

**Improvements**
- ⚡ Persistent event loop for better async performance
- 🖼️ Enhanced image processing with media type detection
- 🛡️ Improved security and validation
- 📊 Better error handling and user feedback
- 🎛️ More granular configuration options

---

## 💡 Usage Tips

### Getting the Best Results

**For Coding Tasks**
- Use Claude Opus 4.1 or GPT-4.1 with "Technical Expert" or "AI Engineer" template
- Enable Extended Thinking for complex architecture decisions
- Provide context about your tech stack and constraints

**For Creative Writing**
- Use GPT-4.1 or Claude Sonnet 4.5 with "Creative Writer" template
- Adjust temperature higher (0.8-1.0) for more creativity
- Provide examples of the style you want

**For Research & Analysis**
- Use Gemini 2.5 Pro or Claude Opus 4.1 with "Research Assistant" template
- Enable Extended Thinking for complex analysis
- Upload relevant documents for context

**For Business Strategy**
- Use Claude Sonnet 4.5 or GPT-4.1 with "Business Consultant" template
- Provide market data and business context
- Ask for structured outputs (frameworks, action plans)

---

**💡 Note**: This application requires valid API keys from the respective providers. Usage costs vary by provider, model, and features used (e.g., Extended Thinking, 1M Context). Please refer to each provider's pricing documentation for current rates.

**🔗 Repository**: [https://github.com/TGtongga/multimodal-ai-chatbot](https://github.com/TGtongga/multimodal-ai-chatbot)

---

*Built with ❤️ using Gradio, OpenAI, Google Gemini, and Anthropic Claude APIs*