# Enhanced AI Browser Automation Backend

This backend now supports multiple AI providers, smart retry systems, and vision-based automation for bulletproof browser automation.

## Features

- 🤖 **Multi-AI Provider Support**: Groq, OpenAI GPT-4V, Anthropic Claude with automatic fallback
- 🎯 **Smart Retry System**: Automatic alternative selector generation with fuzzy matching
- 👁️ **Vision-Based Automation**: Screenshot analysis for element detection and coordinate-based clicks
- 🌐 Real-time browser automation with enhanced DOM processing
- 📸 Automatic screenshot capture after each step
- 💾 SQLite database for command history
- 🔄 Background task execution with intelligent error recovery
- 🛡️ Bulletproof automation that rarely fails
- 📡 REST API with automatic documentation

## New Capabilities

### Multi-Model AI Support
- **Groq**: Fast and cost-effective (primary)
- **OpenAI**: GPT-4V with vision capabilities  
- **Anthropic**: Claude with vision support
- Automatic fallback chain when one provider fails

### Smart Retry System
- Multiple selector strategies (exact, partial, fuzzy, CSS, XPath)
- Element traversal and hierarchy analysis
- Coordinate-based clicking as last resort
- Intelligent waiting strategies

### Vision-Based Automation
- AI analysis of screenshots to find elements
- Visual verification of successful actions
- Enhanced selector generation based on visual context
- Works even when DOM parsing fails

## Quick Setup

1. **Install Python 3.8+** (if not already installed)

2. **Clone/Download the project**

3. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

4. **Run the setup script**:
   ```bash
   python setup.py
   ```

5. **Set your AI API keys**:
   ```bash
   # Required
   export GROQ_API_KEY="your-groq-api-key-here"
   
   # Optional (for enhanced capabilities)
   export OPENAI_API_KEY="your-openai-api-key-here"
   export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
   ```

6. **Start the backend**:
   ```bash
   python main.py
   ```

The backend will be running at `http://localhost:8000`

## API Endpoints

- `POST /api/automation/command` - Execute automation command
- `GET /api/automation/commands` - Get command history
- `GET /api/automation/command/{id}` - Get specific command
- `GET /api/automation/browser/current-url` - Get current browser URL
- `GET /api/automation/browser/dom` - Get current page DOM
- `GET /docs` - Interactive API documentation

## Example Usage

```bash
# Basic command
curl -X POST "http://localhost:8000/api/automation/command" \
     -H "Content-Type: application/json" \
     -d '{"command": "go to google.com"}'

# Complex UI interaction (now works reliably!)
curl -X POST "http://localhost:8000/api/automation/command" \
     -H "Content-Type: application/json" \
     -d '{"command": "Click on Cash / UPI on Delivery"}'
```

### Smart Automation Examples
Commands that now work reliably with the enhanced system:
- "Click on the red button"
- "Select the cash payment option" 
- "Fill the email field with test@example.com"
- "Click on submit button"
- "Choose the dropdown option 'Premium'"

## Getting Groq API Key

1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Go to API Keys section
4. Create a new API key
5. Copy and use it in your environment

## Troubleshooting

- **Playwright browser not found**: Run `playwright install chromium`
- **Permission errors**: Try running with `sudo` on Linux/Mac
- **Port 8000 in use**: Change port in `main.py` or kill existing process
- **Groq API errors**: Check your API key and internet connection

## Development

- The backend uses SQLite database (`automation.db`)
- Screenshots are saved in `screenshots/` directory
- Logs are printed to console
- Auto-reload with: `uvicorn main:app --reload`