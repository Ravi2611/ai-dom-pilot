# AI Browser Automation Backend

A Python FastAPI backend that converts natural language commands into Playwright automation code.

## Features

- 🤖 Natural language to Playwright code conversion using Groq AI
- 🌐 Real-time browser automation with Chromium
- 📸 Automatic screenshot capture after each step
- 💾 SQLite database for command history
- 🔄 Background task execution
- 🛡️ Error handling and resilient execution
- 📡 REST API with automatic documentation

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

5. **Set your Groq API key**:
   ```bash
   export GROQ_API_KEY="your-groq-api-key-here"
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
curl -X POST "http://localhost:8000/api/automation/command" \
     -H "Content-Type: application/json" \
     -d '{"command": "go to google.com"}'
```

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